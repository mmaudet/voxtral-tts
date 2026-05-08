"""
OpenAI-compatible /v1/audio/speech proxy in front of vllm-omni's Qwen3-TTS-Base.

LiteLLM's `aspeech()` strips non-OpenAI fields (task_type, ref_audio, ref_text,
language) before forwarding upstream, which makes voice cloning impossible
through the standard `model: qwen-clone` alias. This proxy presents an
OpenAI-shaped /v1/audio/speech (so LiteLLM is happy) and translates the
`voice` field (a manifest key like `fr_grand_public`) into the
ref_audio + ref_text + language combo Qwen-Base actually needs.

Listens on 127.0.0.1:8005 — only LiteLLM (same container, loopback) reaches it.
LiteLLM does the auth via its custom_auth allowlist; we trust the upstream.

Manifest format (/workspace/qwen_voices/manifest.json):
  {
    "fr_grand_public": {
      "audio_path": "/workspace/qwen_voices/fr_grand_public.mp3",
      "language": "French",
      "ref_text": "Murmure. Le patrimoine français…"
    },
    ...
  }
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from threading import Lock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
MANIFEST_PATH = Path(os.environ.get(
    "QWEN_CLONE_MANIFEST", "/workspace/qwen_voices/manifest.json"))
UPSTREAM = os.environ.get(
    "QWEN_CLONE_UPSTREAM", "http://127.0.0.1:8004/v1/audio/speech")
UPSTREAM_MODEL = os.environ.get(
    "QWEN_CLONE_UPSTREAM_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwen_clone")


def load_manifest() -> dict[str, dict[str, str]]:
    if not MANIFEST_PATH.exists():
        log.warning("manifest %s missing — qwen-clone voices unavailable", MANIFEST_PATH)
        return {}
    return json.loads(MANIFEST_PATH.read_text())


_MANIFEST: dict[str, dict[str, str]] = load_manifest()
log.info("loaded %d voices from %s", len(_MANIFEST), MANIFEST_PATH)


# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Qwen3-TTS-Base voice-cloning proxy")


@app.get("/health")
def health():
    return {"status": "ok", "voices": len(_MANIFEST)}


@app.get("/v1/voices")
def list_voices():
    return {"voices": sorted(_MANIFEST.keys())}


class SpeechReq(BaseModel):
    """Subset of OpenAI /v1/audio/speech that LiteLLM forwards.

    `voice` here is a manifest key (e.g. "fr_grand_public"), NOT a Qwen
    preset. The proxy expands it into ref_audio + ref_text + language for
    Qwen-Base.
    """
    model: str = Field("qwen-clone")
    input: str
    voice: str
    response_format: str = "wav"
    speed: float | None = None  # accepted but ignored


@app.post("/v1/audio/speech")
def speech(req: SpeechReq):
    if req.voice not in _MANIFEST:
        avail = ", ".join(sorted(_MANIFEST))
        raise HTTPException(404, f"voice '{req.voice}' not in manifest. Available: {avail}")

    v = _MANIFEST[req.voice]
    upstream_payload: dict[str, Any] = {
        "model": UPSTREAM_MODEL,
        "input": req.input,
        # Qwen-Base requires `voice` to be present; it's ignored when
        # task_type=Base, but the API rejects requests without one.
        "voice": "Aiden",
        "task_type": "Base",
        "ref_audio": f"file://{v['audio_path']}",
        "ref_text": v["ref_text"],
        "language": v["language"],
        "response_format": req.response_format,
    }

    log.info("→ qwen-clone voice=%s lang=%s text_len=%d",
             req.voice, v["language"], len(req.input))

    request = urllib.request.Request(
        UPSTREAM,
        data=json.dumps(upstream_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "audio/wav")
    except urllib.error.HTTPError as e:
        body = e.read()
        log.error("upstream HTTPError %d: %s", e.code, body[:300])
        raise HTTPException(e.code, body.decode("utf-8", errors="replace"))
    except Exception as e:
        log.exception("upstream call failed")
        raise HTTPException(502, f"upstream error: {e!s}")

    log.info("← %d bytes in %.2fs", len(body), time.time() - t0)
    return Response(content=body, media_type=ctype)


# ──────────────────────────────────────────────────────────────────────────────
# /v1/audio/speech-with-alignment — qwen-clone synth + faster-whisper post-align
# ──────────────────────────────────────────────────────────────────────────────

# Lazy-loaded faster-whisper model. ~2.6 GB VRAM at fp16, kept resident across
# requests. Module-load cost (~5-15s) only paid on first alignment request.
_WHISPER_MODEL: Any = None
_WHISPER_LOCK = Lock()
_WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
_WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
_WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
_WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/workspace/models/whisper")

# Map Qwen language strings → ISO codes for Whisper.
_LANG_MAP = {
    "French": "fr", "English": "en", "Spanish": "es",
    "German": "de", "Italian": "it", "Dutch": "nl",
    "Japanese": "ja", "Chinese": "zh",
}


def _whisper_model() -> Any:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                from faster_whisper import WhisperModel
                t0 = time.time()
                _WHISPER_MODEL = WhisperModel(
                    _WHISPER_MODEL_NAME,
                    device=_WHISPER_DEVICE,
                    compute_type=_WHISPER_COMPUTE_TYPE,
                    download_root=_WHISPER_MODEL_DIR,
                )
                log.info("loaded faster-whisper %s in %.1fs", _WHISPER_MODEL_NAME, time.time() - t0)
    return _WHISPER_MODEL


def _align_with_whisper(audio_bytes: bytes, language_code: str) -> list[dict[str, Any]]:
    """Run faster-whisper word_timestamps=True on raw audio. Returns
    [{word, start, end}, ...]. Empty list on failure (degraded mode)."""
    model = _whisper_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language=language_code,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    out: list[dict[str, Any]] = []
    for seg in segments:
        words = getattr(seg, "words", None)
        if words is None:
            continue
        for w in words:
            out.append({
                "word": (w.word or "").strip(),
                "start": float(w.start),
                "end": float(w.end),
            })
    return out


@app.post("/v1/audio/speech-with-alignment")
def speech_with_alignment(req: SpeechReq):
    """Synth via qwen-clone (always WAV upstream so Whisper has raw PCM),
    then post-process with faster-whisper to get word-level timestamps.

    Response shape:
      {
        "audio_base64": "<base64-encoded WAV>",
        "audio_mime": "audio/wav",
        "alignments": [{"word": "L'église", "start": 0.0, "end": 0.42}, ...],
        "language": "French",
        "duration_s": 4.81,                # from last alignment.end (or null)
        "synth_ms": 1234,                  # qwen-clone wall-clock
        "align_ms": 5678                   # whisper wall-clock
      }

    On Whisper failure: returns audio + alignments=[] (degraded mode, client
    falls back to transcript-only mode).
    """
    if req.voice not in _MANIFEST:
        avail = ", ".join(sorted(_MANIFEST))
        raise HTTPException(404, f"voice '{req.voice}' not in manifest. Available: {avail}")

    v = _MANIFEST[req.voice]
    upstream_payload: dict[str, Any] = {
        "model": UPSTREAM_MODEL,
        "input": req.input,
        "voice": "Aiden",
        "task_type": "Base",
        "ref_audio": f"file://{v['audio_path']}",
        "ref_text": v["ref_text"],
        "language": v["language"],
        "response_format": "wav",  # WAV required for Whisper
    }

    log.info("→ qwen-clone+whisper voice=%s lang=%s text_len=%d",
             req.voice, v["language"], len(req.input))

    request = urllib.request.Request(
        UPSTREAM,
        data=json.dumps(upstream_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            audio_bytes = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read()
        log.error("upstream HTTPError %d: %s", e.code, body[:300])
        raise HTTPException(e.code, body.decode("utf-8", errors="replace"))
    except Exception as e:
        log.exception("upstream call failed")
        raise HTTPException(502, f"upstream error: {e!s}")
    synth_ms = int((time.time() - t0) * 1000)

    lang_code = _LANG_MAP.get(v["language"], "fr")
    t1 = time.time()
    alignments: list[dict[str, Any]] = []
    try:
        alignments = _align_with_whisper(audio_bytes, lang_code)
    except Exception as e:
        log.exception("whisper alignment failed (degraded mode): %s", e)
    align_ms = int((time.time() - t1) * 1000)

    log.info("← audio=%dB synth=%dms align=%dms words=%d",
             len(audio_bytes), synth_ms, align_ms, len(alignments))

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "audio_mime": "audio/wav",
        "alignments": alignments,
        "language": v["language"],
        "duration_s": alignments[-1]["end"] if alignments else None,
        "synth_ms": synth_ms,
        "align_ms": align_ms,
    }


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="info")
