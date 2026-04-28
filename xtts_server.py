"""
OpenAI-compatible `/v1/audio/speech` for Coqui XTTS v2 voice cloning.

Reference voices live as raw WAV files in /workspace/xtts_voices/<name>.wav.
Drop a 5-15 s clean mono sample (any sample rate, ≥ 16 kHz recommended) there
and reference it via `voice: "<name>"` in the request.

Listens on 127.0.0.1:8002 — only LiteLLM on the same pod (loopback) reaches it.
The public surface is LiteLLM 4000 with the existing `VOXTRAL_KEY_*` allowlist.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

VOICES_DIR = Path(os.environ.get("XTTS_VOICES_DIR", "/workspace/xtts_voices"))
VOICES_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("TTS_HOME", "/workspace/xtts_models")

# XTTS v2 supported languages
SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xtts")

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model load
# ──────────────────────────────────────────────────────────────────────────────

_tts = None

def get_tts():
    global _tts
    if _tts is None:
        from TTS.api import TTS
        log.info("loading xtts_v2 (gpu=%s)...", torch.cuda.is_available())
        t0 = time.time()
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        log.info("xtts_v2 loaded in %.1fs", time.time() - t0)
    return _tts


# ──────────────────────────────────────────────────────────────────────────────
# HTTP API
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="XTTS v2 (OpenAI-compatible TTS with voice cloning)")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _tts is not None}


@app.get("/v1/voices")
def list_voices():
    return {"voices": sorted(p.stem for p in VOICES_DIR.glob("*.wav"))}


class SpeechReq(BaseModel):
    """OpenAI /v1/audio/speech, with `voice` pointing to a WAV in VOICES_DIR."""
    model: str = Field("xtts-clone", description="Cosmetic — only one model is served")
    input: str = Field(..., description="Text to synthesize (max ~250 chars per call recommended)")
    voice: str = Field(..., description="Name of a registered voice (file VOICES_DIR/<voice>.wav must exist)")
    language: str = Field("fr", description="ISO-639-1 (or 'zh-cn'); see SUPPORTED_LANGS")
    response_format: str = Field("wav", description="Only 'wav' is supported (XTTS native)")


@app.post("/v1/audio/speech")
def speech(req: SpeechReq):
    if req.language not in SUPPORTED_LANGS:
        raise HTTPException(400, f"unsupported language '{req.language}' — supported: {sorted(SUPPORTED_LANGS)}")
    if req.response_format.lower() != "wav":
        raise HTTPException(400, "only response_format='wav' is supported (XTTS v2 native output)")

    sample = VOICES_DIR / f"{req.voice}.wav"
    if not sample.exists():
        available = sorted(p.stem for p in VOICES_DIR.glob("*.wav"))
        raise HTTPException(
            404,
            f"voice '{req.voice}' not found. Drop a WAV at /workspace/xtts_voices/{req.voice}.wav "
            f"(available: {available})",
        )

    tts = get_tts()
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    try:
        t0 = time.time()
        tts.tts_to_file(
            text=req.input,
            speaker_wav=str(sample),
            language=req.language,
            file_path=out.name,
        )
        wav = Path(out.name).read_bytes()
        log.info("synth voice=%s lang=%s text_len=%d size=%dB in %.2fs",
                 req.voice, req.language, len(req.input), len(wav), time.time() - t0)
    except Exception as e:
        log.exception("synth failure")
        raise HTTPException(500, f"xtts synthesis failed: {e!s}")
    finally:
        Path(out.name).unlink(missing_ok=True)

    return Response(content=wav, media_type="audio/wav")


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Bind to loopback only — LiteLLM is the only reachable surface.
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
