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

import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
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
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="info")
