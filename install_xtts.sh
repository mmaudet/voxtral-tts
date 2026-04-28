#!/bin/bash
# Install Coqui XTTS v2 in an isolated venv next to the Voxtral one.
# Voice cloning model — license CPML (non-commercial).
#
# Coqui is dead since 2024; we use the actively-maintained `coqui-tts` package
# from the idiap fork. PyTorch is pinned alongside to keep the cuda runtime
# coherent with what's already on the pod.

set -euo pipefail
exec > >(tee -a /workspace/logs/xtts_install.log) 2>&1
echo "=== xtts install start: $(date -u +%FT%TZ) ==="

mkdir -p /workspace/{xtts_models,xtts_voices,logs}

if [ ! -d /workspace/xtts-env ]; then
  python3.10 -m venv /workspace/xtts-env
fi
# shellcheck disable=SC1091
source /workspace/xtts-env/bin/activate
python -V

echo "=== [1/3] pip + uv ==="
pip install --quiet --upgrade pip wheel
pip install --quiet uv

echo "=== [2/3] coqui-tts + torch + FastAPI server deps ==="
# Three pins matter here, learnt the hard way:
#  - coqui-tts 0.27.5 doesn't list torch as a hard dep (so users pick a
#    CUDA-matching wheel themselves) — without an explicit pull, first import
#    raises `PackageNotFoundError: No package metadata was found for torch`.
#  - The `[codec]` extra pulls torchcodec, mandatory for torch >= 2.9 audio IO.
#  - transformers must stay on the 4.x line. coqui-tts 0.27.5 imports
#    `from transformers.pytorch_utils import isin_mps_friendly` which doesn't
#    exist in transformers 5.x.
# Different venv from Voxtral, so no conflict with its torch 2.10 pin.
uv pip install \
  "coqui-tts[codec]" \
  "torch" "torchaudio" \
  "transformers>=4.46,<5" \
  "fastapi" "uvicorn[standard]" "pydantic" \
  "soundfile" "numpy<2"

echo "=== [3/3] verify imports + auto-accept XTTS license ==="
export COQUI_TOS_AGREED=1
export TTS_HOME=/workspace/xtts_models
python - <<'PY'
import os, importlib.metadata as md
print("coqui-tts:", md.version("coqui-tts"))
print("torch:    ", md.version("torch"))
print("fastapi:  ", md.version("fastapi"))
print("uvicorn:  ", md.version("uvicorn"))

import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))

# Trigger the XTTS v2 model download (~2 GB) into TTS_HOME so it's persisted
# on the volume and doesn't redownload on every pod restart.
print("\nDownloading XTTS v2 weights...")
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
print("XTTS v2 ready, languages:", getattr(tts, "languages", "?"))
PY

echo "=== XTTS INSTALL OK: $(date -u +%FT%TZ) ==="
