#!/bin/bash
# Downloads Voxtral-4B-TTS-2603 to /workspace/models/. Idempotent.
# Requires HF_TOKEN in env.

set -euo pipefail
exec > >(tee -a /workspace/logs/download.log) 2>&1
echo "=== download start: $(date -u +%FT%TZ) ==="

# RunPod sets pod env vars on container PID 1 only; login shells don't get them.
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' kv; do
    case "$kv" in
      HF_TOKEN=*|HF_HOME=*) export "$kv" ;;
    esac
  done < /proc/1/environ
fi

# shellcheck disable=SC1091
source /workspace/voxtral-env/bin/activate

MODEL_DIR=/workspace/models/Voxtral-4B-TTS-2603
mkdir -p "$MODEL_DIR"

# `huggingface-cli` was renamed to `hf` in recent huggingface_hub versions.
hf download mistralai/Voxtral-4B-TTS-2603 \
  --local-dir "$MODEL_DIR" \
  --token "$HF_TOKEN" \
  --format quiet

echo "=== model layout ==="
ls -lh "$MODEL_DIR"
echo "=== voice_embedding count ==="
ls "$MODEL_DIR/voice_embedding/" | wc -l

echo "=== DOWNLOAD OK: $(date -u +%FT%TZ) ==="
