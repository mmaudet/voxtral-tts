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

MODEL_DIR=/workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
mkdir -p "$MODEL_DIR"

# Qwen3-TTS-12Hz-1.7B-CustomVoice — Apache 2.0, replaces Voxtral.
# `huggingface-cli` was renamed to `hf` in recent huggingface_hub versions.
# (huggingface_hub 0.x — pinned <1.0 elsewhere — has no `--format` flag, so
# we just let it print progress to the log.)
hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir "$MODEL_DIR" \
  --token "$HF_TOKEN"

echo "=== model layout ==="
ls -lh "$MODEL_DIR"
echo "=== files (top by size) ==="
ls -lhS "$MODEL_DIR/" | head -10

echo "=== DOWNLOAD OK: $(date -u +%FT%TZ) ==="
