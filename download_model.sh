#!/bin/bash
# Downloads the 3 TTS models used by start_services.sh into /workspace/models.
# Idempotent — `hf download` is a no-op if files already exist.
# Requires HF_TOKEN exported (sourced from /proc/1/environ if not in shell env).

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
source /workspace/voice-factory-env/bin/activate

mkdir -p /workspace/models

pull() {
  local repo="$1" local_dir="$2"
  echo "→ $repo → $local_dir"
  mkdir -p "$local_dir"
  hf download "$repo" --local-dir "$local_dir" --token "$HF_TOKEN"
}

# Voxtral (~8 GB, CC BY-NC 4.0)
pull mistralai/Voxtral-4B-TTS-2603 /workspace/models/Voxtral-4B-TTS-2603

# Qwen3-TTS-CustomVoice (~4 GB, Apache 2.0, preset voices)
pull Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice /workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Qwen3-TTS-Base (~4 GB, Apache 2.0, voice cloning)
pull Qwen/Qwen3-TTS-12Hz-1.7B-Base /workspace/models/Qwen3-TTS-12Hz-1.7B-Base

echo "=== model layout ==="
for d in Voxtral-4B-TTS-2603 Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen3-TTS-12Hz-1.7B-Base; do
  echo "--- $d ---"
  du -sh "/workspace/models/$d" 2>/dev/null
done

echo "=== DOWNLOAD OK: $(date -u +%FT%TZ) ==="
