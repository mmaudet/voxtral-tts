#!/bin/bash
# Starts 3 vLLM-Omni instances + LiteLLM proxy, sequentially:
#   :8003  Voxtral-4B-TTS-2603         (CC BY-NC, native EU voices)
#   :8000  Qwen3-TTS-12Hz-1.7B-CustomVoice  (Apache 2.0, 9 presets)
#   :8004  Qwen3-TTS-12Hz-1.7B-Base    (Apache 2.0, voice cloning via ref_audio)
#   :4000  LiteLLM proxy with 3 model aliases (voxtral-tts, qwen-tts, qwen-clone)
#
# vLLM-Omni overrides the CLI --gpu-memory-utilization with hardcoded YAML
# caps; install_voxtral.sh patches the YAMLs down so 3 models fit on a 48 GB
# card (Voxtral 0.20+0.05 = 12 GiB, Qwen-CV 0.10+0.10 = 10 GiB,
# Qwen-Base 0.10+0.10 = 10 GiB → ~32 GiB total, ~16 GiB margin).

set -euo pipefail
mkdir -p /workspace/logs

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

# ── helper ────────────────────────────────────────────────────────────────────
start_vllm() {
  local label="$1" port="$2" model_path="$3" served_name="$4"
  shift 4
  local log="/workspace/logs/vllm-${label}.log"

  if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
    echo "vLLM[${label}] already healthy on :${port} — skipping start"
    return 0
  fi
  echo "Starting vLLM[${label}] on :${port}..."
  nohup vllm serve "$model_path" \
    --omni \
    --port "$port" \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --served-model-name "$served_name" \
    "$@" \
    > "$log" 2>&1 &
  echo "  vLLM[${label}] PID: $!"
}

wait_health() {
  local label="$1" port="$2" iters="$3" sleep_s="$4"
  echo "Waiting for vLLM[${label}] /health (up to $((iters*sleep_s))s)..."
  for ((i=1; i<=iters; i++)); do
    if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "  vLLM[${label}] healthy after $((i*sleep_s))s"
      return 0
    fi
    sleep "$sleep_s"
  done
  echo "vLLM[${label}] did NOT become healthy"
  tail -60 "/workspace/logs/vllm-${label}.log"
  return 1
}

# ── 1. boot 3 vLLM instances sequentially ─────────────────────────────────────
# Voxtral first (largest KV footprint), then the two Qwen variants.

start_vllm voxtral 8003 \
  /workspace/models/Voxtral-4B-TTS-2603 \
  mistralai/Voxtral-4B-TTS-2603 \
  --max-model-len 4096
wait_health voxtral 8003 180 5

start_vllm qwen 8000 \
  /workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
wait_health qwen 8000 180 5

start_vllm qwen-clone 8004 \
  /workspace/models/Qwen3-TTS-12Hz-1.7B-Base \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --allowed-local-media-path /workspace/qwen_voices
wait_health qwen-clone 8004 180 5

# ── 1.5 qwen_clone_proxy ──────────────────────────────────────────────────────
# Tiny FastAPI proxy on :8005 that maps `voice: "<manifest_key>"` into
# Qwen-Base's task_type=Base / ref_audio / ref_text / language combo. LiteLLM's
# aspeech() strips those fields when it sees a `model:` alias, so we go
# through this proxy instead of pointing the alias straight at :8004.
if [ -f /workspace/qwen_clone_proxy.py ]; then
  if curl -sf http://127.0.0.1:8005/health >/dev/null 2>&1; then
    echo "qwen_clone_proxy already up on :8005 — skipping start"
  else
    echo "Starting qwen_clone_proxy..."
    nohup python3 /workspace/qwen_clone_proxy.py \
      > /workspace/logs/qwen-clone-proxy.log 2>&1 &
    echo "  qwen_clone_proxy PID: $!"
  fi
  for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8005/health >/dev/null 2>&1; then
      echo "  qwen_clone_proxy healthy after $((i*2))s"
      break
    fi
    sleep 2
  done
fi

# ── 2. LiteLLM ────────────────────────────────────────────────────────────────
if curl -sf http://localhost:4000/health/liveliness >/dev/null 2>&1 \
   || curl -sf http://localhost:4000/health >/dev/null 2>&1; then
  echo "LiteLLM already up on :4000 — skipping start"
else
  echo "Starting LiteLLM..."
  if [ -r /workspace/.litellm.env ]; then
    # shellcheck disable=SC1091
    source /workspace/.litellm.env
  else
    echo "WARN: /workspace/.litellm.env missing — LiteLLM will start with no virtual keys"
  fi
  export PYTHONPATH="/workspace:${PYTHONPATH:-}"
  nohup litellm --config /workspace/litellm_config.yaml \
    --port 4000 --host 0.0.0.0 \
    > /workspace/logs/litellm.log 2>&1 &
  echo "LiteLLM PID: $!"
fi

for i in $(seq 1 30); do
  if curl -sf http://localhost:4000/health/liveliness >/dev/null 2>&1 \
     || curl -sf http://localhost:4000/health >/dev/null 2>&1; then
    echo "LiteLLM up after $((i*2))s"
    break
  fi
  sleep 2
done

echo "=== SERVICES READY ==="
echo "  voxtral-tts → http://localhost:8003/v1 (loopback)"
echo "  qwen-tts    → http://localhost:8000/v1 (loopback)"
echo "  qwen-clone  → http://localhost:8004/v1 (loopback)"
echo "  LiteLLM     → http://localhost:4000/v1/audio/speech (Bearer \$VOXTRAL_KEY_*)"
