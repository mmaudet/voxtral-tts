#!/bin/bash
# Default fast path: only Qwen3-TTS-Base for voice cloning + qwen_clone_proxy
# + LiteLLM. The 3-model setup (Voxtral + Qwen-CV + Qwen-Base) is preserved
# below as commented-out blocks; uncomment to re-enable.
#
# With a single Qwen-Base instance the YAML (qwen3_tts_batch.yaml installed
# by install_voxtral.sh) configures Stage 0 max_num_seqs=4, async_scheduling,
# CUDA graphs, gpu_memory_utilization 0.30/0.20 — ~3× the throughput of the
# stock single-stream config. Pair with concurrency=4 client-side and an SSH
# tunnel to bypass Cloudflare's 100s ceiling on /v1/audio/speech.

set -euo pipefail
mkdir -p /workspace/logs

# RunPod pod env is on PID 1 only; copy what we need.
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

# ── 1. boot active models ─────────────────────────────────────────────────────

# (DEFAULT: only Qwen-Base for voice cloning. Uncomment Voxtral / Qwen-CV
#  blocks below to re-enable those aliases.)

# # Voxtral-4B-TTS-2603 (CC BY-NC) — 20 native EU voices for non-commercial use.
# start_vllm voxtral 8003 \
#   /workspace/models/Voxtral-4B-TTS-2603 \
#   mistralai/Voxtral-4B-TTS-2603 \
#   --max-model-len 4096
# wait_health voxtral 8003 180 5

# # Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache 2.0) — 9 preset voices, best ZH/JA/KO.
# start_vllm qwen 8000 \
#   /workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
#   Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
# wait_health qwen 8000 180 5

# Qwen3-TTS-12Hz-1.7B-Base (Apache 2.0) — voice cloning via ref_audio + ref_text.
# `--allowed-local-media-path` is required so the model can read the
# reference audios at /workspace/qwen_voices/<id>.mp3 (file:// URIs).
start_vllm qwen-clone 8004 \
  /workspace/models/Qwen3-TTS-12Hz-1.7B-Base \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --allowed-local-media-path /workspace/qwen_voices
wait_health qwen-clone 8004 180 5

# ── 2. qwen_clone_proxy (FastAPI) ─────────────────────────────────────────────
# Translates `voice: "<manifest-key>"` into Qwen-Base's task_type=Base /
# ref_audio / ref_text combo (LiteLLM strips those non-OpenAI fields).
if [ -f /workspace/qwen_clone_proxy.py ]; then
  if curl -sf http://127.0.0.1:8005/health >/dev/null 2>&1; then
    echo "qwen_clone_proxy already up on :8005 — skipping start"
  else
    echo "Starting qwen_clone_proxy..."
    nohup bash -c '
      source /workspace/voxtral-env/bin/activate
      exec python /workspace/qwen_clone_proxy.py
    ' > /workspace/logs/qwen-clone-proxy.log 2>&1 &
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

# ── 3. LiteLLM ────────────────────────────────────────────────────────────────
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
echo "  qwen-clone  → http://localhost:8004/v1 (loopback) via :8005 proxy"
echo "  LiteLLM     → http://localhost:4000/v1/audio/speech (Bearer \$VOXTRAL_KEY_*)"
echo
echo "Long audios (30-45 s) exceed Cloudflare's 100s timeout on the public"
echo "*.proxy.runpod.net URL. For batch generation, use ./tunnel.sh on your"
echo "Mac and point clients at http://localhost:14000/v1/audio/speech."
