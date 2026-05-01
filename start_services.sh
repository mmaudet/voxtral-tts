#!/bin/bash
# Starts vLLM-Omni × 2 (Voxtral on :8003, Qwen3-TTS on :8000) and the LiteLLM
# proxy (:4000) in background. Idempotent: skips a service that's already healthy.
#
# Both vLLM instances cap their --gpu-memory-utilization so they coexist on
# one GPU. NOTE: vllm-omni applies the cap per-stage (Stage-0 + Stage-1), so
# the *peak* allocation per model is roughly 2× the flag value. On a 48 GB
# card we use:
#   Voxtral  0.20  ≈ 9 GiB / stage  → ~18 GiB peak (model 7.8 + KV)
#   Qwen-TTS 0.15  ≈ 7 GiB / stage  → ~14 GiB peak
# Sequential boot (Voxtral first → healthy → then Qwen) further avoids the
# CUDA-graph-capture overlap that triggers OOM during init.

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
# start_vllm <label> <port> <model-path> <served-model-name> <gpu-mem-util> [extra-args…]
start_vllm() {
  local label="$1" port="$2" model_path="$3" served_name="$4" gpu_mem="$5"
  shift 5
  local log="/workspace/logs/vllm-${label}.log"

  if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
    echo "vLLM[${label}] already healthy on :${port} — skipping start"
    return 0
  fi
  echo "Starting vLLM[${label}] on :${port}..."
  # Bind to loopback only — the public RunPod proxy URL on this port will 502
  # since CUDA-net can't reach 127.0.0.1. All inference must go through LiteLLM.
  nohup vllm serve "$model_path" \
    --omni \
    --port "$port" \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --served-model-name "$served_name" \
    --gpu-memory-utilization "$gpu_mem" \
    "$@" \
    > "$log" 2>&1 &
  echo "  vLLM[${label}] PID: $!"
}

# wait_health <label> <port> <iters> <sleep>  — polls /health until 200 or exhausted
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

# ── 1. boot vLLM instances sequentially ──────────────────────────────────────
# Sequential to avoid contention during CUDA graph capture (parallel boot
# triggered OOM with two 2-stage vllm-omni engines competing for the same
# GPU). Voxtral first because it has the bigger KV-cache footprint.

# Voxtral-4B-TTS-2603 (CC BY-NC) — best for EU languages with native voices
start_vllm voxtral 8003 \
  /workspace/models/Voxtral-4B-TTS-2603 \
  mistralai/Voxtral-4B-TTS-2603 \
  0.20 \
  --max-model-len 4096
wait_health voxtral 8003 180 5

# Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache 2.0) — best for ZH/JA/KO + commercial
start_vllm qwen 8000 \
  /workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  0.15
wait_health qwen 8000 180 5

# ── 2. LiteLLM ────────────────────────────────────────────────────────────────
if curl -sf http://localhost:4000/health/liveliness >/dev/null 2>&1 \
   || curl -sf http://localhost:4000/health >/dev/null 2>&1; then
  echo "LiteLLM already up on :4000 — skipping start"
else
  echo "Starting LiteLLM..."
  # Source the per-pod LiteLLM secrets file (master key + virtual keys).
  # Pushed onto the pod by restart-pod.sh from the local .voxtral.env.
  # auth.py lives in /workspace, so PYTHONPATH must include it for
  # `custom_auth: auth.user_api_key_auth` to resolve.
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
echo "  LiteLLM     → http://localhost:4000/v1/audio/speech (Bearer \$VOXTRAL_KEY_*)"
