#!/bin/bash
# Starts vLLM (port 8000) and LiteLLM proxy (port 4000) in background.
# Idempotent: skips start if a healthy service is already running.

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

# --- vLLM ---
if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
  echo "vLLM already healthy on :8000 — skipping start"
else
  echo "Starting vLLM..."
  # Bind to loopback only: LiteLLM (same container) reaches it via localhost,
  # but RunPod's external HTTPS proxy can't connect → the public 8000 URL
  # `https://<pod-id>-8000.proxy.runpod.net` returns 502 even though the port
  # is declared in the pod metadata. All inference must go through LiteLLM.
  #
  # Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache 2.0) replaces Voxtral here. Same
  # vllm-omni 0.18 binary, different model + parser. Voxtral weights stay on
  # disk under /workspace/models/Voxtral-4B-TTS-2603 in case of rollback.
  nohup vllm serve /workspace/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni \
    --port 8000 \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --served-model-name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    > /workspace/logs/vllm.log 2>&1 &
  echo "vLLM PID: $!"
fi

# --- wait for vLLM health (up to 15 min — vllm-omni's 2-stage engine
#     plus first-time CUDAGraph capture / flashinfer JIT can take 5-10 min) ---
echo "Waiting for vLLM /health..."
for i in $(seq 1 180); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "vLLM healthy after $((i*5))s"
    break
  fi
  sleep 5
done
curl -sf http://localhost:8000/health >/dev/null || { echo "vLLM did NOT become healthy"; tail -50 /workspace/logs/vllm.log; exit 1; }

# --- XTTS v2 (voice cloning, optional) ---
# Skipped silently if /workspace/xtts-env hasn't been created (i.e.
# install_xtts.sh hasn't been run on this pod). LiteLLM still boots without it.
# 8002 picked instead of 8001 because the RunPod base image runs an nginx on
# 8001 that returns 200 to any /health probe — false positives confuse the
# `already up` check.
if [ -d /workspace/xtts-env ] && [ -f /workspace/xtts_server.py ]; then
  # Check for an *xtts-specific* route, not just /health (which any nginx
  # could 200 on). /v1/voices is the cheapest XTTS-only endpoint.
  if curl -sf http://127.0.0.1:8002/v1/voices >/dev/null 2>&1; then
    echo "XTTS already up on :8002 — skipping start"
  else
    echo "Starting XTTS server..."
    nohup bash -c '
      source /workspace/xtts-env/bin/activate
      export COQUI_TOS_AGREED=1
      export TTS_HOME=/workspace/xtts_models
      exec python /workspace/xtts_server.py
    ' > /workspace/logs/xtts.log 2>&1 &
    echo "XTTS PID: $!"
  fi

  # XTTS first-load is fast (~30 s) since the model lives on the volume already
  echo "Waiting for XTTS /health..."
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8002/health >/dev/null 2>&1; then
      echo "XTTS healthy after $((i*2))s"
      break
    fi
    sleep 2
  done
fi

# --- LiteLLM ---
if curl -sf http://localhost:4000/health/liveliness >/dev/null 2>&1 \
   || curl -sf http://localhost:4000/health >/dev/null 2>&1; then
  echo "LiteLLM already up on :4000 — skipping start"
else
  echo "Starting LiteLLM..."
  # Source the per-pod LiteLLM secrets file (master key + virtual keys).
  # Pushed onto the pod by restart-pod.sh from the local .voxtral.env.
  # `auth.py` lives next to litellm_config.yaml in /workspace, so PYTHONPATH
  # must include /workspace for `custom_auth: auth.user_api_key_auth` to resolve.
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

# --- wait for LiteLLM ---
for i in $(seq 1 30); do
  if curl -sf http://localhost:4000/health/liveliness >/dev/null 2>&1 \
     || curl -sf http://localhost:4000/health >/dev/null 2>&1; then
    echo "LiteLLM up after $((i*2))s"
    break
  fi
  sleep 2
done

echo "=== SERVICES READY ==="
echo "vLLM:    http://localhost:8000/v1/audio/speech"
echo "LiteLLM: http://localhost:4000/v1/audio/speech (model=voxtral-tts, key=sk-voxtral-local)"
