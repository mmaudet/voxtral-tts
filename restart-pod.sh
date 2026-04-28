#!/bin/bash
# Restart a stopped voxtral-tts pod end-to-end:
#   1. POST /v1/pods/<id>/start
#   2. poll until pod is RUNNING with publicIp + ssh port assigned
#   3. wait for sshd to accept connections
#   4. scp the latest start_services.sh + litellm_config.yaml
#   5. run start_services.sh on the pod (blocks until SERVICES READY, ~4-5 min)
#   6. print the public proxy URLs
#
# Usage:
#   ./restart-pod.sh                 # uses runpod-pod-info.json[voxtral-main]
#   ./restart-pod.sh <pod-id>        # explicit pod id
#
# Requires:
#   .voxtral.env with RUNPOD_API_KEY
#   ~/.ssh/id_ed25519  (the key that was injected at pod creation time)

set -euo pipefail
cd "$(dirname "$0")"

# ---------------- 1. secrets + pod id ----------------
if [ ! -f .voxtral.env ]; then
  echo "✗ .voxtral.env not found — copy from .voxtral.env.example, fill, chmod 600" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .voxtral.env

if [ -n "${1:-}" ]; then
  POD_ID="$1"
elif [ -f runpod-pod-info.json ]; then
  POD_ID=$(python3 -c 'import json,sys; d=json.load(open("runpod-pod-info.json")); print(d["voxtral-main"]["podId"])')
else
  echo "✗ no pod id — pass one as arg, or have runpod-pod-info.json present" >&2
  exit 1
fi
echo "→ pod $POD_ID"

# ---------------- 2. POST /start ----------------
HTTP=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
  "https://rest.runpod.io/v1/pods/$POD_ID/start" \
  -o /tmp/voxtral_restart.json -w "%{http_code}")
case "$HTTP" in
  200|409)  # 200 = started, 409 = already running (treat as success)
    DESIRED=$(python3 -c 'import json; d=json.load(open("/tmp/voxtral_restart.json")); print(d.get("desiredStatus","?"))' 2>/dev/null || echo "?")
    echo "  start API: HTTP $HTTP, desiredStatus=$DESIRED"
    ;;
  *)
    echo "✗ start failed (HTTP $HTTP):"
    cat /tmp/voxtral_restart.json
    exit 1
    ;;
esac

# ---------------- 3. poll for RUNNING + ports assigned ----------------
echo "→ waiting for pod to be reachable (publicIp + ssh port)..."
IP=""; SSHP=""
for i in $(seq 1 60); do
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD_ID" -o /tmp/voxtral_pod.json
  read -r DESIRED IP SSHP <<< "$(python3 -c '
import json
d = json.load(open("/tmp/voxtral_pod.json"))
print(d.get("desiredStatus") or "-",
      d.get("publicIp") or "-",
      (d.get("portMappings") or {}).get("22") or "-")
')"
  if [ "$DESIRED" = "RUNNING" ] && [ "$IP" != "-" ] && [ "$SSHP" != "-" ]; then
    echo "  pod RUNNING after ~$((i*5))s — ip=$IP ssh=$SSHP"
    break
  fi
  printf "  [%2d] desiredStatus=%s ip=%s ssh=%s\n" "$i" "$DESIRED" "$IP" "$SSHP"
  sleep 5
done
if [ "$IP" = "-" ] || [ "$SSHP" = "-" ]; then
  echo "✗ timeout waiting for pod to expose IP+SSH after 5 min"
  exit 1
fi

# ---------------- 4. wait for sshd ----------------
echo "→ waiting for sshd..."
SSH_OPTS=(-p "$SSHP" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
for i in $(seq 1 30); do
  if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=5 -o BatchMode=yes \
       "root@$IP" 'echo ok' 2>/dev/null | grep -q ok; then
    echo "  sshd up after ~$((i*3))s"
    break
  fi
  sleep 3
done

# ---------------- 5. push latest local scripts + LiteLLM secrets ----------------
echo "→ syncing start_services.sh + litellm_config.yaml + auth.py to pod..."
scp -P "$SSHP" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  start_services.sh litellm_config.yaml auth.py \
  "root@$IP:/workspace/" >/dev/null
ssh "${SSH_OPTS[@]}" "root@$IP" 'chmod +x /workspace/start_services.sh'

# Push LiteLLM virtual keys via stdin (never in argv). Keys are read from the
# local .voxtral.env which is sourced at the top of this script.
if [ -n "${VOXTRAL_KEY_OWNER:-}" ] && [ -n "${VOXTRAL_LITELLM_MASTER_KEY:-}" ]; then
  echo "→ syncing /workspace/.litellm.env (custom_auth keys)..."
  ssh "${SSH_OPTS[@]}" "root@$IP" 'umask 077; cat > /workspace/.litellm.env && chmod 600 /workspace/.litellm.env' <<EOF
export VOXTRAL_KEY_OWNER=$VOXTRAL_KEY_OWNER
export VOXTRAL_KEY_COLLEAGUE=${VOXTRAL_KEY_COLLEAGUE:-}
export VOXTRAL_LITELLM_MASTER_KEY=$VOXTRAL_LITELLM_MASTER_KEY
EOF
else
  echo "WARN: VOXTRAL_KEY_OWNER/MASTER not set in .voxtral.env — LiteLLM will start with no virtual keys"
fi

# ---------------- 6. launch services (foreground, blocks ~4-5 min) ----------------
echo "→ launching services on the pod (this will block ~4-5 min on first cold boot)..."
ssh "${SSH_OPTS[@]}" "root@$IP" '/workspace/start_services.sh'

# ---------------- 7. print summary ----------------
URL_VLLM="https://${POD_ID}-8000.proxy.runpod.net/v1/audio/speech"
URL_LITELLM="https://${POD_ID}-4000.proxy.runpod.net/v1/audio/speech"

cat <<EOF

✅ pod ready

  vLLM    : $URL_VLLM
  LiteLLM : $URL_LITELLM    (Authorization: Bearer sk-voxtral-local)
  SSH     : ssh -p $SSHP root@$IP

The HTTPS proxy may take 20-30 s to start routing to the new container.
If your first external curl returns 502, just retry.
EOF
