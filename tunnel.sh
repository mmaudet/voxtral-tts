#!/bin/bash
# Open an SSH tunnel from local :14000 to the pod's :4000 (LiteLLM internal).
# Required for long audios (>~12 s out / >~50 s gen time) because the public
# RunPod proxy *.proxy.runpod.net is fronted by Cloudflare which 524s any
# request taking longer than ~100 s.
#
# Usage:
#   ./tunnel.sh                       # uses runpod-pod-info.json
#   ./tunnel.sh <ip> <port>           # explicit
#   ./tunnel.sh --close               # stop any running tunnel
#
# Once open, point clients at http://localhost:14000/v1/audio/speech instead
# of the public proxy URL. The Bearer auth header is unchanged.

set -euo pipefail
cd "$(dirname "$0")"

LOCAL_PORT=14000
REMOTE_PORT=4000

if [ "${1:-}" = "--close" ]; then
  echo "Closing any tunnel on local :$LOCAL_PORT..."
  pkill -f "ssh.*-L $LOCAL_PORT:127.0.0.1:$REMOTE_PORT" 2>/dev/null || true
  echo "done."
  exit 0
fi

if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
  IP="$1"; SSHP="$2"
elif [ -f runpod-pod-info.json ]; then
  IP=$(python3 -c 'import json; print(json.load(open("runpod-pod-info.json"))["voice-factory-main"]["publicIp"])')
  SSHP=$(python3 -c 'import json; print(json.load(open("runpod-pod-info.json"))["voice-factory-main"]["sshPort"])')
else
  echo "✗ usage: $0 [<ip> <ssh_port>] | $0 --close"
  echo "  (or have runpod-pod-info.json with publicIp + sshPort)"
  exit 1
fi

# Skip if already open
if lsof -i :"$LOCAL_PORT" >/dev/null 2>&1; then
  echo "tunnel on local :$LOCAL_PORT is already open"
  exit 0
fi

ssh -p "$SSHP" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -fN -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "root@$IP"

# Verify
sleep 1
if curl -sf http://localhost:"$LOCAL_PORT"/health/liveliness >/dev/null 2>&1; then
  echo "tunnel UP — http://localhost:$LOCAL_PORT/v1/audio/speech (LiteLLM in pod)"
else
  echo "✗ tunnel opened but health check failed; LiteLLM might be down on the pod"
  exit 1
fi
