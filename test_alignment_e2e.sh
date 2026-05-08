#!/bin/bash
# E2E test for /v1/audio/speech-with-alignment.
# Hits the qwen_clone_proxy directly (bypasses LiteLLM, which strips
# custom fields). Use a dedicated SSH tunnel localhost:18005 → pod:8005.
#
# Usage:
#   ssh -fNT -L 18005:127.0.0.1:8005 root@<pod_ip> -p <ssh_port>
#   bash test_alignment_e2e.sh

set -euo pipefail

PORT=${PORT:-18005}
VOICE=${VOICE:-fr_grand_public}
TEXT=${TEXT:-"L'église Saint-Denis a été bâtie au douzième siècle. Ses vitraux racontent une histoire incroyable."}

echo "=== test_alignment_e2e.sh ==="
echo "  endpoint : http://localhost:${PORT}/v1/audio/speech-with-alignment"
echo "  voice    : ${VOICE}"
echo "  text     : ${TEXT}"
echo

# Build payload
python3 - > /tmp/align-payload.json <<PY
import json
print(json.dumps({"voice": "${VOICE}", "input": """${TEXT}"""}))
PY

t0=$(date +%s)
curl -sS -X POST "http://localhost:${PORT}/v1/audio/speech-with-alignment" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/align-payload.json \
  -m 360 \
  -o /tmp/align-response.json \
  -w "HTTP %{http_code} | size=%{size_download}B time=%{time_total}s\n"
t1=$(date +%s)

echo "wall clock: $((t1 - t0))s"
echo

# Parse response
if file /tmp/align-response.json | grep -q JSON; then
  python3 - <<'PY'
import json, base64, sys
with open('/tmp/align-response.json') as f:
    d = json.load(f)
print(f"audio_mime  : {d.get('audio_mime')}")
print(f"language    : {d.get('language')}")
print(f"duration_s  : {d.get('duration_s')}")
print(f"synth_ms    : {d.get('synth_ms')}")
print(f"align_ms    : {d.get('align_ms')}")
audio = base64.b64decode(d['audio_base64'])
print(f"audio_bytes : {len(audio)} ({len(audio)/1024:.0f} KB)")
print(f"alignments  : {len(d.get('alignments', []))} words")
print()
print("first 12 word alignments:")
for w in d.get('alignments', [])[:12]:
    print(f"  {w['word']:30s} {w['start']:6.2f}s → {w['end']:6.2f}s  ({(w['end']-w['start']):4.2f}s)")
# Save the audio for verification
with open('/tmp/align-test.wav', 'wb') as f:
    f.write(audio)
print()
print(f"audio saved to /tmp/align-test.wav")
PY
else
  echo "response (non-JSON):"
  head -c 500 /tmp/align-response.json
  echo
  exit 1
fi
