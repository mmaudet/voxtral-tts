#!/bin/bash
# Probe vllm-omni :8004 for native timestamp / alignment emission.
# Tests several payload extensions to see what (if any) makes the upstream
# return alignment data alongside audio.
#
# Run on the pod (after start_services.sh) OR via SSH tunnel localhost:18004.

set -euo pipefail
PORT=${PORT:-18004}
HOST=${HOST:-localhost}

REF_AUDIO=${REF_AUDIO:-/workspace/qwen_voices/fr_grand_public.mp3}
REF_TEXT=${REF_TEXT:-"Murmure. Le patrimoine français comme vous ne l'avez jamais entendu."}
INPUT_TEXT=${INPUT_TEXT:-"L'église Saint-Denis a été bâtie au douzième siècle."}

probe() {
  local label="$1" extra="$2"
  echo "=== ${label} ==="
  local body
  body=$(cat <<EOF
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "input": "${INPUT_TEXT}",
  "voice": "Aiden",
  "task_type": "Base",
  "ref_audio": "file://${REF_AUDIO}",
  "ref_text": "${REF_TEXT}",
  "language": "French",
  "response_format": "wav"
  ${extra}
}
EOF
)
  curl -sS -X POST "http://${HOST}:${PORT}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "${body}" \
    -o /tmp/probe-${label}.bin -D /tmp/probe-${label}.headers \
    -w "  http_code=%{http_code} time=%{time_total}s\n"
  echo "  bytes: $(wc -c </tmp/probe-${label}.bin)"
  echo "  content-type: $(grep -i '^content-type' /tmp/probe-${label}.headers || echo MISSING)"
  # If body looks like text (likely JSON or error), print first 500 chars
  if file /tmp/probe-${label}.bin 2>/dev/null | grep -qiE 'JSON|ASCII|UTF-8|text'; then
    echo "  body (text):"
    head -c 500 /tmp/probe-${label}.bin | sed 's/^/    /'
    echo
  else
    echo "  body (binary, first 4 bytes):"
    head -c 4 /tmp/probe-${label}.bin | xxd | head -1
  fi
  echo
}

echo "Target: http://${HOST}:${PORT}/v1/audio/speech"
echo "Ref audio: ${REF_AUDIO}"
echo "Input: ${INPUT_TEXT}"
echo

# Baseline (no extra) for sanity
probe "h0-baseline" ""

# H1.a : flag OpenAI-style return_timestamps
probe "h1a-return-timestamps" ', "return_timestamps": true'

# H1.b : flag style Whisper (granularities)
probe "h1b-timestamp-granularities" ', "timestamp_granularities": ["word"]'

# H1.c : extra_outputs (vllm-pattern)
probe "h1c-extra-outputs" ', "extra_outputs": ["alignments"]'

# H1.d : extra_body (OpenAI-SDK-style)
probe "h1d-extra-body" ', "extra_body": {"return_alignments": true}'

# H1.e : output_format JSON
probe "h1e-format-json" ', "response_format": "json"'

# H1.f : Qwen-specific candidates worth trying
probe "h1f-include-timestamps" ', "include_timestamps": true'
probe "h1g-with-timestamps" ', "with_timestamps": true'
probe "h1h-include-alignment" ', "include_alignment": true'
probe "h1i-stream-true" ', "stream": true'
probe "h1j-output-alignment" ', "output": "alignment"'

echo "=== summary ==="
for f in /tmp/probe-*.bin; do
  label=$(basename "$f" .bin | sed 's/probe-//')
  size=$(wc -c <"$f")
  ctype=$(grep -i '^content-type' /tmp/probe-${label}.headers | head -1 | tr -d '\r' || echo "?")
  printf "  %-30s %10s bytes  %s\n" "$label" "$size" "$ctype"
done
