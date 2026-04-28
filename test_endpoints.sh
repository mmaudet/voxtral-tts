#!/bin/bash
# Smoke tests /v1/audio/speech via vLLM (8000) and LiteLLM (4000).
# Multilingual: en, fr, de, es, it, nl, pt.

set -uo pipefail
mkdir -p /workspace/test_audio

build_payload() {
  local model="$1" voice="$2" input="$3"
  python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"input":sys.argv[2],"voice":sys.argv[3],"response_format":"wav"}))' "$model" "$input" "$voice"
}

test_vllm() {
  local label="$1" voice="$2" input="$3"
  local out="/workspace/test_audio/${label}.wav"
  local payload; payload=$(build_payload "mistralai/Voxtral-4B-TTS-2603" "$voice" "$input")
  local code; code=$(curl -s -o "$out" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "http://localhost:8000/v1/audio/speech")
  local size; size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null || echo 0)
  printf "  %-32s http=%s  size=%9sB  %s\n" "$label" "$code" "$size" "$([ "$size" -gt 1000 ] && echo OK || echo FAIL)"
}

test_litellm() {
  local label="$1" voice="$2" input="$3"
  local out="/workspace/test_audio/${label}.wav"
  local payload; payload=$(build_payload "voxtral-tts" "$voice" "$input")
  local code; code=$(curl -s -o "$out" -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-voxtral-local" \
    -d "$payload" \
    "http://localhost:4000/v1/audio/speech")
  local size; size=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out" 2>/dev/null || echo 0)
  printf "  %-32s http=%s  size=%9sB  %s\n" "$label" "$code" "$size" "$([ "$size" -gt 1000 ] && echo OK || echo FAIL)"
}

echo "=== vLLM direct (port 8000) — 7 European languages ==="
test_vllm en_neutral_female "neutral_female" "Hello, this is a test of Voxtral TTS in English."
test_vllm fr_male           "fr_male"        "Bonjour, ceci est un test de Voxtral TTS en français."
test_vllm de_female         "de_female"      "Hallo, dies ist ein Test von Voxtral TTS auf Deutsch."
test_vllm es_male           "es_male"        "Hola, esta es una prueba de Voxtral TTS en español."
test_vllm it_female         "it_female"      "Ciao, questo è un test di Voxtral TTS in italiano."
test_vllm nl_male           "nl_male"        "Hallo, dit is een test van Voxtral TTS in het Nederlands."
test_vllm pt_female         "pt_female"      "Olá, este é um teste de Voxtral TTS em português."

echo
echo "=== LiteLLM proxy (port 4000) — sanity ==="
test_litellm fr_female_proxy "fr_female"    "Test via le proxy LiteLLM."
test_litellm en_proxy        "neutral_male" "Test through the LiteLLM proxy."

echo
echo "=== outputs ==="
ls -lh /workspace/test_audio/
echo
echo "=== file headers (should start with RIFF for WAV) ==="
for f in /workspace/test_audio/*.wav; do
  head -c 4 "$f" | od -An -c | tr -s ' ' | sed "s|^|$(basename "$f"):|"
done
