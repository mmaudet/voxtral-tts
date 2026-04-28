#!/usr/bin/env python3
"""
Generate the "Murmure" pitch as TTS samples in 6 European languages × 2
voices, hitting the vLLM endpoint of the running voxtral-tts pod through
RunPod's HTTPS proxy.

Pod id is read from runpod-pod-info.json (see runpod-pod-info.example.json).

Output: 12 WAV files in samples/murmure/.

Usage:
  ./generate-murmure-samples.py              # uses runpod-pod-info.json
  ./generate-murmure-samples.py <pod-id>     # explicit
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

# ── Source text (FR original) and translations ──────────────────────────────
# "Murmure" is the product name, kept as-is across languages.

TEXTS = {
    "fr": (
        "Murmure.\n\n"
        "Le patrimoine français, chuchoté à ton oreille.\n"
        "Active le Mode Balade, range ton téléphone. "
        "Quand tu passes devant un lieu qui a quelque chose à raconter, "
        "une voix te le dit."
    ),
    "en": (
        "Murmure.\n\n"
        "French heritage, whispered into your ear.\n"
        "Switch on Walk Mode and put your phone away. "
        "When you pass a place with a story to tell, "
        "a voice tells it to you."
    ),
    "de": (
        "Murmure.\n\n"
        "Französisches Kulturerbe, dir ins Ohr geflüstert.\n"
        "Aktiviere den Spaziermodus und steck dein Handy weg. "
        "Wenn du an einem Ort vorbeigehst, der etwas zu erzählen hat, "
        "erzählt es dir eine Stimme."
    ),
    "it": (
        "Murmure.\n\n"
        "Il patrimonio francese, sussurrato al tuo orecchio.\n"
        "Attiva la Modalità Passeggiata e metti via il telefono. "
        "Quando passi davanti a un luogo che ha qualcosa da raccontare, "
        "una voce te lo dice."
    ),
    "es": (
        "Murmure.\n\n"
        "El patrimonio francés, susurrado a tu oído.\n"
        "Activa el Modo Paseo y guarda tu teléfono. "
        "Cuando pasas junto a un lugar que tiene algo que contar, "
        "una voz te lo cuenta."
    ),
    "nl": (
        "Murmure.\n\n"
        "Frans erfgoed, in je oor gefluisterd.\n"
        "Zet de Wandelmodus aan en stop je telefoon weg. "
        "Wanneer je langs een plek loopt met iets te vertellen, "
        "fluistert een stem het je toe."
    ),
}

# Voice presets per language. The model has no en_* voices — use the generic
# neutral_* presets for English (those are English-leaning by default).
VOICES = {
    "en": {"male": "neutral_male", "female": "neutral_female"},
    "fr": {"male": "fr_male",      "female": "fr_female"},
    "de": {"male": "de_male",      "female": "de_female"},
    "it": {"male": "it_male",      "female": "it_female"},
    "es": {"male": "es_male",      "female": "es_female"},
    "nl": {"male": "nl_male",      "female": "nl_female"},
}

LANGS = ["fr", "en", "de", "it", "es", "nl"]
GENDERS = ["female", "male"]
MODEL = "mistralai/Voxtral-4B-TTS-2603"
OUT_DIR = "samples/murmure"


def get_pod_id() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        with open("runpod-pod-info.json") as f:
            return json.load(f)["voxtral-main"]["podId"]
    except FileNotFoundError:
        sys.exit("✗ runpod-pod-info.json not found — pass pod id as arg")
    except KeyError:
        sys.exit("✗ runpod-pod-info.json has no .voxtral-main.podId")


def synth(url: str, payload: dict, out_path: str):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
    with open(out_path, "wb") as f:
        f.write(body)
    return time.time() - t0, len(body)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    pod_id = get_pod_id()
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1/audio/speech"

    print(f"target : {url}")
    print(f"output : {OUT_DIR}/")
    print()

    failures = 0
    total_t = 0.0
    total_b = 0
    for lang in LANGS:
        text = TEXTS[lang]
        for gender in GENDERS:
            voice = VOICES[lang][gender]
            out = f"{OUT_DIR}/{lang}_{gender}_{voice}.wav"
            payload = {
                "model": MODEL,
                "input": text,
                "voice": voice,
                "response_format": "wav",
            }
            label = f"{lang} {gender:6} ({voice:>16})"
            try:
                dt, size = synth(url, payload, out)
                total_t += dt
                total_b += size
                ok = "OK" if size > 1000 else "FAIL"
                if ok == "FAIL":
                    failures += 1
                print(f"  {label} -> {size:>7} B  {dt:5.2f}s  {ok}")
            except urllib.error.HTTPError as e:
                failures += 1
                print(f"  {label} -> HTTP {e.code}: {e.read()[:200]!r}")
            except Exception as e:
                failures += 1
                print(f"  {label} -> ERROR: {e}")

    print()
    print(f"summary: {len(LANGS) * len(GENDERS) - failures}/{len(LANGS) * len(GENDERS)} OK, "
          f"{total_b/1024:.0f} KB total, {total_t:.1f}s wall clock")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
