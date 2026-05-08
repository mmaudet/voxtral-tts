# Qwen3-TTS-Base timestamps — Investigation

**Date** : 2026-05-08
**Pod testé** : `gxuo53n1ckl1a6` (NVIDIA L40S 46 GB, vllm-omni 0.18, Qwen/Qwen3-TTS-12Hz-1.7B-Base)
**Question** : peut-on obtenir des word-level timestamps natifs au moment de la synthèse, sans re-traitement ASR ?

## Décision

**Phase A.2b — faster-whisper post-process** sur le pod.

Raisonnement : Qwen3-TTS-Base est un modèle TTS récent (Apache 2.0) sans documentation
publique sur l'émission de timestamps. La probabilité de succès du spike natif (Tasks 5/6/7)
est faible (~10-20% à vue de nez), et même en cas de succès on devrait valider les timings
contre Whisper comme ground truth — donc on aurait Whisper de toute façon.

Aller direct sur 8b est pragmatique : Whisper est l'industrie-standard pour le forced alignment,
faster-whisper est rapide sur GPU, et l'overhead réel mesuré est faible.

## H1 — Flag `return_timestamps` upstream

**SKIPPED** — décision pragmatique de pivoter direct vers 8b sans tester les flags. Le script
`test_alignment.sh` reste dans le repo si quelqu'un veut explorer plus tard.

## H2 / H3 — Stream + source inspection

**SKIPPED** — même raison.

## Validation Phase A.2b (faster-whisper)

### Setup

```bash
# Sur le pod, après install_voxtral.sh :
. /workspace/voxtral-env/bin/activate
pip install faster-whisper==1.2.0
python -c "
from faster_whisper import WhisperModel
WhisperModel('large-v3', device='cuda', compute_type='float16',
             download_root='/workspace/models/whisper')
"
```

Disk : 2.9 GB pour `whisper-large-v3` à `/workspace/models/whisper/`.
VRAM : ~2.6 GB en fp16 (lazy-loaded, gardé résident entre requêtes).

### Mesures empiriques

**Test 1 — texte court (5.14s audio, 15 mots)**

```
synth   : 40.7s   (cold start de la session)
align   :  5.1s   (premier appel : load whisper + transcribe)
total   : 47.0s   wall clock
```

**Test 2 — texte mémo-réaliste (19.9s audio, 70 mots)**

```
synth   : 18.9s   (≈ real-time, batch=8 + cudagraph chauds)
align   :  1.3s   (whisper chaud)
total   : 21.3s   wall clock
words   : 75 alignments {word, start, end}
```

**Conclusion** : en steady state, l'alignment whisper coûte ~5-10% du temps de synth.
Sur l'ensemble de la prod 56k audios à batch=8 / 2 pods :
- Sans alignment : ~13 jours, ~$546
- Avec alignment : **~14-15 jours, ~$600** (+10-15% wall-clock, +$60)

### Qualité des alignments

Whisper transcrit + aligne — ce n'est PAS du forced alignment pur. Conséquences observées :

| Pattern | Exemple | Impact |
|---|---|---|
| Mauvaise transcription | "L'église" → "Selglise" | mot affiché incorrect, timing OK |
| Split sur hyphens | "Saint-Étienne-du-Mont" → 4 segments | granularité plus fine que prévu |
| Substitution | "douzième" → "XIIe" | sémantique préservée, mot != original |
| Lost feminine -e | "bâtie" → "bâti" | orthographe approximative |

**Stratégies de mitigation côté client** :
1. (simple) Garder les mots de Whisper tels quels — timings exacts, mots parfois faux
2. (robuste) Remapper les mots du texte source sur les timings Whisper par position — forced alignment "à la main"

À choisir au moment de l'intégration côté `synth.ts` (Tasks 12-15).

### Endpoint exposé

`POST http://<pod>:8005/v1/audio/speech-with-alignment` (bypass LiteLLM, tunnel SSH dédié)

Request:
```json
{ "voice": "fr_grand_public", "input": "..." }
```

Response:
```json
{
  "audio_base64": "<base64-encoded WAV>",
  "audio_mime": "audio/wav",
  "alignments": [{ "word": "...", "start": 0.0, "end": 0.42 }, ...],
  "language": "French",
  "duration_s": 19.9,
  "synth_ms": 18927,
  "align_ms": 1277
}
```

Failure mode : si Whisper crash, on retourne `alignments: []` + audio quand même.
Le client peut détecter et fallback en mode transcript-only.
