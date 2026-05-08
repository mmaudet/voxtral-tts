# voice-factory

Self-hostable recipe for OpenAI-compatible `/v1/audio/speech` on a single RunPod GPU, fronted by LiteLLM with per-user API keys.

**Default config (fast path, recommended)**: only `qwen-clone` is active — Alibaba's `Qwen3-TTS-12Hz-1.7B-Base` (Apache 2.0) doing voice cloning from your reference audio + transcript. Configured for batch=4 with CUDA graphs (the upstream `qwen3_tts_batch.yaml`), giving ~3× the throughput of the stock single-stream config. The repo ships a tiny FastAPI proxy ([`qwen_clone_proxy.py`](qwen_clone_proxy.py)) that translates `voice: "<id>"` into Qwen-Base's `task_type=Base` / `ref_audio` / `ref_text` / `language` combo (LiteLLM strips those non-OpenAI fields, so it can't hit Qwen-Base directly).

**Optional opt-in** (commented out by default in [`start_services.sh`](start_services.sh) and [`litellm_config.yaml`](litellm_config.yaml)):

- **`voice-factory`** — Mistral's `Voxtral-4B-TTS-2603` (CC BY-NC 4.0). 20 stock voices including native French / German / Spanish / Italian / Dutch / Portuguese / Hindi / Arabic. Useful when a non-commercial license is acceptable and you want native EU prosody without supplying your own samples.
- **`qwen-tts`** — Alibaba's `Qwen3-TTS-12Hz-1.7B-CustomVoice` (Apache 2.0). 9 preset voices (mostly Chinese, plus Aiden/Ryan in English, Ono_Anna JP, Sohee KR). Best when you want commercial-friendly *preset* voices for ZH/JA/KO without cloning.

To re-enable either: uncomment the relevant `start_vllm` block in `start_services.sh` and the matching `model_list` entry in `litellm_config.yaml`, then `./restart-pod.sh`. Both models' weights stay on the volume; nothing to re-download.

## What you get

- A single HTTPS endpoint speaking the OpenAI `/v1/audio/speech` schema, gated by per-user keys (LiteLLM with `custom_auth`, no DB).
- Voice cloning under **Apache 2.0** (Qwen3-TTS-Base): bring your own reference samples (16+ kHz mono, **≤ 25 s** after trim) and Qwen produces same-voice output in any of its 10 languages.
- Default fast path uses one model on a 48 GB card with batch=4 + CUDA graphs (~13 GB peak, plenty of headroom). Multi-model mode available by uncommenting blocks in `start_services.sh` and `litellm_config.yaml` (no re-install needed).
- Output: 24 kHz WAV / PCM / FLAC / MP3 / AAC / Opus.

### Throughput (default fast path)

| Audio length per request | Generation time @ batch=4 | Throughput |
|---|---|---|
| ~12 s out (~400-char text) | ~120 s | ~120 audios/h |
| ~30 s out (~1 200-char text) | ~270 s | ~50 audios/h |
| ~45 s out (~1 800-char text) | ~400 s | ~36 audios/h |

For long audios (>~12 s out), individual requests exceed Cloudflare's 100 s timeout on `*.proxy.runpod.net`. Use [`tunnel.sh`](tunnel.sh) to route through SSH and bypass that ceiling.

## What you DON'T get

- Multi-tenant routing, queueing, autoscaling — single-pod recipe.
- Hidden license trade-offs. Voxtral is **CC BY-NC 4.0** (non-commercial). Qwen is **Apache 2.0** (commercial OK). **You** pick the right `model` per use case.
- Native EU-language voices in `qwen-tts`. The 9 Qwen presets are mostly Chinese; for native FR/DE/ES/IT/NL/PT either use `voice-factory` (CC BY-NC) or clone your own EU voices into `qwen-clone` (Apache 2.0).
- Reference audio longer than 25 s. Qwen-Base rejects clips over 30 s; the recipe trims to 25 s on upload.

## Architecture

```
┌──────────────────── RunPod pod (single GPU 48 GB, BF16) ────────────────────────┐
│                                                                                  │
│                                  ┌──► 127.0.0.1:8003  vllm-omni  Voxtral         │
│   :4000  LiteLLM proxy           │                    model=voice-factory          │
│   custom_auth (auth.py) ─────────┤                                                │
│   N keys, no DB                  ├──► 127.0.0.1:8000  vllm-omni  Qwen3-CV        │
│   model alias routing            │                    model=qwen-tts             │
│                                  │                                                │
│                                  └──► 127.0.0.1:8005  qwen_clone_proxy.py        │
│                                            │           model=qwen-clone          │
│                                            └─► 127.0.0.1:8004  vllm-omni Qwen3-Base│
│                                                  task_type=Base + ref_audio +    │
│                                                  ref_text translated by proxy    │
│                                                  from manifest.json              │
│                                                                                  │
│   all upstream services bind to loopback — only :4000 LiteLLM is externally      │
└──────────────────────────────────────────────────────────────────────────────────┘
        ▲                          ▲
        │                          │
   owner key                  colleague key
   sk-voxtral-owner-…         sk-voxtral-colleague-…
        │                          │
        └──────► :4000 ◄───────────┘

       Public proxy:  https://<pod-id>-4000.proxy.runpod.net
```

LiteLLM on port 4000 is the **only** externally reachable inference endpoint. All three vLLM-Omni instances and `qwen_clone_proxy.py` bind to `127.0.0.1` so the RunPod public proxy can't connect to them — every external call has to come through LiteLLM and present a valid `VOICE_FACTORY_KEY_*`.

The proxy `qwen_clone_proxy.py` exists because LiteLLM's `aspeech()` strips fields not in OpenAI's `/v1/audio/speech` schema (`task_type`, `ref_audio`, `ref_text`, `language`). The proxy reads `voice: "<id>"` from a manifest and forwards a complete cloning payload to Qwen-Base on `:8004`.

## Quick start

You need a [RunPod account](https://www.runpod.io/) with billing set up, a [HuggingFace token](https://huggingface.co/settings/tokens), and a 48 GB GPU pod (RTX A6000, L40, L40S, or modded 4090 48GB — the regular 4090 24GB is not enough for three models).

```bash
git clone git@github.com:mmaudet/voice-factory.git
cd voice-factory

cp .voice-factory.env.example .voice-factory.env
chmod 600 .voice-factory.env
$EDITOR .voice-factory.env   # paste RUNPOD_API_KEY, HF_TOKEN, VOICE_FACTORY_KEY_*

# Provision a 48 GB pod (see "Deploying from scratch" below for the exact API call),
# scp the scripts + manifest, install, download, boot.
```

## Endpoints

```bash
# voice-factory: 20 native EU voices (Voxtral, CC BY-NC 4.0)
curl -s https://<pod-id>-4000.proxy.runpod.net/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VOICE_FACTORY_KEY_OWNER" \
  -d '{
    "model": "voice-factory",
    "input": "Bonjour, ceci est Voxtral en français natif.",
    "voice": "fr_male",
    "response_format": "wav"
  }' \
  --output sample-voxtral.wav

# qwen-tts: 9 Qwen presets (Apache 2.0, best in ZH/JA/KO)
curl -s https://<pod-id>-4000.proxy.runpod.net/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VOICE_FACTORY_KEY_OWNER" \
  -d '{
    "model": "qwen-tts",
    "input": "你好,这是 Qwen 中文测试。",
    "voice": "Vivian",
    "language": "Chinese",
    "response_format": "wav"
  }' \
  --output sample-qwen.wav

# qwen-clone: cloned voice (Apache 2.0). `voice` is a manifest key.
curl -s https://<pod-id>-4000.proxy.runpod.net/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VOICE_FACTORY_KEY_OWNER" \
  -d '{
    "model": "qwen-clone",
    "input": "Bonjour, ceci est ma voix clonée pour mon produit.",
    "voice": "fr_grand_public",
    "response_format": "wav"
  }' \
  --output sample-clone.wav
```

The `pod-id` is whatever RunPod assigns at create time. URLs survive a stop/start cycle but **change** if you re-create the pod.

If your pod is stopped (`EXITED`) to save cost:

```bash
./restart-pod.sh                # uses runpod-pod-info.json[voice-factory-main].podId
# or
./restart-pod.sh <pod-id>       # explicit
```

The script POSTs `/v1/pods/<id>/start`, polls until `RUNNING`, syncs the latest local `start_services.sh` + `litellm_config.yaml` to the pod, runs `start_services.sh` (3 vLLM sequential boot + proxy + LiteLLM, ≈ 10–12 min), and prints the proxy URLs.

## Voices

### `voice-factory` (Voxtral-4B-TTS-2603 — 20 presets, CC BY-NC 4.0)

| Style / Language | Voices |
|---|---|
| Generic (English-leaning) | `casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male` |
| French | `fr_female`, `fr_male` |
| German | `de_female`, `de_male` |
| Spanish | `es_female`, `es_male` |
| Italian | `it_female`, `it_male` |
| Dutch | `nl_female`, `nl_male` |
| Portuguese | `pt_female`, `pt_male` |
| Hindi | `hi_female`, `hi_male` |
| Arabic | `ar_male` (no female) |

`voice` selects the embedding (timbre + accent + prosody). The `input` text drives the language. Voxtral does **not** take a `language` field — it infers from the text and the chosen voice.

### `qwen-tts` (Qwen3-TTS-12Hz-1.7B-CustomVoice — 9 presets, Apache 2.0)

| Voice | Native language | Description |
|---|---|---|
| `Vivian`, `Serena` | Chinese (female) | Bright young / warm gentle |
| `Uncle_Fu`, `Dylan`, `Eric` | Chinese (male) | Mellow / Beijing dialect / Sichuan dialect |
| `Ryan`, `Aiden` | English (male) | Dynamic / sunny American |
| `Ono_Anna` | Japanese (female) | Light, nimble |
| `Sohee` | Korean (female) | Warm, rich emotion |

`language` field uses **English language names capitalised** (`"French"`, not `"fr"`) plus `"Auto"` for autodetection. Sending an ISO code yields HTTP 400. Supported: `Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian` (no Dutch).

**No native EU voices.** French/German/etc. through Aiden = English-accented, through Vivian = Chinese-accented. For native-quality EU under Apache 2.0, use `qwen-clone` with your own reference samples.

### `qwen-clone` (Qwen3-TTS-12Hz-1.7B-Base — voice cloning, Apache 2.0)

`voice` is a **manifest key** in `/workspace/qwen_voices/manifest.json`. The proxy looks up the key and translates it into Qwen-Base's `task_type=Base` + `ref_audio` (a `file:///workspace/qwen_voices/<id>.mp3` URL) + `ref_text` (the transcript) + `language`. Languages: same 10 as `qwen-tts`.

Constraints on reference audio:

- **≤ 25 s** after trim (Qwen-Base rejects > 30 s; the recipe trims to 25 s on upload).
- 16+ kHz mono, ideally 22 or 24 kHz.
- Clean recording: no background noise, no echo, no overlap, natural pace.
- Same speaker throughout.
- One voice = one MP3 + one transcript entry. Add as many as you want.

Output language is independent of reference language: a French reference can speak English, Chinese, Japanese, etc. — Qwen retains the speaker's timbre but shifts accent toward the target.

## Reference voices for `qwen-clone`

The manifest format (one `voice_id` per entry) is in [`qwen_voices_manifest.example.json`](qwen_voices_manifest.example.json). The real manifest at `/workspace/qwen_voices/manifest.json` on the pod is git-ignored locally as `qwen_voices_manifest.local.json` because transcripts are usually proprietary.

Adding a new voice:

```bash
# 1. trim to ≤25 s, mono, 22 kHz on your Mac
sox raw.wav -c 1 -r 22050 -b 16 my_voice.mp3 trim 0 25

# 2. scp to the pod
scp -P <ssh_port> my_voice.mp3 root@<pod_ip>:/workspace/qwen_voices/my_voice.mp3

# 3. on the pod, append to the manifest
ssh root@<pod_ip>
python3 -c '
import json
m = json.load(open("/workspace/qwen_voices/manifest.json"))
m["my_voice"] = {
    "audio_path": "/workspace/qwen_voices/my_voice.mp3",
    "language": "French",
    "ref_text": "<exact transcript of my_voice.mp3>",
}
json.dump(m, open("/workspace/qwen_voices/manifest.json", "w"), indent=2, ensure_ascii=False)'

# 4. restart the proxy (it reloads the manifest at startup)
pkill -f qwen_clone_proxy && nohup python3 /workspace/qwen_clone_proxy.py > /workspace/logs/qwen-clone-proxy.log 2>&1 &
```

Then `voice: "my_voice"` is live.

## Repository layout

```
.
├── README.md                         this file
├── LICENSE                           MIT for scripts; Voxtral is CC BY-NC, Qwen is Apache 2.0
├── .voice-factory.env.example              template for secrets — copy to .voice-factory.env (gitignored)
├── .gitignore
├── runpod-pod-info.example.json      schema for the per-pod state file (gitignored real one)
├── qwen_voices_manifest.example.json schema for the qwen-clone reference manifest
├── versions.lock.json                exact package versions known to work
├── litellm_config.yaml               proxy config: 3 aliases, custom_auth → auth.py
├── auth.py                           custom_auth: any VOICE_FACTORY_KEY_<NAME> env var becomes a Bearer token
├── install_voice_factory.sh                install vllm-omni + LiteLLM + patch per-stage YAML caps for 3-model fit
├── download_model.sh                 pull all 3 models (~16 GB total) to /workspace/models
├── start_services.sh                 sequential boot of 3 vLLM-Omni + qwen_clone_proxy + LiteLLM
├── qwen_clone_proxy.py               FastAPI on :8005 — translates voice="<id>" to Qwen-Base's task_type=Base
├── restart-pod.sh                    local: start a stopped pod end-to-end (start API → SSH → services → URLs)
├── tunnel.sh                         local: open SSH tunnel localhost:14000 → pod:4000 (bypasses Cloudflare 524 on long audios)
└── test_endpoints.sh                 smoke-test 7 European languages on the voice-factory endpoint (only meaningful if voice-factory is enabled)
```

## Deploying from scratch

### 1. Provision the pod (≥ 48 GB GPU)

```bash
source .voice-factory.env

python3 - <<'PY' | curl -sS -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @-
import json, os
print(json.dumps({
  "name": "voice-factory-main",
  "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  "gpuTypeIds": ["NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S"],
  "gpuTypePriority": "availability",
  "gpuCount": 1,
  "containerDiskInGb": 50,
  "volumeInGb": 60,
  "volumeMountPath": "/workspace",
  "ports": ["8000/http", "4000/http", "22/tcp"],
  "cloudType": "SECURE",
  "interruptible": False,
  "env": {
    "HF_TOKEN":   os.environ["HF_TOKEN"],
    "HF_HOME":    "/workspace/hf_cache",
    "PUBLIC_KEY": open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip(),
  },
}))
PY
```

Note: pick the *guaranteed-48GB* GPUs (A6000, L40, L40S). The plain `NVIDIA GeForce RTX 4090` ID can land on a 24 GB card; only the 48 GB modded variant works for three models, and you can't filter for it via the public API.

### 2. Install + download

```bash
# from your local checkout
scp -P <ssh_port> install_voice_factory.sh download_model.sh start_services.sh \
                  litellm_config.yaml auth.py qwen_clone_proxy.py test_endpoints.sh \
                  root@<publicIp>:/workspace/
ssh -p <ssh_port> root@<publicIp>
cd /workspace
chmod +x install_voice_factory.sh download_model.sh start_services.sh test_endpoints.sh
./install_voice_factory.sh   # ≈ 12 min — installs vllm-omni 0.18 + LiteLLM and patches YAML caps
./download_model.sh    # ≈ 1 min — pulls all 3 models (~16 GB)
```

### 3. Upload reference voices (optional, for qwen-clone)

```bash
# locally: stage your reference samples with the canonical naming
mkdir staging
for lang_persona in fr_grand_public en_grand_public ...; do
  cp my-elevenlabs/$lang_persona.mp3 staging/
done
cp qwen_voices_manifest.local.json staging/manifest.json   # your filled-in manifest

scp -P <ssh_port> staging/* root@<publicIp>:/workspace/qwen_voices/

# on the pod: trim to ≤25 s (Qwen-Base hard limit is 30 s)
ssh -p <ssh_port> root@<publicIp> '
cd /workspace/qwen_voices/
TMPD=$(mktemp -d)
for f in *.mp3; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  if [ "${d%.*}" -gt 25 ]; then
    ffmpeg -nostdin -y -loglevel error -i "$f" -t 25 -acodec libmp3lame -b:a 128k "$TMPD/$f"
    mv "$TMPD/$f" "$f"
  fi
done
rm -rf "$TMPD"
'
```

### 4. Run + test

```bash
./start_services.sh   # ≈ 10-12 min — Voxtral, Qwen-CV, Qwen-Base sequential, then proxy + LiteLLM
./test_endpoints.sh   # smoke tests on voice-factory
```

## Authentication

LiteLLM uses [custom_auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth) ([`auth.py`](auth.py)) so we hand out **multiple pre-shared keys without a database**. Every env var named `VOICE_FACTORY_KEY_<NAME>` becomes a valid Bearer token; `<NAME>` (lowercased) becomes the LiteLLM `user_id` for log tagging.

| Env var | Intended user |
|---|---|
| `VOICE_FACTORY_KEY_OWNER` | the operator (you) |
| `VOICE_FACTORY_KEY_COLLEAGUE` | someone you trust to test |

Add a third (or fifth): another `VOICE_FACTORY_KEY_<NAME>=sk-voxtral-…` in `.voice-factory.env`, run `./restart-pod.sh`. To revoke: delete the line, restart. The change reaches the pod via SSH stdin (keys never touch git, never appear in `ps`).

`VOICE_FACTORY_LITELLM_MASTER_KEY` is admin-only with `custom_auth` on: it accesses `/key/*` and `/metrics` but **not** `/v1/audio/speech`. Leaking it lets nobody synthesize audio.

```bash
python3 -c 'import secrets; print("sk-voxtral-owner-" + secrets.token_urlsafe(24))'
```

## Operating notes

- **`--omni`** is mandatory on `vllm serve`. Without it, `/v1/audio/speech` returns 404.
- **Per-stage GPU caps come from YAML, not the CLI.** vLLM-Omni overrides `--gpu-memory-utilization` with hardcoded values in `vllm_omni/model_executor/stage_configs/*.yaml`. Three 2-stage models on one GPU OOM under upstream defaults (Voxtral 0.8, Qwen 0.3 each). `install_voice_factory.sh` `sed`s them down to **Voxtral 0.20/0.05** and **Qwen 0.10/0.10** (qwen3_tts.yaml is shared by both Qwen variants). Re-run `install_voice_factory.sh` if you ever pip-reinstall vllm-omni — YAMLs revert to upstream defaults.
- **Sequential boot.** The three vLLM instances are launched one after the other in `start_services.sh`. Parallel boot OOMs during CUDA-graph capture.
- **`--allowed-local-media-path`** is required on Qwen-Base for `file://` `ref_audio` to work. The recipe sets it to `/workspace/qwen_voices`.
- **LiteLLM strips non-OpenAI fields** (`task_type`, `ref_audio`, `ref_text`, `language`) before forwarding `aspeech()` upstream. That's why `qwen-clone` routes through `qwen_clone_proxy.py` (an OpenAI-shaped proxy that expands `voice: "<id>"` into the cloning fields) instead of straight at vLLM.
- **Reference audio ≤ 25 s.** Qwen-Base hard-rejects > 30 s; the recipe trims with ffmpeg on upload.
- **Cold start is dominated by stage-1 init.** `start_services.sh` polls each `/health` for 15 minutes — intentional.
- **Secrets**: `.voice-factory.env` is git-ignored and chmod 600. `HF_TOKEN` is also pushed into the pod's container env at creation time; install scripts source it from `/proc/1/environ` because RunPod doesn't expose pod env in interactive SSH shells.

## Versions (exact, known-good)

See [`versions.lock.json`](versions.lock.json) for the full lockfile. Key pins:

| Package | Version | Why pinned |
|---|---|---|
| `vllm` | 0.18.1 | vllm-omni 0.18.0 imports `vllm.inputs.data.TokensPrompt`, removed in vllm ≥ 0.20 |
| `vllm-omni` | 0.18.0 | Latest released; supports Voxtral and both Qwen3-TTS variants |
| `torch` | 2.10.0 | vllm 0.18.1's `_C.abi3.so` is ABI-linked against torch 2.10; 2.11 yields `undefined symbol _ZN3c10…` |
| `torchaudio`/`torchvision` | 2.10.0 / 0.25.0 | Match torch |
| `transformers` | 4.57.6 | Compatible with vllm 0.18 |
| `flashinfer-python`/`flashinfer-cubin` | 0.6.6 / 0.6.6 | Both must match — newer cubin refuses to load against 0.6.6 python |
| `mistral_common` | ≥ 1.10 | Required by vllm-omni |
| `huggingface_hub[cli]` | < 1.0 | `transformers 4.57` hard-requires `<1.0` |
| `litellm[proxy]` | 1.83.x | Any recent should work |

Apt: `python3.10-venv python3.10-dev build-essential ffmpeg libsndfile1`. The `-dev` headers are needed by Triton's first-run gcc compile of `cuda_utils.c`. `ffmpeg` is needed for the qwen-clone trim step.

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `Free memory on device cuda:0 (X/Y GiB) on startup is less than desired GPU memory utilization` | YAML caps too generous after a vllm-omni reinstall, or starting a 4th model | Re-run `install_voice_factory.sh` (re-applies caps), or drop one model |
| `Failed to move speech tokenizer to cuda:0: CUDA out of memory` | Cloning loads a speech tokenizer on first `qwen-clone` request; not enough free VRAM | Same as above. Pod likely has zombie processes from previous runs holding GPU; `pkill -9 -f vllm; sleep 5; ./start_services.sh` |
| `Reference audio too long (XX.Xs). Maximum 30s supported` | Qwen-Base rejects long ref_audio | Trim to ≤25 s with ffmpeg (recipe does this on upload) |
| `Cannot load local files without --allowed-local-media-path` | Qwen-Base default-rejects `file://` URLs | `start_services.sh` passes `--allowed-local-media-path /workspace/qwen_voices` |
| `qwen-clone` returns `400 invalid api key` from LiteLLM (yet other models work) | Authorization header valid, but proxy on :8005 not reachable | `pgrep -f qwen_clone_proxy` should show one PID; if missing, `nohup python3 /workspace/qwen_clone_proxy.py > /workspace/logs/qwen-clone-proxy.log 2>&1 &` |
| LiteLLM returns 500 from `qwen-clone` with `Speech generation failed: Cannot load local files` | proxy got the request but Qwen-Base wasn't started with the allowed-local-media-path flag | Restart Qwen-Base with the flag (see `start_services.sh`) |
| `qwen-tts` returns `400 Invalid language 'fr'` | Qwen takes English language names, not ISO codes | Send `"French"` (or `"Auto"`), not `"fr"` |
| `Router.aspeech() missing 1 required positional argument: 'voice'` | LiteLLM requires `voice` even for cloning | The proxy always sets a voice; if you bypass the proxy, include `"voice": "anything"` |
| `huggingface-cli: deprecated, use hf` | huggingface_hub 1.x renamed the binary | `download_model.sh` uses `hf download` |
| `ImportError: huggingface-hub>=0.34.0,<1.0 is required …` | Mismatch between transformers 4.57 and huggingface_hub 1.x | Pin `huggingface_hub[cli]<1.0` (already done) |
| `ModuleNotFoundError: No module named 'vllm.inputs.data'` | vllm 0.20 with vllm-omni 0.18 | Pin `vllm==0.18.1` |
| `ImportError: undefined symbol _ZN3c1013MessageLogger…` | torch 2.11 with vllm 0.18 ABI mismatch | Pin `torch==2.10.0` |
| `vllm: error: unrecognized arguments: --omni` | A later `pip install vllm` overwrote the vllm-omni entrypoint | `install_voice_factory.sh` rewrites `<venv>/bin/vllm` after install |
| `flashinfer-cubin version (X.X) does not match flashinfer version (Y.Y)` | Mismatch between python and cubin packages | Pin both to the same version |
| `InductorError: … Python.h: No such file or directory` | Triton's gcc-based JIT can't find Python headers | `apt install python3.10-dev build-essential` |
| `vLLM did NOT become healthy` after 15 min | Stage-1 (audio decoder) genuinely failed | `tail /workspace/logs/vllm-{voxtral,qwen,qwen-clone}.log` for the actual error |
| LiteLLM 401 `invalid api key` | Bearer not in `VOICE_FACTORY_KEY_*` allowlist | Use `$VOICE_FACTORY_KEY_OWNER`; the master key alone won't work on `/v1/audio/speech` |
| Audio file is 0 bytes / no RIFF | Bad voice name | Pick from the relevant table |
| HTTP 403 + `error code: 1010` from public proxy URL | Cloudflare in front of `*.proxy.runpod.net` rejects `Python-urllib/*` UA | Set any non-default `User-Agent` |

## Credits

- **[Mistral AI](https://mistral.ai/news/voice-factory)** — Voxtral with open weights and 20 voices (CC BY-NC).
- **[Alibaba Qwen team](https://github.com/QwenLM/Qwen3-TTS)** — Qwen3-TTS family under Apache 2.0, including the Base variant that makes self-hosted commercial cloning possible.
- **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** — Han Gao, Hongsheng Liu, Roger Wang, Yueqian Lin — for the audio-capable vLLM fork that makes `/v1/audio/speech` possible for both Voxtral and Qwen.
- **[BerriAI/LiteLLM](https://github.com/BerriAI/LiteLLM)** — OpenAI-shaped proxy with a usable `custom_auth`.
- **[RunPod](https://runpod.io/)** — GPU billing-by-the-second + HTTPS proxy that makes single-pod hosting viable.

## License

- Scripts in this repository: **MIT** (see [LICENSE](LICENSE)).
- **Voxtral-4B-TTS-2603** (the `voice-factory` model) is **CC BY-NC 4.0** by Mistral AI. **Non-commercial only.** Same for the 20 voice presets.
- **Qwen3-TTS-12Hz-1.7B-CustomVoice** (`qwen-tts`) and **Qwen3-TTS-12Hz-1.7B-Base** (`qwen-clone`) are **Apache 2.0** by Alibaba Cloud. Commercial use is allowed.
- For voice cloning under `qwen-clone`: the **cloned audio inherits whatever rights you have on the source samples**. If your reference is an ElevenLabs export, check ElevenLabs' T&Cs on commercial use of synthesized output. If your reference is your own voice, you're fine.
