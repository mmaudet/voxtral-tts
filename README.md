# voxtral-tts

Self-hostable recipe running **two TTS models in parallel** on a single RunPod GPU, both fronted by a unified LiteLLM proxy with per-user API keys:

- **Voxtral-4B-TTS-2603** (Mistral, CC BY-NC 4.0) — best for European languages, with native French / German / Spanish / Italian / Dutch / Portuguese / Hindi / Arabic voices.
- **Qwen3-TTS-12Hz-1.7B-CustomVoice** (Alibaba, Apache 2.0) — best for Mandarin / Japanese / Korean and the only commercial-friendly preset-voice path here.

You pick which backend to call by sending `model: "voxtral-tts"` or `model: "qwen-tts"` on the same OpenAI-compatible `/v1/audio/speech` endpoint.

## What you get

- A single HTTPS endpoint speaking the OpenAI `/v1/audio/speech` schema, gated by per-user keys (LiteLLM with `custom_auth`, no DB).
- Two TTS models behind it, picked via the `model` field of the request:
  - **`voxtral-tts`** → vLLM-Omni serving `mistralai/Voxtral-4B-TTS-2603`. 9 languages (EN/FR/ES/PT/IT/NL/DE/AR/HI), 20 stock voices including native FR/DE/ES/IT/NL/PT pairs.
  - **`qwen-tts`** → vLLM-Omni serving `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. 10 languages (ZH/EN/JA/KO/DE/FR/RU/PT/ES/IT), 9 stock voices (mostly Chinese, plus Aiden/Ryan in English, Ono_Anna JP, Sohee KR).
- Both models share one GPU thanks to a YAML-level patch to vllm-omni's per-stage `gpu_memory_utilization` (the CLI flag alone is ignored — see [Operating notes](#operating-notes)).
- Single-pod deployment on a 48 GB card: ~17 GiB Voxtral + ~14 GiB Qwen + ~17 GiB headroom.
- Output: 24 kHz WAV / PCM / FLAC / MP3 / AAC / Opus.

## What you DON'T get

- Multi-tenant routing, queueing, autoscaling — this is a single-pod recipe.
- A model gateway that hides each backend's license — Voxtral is **CC BY-NC 4.0** (non-commercial only), Qwen is **Apache 2.0** (commercial OK). **You** are responsible for picking the right `model` for your use case.
- Voice cloning. Both models ship preset voices only. For cloning under Apache 2.0, switch one of the aliases to the `Qwen/Qwen3-TTS-12Hz-1.7B-Base` variant (not deployed by default in this recipe).

## Architecture

```
┌──────────────────── RunPod pod (single GPU 48 GB, BF16) ──────────────────────┐
│                                                                                │
│                                  ┌──► 127.0.0.1:8003  vLLM-Omni                │
│   :4000  LiteLLM proxy           │                    model=voxtral-tts        │
│   custom_auth (auth.py) ─────────┤                    20 stock voices          │
│   N keys, no DB                  │                    (CC BY-NC 4.0)           │
│   model alias routing            │                                             │
│                                  └──► 127.0.0.1:8000  vLLM-Omni                │
│                                                       model=qwen-tts           │
│                                                       9 stock voices           │
│                                                       (Apache 2.0)             │
│   both upstream services bind to loopback — NOT externally exposed             │
└────────────────────────────────────────────────────────────────────────────────┘
        ▲                          ▲
        │                          │
   owner key                  colleague key
   sk-voxtral-owner-…         sk-voxtral-colleague-…
        │                          │
        └──────► :4000 ◄───────────┘

       Public proxy:  https://<pod-id>-4000.proxy.runpod.net
```

LiteLLM on port 4000 is the **only** externally reachable inference endpoint. Both upstream vLLM-Omni instances bind to `127.0.0.1` so the RunPod public proxy can't connect to them — every external call has to come through LiteLLM and present a valid `VOXTRAL_KEY_*`. Each consumer gets a distinct key, rotated/revoked by editing `.voxtral.env` and running `./restart-pod.sh`. The `model` field of the OpenAI payload picks which backend handles the request.

## Quick start

You'll need a [RunPod account](https://www.runpod.io/) with billing set up and a [HuggingFace token](https://huggingface.co/settings/tokens). Neither model is gated.

```bash
git clone git@github.com:mmaudet/voxtral-tts.git
cd voxtral-tts

# 1. Configure secrets
cp .voxtral.env.example .voxtral.env
chmod 600 .voxtral.env
$EDITOR .voxtral.env   # paste RUNPOD_API_KEY, HF_TOKEN, VOXTRAL_KEY_*

# 2. Provision a pod with a 48 GB GPU (RTX 4090 48GB, A6000, L40S, …)
#    See "Deploying from scratch" below for the POST /v1/pods call.

# 3. Once the pod is RUNNING, scp scripts onto it and run:
#       install_voxtral.sh   ≈ 12 min   (vllm-omni + LiteLLM + YAML caps)
#       download_model.sh    ≈ 1 min    (Qwen3-TTS, ~4 GB)
#       (Voxtral weights ~8 GB are downloaded the same way; both stay on the volume)
#       start_services.sh    ≈ 9 min    (Voxtral cold + Qwen cold, sequential)
```

## Endpoints

Once the pod is up, the LiteLLM proxy is the only externally reachable surface. Pick a backend with the `model` field:

```bash
# voxtral-tts: 20 native EU voices (Voxtral, CC BY-NC 4.0)
curl -s https://<pod-id>-4000.proxy.runpod.net/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VOXTRAL_KEY_OWNER" \
  -d '{
    "model": "voxtral-tts",
    "input": "Bonjour, ceci est Voxtral en français natif.",
    "voice": "fr_male",
    "response_format": "wav"
  }' \
  --output sample-voxtral.wav

# qwen-tts: 9 voices, Apache 2.0 (best in ZH/JA/KO; English/Chinese accent on EU text)
curl -s https://<pod-id>-4000.proxy.runpod.net/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $VOXTRAL_KEY_OWNER" \
  -d '{
    "model": "qwen-tts",
    "input": "你好,这是 Qwen 中文测试。",
    "voice": "Vivian",
    "language": "Chinese",
    "response_format": "wav"
  }' \
  --output sample-qwen.wav
```

Both upstream services bind to `127.0.0.1` (Voxtral on 8003, Qwen on 8000). For debugging, SSH in and curl localhost directly — there is no public route to either.

The `pod-id` is whatever RunPod assigns at create-time. URLs survive a stop/start cycle but **change** if you re-create the pod. See `runpod-pod-info.example.json` for the metadata schema; the actual `runpod-pod-info.json` is git-ignored because it's per-pod state.

If your pod is stopped (`EXITED`) to save cost, the easiest way back is:

```bash
./restart-pod.sh                # uses runpod-pod-info.json[voxtral-main].podId
# or
./restart-pod.sh <pod-id>       # explicit
```

The script POSTs `/v1/pods/<id>/start`, polls until `RUNNING` with `publicIp` and SSH port assigned, syncs the latest local `start_services.sh` + `litellm_config.yaml` to the pod, runs `start_services.sh` (both vLLM-Omni instances + LiteLLM, sequential boot), and prints the proxy URLs. Total cold restart ≈ 9-10 min on a fresh pod (Voxtral ~4 min + Qwen ~5 min + LiteLLM ~30 s).

## Voices

### `voxtral-tts` (Voxtral-4B-TTS-2603 — 20 presets, CC BY-NC 4.0)

| Style / Language | Voices |
|---|---|
| Generic (English-leaning) | `casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male` |
| French (FR) | `fr_female`, `fr_male` |
| German (DE) | `de_female`, `de_male` |
| Spanish (ES) | `es_female`, `es_male` |
| Italian (IT) | `it_female`, `it_male` |
| Dutch (NL) | `nl_female`, `nl_male` |
| Portuguese (PT) | `pt_female`, `pt_male` |
| Hindi (HI) | `hi_female`, `hi_male` |
| Arabic (AR) | `ar_male` *(no female variant)* |

The `voice` selects the embedding (timbre + accent + prosody). The `input` text drives the language. Pairing a `de_*` voice with an English text usually works but the prosody favours the voice's native language. Voxtral does **not** take a `language` field — it infers from the text and the chosen voice.

### `qwen-tts` (Qwen3-TTS-12Hz-1.7B-CustomVoice — 9 presets, Apache 2.0)

| Voice | Native language | Description (per Qwen model card) |
|---|---|---|
| `Vivian` | Chinese | Bright, slightly edgy young female voice |
| `Serena` | Chinese | Warm, gentle young female voice |
| `Uncle_Fu` | Chinese | Seasoned male voice with a low, mellow timbre |
| `Dylan` | Chinese (Beijing dialect) | Youthful Beijing male, clear natural timbre |
| `Eric` | Chinese (Sichuan dialect) | Lively Chengdu male, slightly husky brightness |
| `Ryan` | English | Dynamic male voice with strong rhythmic drive |
| `Aiden` | English | Sunny American male voice with a clear midrange |
| `Ono_Anna` | Japanese | Playful Japanese female, light nimble timbre |
| `Sohee` | Korean | Warm Korean female voice with rich emotion |

Qwen3-TTS supports 10 languages: `Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian` (no Dutch). The `language` field uses **English language names capitalised** (`"French"`, not `"fr"`) plus the special `"Auto"` value for autodetection. Sending an ISO code yields `400`.

**No native EU voices.** French/German/etc. through Aiden = English-accented, through Vivian = Chinese-accented. For native EU quality, use `voxtral-tts`. Qwen is the right choice for ZH/JA/KO (where it has native voices), or for any commercial workflow that needs an Apache 2.0 model.

Response formats: `wav` (default), `pcm`, `flac`, `mp3`, `aac`, `opus`. Output is 24 kHz mono.

## Repository layout

```
.
├── README.md                       this file
├── LICENSE                         MIT for scripts; Voxtral is CC BY-NC 4.0, Qwen is Apache 2.0
├── .voxtral.env.example            template for secrets — copy to .voxtral.env (gitignored)
├── .gitignore
├── runpod-pod-info.example.json    schema for the per-pod state file (gitignored real one)
├── versions.lock.json              the *exact* package versions known to work
├── litellm_config.yaml             proxy config: aliases `voxtral-tts` + `qwen-tts`, custom_auth → auth.py
├── auth.py                         custom_auth module: any `VOXTRAL_KEY_<NAME>` env var becomes a valid Bearer token (no DB)
├── install_voxtral.sh              install vLLM-Omni + LiteLLM + patch the per-stage YAML caps (idempotent)
├── download_model.sh               pull the Qwen3-TTS weights to /workspace/models/ (idempotent)
├── start_services.sh               launch Voxtral (8003) + Qwen (8000) sequentially + LiteLLM (idempotent)
├── restart-pod.sh                  local: start a stopped pod end-to-end (start API → SSH → services → URLs)
└── test_endpoints.sh               smoke-test 7 European languages on the voxtral-tts endpoint
```

## Deploying from scratch

The recipe runs in three stages. Each stage's script is idempotent: rerunning is safe.

### 1. Provision the pod (RunPod REST API)

A bare `POST /v1/pods` call. **Pick a 48 GB+ GPU** so two TTS models fit comfortably (RTX 4090 48GB modded, A6000 48GB, L40S 48GB, etc.). Smaller cards work for one model only.

```bash
source .voxtral.env

python3 - <<'PY' | curl -sS -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @-
import json, os
print(json.dumps({
  "name": "voxtral-main",
  "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  "gpuTypeIds": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA L40S"],
  "gpuTypePriority": "availability",
  "gpuCount": 1,
  "containerDiskInGb": 50,
  "volumeInGb": 50,
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

Wait until `desiredStatus == RUNNING` and `portMappings.22` is set, then SSH to `root@<publicIp>` on that port. From there:

### 2. Install + download (one-time per pod)

```bash
# from your local checkout, push the scripts up
scp -P <ssh_port> install_voxtral.sh download_model.sh start_services.sh \
                  litellm_config.yaml auth.py test_endpoints.sh \
                  root@<publicIp>:/workspace/

# on the pod
ssh -p <ssh_port> root@<publicIp>
cd /workspace
chmod +x install_voxtral.sh download_model.sh start_services.sh test_endpoints.sh
./install_voxtral.sh    # ≈ 12 min — installs vllm-omni 0.18 + LiteLLM with the
                        # exact pinned versions from versions.lock.json,
                        # then patches per-stage YAML caps for dual-model fit
./download_model.sh     # ≈ 1 min  — pulls Qwen3-TTS (~4 GB)
# Voxtral (~8 GB) needs to be downloaded too. Easiest: re-run download_model.sh
# with `Qwen/Qwen3-TTS-...` swapped for `mistralai/Voxtral-4B-TTS-2603`, or
# `hf download mistralai/Voxtral-4B-TTS-2603 --local-dir /workspace/models/Voxtral-4B-TTS-2603`.
```

### 3. Run + test

```bash
./start_services.sh     # ≈ 9 min — Voxtral cold + Qwen cold + LiteLLM
./test_endpoints.sh     # 9 calls on voxtral-tts, prints HTTP/size for each
```

If `start_services.sh` reports `vLLM did NOT become healthy`, the actual vllm process may still be booting (cold-start CUDAGraph capture is slow). Re-run the script — it'll skip starting if a healthy server is already running.

## Authentication

LiteLLM uses a [custom_auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth) module ([`auth.py`](auth.py)) so we can hand out **multiple pre-shared keys without running a database**. Every env var named `VOXTRAL_KEY_<NAME>` becomes a valid Bearer token; the `<NAME>` suffix (lower-cased) becomes the LiteLLM `user_id` for logging.

The repo's defaults define two:

| Env var | Intended user | Where it's used |
|---|---|---|
| `VOXTRAL_KEY_OWNER` | the operator (you) | private code, scripts, dashboards |
| `VOXTRAL_KEY_COLLEAGUE` | someone you trust to test the endpoint | hand off via Bitwarden / 1Password |

To add a third (or fifth), just add another `VOXTRAL_KEY_<NAME>=sk-voxtral-...` to `.voxtral.env`, run `./restart-pod.sh`, and the new key is live. To revoke one, delete the line and `restart-pod.sh` again — the change reaches the pod via SSH stdin (the keys never touch git, never appear in `ps`).

The **master key** (`VOXTRAL_LITELLM_MASTER_KEY`) is admin-only with `custom_auth` on: it grants access to LiteLLM's `/key/*` and `/metrics` routes but **not** to `/v1/audio/speech`. So leaking the master key doesn't let anyone synthesize audio — they'd only see proxy metadata. Still, treat it as a secret.

Generate fresh keys with:

```bash
python3 -c 'import secrets; print("sk-voxtral-owner-" + secrets.token_urlsafe(24))'
```

## Operating notes

- **`--omni` is mandatory** on `vllm serve`. Without it, `/v1/audio/speech` returns 404 — vanilla vLLM doesn't expose that route.
- **Per-stage GPU caps come from YAML, not the CLI.** vLLM-Omni overrides `--gpu-memory-utilization` with hardcoded values in `vllm_omni/model_executor/stage_configs/*.yaml` (Voxtral S0=0.8, Qwen S0=0.3 by default). Two 2-stage models on one GPU OOM under those defaults. `install_voxtral.sh` `sed`s them down to `Voxtral S0=0.30/S1=0.05` and `Qwen S0=0.15/S1=0.15` so both fit on a 48 GB card with ~17 GiB margin. Re-run `install_voxtral.sh` if you ever pip-reinstall vllm-omni — the YAMLs would otherwise revert to the upstream defaults.
- **Sequential boot.** The two vLLM instances are launched one after the other in `start_services.sh`. Parallel boot triggered OOM during CUDA-graph capture; sequential adds ~5 min total but is reliable.
- **Install order matters** between vllm and vllm-omni. `install_voxtral.sh` resolves them in a single `uv pip install` to avoid the entrypoint and ABI traps documented in [Troubleshooting](#troubleshooting).
- **Cold start is dominated by stage-1 init** (audio decoder warmup + CUDA graph capture). The default `start_services.sh` polls each `/health` for 15 minutes; that's intentional.
- **LiteLLM auth** is backed by a tiny `custom_auth` module ([`auth.py`](auth.py)) that reads `VOXTRAL_KEY_*` env vars on the pod. The local `.voxtral.env` is the source of truth; `restart-pod.sh` syncs the keys onto the pod via SSH stdin so they never appear in argv. See the [Authentication](#authentication) section above.
- **Secrets**: `.voxtral.env` is git-ignored and chmod 600. `HF_TOKEN` is also written into the pod's container env at creation time; it is NOT visible to interactive SSH shells (RunPod only exposes pod env on PID 1, so the install scripts source it from `/proc/1/environ`).

## Versions (exact, known-good)

See [`versions.lock.json`](versions.lock.json) for the full lockfile. Key pins:

| Package | Version | Why pinned |
|---|---|---|
| `vllm` | 0.18.1 | vllm-omni 0.18.0 imports `vllm.inputs.data.TokensPrompt`, removed in vllm ≥ 0.20 |
| `vllm-omni` | 0.18.0 | Latest released (no 0.19/0.20 exist yet); supports both Voxtral and Qwen3-TTS |
| `torch` | 2.10.0 | vllm 0.18.1's `_C.abi3.so` is ABI-linked against torch 2.10; 2.11 yields `undefined symbol _ZN3c1013MessageLoggerC1...` |
| `torchaudio`/`torchvision` | 2.10.0 / 0.25.0 | Match torch |
| `transformers` | 4.57.6 | Compatible with vllm 0.18 |
| `flashinfer-python`/`flashinfer-cubin` | 0.6.6 / 0.6.6 | Both must match — newer cubin (0.6.8) refuses to load against 0.6.6 python |
| `mistral_common` | ≥ 1.10 | Required by vllm-omni for Voxtral's tokenizer parser |
| `huggingface_hub[cli]` | < 1.0 | `transformers 4.57` hard-requires `<1.0`; the 1.x release ships a different CLI shape |
| `litellm[proxy]` | 1.83.x | Any recent should work; 1.83.14 verified |

Apt packages: `python3.10-venv python3.10-dev build-essential ffmpeg libsndfile1`. The `-dev` headers are needed by Triton's first-run gcc compile of `cuda_utils.c`.

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `Free memory on device cuda:0 (X/Y GiB) on startup is less than desired GPU memory utilization` | vllm-omni stage-config YAML cap > free memory after the first model loaded | Re-run `install_voxtral.sh` to re-apply the YAML caps, OR drop them lower by hand in `vllm_omni/model_executor/stage_configs/*.yaml` |
| `huggingface-cli: deprecated, use hf` | huggingface_hub 1.x renamed the binary | `download_model.sh` already uses `hf download` |
| `ImportError: huggingface-hub>=0.34.0,<1.0 is required ...` | huggingface_hub 1.x pulled in by `-U`, but transformers 4.57 wants <1.0 | Pin `huggingface_hub[cli]<1.0` (already done in `install_voxtral.sh`) |
| `ModuleNotFoundError: No module named 'vllm.inputs.data'` | vllm 0.20 with vllm-omni 0.18 | Pin `vllm==0.18.1` (already in `install_voxtral.sh`) |
| `ImportError: undefined symbol _ZN3c1013MessageLogger...` | torch 2.11 with vllm 0.18 (ABI mismatch) | Pin `torch==2.10.0` (already in `install_voxtral.sh`) |
| `vllm: error: unrecognized arguments: --omni` | A later `pip install vllm` overwrote the vllm-omni entrypoint | `install_voxtral.sh` rewrites `<venv>/bin/vllm` after install — re-run it, or do it by hand |
| `flashinfer-cubin version (X.X) does not match flashinfer version (Y.Y)` | Mismatch between `flashinfer-python` and `flashinfer-cubin` | Pin both to the same version |
| `InductorError: ... Python.h: No such file or directory` | Triton's gcc-based JIT can't find Python headers | `apt install python3.10-dev build-essential` |
| `vLLM did NOT become healthy` after 15 min | Stage-1 (audio decoder) init genuinely failed | `tail /workspace/logs/vllm-{voxtral,qwen}.log` and grep for the actual error; restart with `start_services.sh` |
| LiteLLM returns 401 `invalid api key` | Bearer token isn't in the `VOXTRAL_KEY_*` allowlist | Use `$VOXTRAL_KEY_OWNER` (or `$VOXTRAL_KEY_COLLEAGUE`); the master key alone won't work on `/v1/audio/speech` by design |
| LiteLLM returns 401 `missing api key` | No `Authorization: Bearer …` header at all | Add the header |
| `qwen-tts` returns 400 `Invalid language 'fr'` | Qwen takes capitalised English language names, not ISO codes | Send `"French"` (or `"Auto"`), not `"fr"` |
| Audio file is 0 bytes / WAV without RIFF header | Bad voice name | Pick from the 20 listed in the Voxtral table or the 9 in the Qwen table |
| `HTTP 403, error code: 1010` from the public proxy URL | Cloudflare in front of `*.proxy.runpod.net` rejects `Python-urllib/*` UA | Send any non-default `User-Agent` header — curl works out of the box |

## Credits

- **[Mistral AI](https://mistral.ai/news/voxtral-tts)** for releasing Voxtral with open weights and 20 reference voices, including native European-language pairs.
- **[Alibaba Qwen team](https://github.com/QwenLM/Qwen3-TTS)** for releasing Qwen3-TTS under Apache 2.0.
- **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** team — Han Gao, Hongsheng Liu, Roger Wang, Yueqian Lin — for the audio-capable vLLM fork that makes `/v1/audio/speech` possible for both models.
- **[BerriAI/LiteLLM](https://github.com/BerriAI/LiteLLM)** for the OpenAI-shaped proxy.
- **[RunPod](https://runpod.io/)** for the GPU billing-by-the-second + HTTPS proxy that makes single-pod hosting viable.

## License

- Scripts in this repository: **MIT** (see [LICENSE](LICENSE)).
- **Voxtral-4B-TTS-2603** (the `voxtral-tts` model), retrieved at runtime from HuggingFace, is licensed **CC BY-NC 4.0** by Mistral AI. **Use is non-commercial only.** This includes the 20 voice presets. This repo neither redistributes nor relicenses the model.
- **Qwen3-TTS-12Hz-1.7B-CustomVoice** (the `qwen-tts` model), retrieved at runtime from HuggingFace, is licensed **Apache 2.0** by Alibaba Cloud. Commercial use is allowed. If your use case is commercial, route everything to `model: "qwen-tts"`.
