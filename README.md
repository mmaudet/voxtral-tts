# voxtral-tts

Self-hostable recipe for **Mistral AI's Voxtral-4B-TTS-2603** on a single RunPod GPU, served by **vLLM-Omni** with an OpenAI-compatible `/v1/audio/speech` endpoint and a **LiteLLM** proxy in front of it.

> ⚠ **Non-commercial only.** Voxtral and its 20 voice presets are licensed CC BY-NC 4.0. The scripts in this repo are MIT, but the *model* you'll run is not — see [LICENSE](LICENSE) and the [Mistral model card](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603).

## What you get

- Two HTTPS endpoints, both speaking the OpenAI `/v1/audio/speech` schema:
  - **vLLM direct** — anonymous, low-overhead, recommended for internal calls.
  - **LiteLLM proxy** — gated by a master key, drops in for any client that already speaks OpenAI's API (Python `openai` SDK, LangChain, n8n, custom HTTP).
- 9 supported languages: English, French, Spanish, Portuguese, Italian, Dutch, German, Arabic, Hindi.
- 20 reference voices (5 generic + 12 language-tagged + 1 ar_male + 2 hi).
- Output: 24 kHz WAV / PCM / FLAC / MP3 / AAC / Opus.
- Boots cold in ≈ 3-4 min on an RTX A5000 (24 GB) at ≈ **$0.27 / hour**.

## What you DON'T get

- Multi-tenant routing, queueing, autoscaling — this is a single-pod recipe.
- A model gateway that hides the CC BY-NC license. **You** are responsible for compliance.
- Voice cloning out of the box. The model card mentions adaptation; this repo only exposes the 20 stock voices.

## Architecture

```
┌─────────────────────── RunPod pod (RTX A5000 24 GB) ────────────────────────┐
│                                                                             │
│   client ─► :4000  LiteLLM proxy  (model alias `voxtral-tts`, master_key)   │
│                       │                                                     │
│                       └─► :8000  vLLM-Omni                                  │
│                                  Voxtral-4B-TTS-2603 (BF16, 7.78 GiB)       │
│                                  + 10.4 GiB KV cache (≈ 25× concurrency)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                       ▲                                  ▲
            client (anywhere)                   client (anywhere)
            via https://<pod>-4000             via https://<pod>-8000
            .proxy.runpod.net                  .proxy.runpod.net
```

Both ports are reachable through RunPod's public HTTPS proxy. The vLLM port is unauthenticated by design (vLLM does not enforce auth on `/v1/audio/speech`); use the LiteLLM port if you need a key gate.

## Quick start

You'll need a [RunPod account](https://www.runpod.io/) with billing set up and a [HuggingFace token](https://huggingface.co/settings/tokens). The model itself is not gated.

```bash
git clone git@github.com:mmaudet/voxtral-tts.git
cd voxtral-tts

# 1. Configure secrets
cp .voxtral.env.example .voxtral.env
chmod 600 .voxtral.env
$EDITOR .voxtral.env   # paste RUNPOD_API_KEY and HF_TOKEN

# 2. Provision the pod (RTX A5000, ports 8000+4000+22 exposed, your SSH key injected)
source .voxtral.env
./provision-pod.sh     # see "Provisioning" below — not yet a single script,
                       # this README walks through the API call
```

Once the pod is `RUNNING`, copy the install + serve scripts onto it and run them. The first boot takes ≈ 12 min (install) + ≈ 1 min (download) + ≈ 4 min (vLLM 2-stage init); subsequent restarts skip the install and the download, taking ≈ 4 min.

Detailed steps are in [Deploying from scratch](#deploying-from-scratch).

## Endpoints

Once the pod is up, both URLs are reachable from any machine:

```bash
# vLLM direct (no auth)
curl -s https://<pod-id>-8000.proxy.runpod.net/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "input": "Bonjour, ceci est Voxtral.",
    "voice": "fr_male",
    "response_format": "wav"
  }' \
  --output sample.wav
```

```bash
# LiteLLM proxy (auth required)
curl -s https://<pod-id>-4000.proxy.runpod.net/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-voxtral-local' \
  -d '{
    "model": "voxtral-tts",
    "input": "Test via LiteLLM.",
    "voice": "fr_female",
    "response_format": "wav"
  }' \
  --output sample-via-proxy.wav
```

The `pod-id` is whatever RunPod assigns at create-time. URLs survive a stop/start cycle but **change** if you re-create the pod. See `runpod-pod-info.example.json` for the metadata schema; the actual `runpod-pod-info.json` is git-ignored because it's per-pod state.

If your pod is stopped (`EXITED`) to save cost, the easiest way back is:

```bash
./restart-pod.sh                # uses runpod-pod-info.json[voxtral-main].podId
# or
./restart-pod.sh <pod-id>       # explicit
```

The script POSTs `/v1/pods/<id>/start`, polls until `RUNNING` with `publicIp` and SSH port assigned, syncs the latest local `start_services.sh` + `litellm_config.yaml` to the pod, runs `start_services.sh` (vLLM + LiteLLM), and prints the proxy URLs. Total cold restart ≈ 5-6 min.

## Voices (20 presets)

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

The `voice` field selects the voice embedding (prosody, timbre, accent). The `input` text drives the language. A non-matching pair (e.g. `de_female` + English text) usually works but the prosody favours the voice's native language.

Response formats: `wav` (default), `pcm`, `flac`, `mp3`, `aac`, `opus`. Output is 24 kHz mono.

## Repository layout

```
.
├── README.md                       this file
├── LICENSE                         MIT for scripts; model is CC BY-NC 4.0
├── .voxtral.env.example            template for secrets — copy to .voxtral.env (gitignored)
├── .gitignore
├── runpod-pod-info.example.json    schema for the per-pod state file (gitignored real one)
├── versions.lock.json              the *exact* package versions known to work
├── litellm_config.yaml             proxy config: alias `voxtral-tts`, master_key sk-voxtral-local
├── install_voxtral.sh              one-shot install on the pod (idempotent)
├── download_model.sh               pull the model to /workspace/models (idempotent)
├── start_services.sh               launch vLLM + LiteLLM on the pod (idempotent)
├── restart-pod.sh                  local: start a stopped pod end-to-end (start API → SSH → services → URLs)
└── test_endpoints.sh               smoke-test 7 European languages on both endpoints
```

## Deploying from scratch

The recipe runs in three stages. Each stage's script is idempotent: rerunning is safe.

### 1. Provision the pod (RunPod REST API)

A bare `POST /v1/pods` call. Below is what worked for this deployment — adjust GPU type to whatever's in stock at the moment. The OpenAPI spec is at `https://rest.runpod.io/v1/openapi.json` if you want to tweak.

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
  "gpuTypeIds": ["NVIDIA RTX A5000", "NVIDIA RTX A4500", "NVIDIA RTX A6000"],
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
                  litellm_config.yaml test_endpoints.sh \
                  root@<publicIp>:/workspace/

# on the pod
ssh -p <ssh_port> root@<publicIp>
cd /workspace
chmod +x install_voxtral.sh download_model.sh start_services.sh test_endpoints.sh
./install_voxtral.sh    # ≈ 12 min — installs vllm/vllm-omni/torch/litellm with the
                        # exact pinned versions from versions.lock.json
./download_model.sh     # ≈ 1 min — pulls 8 GB from HF to /workspace/models/
```

### 3. Run + test

```bash
./start_services.sh     # ≈ 4 min — vLLM stage-0 + stage-1 init, then LiteLLM
./test_endpoints.sh     # 9 calls across vLLM + LiteLLM, prints HTTP/size for each
```

If `start_services.sh` reports `vLLM did NOT become healthy`, the actual vllm process may still be booting (cold-start CUDAGraph capture is slow). Re-run the script — it'll skip starting if a healthy server is already running.

## Test report (from this repo's deployment, 2026-04-28)

7 languages on vLLM, 2 on the LiteLLM proxy, 0 failures:

| Endpoint | Voice | Lang | HTTP | WAV size | Mac→pod→Mac latency |
|---|---|---|---|---|---|
| vLLM | `neutral_female` | EN | 200 | 376 KB | — |
| vLLM | `fr_male` | FR | 200 | 282 KB | 3.3 s |
| vLLM | `de_female` | DE | 200 | 169 KB | — |
| vLLM | `es_male` | ES | 200 | 184 KB | — |
| vLLM | `it_female` | IT | 200 | 184 KB | — |
| vLLM | `nl_male` | NL | 200 | 203 KB | — |
| vLLM | `pt_female` | PT | 200 | 181 KB | — |
| LiteLLM | `fr_female` | FR | 200 | 214 KB | 2.6 s |
| LiteLLM | `neutral_male` | EN | 200 | 128 KB | — |

All outputs: PCM 16-bit mono 24 kHz WAV. The first call after a cold boot is roughly 10× slower than steady-state.

## Cost

| Phase | Duration | Cost |
|---|---|---|
| First-time deploy (install + download + boot + test) | ≈ 1 h 30 min | ≈ $0.41 |
| Steady state, pod running idle | per hour | $0.27 |
| Pod stopped, volume retained | per day | ≈ $0.007 (50 GB × $0.07/mo) |

Pricing is whatever Secure Cloud charges for an A5000 at the time you run it. Always check `nvidia-smi` after a stop/start cycle: stopped pods don't burn GPU dollars but mistakenly leaving one `RUNNING` overnight does.

## Operating notes

- **`--omni` is mandatory** on `vllm serve`. Without it, `/v1/audio/speech` returns 404 — vanilla vLLM doesn't expose that route.
- **Install order matters** between vllm and vllm-omni. The current `install_voxtral.sh` resolves them in a single `uv pip install` to avoid the entrypoint and ABI traps documented in [Troubleshooting](#troubleshooting).
- **Cold start is dominated by stage-1 init** (audio decoder warmup + CUDA graph capture). The default `start_services.sh` polls `/health` for 15 minutes; that's intentional.
- **The LiteLLM master_key is `sk-voxtral-local`**, hard-coded for sandbox use. Rotate before you point anything you care about at this. Easy fix: edit `litellm_config.yaml` and restart the proxy.
- **The vLLM 8000 port is anonymous on the public proxy.** If that's not OK for you, either firewall it via RunPod's port settings, or front everything through LiteLLM 4000 only.
- **Secrets**: `.voxtral.env` is git-ignored and chmod 600. `HF_TOKEN` is also written into the pod's container env at creation time; it is NOT visible to interactive SSH shells (RunPod only exposes pod env on PID 1, so the install scripts source it from `/proc/1/environ`).

## Versions (exact, known-good)

See [`versions.lock.json`](versions.lock.json) for the full lockfile. Key pins:

| Package | Version | Why pinned |
|---|---|---|
| `vllm` | 0.18.1 | vllm-omni 0.18.0 imports `vllm.inputs.data.TokensPrompt`, removed in vllm ≥ 0.20 |
| `vllm-omni` | 0.18.0 | Latest released (no 0.19/0.20 exist yet) |
| `torch` | 2.10.0 | vllm 0.18.1's `_C.abi3.so` is ABI-linked against torch 2.10; 2.11 yields `undefined symbol _ZN3c1013MessageLoggerC1...` |
| `torchaudio`/`torchvision` | 2.10.0 / 0.25.0 | Match torch |
| `transformers` | 4.57.6 | Compatible with vllm 0.18 |
| `flashinfer-python`/`flashinfer-cubin` | 0.6.6 / 0.6.6 | Both must match — newer cubin (0.6.8) refuses to load against 0.6.6 python |
| `mistral_common` | ≥ 1.10 | Required by vllm-omni for the Voxtral tokenizer parser |
| `huggingface_hub[cli]` | < 1.0 | `transformers 4.57` hard-requires `<1.0`; the 1.x release ships a different CLI shape |
| `litellm[proxy]` | 1.83.x | Any recent should work; 1.83.14 verified |

Apt packages: `python3.10-venv python3.10-dev build-essential ffmpeg libsndfile1`. The `-dev` headers are needed by Triton's first-run gcc compile of `cuda_utils.c`.

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `huggingface-cli: deprecated, use hf` | huggingface_hub 1.x renamed the binary | `download_model.sh` already uses `hf download` |
| `ImportError: huggingface-hub>=0.34.0,<1.0 is required ...` | huggingface_hub 1.x pulled in by `-U`, but transformers 4.57 wants <1.0 | Pin `huggingface_hub[cli]<1.0` (already done in `install_voxtral.sh`) |
| `ModuleNotFoundError: No module named 'vllm.inputs.data'` | vllm 0.20 with vllm-omni 0.18 | Pin `vllm==0.18.1` (already in `install_voxtral.sh`) |
| `ImportError: undefined symbol _ZN3c1013MessageLogger...` | torch 2.11 with vllm 0.18 (ABI mismatch) | Pin `torch==2.10.0` (already in `install_voxtral.sh`) |
| `vllm: error: unrecognized arguments: --omni` | A later `pip install vllm` overwrote the vllm-omni entrypoint | `install_voxtral.sh` rewrites `<venv>/bin/vllm` after install — re-run it, or do it by hand |
| `flashinfer-cubin version (X.X) does not match flashinfer version (Y.Y)` | Mismatch between `flashinfer-python` and `flashinfer-cubin` | Pin both to the same version |
| `InductorError: ... Python.h: No such file or directory` | Triton's gcc-based JIT can't find Python headers | `apt install python3.10-dev build-essential` |
| `vLLM did NOT become healthy` after 15 min | Stage-1 (audio decoder) init genuinely failed | `tail /workspace/logs/vllm.log` and grep for the actual error; restart with `start_services.sh` |
| LiteLLM returns 401 | Wrong/missing master_key | Header must be `Authorization: Bearer sk-voxtral-local` |
| Audio file is 0 bytes / WAV without RIFF header | Bad voice name | Pick from the 20 listed above |

## Credits

- **[Mistral AI](https://mistral.ai/news/voxtral-tts)** for releasing Voxtral with open weights and reference voices.
- **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)** team — Han Gao, Hongsheng Liu, Roger Wang, Yueqian Lin — for the audio-capable vLLM fork that makes `/v1/audio/speech` possible.
- **[BerriAI/LiteLLM](https://github.com/BerriAI/LiteLLM)** for the OpenAI-shaped proxy.
- **[RunPod](https://runpod.io/)** for the GPU billing-by-the-second + HTTPS proxy that makes single-pod hosting viable.

## License

- Scripts in this repository: **MIT** (see [LICENSE](LICENSE)).
- The Voxtral-4B-TTS-2603 model and its 20 voice presets, retrieved at runtime from HuggingFace, are licensed **CC BY-NC 4.0** by Mistral AI. **Use is non-commercial only.** This repo neither redistributes nor relicenses the model.
