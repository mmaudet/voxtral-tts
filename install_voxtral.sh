#!/bin/bash
# Installs vLLM + vllm-omni + audio deps + LiteLLM for Voxtral-4B-TTS-2603.
# Idempotent. Critical: vllm MUST be installed before vllm-omni.

set -euo pipefail
exec > >(tee -a /workspace/logs/install.log) 2>&1
echo "=== install start: $(date -u +%FT%TZ) ==="

mkdir -p /workspace/models /workspace/logs /workspace/hf_cache

echo "=== [1/8] apt deps ==="
# python3.10-dev provides Python.h headers — without it, vLLM's first-run
# triton/inductor JIT cache build fails when gcc tries to compile cuda_utils.c
# (`fatal error: Python.h: No such file or directory`).
# build-essential is needed by triton's gcc-based runtime kernel build.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.10-venv python3.10-dev build-essential ffmpeg libsndfile1 >/dev/null

echo "=== [2/8] venv ==="
if [ ! -d /workspace/voxtral-env ]; then
  python3.10 -m venv /workspace/voxtral-env
fi
# shellcheck disable=SC1091
source /workspace/voxtral-env/bin/activate
python -V

echo "=== [3/8] pip + uv ==="
pip install --quiet --upgrade pip wheel
pip install --quiet uv

echo "=== [4/8] vLLM + vllm-omni + torch (pinned together) ==="
# Three coupled pins:
#  - vllm-omni 0.18.0 is the only released version >= 0.18 (no 0.19/0.20 exist).
#  - vllm-omni 0.18.0 imports `vllm.inputs.data.TokensPrompt` which only exists
#    in vllm 0.18.x; in 0.19+ it moved/was renamed.
#  - vllm 0.18.1's C++ extension is ABI-linked against torch 2.10.0; bumping
#    torch to 2.11 yields `undefined symbol _ZN3c1013MessageLoggerC1...`.
# Resolving them in one uv invocation forces a coherent install graph.
uv pip install \
  "vllm==0.18.1" \
  "vllm-omni==0.18.0" \
  "torch==2.10.0" \
  "torchaudio==2.10.0" \
  "torchvision==0.25.0" \
  "transformers==4.57.6"

# vllm-omni hijacks <venv>/bin/vllm to add its --omni CLI flag, but if uv
# resolves vllm AFTER vllm-omni in any subsequent install, it overwrites the
# entrypoint. Rewriting it explicitly is idempotent and survives later pip ops.
cat > /workspace/voxtral-env/bin/vllm <<'PY'
#!/workspace/voxtral-env/bin/python3
import sys
from vllm_omni.entrypoints.cli.main import main
if __name__ == "__main__":
    sys.exit(main() or 0)
PY
chmod +x /workspace/voxtral-env/bin/vllm

echo "=== [5/8] verify mistral_common >= 1.10 ==="
python -c "import mistral_common; v=mistral_common.__version__; print('mistral_common:', v); assert tuple(map(int, v.split('.')[:2])) >= (1,10), 'too old'"

echo "=== [6/8] sanity: vllm-omni imports clean ==="
python -c "import vllm_omni; from vllm.inputs.data import TokensPrompt; print('vllm_omni + TokensPrompt: OK')"

echo "=== [7/8] audio deps + HF CLI ==="
uv pip install librosa soundfile pydub
# transformers 4.57 hard-requires huggingface_hub<1.0; the 1.x line ships a
# new CLI shape and a different versioning policy, and trips
# `ImportError: huggingface-hub>=0.34.0,<1.0 is required ...` at vllm import.
# Pin under 1.0; `hf` CLI is available from 0.27+, no functionality lost.
uv pip install "huggingface_hub[cli]<1.0"

echo "=== [8/8] LiteLLM (with proxy extras) ==="
uv pip install "litellm[proxy]"

echo "=== verify all imports ==="
python - <<'PY'
import vllm, librosa, soundfile, mistral_common, litellm
import importlib.metadata as md
def v(p):
    try: return md.version(p)
    except md.PackageNotFoundError: return "MISSING"
print("vllm:", v("vllm"))
print("vllm-omni:", v("vllm-omni"))
print("mistral_common:", v("mistral_common"))
print("litellm:", v("litellm"))
print("librosa:", v("librosa"))
print("soundfile:", v("soundfile"))
PY

echo "=== ALL INSTALLED OK: $(date -u +%FT%TZ) ==="
