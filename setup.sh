#!/usr/bin/env bash
# Environment setup on the GPU machine (tested: one NVIDIA L4 24 GB, driver 580, Python 3.12). Re-runnable.
# One venv per tool: vLLM, bfcl-eval and Heretic pin conflicting dependencies.
# Versions are the exact ones the published results were produced with.
# Training-stage envs (Unsloth/TRL, GSQ/RCO) get added here when those stages run.
set -euo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}            # persistent work dir: models/, results/, venvs
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3.5-4B}     # swap to the Qwen 4 small sibling if one ships

command -v uv >/dev/null || pip install -U uv
venv() { [ -d "$W/.venv-$1" ] || uv venv --python 3.12 "$W/.venv-$1"; }

venv eval     # vLLM serving + every eval except BFCL
VIRTUAL_ENV=$W/.venv-eval uv pip install "vllm==0.31.0" "lm-eval[api,ruler]==0.4.13" "evalplus==0.2.1" \
  "math-verify==0.9.0" datasets openai hf_transfer langdetect immutabledict nltk

venv bfcl     # bfcl-eval's pins conflict with vLLM's; it only talks HTTP to the server
VIRTUAL_ENV=$W/.venv-bfcl uv pip install "bfcl-eval==2026.3.23" soundfile

venv heretic
VIRTUAL_ENV=$W/.venv-heretic uv pip install \
  "heretic-llm @ git+https://github.com/p-e-w/heretic@49b56e5af43f8f602af0f251a118b1bf8b19667f" flash-linear-attention
# Heretic's pyproject points torch at the CPU wheel index; replace it with the CUDA build from PyPI.
VIRTUAL_ENV=$W/.venv-heretic uv pip install --reinstall torch torchvision --index-url https://pypi.org/simple

HF_HUB_ENABLE_HF_TRANSFER=1 "$W/.venv-eval/bin/hf" download "$BASE_MODEL" --local-dir "$W/models/base"
echo "done: base model in $W/models/base"
