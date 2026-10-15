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

# Stage 2: GSQ (per-tensor low-bit quantisation) and RCO (bit allocation under a size budget).
# Each has its own lockfile/pins; flash-linear-attention keeps Qwen3.5's Gated DeltaNet layers fast.
[ -d "$W/GSQ" ] || git clone https://github.com/IST-DASLab/GSQ "$W/GSQ"
git -C "$W/GSQ" checkout -q 03fc164
(cd "$W/GSQ" && uv sync)   # flash-attn build skipped: not needed
VIRTUAL_ENV=$W/GSQ/.venv uv pip install flash-linear-attention
[ -d "$W/RCO" ] || git clone https://github.com/IST-DASLab/RCO "$W/RCO"
git -C "$W/RCO" checkout -q 9a1e09c
# Patches (see LOG.md): rco-single-gpu skips NCCL init on 1 GPU (its barrier segfaults here);
# rco-resume makes run_quantize skip layers already in the database (preemptible GPUs);
# rco-split-symmetric imports GSQ's symmetric low-bit codes; rco-streamed-deltas fits the search in 24 GB.
for p in "$(cd "$(dirname "$0")" && pwd)"/patches/rco-*.patch; do
  git -C "$W/RCO" apply --check "$p" 2>/dev/null && git -C "$W/RCO" apply "$p"
done
venv rco
VIRTUAL_ENV=$W/.venv-rco uv pip install -r "$W/RCO/requirements.txt" flash-linear-attention \
  --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
# RCO pins transformers 5.7 for an MoE code path; 5.7 can't build Qwen3.5 from its config, 5.8.1 can.
VIRTUAL_ENV=$W/.venv-rco uv pip install transformers==5.8.1
# Humming kernels let vLLM serve 2/3/5/6/7-bit compressed-tensors weights (Marlin covers 4/8).
VIRTUAL_ENV=$W/.venv-eval uv pip install humming-kernels
# Stage 4+ training (train/train_qat.py)
venv train
VIRTUAL_ENV=$W/.venv-train uv pip install torch==2.11.0 transformers==5.8.1 flash-linear-attention==0.5.2 safetensors datasets
# Stage 5 runs generated code against unit tests
VIRTUAL_ENV=$W/.venv-eval uv pip install pytest

HF_HUB_ENABLE_HF_TRANSFER=1 "$W/.venv-eval/bin/hf" download "$BASE_MODEL" --local-dir "$W/models/base"
echo "done: base model in $W/models/base"
