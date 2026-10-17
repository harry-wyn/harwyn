#!/usr/bin/env bash
# Build the CPU (llama.cpp) version of harwyn-4b from the released weights.
#   1. merge the final LoRA into h0, fake-quantised exactly as trained/evaluated (export/merge_lora.py)
#   2. convert to GGUF (llama.cpp b11709 converter, transformers 5.8.1 for the Qwen3.5 tokenizer)
#   3. quantise with the model's own bit map: MLP gate/up -> Q6_K, everything else -> Q8_0
#      (the vLLM checkpoint keeps embeddings in bf16; Q8_0 here saves ~0.6 GB of RAM at negligible cost)
# usage: H0=<stage1-h0 dir> LORA=<stage7-final/lora_step218.pt> LLAMA=<llama.cpp dir with src/ and bin/> OUT=<dir> bash build_gguf.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
python "$REPO/export/merge_lora.py" "$H0" "$LORA" "$REPO/results/stage2/assignment_q7.json" "$OUT/harwyn-4b-hf"
python "$LLAMA/src/convert_hf_to_gguf.py" "$OUT/harwyn-4b-hf" --outtype bf16 --outfile "$OUT/harwyn-4b-bf16.gguf"
"$LLAMA/bin/llama-quantize" --tensor-type ffn_gate=q6_k --tensor-type ffn_up=q6_k \
  "$OUT/harwyn-4b-bf16.gguf" "$OUT/harwyn-4b-q7.gguf" q8_0
