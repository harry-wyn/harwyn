#!/usr/bin/env bash
# Stage 4: QAT SFT of h0 on data/sft_v1.jsonl, then the three quick evals on vLLM with int4 KV.
#   T  train/train_qat.py: LoRA r32, merged weights fake-quantised to the stage-2 q7 widths, K/V int4 fake-quant
#   F  write the trained weights into a fake-quant copy of h0 (same layout as the stage-2 q7 eval model)
#   E  HumanEval+, GPQA Diamond first 99, RULER 32k 25/task; served with --kv-cache-dtype int4_per_token_head
# Each phase writes a .done marker and is skipped on rerun (T also resumes from its checkpoint).
#   EPOCHS  fraction of data/sft_v1.jsonl to train on (default 1; the first run used 0.65 to save compute)
#   HARWYN_WORK  work dir (required); expects models/h0, data/sft_v1.jsonl, .venv-train, .venv-eval
set -euo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
trap "${ON_EXIT:-true}" EXIT
REPO=$(cd "$(dirname "$0")/.." && pwd)
A=$REPO/results/stage2/assignment_q7.json
RUN=$W/runs/sft_v1
M=$W/models/harwyn-4b-sft-q7-fq
OUT=$W/results/stage4; mkdir -p $OUT
done_() { [ -f $OUT/.done_$1 ]; }
mark() { touch $OUT/.done_$1; echo "$(date +%T) phase $1 done"; }

if ! done_ T; then
  echo "$(date +%T) T: QAT SFT"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $W/.venv-train/bin/python $REPO/train/train_qat.py \
    $W/models/h0 $W/data/sft_v1.jsonl $A $RUN --max-len 16384 --rank 32 --lr 1e-4 --accum 16 --epochs ${EPOCHS:-1} \
    >> $OUT/T_train.log 2>&1
  mark T
fi

if ! done_ F; then
  echo "$(date +%T) F: build $M"
  $W/.venv-train/bin/python $REPO/evals/fakequant_from_db.py $W/models/h0 $RUN/merged_fq.pt $A $M > $OUT/F_build.log 2>&1
  mark F
fi

if ! done_ E; then
  source $W/.venv-eval/bin/activate
  cd $REPO/evals
  R=$OUT/eval; mkdir -p $R
  echo "$(date +%T) E: serve $M"
  vllm serve $M --served-model-name m --port 8000 --max-model-len 40960 --language-model-only \
    --reasoning-parser qwen3 --gpu-memory-utilization 0.92 --kv-cache-dtype int4_per_token_head \
    --override-generation-config '{"max_new_tokens": 8192}' > $R/vllm.log 2>&1 &
  V=$!
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; kill -0 $V 2>/dev/null || { echo "vllm died"; exit 1; }; done
  python gen.py $R humaneval > $R/gen_humaneval.log 2>&1 & P1=$!
  LIMIT=99 python gen.py $R gpqa > $R/gen_gpqa.log 2>&1 & P2=$!
  lm_eval --model local-completions \
    --model_args model=m,base_url=http://localhost:8000/v1/completions,tokenizer=$M,num_concurrent=4,timeout=3600,tokenized_requests=False \
    --tasks niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue \
    --metadata "{\"max_seq_lengths\":[32768],\"tokenizer\":\"$M\"}" --limit 25 \
    --use_cache $R/ruler_cache --output_path $R/ruler > $R/ruler.log 2>&1 & P3=$!
  wait $P1 $P2 $P3
  kill -9 $V || true; pkill -9 -f "[V]LLM::" || true
  python score.py $R > $R/score.log 2>&1
  mark E
fi
echo "$(date +%T) stage 4 done"
