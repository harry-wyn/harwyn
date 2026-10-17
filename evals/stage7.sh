#!/usr/bin/env bash
# Stage 7: short QAT SFT top-up (concise-reasoning coding + replay) on the stage-6 LoRA, then the three main
# benchmarks for the final model (q7 weights, int4 KV) AND the base model (Qwen3.5-4B as shipped), both at the
# 32k budget. gen.py records every answer's token count, so thinking length is reported too.
#   T  train   F  build final model   EF  evals, final model   EB  evals, base model
#   HARWYN_WORK  work dir; expects models/h0, models/base, data/sft_v2.jsonl, runs/grpo_v1/lora_it16.pt,
#                .venv-train, .venv-eval
set -euo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
REPO=$(cd "$(dirname "$0")/.." && pwd)
A=$REPO/results/stage2/assignment_q7.json
TP=$W/.venv-train/bin/python
RUN=$W/runs/sft_v2
M=$W/models/harwyn-4b-final-q7-fq
OUT=$W/results/stage7; mkdir -p $OUT
export PATH=$W/.venv-eval/bin:$PATH
done_() { [ -f $OUT/.done_$1 ]; }
mark() { touch $OUT/.done_$1; echo "$(date +%T) phase $1 done"; }

evals() {  # evals MODEL OUTDIR [extra vllm args...]
  local m=$1 R=$2; shift 2; mkdir -p $R
  vllm serve $m --served-model-name m --port 8000 --max-model-len 40960 --language-model-only --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.85 --max-num-batched-tokens 16384 "$@" \
    --override-generation-config "{\"max_new_tokens\": ${MAX_TOKENS:-32768}}" > $R/vllm.log 2>&1 &
  local V=$!
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; kill -0 $V 2>/dev/null || { echo "vllm died"; exit 1; }; done
  (cd $REPO/evals
   python gen.py $R humaneval > $R/gen_humaneval.log 2>&1 & P1=$!
   LIMIT=99 python gen.py $R gpqa > $R/gen_gpqa.log 2>&1 & P2=$!
   lm_eval --model local-completions \
     --model_args model=m,base_url=http://localhost:8000/v1/completions,tokenizer=$m,num_concurrent=4,timeout=3600,tokenized_requests=False \
     --tasks niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue \
     --metadata "{\"max_seq_lengths\":[32768],\"tokenizer\":\"$m\"}" --limit 25 \
     --use_cache $R/ruler_cache --output_path $R/ruler > $R/ruler.log 2>&1 & P3=$!
   wait $P1 $P2 $P3)
  kill -9 $V 2>/dev/null || true; pkill -9 -f "[V]LLM::" || true; sleep 5
  (cd $REPO/evals && python score.py $R > $R/score.log 2>&1)
}

if ! done_ T; then
  echo "$(date +%T) T: QAT SFT top-up"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TP $REPO/train/train_qat.py $W/models/h0 $W/data/sft_v2.jsonl $A $RUN \
    --init $W/runs/grpo_v1/lora_it16.pt --max-len 16384 --rank 32 --lr 2e-5 --accum 16 --epochs 1 >> $OUT/T_train.log 2>&1
  mark T
fi
if ! done_ F; then
  $TP $REPO/evals/fakequant_from_db.py $W/models/h0 $RUN/merged_fq.pt $A $M > $OUT/F_build.log 2>&1
  mark F
fi
if ! done_ EF; then echo "$(date +%T) EF: final model"; evals $M $OUT/eval_final --kv-cache-dtype int4_per_token_head; mark EF; fi
if ! done_ EB; then echo "$(date +%T) EB: base model"; evals $W/models/base $OUT/eval_base; mark EB; fi
echo "$(date +%T) stage 7 done"
