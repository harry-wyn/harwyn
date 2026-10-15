#!/usr/bin/env bash
# Stage 5: rejection-sampling SFT (RFT) on top of the stage-4 model, still QAT (q7 weights, int4 KV).
#   S  rebuild the stage-4 model from its LoRA (models/harwyn-4b-sft-q7-fq)
#   G  serve it on vLLM (int4 KV) and run data/rft_gen.py: KodCode problems x 4 samples, unit-tested,
#      keep the shortest passing answer per problem, plus agentic replay from sft_v1
#   T  continue training the stage-4 LoRA on that data (lower LR)
#   F  build models/harwyn-4b-rft-q7-fq
#   E  HumanEval+, GPQA Diamond first 99, RULER 32k on vLLM with int4 KV (MAX_TOKENS, default 32k)
#   HARWYN_WORK  work dir; expects models/h0, data/sft_v1.jsonl, runs/sft_v1/lora_step212.pt, .venv-train, .venv-eval
set -euo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
trap "${ON_EXIT:-true}" EXIT
REPO=$(cd "$(dirname "$0")/.." && pwd)
A=$REPO/results/stage2/assignment_q7.json
TP=$W/.venv-train/bin/python
SFT=$W/models/harwyn-4b-sft-q7-fq
RUN=$W/runs/rft_v1
M=$W/models/harwyn-4b-rft-q7-fq
OUT=$W/results/stage5; mkdir -p $OUT
done_() { [ -f $OUT/.done_$1 ]; }
mark() { touch $OUT/.done_$1; echo "$(date +%T) phase $1 done"; }
serve() {  # serve MODEL LOG
  $W/.venv-eval/bin/vllm serve $1 --served-model-name m --port 8000 --max-model-len 40960 --language-model-only \
    --reasoning-parser qwen3 --gpu-memory-utilization 0.92 --kv-cache-dtype int4_per_token_head \
    --override-generation-config "{\"max_new_tokens\": ${MAX_TOKENS:-32768}}" > $2 2>&1 &
  V=$!
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; kill -0 $V 2>/dev/null || { echo "vllm died"; exit 1; }; done
}

if ! done_ S; then
  $TP $REPO/train/train_qat.py $W/models/h0 none $A $W/runs/sft_export --init $W/runs/sft_v1/lora_step212.pt --export-only > $OUT/S_export.log 2>&1
  $TP $REPO/evals/fakequant_from_db.py $W/models/h0 $W/runs/sft_export/merged_fq.pt $A $SFT >> $OUT/S_export.log 2>&1
  mark S
fi

if ! done_ G; then
  serve $SFT $OUT/G_vllm.log
  (cd $REPO/data && $W/.venv-eval/bin/python rft_gen.py $W/data/rft_v1.jsonl $W/data/sft_v1.jsonl 2400 4) > $OUT/G_gen.log 2>&1
  kill -9 $V || true; pkill -9 -f "[V]LLM::" || true; sleep 5
  cp $W/data/rft_v1.raw.jsonl $OUT/
  mark G
fi

if ! done_ T; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TP $REPO/train/train_qat.py $W/models/h0 $W/data/rft_v1.jsonl $A $RUN \
    --init $W/runs/sft_v1/lora_step212.pt --max-len 16384 --rank 32 --lr 5e-5 --accum 16 --epochs 1 >> $OUT/T_train.log 2>&1
  mark T
fi

if ! done_ F; then
  $TP $REPO/evals/fakequant_from_db.py $W/models/h0 $RUN/merged_fq.pt $A $M > $OUT/F_build.log 2>&1
  mark F
fi

if ! done_ E; then
  R=$OUT/eval; mkdir -p $R
  serve $M $R/vllm.log
  source $W/.venv-eval/bin/activate; cd $REPO/evals
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
echo "$(date +%T) stage 5 done"
