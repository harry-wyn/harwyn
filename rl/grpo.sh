#!/usr/bin/env bash
# Stage 6: GRPO with verifiable rewards, QAT throughout (q7 weights, int4 KV), starting from stage 5.
# Each iteration: serve the current policy on vLLM in its deployment format -> rl/rollout.py
# (TASKS_PER_ITER tasks x GROUP episodes, tools + hidden tests / screenshot judge) -> group-normalised
# advantages -> 2 optimiser steps of advantage-weighted QAT LoRA training -> checkpoint.
# Stops after ITERS iterations or when $W/STOP exists. Then: final model + HumanEval+/GPQA/RULER at 32k.
#   HARWYN_WORK  work dir; expects models/h0, runs/rft_v1/lora_step155.pt, fs/web (MiMo webdev rootfs),
#                rl/tasks.jsonl, .venv-train, .venv-eval; GEMINI_API_KEY in the environment
set -euo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
REPO=$(cd "$(dirname "$0")/.." && pwd)
A=$REPO/results/stage2/assignment_q7.json
TP=$W/.venv-train/bin/python
RUN=$W/runs/grpo_v1; OUT=$W/results/stage6; mkdir -p $RUN $OUT
ITERS=${ITERS:-24}; TASKS_PER_ITER=${TASKS_PER_ITER:-16}; GROUP=${GROUP:-8}
export PATH=$W/.venv-eval/bin:$PATH WEB_ROOTFS=$W/fs/web SANDBOX_ROOT=$W/sandbox

serve() {  # serve MODEL LOG
  vllm serve $1 --served-model-name m --port 8000 --max-model-len 32768 --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.90 --kv-cache-dtype int4_per_token_head > $2 2>&1 &
  V=$!
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; kill -0 $V 2>/dev/null || { echo "vllm died"; exit 1; }; done
}
stop_vllm() { kill -9 $V 2>/dev/null || true; pkill -9 -f "[V]LLM::" || true; sleep 5; }
build() {  # build MERGED_PT -> $W/models/policy
  rm -rf $W/models/policy
  $TP $REPO/evals/fakequant_from_db.py $W/models/h0 $1 $A $W/models/policy > /dev/null
}

if [ ! -f $RUN/merged_fq.pt ]; then
  $TP $REPO/train/train_qat.py $W/models/h0 none $A $RUN/init --init $W/runs/rft_v1/lora_step155.pt --export-only > $OUT/init.log 2>&1
  cp $RUN/init/merged_fq.pt $RUN/merged_fq.pt
fi
[ -f $OUT/log.tsv ] || printf "iter\ttime\tcode_reward\twebdev_reward\tinformative_rows\tmean_turns\tmean_tokens\n" > $OUT/log.tsv

for it in $(seq 1 $ITERS); do
  [ -f $W/STOP ] && { echo "STOP file, ending"; break; }
  [ -f $OUT/.done_it$it ] && continue
  echo "$(date +%T) iter $it: build + serve"
  build $RUN/merged_fq.pt
  serve $W/models/policy $OUT/vllm_it$it.log
  python $REPO/rl/rollout.py $W/rl/tasks.jsonl $OUT/rollouts_it$it.jsonl $TASKS_PER_ITER $GROUP $it > $OUT/rollout_it$it.log 2>&1
  stop_vllm
  python $REPO/rl/advantages.py $OUT/rollouts_it$it.jsonl $OUT/train_it$it.jsonl >> $OUT/rollout_it$it.log 2>&1
  n=$(wc -l < $OUT/train_it$it.jsonl)
  if [ "$n" -ge 4 ]; then
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TP $REPO/train/train_qat.py $W/models/h0 $OUT/train_it$it.jsonl $A $RUN \
      --init $W/runs/rft_v1/lora_step155.pt --advantage --constant-lr --lr 2e-5 --accum $(( (n + 1) / 2 )) \
      --more-steps 2 --max-len 24576 >> $OUT/train.log 2>&1
    $TP -c "import torch; st=torch.load('$RUN/lora.pt', map_location='cpu'); torch.save(st['lora'], '$RUN/lora_it$it.pt')"
  fi
  cr=$(grep -oP "code: .*mean_reward=\K[0-9.]+" $OUT/rollout_it$it.log || echo NA)
  wr=$(grep -oP "webdev: .*mean_reward=\K[0-9.]+" $OUT/rollout_it$it.log || echo NA)
  mt=$(grep -oP "mean_turns=\K[0-9.]+" $OUT/rollout_it$it.log || echo NA)
  mk=$(grep -oP "mean_tokens=\K[0-9.]+" $OUT/rollout_it$it.log || echo NA)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $it $(date +%T) $cr $wr $n $mt $mk >> $OUT/log.tsv
  gzip -f $OUT/rollouts_it$it.jsonl; rm -f $OUT/train_it$it.jsonl
  touch $OUT/.done_it$it
done

# final model + benchmarks (32k budget, int4 KV)
build $RUN/merged_fq.pt; rm -rf $W/models/harwyn-4b-grpo-q7-fq; mv $W/models/policy $W/models/harwyn-4b-grpo-q7-fq
M=$W/models/harwyn-4b-grpo-q7-fq; R=$OUT/eval; mkdir -p $R
vllm serve $M --served-model-name m --port 8000 --max-model-len 40960 --language-model-only --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.92 --kv-cache-dtype int4_per_token_head \
  --override-generation-config "{\"max_new_tokens\": ${MAX_TOKENS:-32768}}" > $R/vllm.log 2>&1 &
V=$!
until curl -sf localhost:8000/health >/dev/null; do sleep 5; kill -0 $V 2>/dev/null || { echo "vllm died"; exit 1; }; done
cd $REPO/evals
python gen.py $R humaneval > $R/gen_humaneval.log 2>&1 & P1=$!
LIMIT=99 python gen.py $R gpqa > $R/gen_gpqa.log 2>&1 & P2=$!
lm_eval --model local-completions \
  --model_args model=m,base_url=http://localhost:8000/v1/completions,tokenizer=$M,num_concurrent=4,timeout=3600,tokenized_requests=False \
  --tasks niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue \
  --metadata "{\"max_seq_lengths\":[32768],\"tokenizer\":\"$M\"}" --limit 25 \
  --use_cache $R/ruler_cache --output_path $R/ruler > $R/ruler.log 2>&1 & P3=$!
wait $P1 $P2 $P3
stop_vllm
python score.py $R > $R/score.log 2>&1
echo "$(date +%T) stage 6 done"
