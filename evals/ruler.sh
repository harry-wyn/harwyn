#!/usr/bin/env bash
# RULER (4 tasks x 50 samples at 32k) against its own vLLM with the given KV cache dtype.
# Resumable: lm-eval caches every response (--use_cache), and a watchdog restarts vLLM if it
# freezes (seen once with int4_per_token_head at 198/200), then lm-eval resumes from the cache.
# usage: ruler.sh OUT_DIR LABEL KV_DTYPE [extra vllm args]
set -uo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
OUT=$1; LABEL=$2; shift 2
MODEL=${MODEL:-$W/models/base}
LOG=$OUT/ruler_$LABEL.log
source $W/.venv-eval/bin/activate

stop_vllm() { pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 3; }  # [x] so pkill can't match its caller

for try in 1 2 3 4; do
  stop_vllm
  vllm serve $MODEL --served-model-name m --port 8000 --max-model-len 40960 --language-model-only \
    --gpu-memory-utilization 0.92 --kv-cache-dtype "$@" >> $OUT/vllm_$LABEL.log 2>&1 &
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; done
  echo "$(date +%T) ruler $LABEL try $try"
  lm_eval --model local-completions \
    --model_args model=m,base_url=http://localhost:8000/v1/completions,tokenizer=$MODEL,num_concurrent=4,timeout=3600,tokenized_requests=False \
    --tasks niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue \
    --metadata "{\"max_seq_lengths\":[32768],\"tokenizer\":\"$MODEL\"}" --limit 50 \
    --use_cache $OUT/ruler_${LABEL}_cache --output_path $OUT/ruler_$LABEL >> $LOG 2>&1 &
  LM=$!
  while kill -0 $LM 2>/dev/null; do  # progress bar touches the log every ~15 s; 10 min silent = frozen
    sleep 60
    if [ $(( $(date +%s) - $(stat -c %Y $LOG) )) -gt 600 ]; then echo "frozen, restarting"; kill -9 $LM; fi
  done
  wait $LM && break
done
stop_vllm
grep 32768 $LOG | grep niah | tail -4
