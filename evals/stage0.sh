#!/usr/bin/env bash
# Stage 0 GPU part (run on the L4): base-model generations, BFCL, RULER at bf16 and int4 KV (ruler.sh).
# RULER_INT4_ONLY=1 skips straight to the int4 RULER run.
# Scoring runs afterwards on CPU (score.py).
#   HARWYN_WORK  work dir holding models/ and results/ (required)
#   ON_EXIT      optional command run when the script ends, e.g. to power off the GPU machine
set -uo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
trap "${ON_EXIT:-true}" EXIT
source $W/.venv-eval/bin/activate
cd "$(dirname "$0")"
OUT=$W/results/stage0; mkdir -p $OUT
MODEL=${MODEL:-$W/models/base}

serve() {  # $1 = kv-cache dtype; extra args after. Reuses a server that's already up.
  if curl -sf localhost:8000/health >/dev/null; then VLLM=$(pgrep -f "vllm serve" | head -1); return; fi
  vllm serve $MODEL --served-model-name m --port 8000 --max-model-len 40960 --language-model-only \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --gpu-memory-utilization 0.92 --override-generation-config "{\"max_new_tokens\": ${MAX_TOKENS:-32768}}" \
    --kv-cache-dtype "$@" > $OUT/vllm_$1.log 2>&1 &
  VLLM=$!
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; kill -0 $VLLM || { echo "vllm died"; exit 1; }; done
}
if [ -z "${RULER_INT4_ONLY:-}" ]; then
echo "$(date +%T) serve bf16 KV"; serve auto
[ "${SMOKE:-1}" = 0 ] || LIMIT=2 python gen.py $OUT/smoke math500,gpqa,humaneval,ifeval || exit 1
python gen.py $OUT math500,gpqa,humaneval,ifeval > $OUT/gen.log 2>&1 & GEN=$!
export BFCL_PROJECT_ROOT=$OUT/bfcl
$W/.venv-bfcl/bin/python bfcl_ids.py 100
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=x \
  $W/.venv-bfcl/bin/python bfcl.py generate --model harwyn-FC --run-ids --num-threads 32 \
  > $OUT/bfcl_gen.log 2>&1 & BFCL=$!
wait $GEN $BFCL
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; sleep 3
bash ruler.sh $OUT bf16 auto
fi
bash ruler.sh $OUT int4 int4_per_token_head --attention-backend TRITON_ATTN
echo "$(date +%T) GPU part done"
