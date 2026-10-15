#!/usr/bin/env bash
# Stage 1: Heretic (default config) on the base model -> $HARWYN_WORK/models/h0, then three quick evals
# on one shared vLLM server: HumanEval+ (164), GPQA Diamond first 99 (paired against stage 0's same 99),
# RULER 32k 25/task. Resumable: Heretic continues its saved study; gen.py and lm-eval's cache resume.
#   HARWYN_WORK     work dir holding models/ and results/ (required)
#   HERETIC_TRIALS       trials before picking, fresh study only (default 100; harwyn-4b stopped early, see LOG.md)
#   HERETIC_EXPORT_ONLY  set to skip the search and export from the existing study
#   HERETIC_TRIAL        export this trial number instead of the rule's pick (harwyn-4b: 3)
#   ON_EXIT         optional command run when the script ends, e.g. to power off the GPU machine
set -uo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
trap "${ON_EXIT:-true}" EXIT
OUT=$W/results/stage1; mkdir -p $OUT
H0=$W/models/h0
HP=$W/.venv-heretic/bin/python
EVALS=$(cd "$(dirname "$0")" && pwd)

if [ ! -f $H0/config.json ]; then
  mkdir -p $W/heretic-run; cd $W/heretic-run
  # Only run-control settings; prompts, scorers and the ARA modifier are Heretic's defaults.
  printf 'checkpoint_action = "continue"\nn_trials = %s\n' "${HERETIC_TRIALS:-100}" > config.toml
  # At the end Heretic asks which trial to use; with no TTY that prompt fails and it exits,
  # leaving the study in checkpoints/. HERETIC_TRIALS only applies to a fresh study: on continue,
  # Heretic restores the stored settings. HERETIC_EXPORT_ONLY=1 skips straight to picking/exporting.
  [ -n "${HERETIC_EXPORT_ONLY:-}" ] || $W/.venv-heretic/bin/heretic $W/models/base < /dev/null >> $OUT/heretic_optimise.log 2>&1
  J=$(ls checkpoints/*.jsonl)
  # Pick the trial and store export settings in the study (see heretic_pick.py), then continue.
  K=$($HP $EVALS/heretic_pick.py $J $H0 2> $OUT/heretic_front.txt)
  echo "$(date +%T) export front[$K]"
  echo 'checkpoint_action = "continue"' > config.toml
  $W/.venv-heretic/bin/heretic $W/models/base < /dev/null > $OUT/heretic_export.log 2>&1
  [ -f $H0/config.json ] || { echo "export failed"; exit 1; }
  # The merged save can skip processor/template files vLLM wants; copy any missing ones from base.
  for f in $W/models/base/*.json $W/models/base/*.jinja $W/models/base/*.txt; do
    [ -e "$f" ] && [ ! -e "$H0/$(basename "$f")" ] && cp "$f" $H0/
  done
fi

source $W/.venv-eval/bin/activate
cd $EVALS
echo "$(date +%T) serve h0"
vllm serve $H0 --served-model-name m --port 8000 --max-model-len 40960 --language-model-only \
  --reasoning-parser qwen3 --gpu-memory-utilization 0.92 \
  --override-generation-config "{\"max_new_tokens\": ${MAX_TOKENS:-32768}}" > $OUT/vllm.log 2>&1 &
until curl -sf localhost:8000/health >/dev/null; do sleep 5; done
echo "$(date +%T) evals"
python gen.py $OUT humaneval > $OUT/gen_humaneval.log 2>&1 & A=$!
LIMIT=99 python gen.py $OUT gpqa > $OUT/gen_gpqa.log 2>&1 & B=$!
lm_eval --model local-completions \
  --model_args model=m,base_url=http://localhost:8000/v1/completions,tokenizer=$H0,num_concurrent=4,timeout=3600,tokenized_requests=False \
  --tasks niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue \
  --metadata "{\"max_seq_lengths\":[32768],\"tokenizer\":\"$H0\"}" --limit 25 \
  --use_cache $OUT/ruler_cache --output_path $OUT/ruler > $OUT/ruler.log 2>&1 & C=$!
wait $A $B $C
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"
python score.py $OUT > $OUT/score.log 2>&1
echo "$(date +%T) stage 1 done"
