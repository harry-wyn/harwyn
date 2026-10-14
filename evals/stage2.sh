#!/usr/bin/env bash
# Stage 2: draft GSQ+RCO quantisation of h0 at TARGET_BITS average (default 3.0), served by vLLM.
#   A  RCO's GPTQ database: every quantisable tensor at 2..8 bits (symmetric, group 128)
#   B  GSQ at 2 and 3 bits on the MLPs, written over those database slots (GSQ's gains are at 2-3 bits).
#      Attention/GDN projections keep their GPTQ entries (see LOG.md: GSQ's Qwen3.5 attention path is broken)
#   C  RCO search: one width per tensor, exact TARGET_BITS average
#   D  build the packed compressed-tensors checkpoint (vLLM) + a fake-quant bf16 copy (QAT reference)
#   E  quick evals on vLLM: HumanEval+, GPQA Diamond first 99, RULER 32k 25/task
# Each phase writes a .done marker and is skipped on rerun.
#   HARWYN_WORK  work dir (required);  ON_EXIT  optional command run at exit
#   SKIP_GSQ=1   skip phase B (GPTQ-only database)
set -euo pipefail
W=${HARWYN_WORK:?set HARWYN_WORK}
trap "${ON_EXIT:-true}" EXIT
EVALS=$(cd "$(dirname "$0")" && pwd)
BITS=${TARGET_BITS:-3.0}
TAG=q${BITS/./p}
M=$W/models/qwen3.5-4b-h0               # GSQ routes Qwen3.5 by "qwen3.5" in the model path
DB=$W/rco-db/h0
LD=$DB/$(basename $M)/4bit                # run_quantize nests layers under <model>/<calibration bits>bit/
Q=$W/models/qwen3.5-4b-h0-$TAG           # packed, for vLLM
QFQ=$W/models/qwen3.5-4b-h0-$TAG-fq      # fake-quant bf16, the QAT reference
OUT=$W/results/stage2; mkdir -p $OUT $DB
RP=$W/.venv-rco/bin/python
GP=$W/GSQ/.venv/bin
# Every Linear GSQ quantises in Qwen3.5: MLPs, full attention, Gated DeltaNet projections.
# in_proj_a/in_proj_b (tiny gates), embeddings and lm_head stay bf16.
MODULES='.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)$'
done_() { [ -f $OUT/.done_$1 ]; }
mark() { touch $OUT/.done_$1; echo "$(date +%T) phase $1 done"; }

if ! done_ A; then
  echo "$(date +%T) A: GPTQ database 2..8 bits"
  (cd $W/RCO && $RP -m torch.distributed.run --standalone --nproc-per-node=1 run_quantize.py \
    --model_name_or_path $M --dtype bfloat16 \
    --quantizable_modules "$MODULES" \
    --pre_block_modules model.embed_tokens --block_modules model.layers \
    --post_block_modules model.norm lm_head \
    --calibration_data fineweb_edu --calibration_tokens 1048576 --calibration_sequence_length 2048 \
    --bitwidth_options 2 3 4 5 6 7 8 --calibration_bitwidth 4 --group_size 128 --sym \
    --save_dir $DB) > $OUT/A_gptq.log 2>&1
  mark A
fi

if [ -z "${SKIP_GSQ:-}" ]; then
  for b in 2 3; do
    done_ B$b && continue
    echo "$(date +%T) B: GSQ $b-bit"
    CFG=$OUT/gsq_${b}bit.yaml
    cat > $CFG <<EOF
model: {name: "$M", device: "cuda", dtype: "bfloat16", dummy: false}
data: {dataset_name: "open_thoughts", batch_size: 8, num_samples: 512, val_samples: 32,
       max_length: 2048, num_workers: 4}
quantization: {gsq_bits: $b, init_method: "gptq", gsq_enabled: true, start_layer: 0, self_attn: false,
               std: 0.01, temperature: [2, 0.05], scale: [100, 500], groupsize: 128, strength: 6,
               logits_dtype: "bfloat16"}
training: {num_epochs: 2, device_microbatch_size: 2, lr1: 0.0001, lr2: 0.00005, weight_decay: 1.0,
           checkpoint_dir: "$W/gsq-runs/${b}bit", log_dir: "$W/gsq-runs/${b}bit-logs",
           act_cache_dir: "$W/gsq-runs/act_cache", eval_baseline: false, ppl_eval_every_n_layers: 0}
gptq: {nsamples: 256, wbits: $b, sym: true, trits: false, percdamp: 0.1, blocksize: 128,
       groupsize: 128, static_groups: false, prunen: 0, prunem: 0}
wandb: false
EOF
    RESUME=(); ls $W/gsq-runs/${b}bit/*/progress.json >/dev/null 2>&1 && RESUME=(--resume)
    (cd $W/GSQ && $GP/torchrun --standalone --nproc-per-node=1 main.py --config $CFG "${RESUME[@]}" \
      && $GP/python save_model.py --config $CFG --out-dir $W/gsq-runs/${b}bit-ct) >> $OUT/B_gsq_${b}bit.log 2>&1
    # keep the GPTQ MLP entries GSQ replaces (undo = copy back)
    for d in $LD/*.mlp.*; do
      mkdir -p $DB/gptq-mlp-backup/$(basename $d)
      cp -n $d/$b.pth $d/${b}_qparams.pt $DB/gptq-mlp-backup/$(basename $d)/
    done
    # only GSQ's quantised MLPs go in; its bf16 attention/GDN copies must not overwrite GPTQ entries
    (cd $W/RCO && $RP run_split_checkpoint.py --model $W/gsq-runs/${b}bit-ct --layer-dir $LD \
      --bitwidth $b --overwrite --exclude-patterns lm_head embed_tokens embed self_attn linear_attn) \
      >> $OUT/B_gsq_${b}bit.log 2>&1
    mark B$b
  done
fi

if ! done_ C$TAG; then
  echo "$(date +%T) C: RCO search, target $BITS bits"
  # Memory (see LOG.md): one bf16 delta per candidate width per layer is ~7 GB/width at 4B, which can't
  # sit on a 24 GB GPU, and RCO's default RAM cache duplicates it. So: no cache, deltas in RAM streamed
  # to the GPU (--cpu-deltas), widths {2,3,4,6,8} (~36 GB), and fewer steps (below).
  # Objective: cross-entropy on answer tokens. RCO's KL objective caches full-vocabulary reference
  # log-probs on CPU (256 x 2047 x 248,320 in fp16 ~ 260 GB for Qwen3.5) and was OOM-killed.
  # Gradient search made to fit 24 GB by patches/rco-streamed-deltas.patch (deltas streamed from RAM in
  # forward and backward). 512-token calibration (1024 OOMed in the GDN backward), 100 steps x 2 Gumbel samples (paper: 2048, 200 x 4).
  # Without the search, RCO's sensitivity init alone gave a degenerate 2-bit-MLP allocation (LOG.md).
  (cd $W/RCO && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $RP rco_search_quant.py --model $M --layer-dir $LD --bitwidths 2,3,4,6,8 \
    --target-avg-bits $BITS --objective ce --n-steps 100 --lr 0.1 --tau-min 0.01 --n-gumbel-samples 2 \
    --batch-size 1 --batches-per-step 1 --gradient-checkpointing --weightstore-cache off --cpu-deltas \
    --calibration-data evol_codealpaca --calibration-samples 256 --calibration-seq-length 512 --device-map balanced \
    --save-json) > $OUT/C_search_$TAG.log 2>&1
  J=$(grep -oE "JSON saved to \S+" $OUT/C_search_$TAG.log | tail -1 | cut -d' ' -f4)
  cp "$J" $OUT/assignment_$TAG.json
  mark C$TAG
fi

if ! done_ D$TAG; then
  echo "$(date +%T) D: build checkpoints"
  for fmt in compressed-tensors fake-quant; do
    dst=$Q; [ $fmt = fake-quant ] && dst=$QFQ
    (cd $W/RCO && $RP run_build_checkpoint.py quant --model $M --config $OUT/assignment_$TAG.json \
      --layer-dir $LD --output $dst --format $fmt --group-size 128 --symmetric) >> $OUT/D_build_$TAG.log 2>&1
    # RCO's config targets every Linear by default; the bf16 GDN gates (in_proj_a/b, fused by vLLM) must be ignored
    $RP -c "import json,sys; p=sys.argv[1]+'/config.json'; c=json.load(open(p)); q=c.get('quantization_config')
if q: q['ignore']=sorted(set(q['ignore']+['re:.*in_proj_a\$','re:.*in_proj_b\$'])); json.dump(c,open(p,'w'),indent=2)" $dst
    for f in $M/*.json $M/*.jinja $M/*.txt; do   # processor/template files vLLM wants (not the weight index)
      [[ $f == *.index.json ]] && continue
      [ -e "$f" ] && [ ! -e "$dst/$(basename "$f")" ] && cp "$f" $dst/ || true
    done
  done
  mark D$TAG
fi

if ! done_ E$TAG; then
  source $W/.venv-eval/bin/activate
  cd $EVALS
  R=$OUT/eval_$TAG; mkdir -p $R
  echo "$(date +%T) E: serve $Q"
  vllm serve $Q --served-model-name m --port 8000 --max-model-len 40960 --language-model-only \
    --reasoning-parser qwen3 --gpu-memory-utilization 0.92 \
    --override-generation-config '{"max_new_tokens": 8192}' > $R/vllm.log 2>&1 &
  V=$!
  # give up if the server dies (a dead vLLM once left this loop polling for 5 h)
  until curl -sf localhost:8000/health >/dev/null; do sleep 5; kill -0 $V 2>/dev/null || { echo "vllm died"; exit 1; }; done
  python gen.py $R humaneval > $R/gen_humaneval.log 2>&1 & A=$!
  LIMIT=99 python gen.py $R gpqa > $R/gen_gpqa.log 2>&1 & B=$!
  lm_eval --model local-completions \
    --model_args model=m,base_url=http://localhost:8000/v1/completions,tokenizer=$Q,num_concurrent=4,timeout=3600,tokenized_requests=False \
    --tasks niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue \
    --metadata "{\"max_seq_lengths\":[32768],\"tokenizer\":\"$Q\"}" --limit 25 \
    --use_cache $R/ruler_cache --output_path $R/ruler > $R/ruler.log 2>&1 & C=$!
  wait $A $B $C
  pkill -9 -f "[v]llm serve" || true; pkill -9 -f "[V]LLM::" || true
  python score.py $R > $R/score.log 2>&1
  mark E$TAG
fi
echo "$(date +%T) stage 2 done"
