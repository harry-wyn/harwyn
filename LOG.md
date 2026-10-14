# harwyn lab notebook

What was run, in order, with the numbers, the decisions and what went wrong. Plan and rationale:
[`README.md`](README.md). Headline numbers: [`results.md`](results.md). Raw outputs: [`results/`](results/).

## Setup (2026-10-10)

**Hardware:** one NVIDIA L4 (24 GB, driver 580.178), Python 3.12. Preemptible instances, so every
script is resumable.

**Software** (exact pins in [`setup.sh`](setup.sh)):

| venv | Packages |
|---|---|
| eval | vLLM 0.31.0, torch 2.13.0, transformers 5.17.0, lm-eval 0.4.13, evalplus 0.2.1, math-verify 0.9.0, datasets 5.1.0 |
| bfcl | bfcl-eval 2026.3.23 (its pins conflict with vLLM's, so it gets its own venv and only talks HTTP) |
| heretic | heretic-llm 2.0.0.dev0 @ `49b56e5`, torch 2.14.1+cu130, transformers 5.19.0, optuna 5.0.0, peft 0.21.2, flash-linear-attention 0.5.2 |

**Model choice: Qwen3.5-4B, not Qwen3-4B.** Qwen3.5 onwards uses a hybrid stack (3 Gated DeltaNet
layers per full-attention layer), which Qwen 4 is expected to keep. Only the 8 full-attention layers
of the 32 have a KV cache, and they're the ones doing precise long-range retrieval, which is exactly what
int4 KV quantisation touches. A plain-transformer stand-in would rehearse the wrong problem.

## Stage 0: base-model baselines (2026-10-11)

### Eval design
- **Thinking mode on**, sampling from the Qwen3.5 model card (thinking/general: T=1.0, top-p 0.95,
  top-k 20, presence penalty 1.5; thinking/coding for HumanEval+: T=0.6, presence penalty 0).
- **Fixed 8,192-token output budget per answer; a truncated answer counts as wrong.** This started as
  an accident: vLLM's `--override-generation-config '{"max_new_tokens": 8192}'` caps every request,
  overriding the 16k we passed per request. Kept on purpose: harwyn's goal includes *concise*
  reasoning, so "accuracy within 8k" is the number to improve, and it's ~3× cheaper than the 32k
  Qwen recommends. For full-length scores, use Qwen's published numbers.
- **GPQA Diamond** from the public mirror `fingertap/GPQA-Diamond` (original is gated); answer letters
  A/B/C/D = 55/54/48/41, so no positional shortcut.
- **BFCL has no Qwen3.5 handler.** `evals/bfcl.py` registers `harwyn-FC` on BFCL's OpenAI-compatible
  handler, so vLLM applies Qwen3.5's own chat template and `qwen3_coder` tool parser. A fixed sample of
  100 cases per category (seed 0, `evals/bfcl_ids.py`) instead of 3,841: ±4–5 points per category.
- **RULER** via lm-eval's `niah_multikey_{1,2,3}` + `niah_multivalue` at 32,768 tokens, 50 per task,
  served once with bf16 KV and once with `--kv-cache-dtype int4_per_token_head`.

### Throughput: the L4 is KV-cache-bound
Qwen3.5-4B's attention layers use 256-dim heads with 4 KV heads, so the KV cache costs
8 layers × 2 × 4 × 256 × 2 B = **32 KB per token**. At `--gpu-memory-utilization 0.92` only ~70–100
sequences of ~5k tokens fit at once, giving **~650–900 generated tokens/s** (falling to ~410 as
answers grow). Running RULER's 32k prompts at the same time, 16 at once, filled the cache (KV usage 99%,
throughput down to ~270 tok/s, RULER requests timing out and retrying), so RULER now runs on its own
at concurrency 4. Stage 0 took ~3 GPU-hours in total.

### Results (8k budget)

| Benchmark | Score | Mean output tokens | Hit 8k |
|---|---|---|---|
| HumanEval+ | 76.8% (plain HumanEval 81.1%) | 3,047 | 12% |
| IFEval (prompt-level strict) | 80.8% | 4,305 | 10% |
| MATH-500 | 60.0% | 5,598 | 40% |
| GPQA Diamond | 10.1% | 7,695 | 81% |

The MATH and GPQA numbers measure verbosity more than knowledge: **all 298 MATH answers that finished
within 8k were correct**, and the 38 GPQA answers that finished scored 52.6%. The 153 GPQA answers
that hit the budget never left the thinking block (empty final answer), which is why GPQA lands below
the 25% guessing rate. Concise reasoning is the biggest lever on this model.

BFCL (100/category): simple_python 92, multiple 94, parallel_multiple 90, irrelevance 90.6,
live_simple 88, live_multiple 59, multi_turn_base 65.

RULER 32k:

| KV cache | multikey_1 | multikey_2 | multikey_3 | multivalue |
|---|---|---|---|---|
| bf16 | 98 | 100 | 88 | 99 |
| int4_per_token_head | 98 | 100 | 88 | 100 |

**int4 KV costs nothing measurable on RULER needle tasks at 32k for the base model.** Good news for
the format (Hadamard rotation + one asymmetric scale per token per head), but it means these tasks
are too easy to show what KV-QAT buys; a harder long-context eval (64k+, multi-hop, own
agentic-retrieval set) is needed before stage 6.

### Incidents
- **vLLM froze on `int4_per_token_head`** at 198/200 RULER samples: 0 tokens/s with one request
  "running" and one waiting at 3.8% KV usage, and no response even to a 5-token request. lm-eval only
  writes results at the end, so restarting lost all 198. Fix: [`evals/ruler.sh`](evals/ruler.sh) runs
  lm-eval with `--use_cache` (every response persisted) and a watchdog that restarts vLLM after 10
  silent minutes. The rerun completed without freezing.
- The preemptible GPU was reclaimed once mid-run; `gen.py` (append-only JSONL) and BFCL (skips
  existing ids) resumed without redoing finished items.

## Stage 1: Heretic (2026-10-11)

Heretic `49b56e5`, default config (ARA modifier, default harmful/harmless prompt sets, keyword
refusal scorer, KL on harmless prompts, 100 trials with 30 random start-up trials). Baseline:
**99/100 refusals**. ~1.5 min per trial on the L4.

The search was stopped early. The random start-up phase had found trial 3 at 11 refusals / KL 0.043;
the first guided trials only found lower refusal counts at 10–25× the drift. Four more trials
(44–47) ran by accident (see incidents) and two of them joined the front. Final Pareto front
(48 trials in [`results/stage1/heretic_study.jsonl`](results/stage1/heretic_study.jsonl)):

| Trial | Refusals /100 | KL divergence |
|---|---|---|
| 7 | 94 | 0.010 |
| 4 | 92 | 0.018 |
| **3 (used)** | **11** | **0.043** |
| 45 | 9 | 0.048 |
| 47 | 6 | 0.134 |
| 38 | 4 | 0.492 |
| 37 | 2 | 0.996 |
| 41 | 1 | 1.108 |
| 27 | 0 | 11.05 |

For reference, the abliterations compared in Heretic's README sit at KL 0.16–1.04.

### Trial 47 vs trial 3: drift shows up as runaway thinking first
The KL ≤ 0.2 rule in `evals/heretic_pick.py` picks **trial 47**, which was exported and partly
evaluated before switching to trial 3 (partial outputs:
[`results/stage1_trial47_partial/`](results/stage1_trial47_partial/)). On the same 128 HumanEval+
problems:

| | Trial 3 (KL 0.043) | Trial 47 (KL 0.134) | Base |
|---|---|---|---|
| Answers hitting the 8k budget | 16 | **36** | 15 |

Trial 47 doesn't obviously get dumber per answer; it **stops ending its thinking**, and under a fixed
budget every runaway trace scores zero. For a thinking model, "KL on harmless prompts" understates
the damage: a small shift in the end-of-thinking decision costs far more than a small shift in token
choices. Trial 3 matches the base model's thinking lengths, so it became `h0` (`HERETIC_TRIAL=3`).

**Trial 3 parameters** (ARA; exact values to replay with Heretic's reproduction mode):

```json
{"start_layer_index": 0, "end_layer_index": 20,
 "preserve_good_behavior_weight": 0.9514856283924104,
 "steer_bad_behavior_weight": 0.00015760083596745767,
 "overcorrect_relative_weight": 0.9416278430429633, "neighbor_count": 12}
```

### Results: h0 vs base (8k budget)
| Benchmark | h0 (trial 3) | Base, same items |
|---|---|---|
| HumanEval+ (164) | **77.4%** (plain 81.1%), 2,985 tok, 13.4% hit 8k | 76.8% (plain 81.1%), 3,047 tok, 12.2% |
| GPQA Diamond, first 99 | **7/99**; 22 finished in budget, 31.8% of those | 11/99; 21 finished, 52.4% |
| RULER 32k bf16 KV | **100 / 100 / 92 / 100** (25/task) | 98 / 100 / 88 / 99 (50/task) |

Coding and long-context retrieval are unchanged. GPQA is the open question: on the 17 questions both
models finished, h0 got 6 right vs 10. That's a single sample per question at temperature 1.0, so
it's within noise, but it's the one sign of a reasoning cost; it gets re-measured at stage 4. The
remaining refusals are left for the stage 4 SFT data to remove.

### Incidents
- Heretic's `pyproject.toml` pins torch to the CPU wheel index, so a plain install gives
  `torch …+cpu`; `setup.sh` reinstalls torch from PyPI (CUDA build).
- **Exporting unattended.** Heretic ends with an interactive trial menu. The first attempt set
  `n_trials`/`trial_index`/export options in `config.toml` and continued the study, but **on continue
  Heretic replaces every setting with the ones stored in the study**, so it silently resumed the
  search with `n_trials = 100` (that's where trials 44–48 came from). Fix: `heretic_pick.py` writes the
  export settings (`n_trials` = trials already run, `trial_index`, merge, save) into the study's
  stored settings, then a plain `checkpoint_action = "continue"` runs zero trials and exports.

## Stage 2: draft GSQ + RCO quantisation of h0 (2026-10-11 → 12)

**Budget: 3.0 bits/weight average.** The 4B could afford ~5–6 bpw with almost no loss, but 6-bit
quantisation leaves QAT nothing to recover, so the rehearsal would test nothing. 3.0 bpw matches the
likely harwyn-27b budget, where ISTA's own Qwen3.8-27B GSQ-RCO results show 3.0–3.5 bpw is close to
lossless and 2.5 bpw costs ~9 points on LiveCodeBench:

| bpw | Qwen3.8-27B GPQA-D | LiveCodeBench v6 |
|---|---|---|
| 16 | 89.9 | 85.7 |
| 3.5 | 89.4 | 85.7 |
| 3.0 | 88.9 | 84.6 |
| 2.75 | 86.4 | 82.3 |
| 2.5 | 84.9 | 76.6 |

**vLLM support.** vLLM 0.31's compressed-tensors loader accepts every weight width 2–8 per layer:
Marlin runs 4/8-bit, Humming (`humming-kernels`, same library GSQ ships a converter for) runs
2/3/5/6/7-bit on sm75+, so the L4 can serve the mixed checkpoint.

**Pipeline** ([`evals/stage2.sh`](evals/stage2.sh), resumable per phase):
1. **A, GPTQ database** (RCO `run_quantize.py`): all 200 quantisable Linears (MLP, full attention,
   Gated DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj`; `in_proj_a/b`, embeddings and `lm_head` stay
   bf16) at 2–8 bits, symmetric, group 128, calibrated on 1M FineWeb-Edu tokens. ~70 min on the L4.
2. **B, GSQ** at 2 and 3 bits on the MLPs, replacing those database entries (GPTQ originals backed up).
3. **C, RCO search** for an exact 3.0 bpw average, calibrated on `evol_codealpaca` (code) since
   coding is the target.
4. **D, build** the packed compressed-tensors checkpoint and a fake-quant bf16 copy (the QAT reference).
5. **E, evals** in vLLM against h0.

### Incidents
- **RCO's NCCL barrier segfaults on a single GPU** (torch 2.11/cu126 here): `run_quantize.py` always
  starts NCCL and calls `dist.barrier()`. [`patches/rco-single-gpu.patch`](patches/rco-single-gpu.patch)
  skips distributed init for one rank and guards the two bare barriers.
- **RCO's transformers pin (5.7.0) can't build Qwen3.5 from its config**; 5.8.1 (GSQ's) can, and loads
  the real checkpoint with zero missing/unexpected keys.
- **RCO's database has an extra level**: `<save_dir>/<model name>/<calibration bits>bit/<layer>/<bw>.pth`.
- **Preemptible GPUs.** Phase A was reclaimed at 168/200 layers; `run_quantize.py` can't resume.
  [`patches/rco-resume.patch`](patches/rco-resume.patch) skips layers whose every width is already saved
  and installs their saved calibration-width weights so later layers calibrate on identical
  activations. A later preemption auto-restarted the machine from an older storage snapshot (losing
  ~12 min of work and leaving the GPU idle), so the rest of stage 2 ran on a non-preemptible GPU.
- **GSQ's dense Qwen3.5 attention path is broken** at its own locked dependencies: with `self_attn: true`
  it calls `PackedQuantizationCompressor.compress_weight`, which compressed-tensors 0.15.0.1 (GSQ's lock)
  no longer has. It was also the slow part (3.7 min/layer of the 7.5). GSQ therefore runs on the MLPs
  only (64% of quantised parameters, 2 epochs; layer 0's loss had flattened by epoch 2), and attention /
  GDN projections use RCO's GPTQ entries at every width.
- **RCO's checkpoint import writes every Linear**, compressed or not, so GSQ's bf16 attention copies would
  have overwritten 2-bit GPTQ entries; the import excludes `self_attn`/`linear_attn`.
- **RCO's importer dropped GSQ's integer codes.** GSQ writes symmetric weights with no zero-point tensor,
  and RCO's importer only extracted codes when one existed. Its first pass wrote GSQ's weights for the
  search but kept GPTQ's codes for packing (a 0.26 max mismatch on one 2-bit MLP), so the shipped
  checkpoint would silently have differed from what the search chose and QAT targets. GSQ also stores
  2/3-bit codes in a 4-bit container, which would have been packed and served as 4-bit and broken the
  3.0 bpw budget. [`patches/rco-split-symmetric.patch`](patches/rco-split-symmetric.patch) uses the
  implicit symmetric zero point (2^(bits−1), RCO's own convention) and re-expresses codes at the slot's
  true width. After re-import, every checked tensor reconstructs its stored weights exactly at 2 bits.
- **RCO's search didn't fit a 24 GB GPU + 60 GB RAM machine at 4B** (it was built for large nodes),
  and was OOM-killed twice, silently: the platform puts each command in a memory cgroup with
  `memory.oom.group`, so the kill took the whole stage script with it and no exit hook ran.
  - The KL objective caches **full-vocabulary** reference log-probs on the CPU: 256 × 2,047 × 248,320
    values in fp16 ≈ 260 GB for Qwen3.5's vocabulary. Switched to `--objective ce` (cross-entropy on the
    calibration set's answer tokens), which needs no cache. A top-k reference cache would restore KL;
    worth patching for harwyn-27b.
  - One bf16 delta per candidate width per layer is ~7 GB/width at 4B: too big for the GPU, and the
    default RAM weight cache would duplicate it. Used `--weightstore-cache off --cpu-deltas` and
    candidate widths {2, 3, 4, 6, 8} (~36 GB in RAM) instead of 2–8.
  - To keep the PCIe-streamed search near schedule: 120 steps × 2 Gumbel samples (paper: 200 × 4).
- **RCO's gradient search doesn't fit a 24 GB GPU at 4B even with CPU deltas.** Plain autograd keeps every
  layer's GPU copy of its candidate deltas alive until backward (CUDA OOM at 2,048 and 1,024 tokens).
  [`patches/rco-streamed-deltas.patch`](patches/rco-streamed-deltas.patch) adds a custom autograd function
  that streams the deltas from RAM in forward *and* backward, so autograd stores none
  ([`patches/test_rco_streamed_deltas.py`](patches/test_rco_streamed_deltas.py) checks it against plain
  autograd: identical loss, gradients within fp32 accumulation error). With it the forward fits, but the
  first backward still OOMed inside the Gated DeltaNet kernel (Triton) at 1,024 tokens; a 512-token run is
  the next attempt. An 80 GB GPU (the plan for the 27B) avoids all of this.

### Finding: RCO's sensitivity initialisation alone is degenerate
With zero optimisation steps, RCO's sensitivity-based initial allocation hits the budget exactly
(3.001 bpw) but splits the model bimodally: **all 96 MLPs and 72 attention/GDN projections at 2 bits,
30 attention/GDN projections at 8 bits** (a few tensors at 3). RCO's own check of that assignment against
the bf16 model gives NLL 10.2 and KL 4.65 nats on code answers, i.e. badly broken (uniform guessing is
ln 248,320 ≈ 12.4), so it was not evaluated further. The per-tensor delta norms that drive the
initialisation favour spending bits on small, sensitive attention tensors and starve the MLPs; the
gradient search exists to correct exactly this. Files: [`results/stage2/sensinit/`](results/stage2/sensinit/).

### Search, build and results (2026-10-13/14, on a 48 GB A40, then an RTX 4090 for evals)
- **The gradient search fits on 48 GB** with the streamed-deltas patch (1,024-token calibration, CE
  objective, 100 steps × 2 Gumbel samples). It allocated exactly 3.002 bpw: **109 tensors at 2 bits,
  33 at 3, 16 at 4, 3 at 6, 39 at 8.** RCO's own check vs h0 improved from KL 4.65 (sensitivity init)
  to **KL 1.17**, NLL 2.59 on code answers: better, but still a large divergence.
- **Serving the packed checkpoint in vLLM 0.31 failed**, three separate problems: (1) the merged-weight
  index from h0 was copied into the single-file checkpoint (fixed in `stage2.sh`); (2) RCO's config
  targets every Linear by default, so the bf16 GDN gates must be ignored (fixed); (3) RCO saves the
  model as text-only `Qwen3_5ForCausalLM`, whose vLLM loader still failed on a fused layer
  (`MergedColumnParallelLinear ... has no attribute 'data'`). Separately, vLLM fuses `gate/up`,
  `q/k/v` and `in_proj_qkv/z`, so each fused group must share a width; 27 groups didn't (equalising them
  to their max gives 3.34 bpw: `results/stage2/assignment_q3p0_fused.json`). For the 27B, the search
  must tie these groups (RCO's `--layer-groups`).
- **Evaluated via a fake-quant copy** ([`evals/fakequant_from_db.py`](evals/fakequant_from_db.py)): h0's
  exact files with every quantised tensor replaced by its database weight at the assigned width.
  Numerically the same as the packed model, and it's the QAT reference.

| Benchmark (8k budget) | Base | h0 (bf16) | **h0 at 3.0 bpw (GSQ+RCO)** |
|---|---|---|---|
| HumanEval+ | 76.8% | 77.4% | **0.0%** |
| GPQA Diamond, first 99 | 11/99 | 7/99 | **2/99** |
| RULER 32k, 25/task | 98/100/88/99 (50/task) | 100/100/92/100 | **20/0/0/16** |

**At 3.0 bpw the 4B model collapses**: 41.5% of HumanEval answers ran to the 8k limit and none passed.
This matches the general rule that small models degrade far faster at low bit widths than large ones
(ISTA's Qwen3.8-27B is near-lossless at 3.0 bpw), and RCO's KL of 1.17 already said so. It is also
the strongest possible baseline for the QAT experiment: stage 4 now has a large, measurable gap to
recover at 3.0 bpw. Two caveats: the search ran a shortened schedule, and no all-8-bit sanity run was
done to rule out a pipeline error (that's the first check for next time, ~$0.30).

GPU spend for stage 2 on the second host: $3.70 (including ~5 h of idle GPU from a wait loop that
didn't notice the server had died; the loop now exits).
