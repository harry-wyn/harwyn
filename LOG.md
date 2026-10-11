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
