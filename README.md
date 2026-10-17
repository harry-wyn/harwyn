# harwyn

Build **harwyn-4b**: rehearse the full pipeline on a ~4B model on a single NVIDIA L4 (24 GB), so
the recipe is proven before paying for H100s to run it on **Qwen 4 27B** → harwyn-27b (announced at
Apsara on 22 Sep 2026; no weights, card or date yet).

**Deployment target, and therefore the training target: vLLM, end to end.**
- Weights: GSQ+RCO mixed-precision **compressed-tensors** checkpoint (RCO exports it directly;
  vLLM loads it natively).
- KV cache: `--kv-cache-dtype int4_per_token_head` (in stable vLLM since mid-2026, Triton attention
  backend, runs on the L4).

Every training stage is QAT against exactly that setup. Evals, training rollouts and serving all
use vLLM; no llama.cpp anywhere.

## Testbed model

**`Qwen/Qwen3.5-4B`, not `Qwen/Qwen3-4B`.** Qwen3.5 onwards (3.6, 3.8-27B, almost certainly Qwen 4)
uses the hybrid stack: 3 Gated DeltaNet (linear attention) layers to every 1 full-attention layer.
- **Only the full-attention layers have a KV cache.** vLLM keeps the GDN layers' recurrent state in a
  separate "mamba" cache with its own dtype (`mamba_cache_dtype`, leave it `auto`), so
  `--kv-cache-dtype` only touches the 1-in-4 attention layers.
- **Those layers do the precise long-range retrieval,** so int4 KV hits exactly what retrieval needs.
  KV-QAT has a small, precise target. On plain Qwen3-4B (every layer cached) you'd be rehearsing the
  wrong problem.

Swap with `BASE_MODEL` in `setup.sh`. If Qwen 4 ships a small sibling, switch to it.

## How the QAT works

**Weight QAT, matched to GSQ+RCO.** RCO gives each tensor its own bit width (2/3/4-bit group-wise
int, sensitive tensors higher), so uniform `int4` QAT would train for the wrong format.
1. Stage 2 runs GSQ+RCO once on the abliterated model → a **bit map** (one width per tensor) plus
   RCO's `--format fake-quant` checkpoint (dequantised weights) as the reference.
2. During training, fake-quantise each tensor to *its* width with the same grid as the
   checkpoint's `quantization_config` (symmetric group-wise, same group size). torchao
   `IntxFakeQuantizeConfig` with a per-module `filter_fn` does this. Unit test: fake-quantising the
   stage 2 weights must reproduce RCO's fake-quant checkpoint.
3. **Fake-quantise the merged weight `W + BA`, not just `W`.** Stock LoRA+QAT quantises only the
   frozen base weight, but the shipped checkpoint quantises `W + BA`, which the model never saw. Use
   a custom LoRA forward with `fake_quant(W + s·B@A)`. That's one extra rank-r matmul per layer,
   cheap at 4B. (Full fine-tuning would also avoid it but doesn't fit 24 GB at 4B.)

**KV-QAT, matched to vLLM `int4_per_token_head`.** That format (read from vLLM's
`v1/attention/ops/int4_per_token_head.py`):
1. Rotate K and V with a **randomized Hadamard transform** (fixed ±1 signs, deterministic seed).
2. **Asymmetric int4, one scale and zero-point per (token, KV head)** over the whole head dim:
   `scale = (max − min)/15` with the low 4 mantissa bits cleared (the zero-point is hidden there),
   `zp = round(−min/scale)`, `q = clamp(round(x/scale) + zp, 0, 15)`.

In each full-attention layer, after RoPE, replace K and V with
`RHT⁻¹(dequant(quant(RHT(x))))` using a straight-through estimator. Because RHT is orthogonal,
attention maths is unchanged; only the rounding error is injected. **Import vLLM's own `single_rht`**
so the signs match, and unit-test bit-exactness against vLLM's `reshape_and_cache_int4` kernel on
the L4 before training on it. One scale per 128–256 values is coarse (llama.cpp q4_0 uses 32), and
the rotation is what makes it workable. That's exactly why KV-QAT should buy a lot here.

**GRPO under QAT.** Rollouts come from vLLM serving the current policy **with
`int4_per_token_head` on** (only the weights are still bf16), so the KV side is on-policy for free.
The trainer forward runs weight + KV fake-quant; enable TRL's vLLM importance-sampling correction
for the small weight mismatch (check the current flag name on day one). Fallback: GRPO in bf16,
then let stage 6 (QAT again) absorb the gap.

## Pipeline order

| # | Stage | L4 hours |
|---|-------|----------|
| 0 | Baseline evals on the base model (RULER at bf16 KV **and** int4 KV); freeze eval set | 2.2 |
| 1 | **Heretic on the base** (default config, ARA) → pick trial → merge → `h0`; 3 quick evals | 2.5 |
| 2 | Draft GSQ+RCO on `h0` → per-tensor bit map + fake-quant reference (the QAT target) | 2–3 |
| 3 | Build datasets incl. the long-context retrieval set; distil, filter, decontaminate | 0 (CPU) |
| 4 | **SFT + weight-QAT + KV-QAT**, one run, 8k ctx: agentic coding 70% / replay 30% | 8–10 |
| 5 | **GRPO (RLVR) + weight-QAT + KV-QAT**: tests pass, page renders, length penalty | 8 |
| 6 | **Long-context QAT** (weight + KV) at 32k: retrieval data + replay | 6–8 |
| 7 | Final GSQ+RCO on the trained weights → compressed-tensors `harwyn-4b` | 2–3 |
| 8 | Final evals at deploy config (RULER at bf16 KV **and** int4 KV) | 2.2 |

**~35–40 L4 hours** (evals ~10 of them); cut the stage 4 data to ~10k examples to save ~4 h.
Every script resumes after an interruption, so preemptible/spot GPUs work; training checkpoints
every ~200 steps.

**Refusal tripwire:** after stages 4, 5 and 6, run Heretic's refusal count on the same prompt sets
(minutes). If refusals climb back above ~2× the `h0` level, do a light Heretic touch-up on the
merged checkpoint *before* stage 6, so the last QAT stage still re-settles the edited weights.

**Why this order:**
- **Heretic first.** Abliteration is a direct weight edit with a small KL cost. Done first, every
  training stage (~25 hours of it) heals that cost instead of the finished model taking the hit, and
  no un-trained edit lands on QAT-settled weights. The risk is refusals training back in: filtering
  in stage 3 is mandatory, and the tripwire catches the rest.
- **Draft quant right after Heretic, before any gradient training.** QAT needs each tensor's final
  bit width. Training shifts per-tensor sensitivity only a little; stage 7 re-runs RCO, and if the map
  moves a lot, do a short (~0.5 h) stage-6-style QAT touch-up with the new map.
- **One SFT run for coding, concise reasoning and replay.** Sequential runs cause the forgetting the
  30% replay exists to prevent. Concise reasoning comes from the data (shortest correct trace).
- **GRPO after SFT.** It needs a model that already hits the reward sometimes. RLVR = GRPO with
  programmatic rewards; same stage.
- **Long context last among training stages.** It's the most memory-hungry and works best as a
  short final stage on an already-capable model (standard context-extension practice).

### Heretic: what it does and doesn't fix
Heretic runs with its **default config** (ARA modifier, default prompt sets and refusal scorer); only
run-control settings are set, so the run is unattended. `evals/heretic_pick.py` picks the trial with
the fewest refusals at **KL ≤ 0.2** (looser than Heretic's usual ~0.1 because ~25 h of training
follows and heals drift). Abliteration removes **refusals**. It does **not** remove hedged or
one-sided answers: a model can answer and still soften the facts. That's fixed in **stage 3 data**
(plain, sourced historical Q&A; drop teacher answers that moralise or hedge).

## Evals: all in vLLM

| Area | Benchmark | KV cache | Why this one |
|------|-----------|----------|--------------|
| Code | **HumanEval+** (EvalPlus) | as-is | OpenAI's 164 HumanEval problems with ~80× more tests; reports plain HumanEval too |
| Agentic | **BFCL** | as-is | Tool-call correctness: the core skill of an agent |
| Agentic | **IFEval** | as-is | Instruction following; agents fail when they ignore format rules |
| Maths | **MATH-500** | as-is | Not saturated at 4B like GSM8K; less noisy than AIME |
| Reasoning | **GPQA Diamond** | as-is | 198 hard questions, standard reasoning yardstick |
| Long context | **RULER** multi-key + multi-value, 32k | **bf16 and int4, side by side** | The int4-KV retrieval target; the gap between the two runs is the number KV-QAT must shrink |
| Refusals | Heretic refusal count | n/a | Free, already part of Heretic |

"As-is" = whatever the stage produced: bf16 base at stage 0, bf16 merged checkpoints in between, the
final compressed-tensors model with int4 KV at stage 8.

| When | What | Time |
|------|------|------|
| Stages 0 and 8 | Everything, RULER twice | ~3 h each (measured: stage 0 ≈ 3 h on the L4) |
| After stage 1 | HumanEval+, GPQA Diamond first 99 (paired vs stage 0), RULER 32k 25/task bf16 KV | <1 h |
| After stages 4, 5 | Everything except RULER | ~1.5 h each |
| After stage 6 | Everything, RULER twice | ~3 h |

Track **mean reasoning tokens** on MATH-500 and GPQA as the conciseness number. Serve with
`vllm serve` and point `lm-evaluation-harness`, `evalplus`, BFCL and RULER at its OpenAI-compatible
endpoint. Log every run to `results.md`. LiveCodeBench, SWE-bench etc. wait for harwyn-27b.

## Stage details

### 3. Datasets
**Stage 4 SFT mix (~20k examples; ~10k to save hours):**
| Share | Content | Starting sources (check licences on HF) |
|-------|---------|------------------------------------------|
| 65% | Agentic coding: multi-turn tool-call trajectories, repo edits, terminal, web design | SWE-smith trajectories, SWE-rebench OpenHands trajectories, Hermes function-calling, own distilled traces |
| 5% | Long-context seed (8k): retrieval inside agent sessions | Own synthetic (below) |
| 8% | Reasoning (short traces only) | OpenThoughts3, Nemotron post-training (length-filtered) |
| 7% | Maths | OpenMathReasoning (short-trace subset) |
| 6% | Chat + prose | Tulu 3 SFT mix, smoltalk |
| 5% | Multilingual | Aya dataset |
| 4% | History, plain and sourced | Own, distilled from a strong teacher, fact-checked |

**Stage 6 long-context mix (~2–3k examples, 16k–32k tokens):**
- 45% **synthetic agentic sessions**: real repos, long tool-call transcripts (file reads, grep
  output, test logs), with the final question answerable only from something read early. Distractors:
  near-duplicate files, a value that changes mid-session (answer must use the latest).
- 25% **long-document analysis**: multi-doc QA over long reports, specs and papers; answers cite
  the passage.
- 30% **replay**: short stage-4 data so short-context skills don't drift.
Generate questions with a strong teacher and keep only items where the teacher fails *without*
the long context (proves retrieval is needed). Don't reuse RULER templates: that's the eval.

Everything: dedupe, strip refusals/moralising, n-gram-decontaminate against the frozen eval set.
Biggest single lever remains **distilling from a stronger teacher** in your real tool-call format
and keeping only trajectories that pass tests.

### 4. SFT + QAT
Unsloth + TRL, LoRA r=64 on all linear layers (attention, MLP **and** GDN projections), merged-weight
fake-quant, KV fake-quant on, bf16 master weights, 8k context, 2 epochs, loss on assistant tokens
only. Expect QAT to cost ~30–50% more time than plain LoRA.

### 5. GRPO + QAT
TRL `GRPOTrainer` with vLLM rollouts (`int4_per_token_head` on), group size 4–8 on the L4.
Rewards: **coding**, hidden unit tests in a sandbox (pass rate); **web design**, HTML parses and
renders in Playwright, no console errors, plus a VLM judge score; **concise reasoning**, penalty per
reasoning token over budget, *correct answers only*.
Prompt sources to evaluate: [XiaomiMiMo/MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss).

### 6. Long-context QAT
Same QAT setup as stage 4 at 32k (fall back to 16k if it OOMs). Gradient checkpointing, batch 1,
grad accumulation, Unsloth's long-context offloading. The GDN layers are linear in memory, so the
4B should fit; the 27B runs this at 128k on H100s. Low LR (≈ 1/3 of stage 4), 1 epoch.

### 2 / 7. GSQ + RCO
IST-DASLab reference code: [GSQ](https://github.com/IST-DASLab/GSQ) quantises each tensor at every
candidate width (one layer on the GPU at a time, so the 27B is fine on one card);
[RCO](https://github.com/IST-DASLab/RCO) picks one width per tensor to hit an exact size budget
(GSQ checkpoints feed RCO's database via `run_split_checkpoint.py`), then
`run_build_checkpoint.py --format compressed-tensors` writes the packed checkpoint vLLM loads.
Calibration data = the stage 4 + 6 mixes **including long sequences**. Make 2–3 budgets
(~3.0 / 3.5 / 4.0 bits/weight); pick on stage 8. GSQ can also convert to Humming kernels (packed
low-bit GEMM in vLLM) if dequant-to-bf16 serving is too slow. Quality reference at 27B:
`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`.

## Also worth adding (ranked)
1. **Teacher distillation + rejection sampling**: usually the largest single gain.
2. **Decontamination**: otherwise gains are partly fake.
3. **Fallback KV format:** if int4 retrieval still lags badly after stage 6, vLLM's
   `turboquant_k8v4` (8-bit K, 4-bit V) is the next step up. It's a different quantiser, so it needs
   its own KV-QAT emulation.
4. **Train in your real agent harness's tool format** (e.g. OpenCode or Hermes).
5. **Checkpoint merging** (TIES/DARE via mergekit) only *before* stage 6: merging after QAT moves
   weights off the grid.
6. **DPO pass for prose/chat** if prose feels flat; run it under QAT like everything else.

## Reproducing
Linux, one NVIDIA GPU with ≥24 GB (results here: one L4), [uv](https://docs.astral.sh/uv/).
Weights, datasets and results live in a work directory outside the repo.

```bash
export HARWYN_WORK=/path/to/work          # models/, results/, venvs
git clone <this repo> $HARWYN_WORK/harwyn && cd $HARWYN_WORK/harwyn
bash setup.sh                             # 3 venvs at the exact versions used + Qwen3.5-4B weights

bash evals/stage0.sh                      # base-model baselines (~3 h on an L4)
$HARWYN_WORK/.venv-eval/bin/python evals/score.py $HARWYN_WORK/results/stage0
OPENAI_API_KEY=x BFCL_PROJECT_ROOT=$HARWYN_WORK/results/stage0/bfcl \
  $HARWYN_WORK/.venv-bfcl/bin/python evals/bfcl.py evaluate --model harwyn-FC --partial-eval

bash evals/stage1.sh                      # Heretic search (default 100 trials) -> pick -> export h0 -> 3 quick evals
HERETIC_EXPORT_ONLY=1 HERETIC_TRIAL=3 bash evals/stage1.sh   # re-export a specific trial (harwyn-4b used 3)
```

`ON_EXIT='<command>'` runs a command when a stage script ends (e.g. power off a rented GPU).
Heretic's search is stochastic, so a rerun finds a different Pareto front; the exact parameters of
the trial behind `h0` are in [`results/stage1/`](results/stage1/) and can be replayed with Heretic's
reproduction mode. Raw per-item outputs for every eval are in [`results/`](results/); the full story,
including what went wrong, is in [`LOG.md`](LOG.md).

## License
[Apache-2.0](LICENSE), the same licence as the Qwen3.5 base weights.

## Download the models
Every stage's weights are attached to a GitHub release (files over 2 GB are split; each release
lists the rejoin command and SHA-256 sums). The base model is
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and is not re-uploaded.

| Stage | Release | What it is |
|---|---|---|
| 1 | [`stage1-h0`](https://github.com/harry-wyn/harwyn/releases/tag/stage1-h0) | Heretic-abliterated base (trial 3), bf16 |
| 2 | [`stage2-q7`](https://github.com/harry-wyn/harwyn/releases/tag/stage2-q7) | h0 at ~7.15 bpw (6-bit MLP gate/up, 8-bit rest), fake-quant |
| 4 | [`stage4-sft`](https://github.com/harry-wyn/harwyn/releases/tag/stage4-sft) | QAT SFT LoRA, all checkpoints (step 212 = final) + eval outputs |
| 5 | [`stage5-rft`](https://github.com/harry-wyn/harwyn/releases/tag/stage5-rft) | Rejection-sampling SFT LoRA on top of stage 4 (step 155 = final), not yet benchmarked |
| 6 | [`stage6-grpo`](https://github.com/harry-wyn/harwyn/releases/tag/stage6-grpo) | GRPO LoRA, every iteration (it16 = final) |
