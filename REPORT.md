# harwyn-4b: technical report

**harwyn-4b** is Qwen3.5-4B, uncensored with Heretic, quantised to ~7.15 bits per weight, and then
trained (supervised fine-tuning, rejection-sampling fine-tuning, GRPO with verifiable rewards, and a final
coding top-up) **entirely quantisation-aware**: every training forward pass saw the exact weight bit map and
the exact 4-bit KV cache that the shipped model runs with. It is a rehearsal of the pipeline planned for a
27B model, done end to end on a 4B for about $39 of rented GPU time.

Every number below comes from files in this repository (`results/`), and every stage can be rerun from the
scripts it names. The lab notebook with every incident is [`LOG.md`](LOG.md).

---

## 1. Headline

Final model vs the base model, same harness, same sampling, same 32,768-token output budget:

| | **harwyn-4b** (~7.15 bpw, int4 KV) | Qwen3.5-4B (bf16, bf16 KV) |
|---|---|---|
| **GPQA Diamond** (first 99) | **45/99 (45.5%)** | 27/99 (27.3%) |
| GPQA reasoning length, median / mean | **3,726 / 5,525 tokens** | 15,908 / 17,530 tokens |
| GPQA answers that ran out of budget | **0** | 18 |
| **RULER 32k** (multikey 1/2/3, multivalue) | **100 / 100 / 96 / 100** | 100 / 100 / 88 / 99 |
| **HumanEval+** | 75.6% (plain 79.3%) | **81.7%** (plain 87.2%) |
| HumanEval+, answers that finished | **89.9%** of 138 | 87.6% of 153 |
| HumanEval answers that ran out of budget | 26 (15.9%) | 11 (6.7%) |
| HumanEval reasoning length, median | 1,949 tokens | 1,094 tokens |
| Weight memory | ~4.3 GB (Linears at 6/8-bit, embeddings bf16) | ~8 GB (bf16) |
| KV cache per token (8 attention layers) | ~8 KB (int4) | ~32 KB (bf16) |

**In one paragraph.** harwyn-4b reasons far better and far more efficiently than its base (+18 points on
GPQA while thinking about a quarter as long, never running out of budget), retrieves better at 32k
context, and does it in roughly half the weight memory and a quarter of the KV memory. On coding it is more
accurate whenever it finishes, but on about 16% of HumanEval problems it reasons until the budget runs out,
which costs it 6 points overall. That runaway tail is the one open problem (section 9).

---

## 2. Why this model, and the rules of the project

- **Testbed: Qwen3.5-4B, not Qwen3-4B.** Qwen3.5 and later use a hybrid stack: three Gated DeltaNet
  (linear attention) layers to every full-attention layer, 32 layers in total. Only the 8 full-attention
  layers have a KV cache (head dim 256, 4 KV heads, 32 KB/token in bf16). The planned 27B target uses the
  same design, so the KV-cache work here rehearses the right problem.
- **Deployment target = training target.** The model is served by vLLM as a mixed-precision
  compressed-tensors checkpoint with `--kv-cache-dtype int4_per_token_head`. Every training stage is QAT
  against exactly that: weights fake-quantised to their per-tensor bit width, K and V fake-quantised the way
  vLLM stores them.
- **Everything measured on vLLM,** thinking mode, the Qwen3.5 model card's sampling settings.
- **Public, reproducible record:** pinned versions (`setup.sh`), raw outputs (`results/`), and every model
  stage released as a download (README, "Download the models").

---

## 3. The pipeline at a glance

| Stage | What | Key result |
|---|---|---|
| 0 | Baselines on the base model | HumanEval+ 76.8%, GPQA 10%, RULER ≈ unchanged with int4 KV |
| 1 | **Heretic** abliteration → `h0` | Refusals 99/100 → 11/100 at KL 0.043, benchmarks unchanged |
| 2 | **GSQ + RCO** quantisation | 3.0 bpw collapses at 4B; **~7.15 bpw** (6-bit MLP gate/up, 8-bit rest) is near-lossless |
| 4 | **QAT SFT**, agentic coding + tool calling | HumanEval+ 76.2 → 78.0–80.5%, GPQA 11 → 18/99, RULER 100 across the board |
| 5 | **Rejection-sampling SFT** (shortest correct answer) | 1,886 self-generated, unit-tested solutions |
| 6 | **GRPO** with verifiable rewards (tools + hidden tests, webdev + vision judge) | Webdev reward 0.27 → 0.44, shorter episodes |
| 7 | **Coding top-up SFT**, concise reasoning, multi-harness replay | Final model (section 1) |

Stage numbers follow the original plan; stage 3 (dataset construction) is folded into 4.

---

## 4. Stages in detail

### Stage 0: baselines (8k budget)
HumanEval+ 76.8% (plain 81.1%), IFEval 80.8%, MATH-500 60.0%, GPQA Diamond 10.1% (81% of answers hit
the 8k budget). BFCL: simple 92, multiple 94, parallel-multiple 90, irrelevance 90.6, live-simple 88,
live-multiple 59, multi-turn 65. RULER 32k NIAH was the same with bf16 and int4 KV
(98/100/88/99 vs 98/100/88/100): int4 KV costs nothing measurable on retrieval for the base model.

### Stage 1: Heretic (p-e-w/heretic, ARA modifier, default config)
Trial 3 chosen: **11/100 refusals, KL 0.043** (base: 99/100). A lower-refusal trial (47, KL 0.134)
was rejected after it produced runaway thinking. h0 vs base: HumanEval+ 77.4 vs 76.8%, RULER
100/100/92/100. Abliteration removes refusals, not hedging; that was left to the training data.

### Stage 2: GSQ + RCO quantisation
GPTQ database at 2-8 bits for all 200 quantisable Linears; GSQ at 2/3 bits on the MLPs; RCO's
gradient search for a per-tensor bit map. Four RCO patches were needed (single-GPU init, resumable
database, importing GSQ's symmetric codes, streaming deltas from RAM in forward and backward so the search
fits 24 GB). **At 3.0 bpw the 4B collapsed** (HumanEval 0%, RULER 20/0/0/16; RCO's KL 1.17): small models
degrade much faster at low bit widths. **The chosen ~7.15 bpw map** (MLP gate/up 6-bit, everything else
8-bit) matched h0 within noise: HumanEval+ 76.2% (plain 80.5%), GPQA 11/99, RULER 100/100/96/100.
This map is the QAT target for every later stage.

### How the QAT works (all training stages)
`train/train_qat.py`:
- **Weights:** LoRA r32 on all 200 quantised Linears. Each forward fake-quantises the *merged* weight
  `W + 2·B·A` to that tensor's width (symmetric, group 128, straight-through estimator), so training sees the
  weights that ship rather than an unquantised LoRA on a quantised base.
- **KV cache:** in the 8 full-attention layers, K (after RoPE) and V are rotated with a fixed randomized
  Hadamard transform, quantised to asymmetric int4 with one scale and zero-point per (token, head), and
  rotated back, matching vLLM's `int4_per_token_head` (rotation signs differ, error statistics match).
- **Loss** only on assistant tokens (reasoning and tool calls included), chunked over the 248k vocabulary.

### Stage 4: QAT SFT
5,232 examples (31.2M tokens) built by `data/build_sft.py`: OpenHands SWE traces, UltraData Code-Agent and
Tool-Use, SWE-QA-Pro, webdev distillation, OpenCodeReasoning, smoltalk replay; ≤ 16k tokens, refusals
dropped, 13-gram decontaminated against HumanEval+ and GPQA. 0.65 epoch (212 steps). Loss
0.511 → 0.434 by quarter. Two runs with identical settings (the first run's weights were lost to an
operations error; see LOG): HumanEval+ 80.5% / 78.0%, GPQA 18/99 both, RULER 100/100/100/100 both, at the
8k budget. The run-to-run HumanEval gap came entirely from answers hitting the budget (5.5% vs 12.8%).

### Stage 5: rejection-sampling SFT
The stage-4 model (served in its deployment format) attempted 2,400 KodCode problems 4 times each; every
attempt ran against the problem's unit tests. 63.2% of attempts passed, 14.7% ran out of an 8k thinking
budget, and 78.6% of problems had at least one correct answer that finished; the shortest was kept.
Training on those 1,886 answers plus 600 agentic replay rows continued the LoRA at half the learning rate.

### Stage 6: GRPO with verifiable rewards
`rl/`. Each iteration served the current policy in its deployment format and ran 16 tasks × 8 episodes:
- **Agentic coding:** an empty sandbox with `bash`, `write_file`, `read_file`, `submit`; reward 1 if the
  problem's hidden pytest file passes.
- **Webdev:** MiMo webdev briefs; the model must produce a self-contained `index.html`; it is rendered with
  Chromium inside MiMo's own webdev image (unpacked and run unprivileged with proot), screenshotted at
  1280px and scored 0-10 by a vision model (Gemini 3.8 Flash, fixed rubric, temperature 0).
- Group-normalised advantages, 2 QAT optimiser steps per iteration, LR 2e-5 then cosine from 5e-5.

| Iterations | Coding reward | Webdev reward | Mean turns |
|---|---|---|---|
| 1-4 | 0.69 | 0.27 | 4.6 |
| 13-16 | 0.73 | **0.44** | **3.3** |

At the 32k budget the stage-6 model scored HumanEval+ 75.0%, GPQA 32/99, RULER 100/100/100/100.
MiMo's SWE environments were not used: their graders aren't in the published images, the images are
multi-GB each, and a 4B rarely solves real-repo issues (no reward variance for GRPO).

### Stage 7: coding top-up (final model)
3,503 examples (~8.8M tokens): 2,400 KodCode R1 solutions that passed their unit tests with reasoning
≤ 3,000 tokens; 500 smoltalk2 reasoning chats; ~600 agentic examples across **five harness styles**
(UltraData Tool-Use with many tool schemas, UltraData Code-Agent, SWE-QA-Pro, OpenHands traces, our own
stage-6 episodes) so the model is not tied to one agent framework. 218 steps, peak LR 5e-5. Loss by
25-step block: 0.627 → 0.583 → 0.568 → 0.557 → 0.548 → … Benchmarks: section 1.

---

## 5. Every benchmark, every stage

| Stage | Budget | HumanEval+ (plain) | GPQA /99 | RULER 32k |
|---|---|---|---|---|
| 0 base | 8k | 76.8% (81.1%) | 11 | 98/100/88/99 (bf16 KV), 98/100/88/100 (int4 KV) |
| 1 h0 | 8k | 77.4% (81.1%) | 7 | 100/100/92/100 |
| 2 h0 at 3.0 bpw | 8k | 0.0% | 2 | 20/0/0/16 |
| 2 h0 at ~7.15 bpw | 8k | 76.2% (80.5%) | 11 | 100/100/96/100 |
| 4 SFT run 1 / run 2 | 8k | 80.5% (88.4%) / 78.0% (82.3%) | 18 / 18 | 100 ×4 / 100 ×4 |
| 6 GRPO | 32k | 75.0% (79.9%) | 32 | 100 ×4 |
| **7 final** | **32k** | **75.6% (79.3%)** | **45** | **100/100/96/100** |
| base | 32k | 81.7% (87.2%) | 27 | 100/100/88/99 |

From stage 4 on, every number is in the deployment format (q7 weights + int4 KV). The 8k and 32k rows
are not directly comparable: at 8k, an answer still thinking at the limit counts as wrong.

---

## 6. Reasoning length

| | median | mean | 90th pct | hit 32k |
|---|---|---|---|---|
| harwyn-4b, GPQA | 3,726 | 5,525 | 10,490 | 0 |
| base, GPQA | 15,908 | 17,530 | 32,768 | 18 |
| harwyn-4b, HumanEval | 1,949 | 6,837 | 32,768 | 26 |
| base, HumanEval | 1,094 | 4,896 | 10,649 | 11 |

On science reasoning the trained model is decisive; on coding, most answers are short but a tail does not
terminate.

---

## 7. Cost

| Stage | GPU time | Cost |
|---|---|---|
| 0-2 baselines, Heretic, quantisation | free credits + ~$4.30 | ~$4.30 |
| 4 SFT (incl. ~9 h idle after a failed upload) + rerun | | ~$18 |
| 5 RFT | 3.5 h A100 40GB | $3.56 |
| 6 GRPO (two host outages, not billed) | ~9.7 h A100 40GB | $9.68 + ~£1 judge |
| 7 top-up + final/base benchmarks | ~3.1 h A100 40GB | ~$3.10 |

About $10 of the total was lost to operational mistakes (an idle GPU after an upload failed, a lost
checkpoint). The fixes became rules: copy and hash-verify every checkpoint off the machine as it is
written, hard time caps on every instance, and never kill processes by a loose name match.

---

## 8. What worked and what didn't

**Worked**
- QAT against the exact deployment format: no measurable loss from 4-bit KV at any stage; retrieval
  improved over the base.
- Near-lossless ~7.15 bpw beat chasing 3 bpw on a 4B.
- Large GPQA gains with much shorter reasoning, from SFT, RFT and GRPO combined.
- A working GRPO harness: real tools, hidden tests, rendered-screenshot judging, all QAT, every checkpoint
  saved.

**Didn't**
- 3.0 bpw at 4B (collapse).
- Removing the HumanEval runaway tail: RFT, GRPO's shorter episodes and the concise-reasoning SFT didn't fix
  it (it went from 21 to 26 answers between stages 6 and 7).
- MiMo's SWE environments (graders not published).

---

## 9. Open problem and next steps

1. **Budget forcing at inference:** stop reasoning after N tokens (e.g. 8k) and make the model answer.
   No training needed; one HumanEval rerun tests it. If it recovers the runaway answers, harwyn-4b beats
   the base on all three benchmarks.
2. **A length-aware reward** in GRPO for coding, so non-terminating reasoning is penalised directly.
3. **Pack the final model** into the real compressed-tensors checkpoint for vLLM and re-run the benchmarks
   on that file.
4. Repeat the whole recipe on the 27B.

---

## 10. Running it

- **GPU (vLLM, the deployment format):** merge `stage7-final/lora_step218.pt` into `stage1-h0`
  (`evals/fakequant_from_db.py` with the merged weights), then
  `vllm serve <model> --kv-cache-dtype int4_per_token_head --reasoning-parser qwen3`.
- **CPU laptop (llama.cpp):** `harwyn-4b` (see README): the same bit map as GGUF (Q6_K MLP gate/up,
  Q8_0 elsewhere) with a q4_0 KV cache, served as an OpenAI-compatible API on localhost.
