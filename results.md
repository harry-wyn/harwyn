# harwyn results

All runs: vLLM 0.31, L4, thinking mode, Qwen3.5 card sampling, **8,192-token output budget** (truncated = wrong).

## Stage 0: base Qwen3.5-4B (2026-10-11)

| Benchmark | Score | Mean output tokens | Hit 8k budget |
|---|---|---|---|
| HumanEval+ | 76.8% (plain 81.1%) | 3,047 | 12% |
| IFEval (prompt strict) | 80.8% | 4,305 | 10% |
| MATH-500 | 60.0% (100% of the 298 that finished in budget) | 5,598 | 40% |
| GPQA Diamond | 10.1% (52.6% of the 38 that finished) | 7,695 | 81% |

BFCL (100 sampled per category, seed 0): simple_python 92, multiple 94, parallel_multiple 90,
irrelevance 90.6, live_simple 88, live_multiple 59, multi_turn_base 65.

RULER 32k, 50/task:
| KV cache | multikey_1 | multikey_2 | multikey_3 | multivalue |
|---|---|---|---|---|
| bf16 | 98 | 100 | 88 | 99 |
| int4_per_token_head | 98 | 100 | 88 | 100 |

**int4 KV costs nothing measurable on RULER NIAH at 32k for the base model** (the 1-point multivalue
difference is noise at n=50). NIAH is too easy to show the KV-QAT benefit; add a harder long-context
check (64k+, variable tracking / multi-hop, own agentic-retrieval set) before stage 6.
Note: first int4 run froze vLLM at 198/200 (lost); rerun with ruler.sh (cache + watchdog) completed cleanly.

## Stage 1: Heretic trial 3 → h0 (2026-10-11)

Refusals on Heretic's default prompts: **11/100** (base 99/100), KL 0.043. Details: [`LOG.md`](LOG.md).

| Benchmark | h0 | Base, same items |
|---|---|---|
| HumanEval+ | 77.4% (plain 81.1%) | 76.8% (plain 81.1%) |
| GPQA Diamond, first 99 | 7/99 (6/17 where both finished) | 11/99 (10/17) |
| RULER 32k bf16 KV | 100 / 100 / 92 / 100 (25/task) | 98 / 100 / 88 / 99 (50/task) |

## Stage 2: h0 at 3.0 bpw (GSQ + RCO), evaluated as fake-quant (2026-10-14)

| Benchmark | h0 | 3.0 bpw |
|---|---|---|
| HumanEval+ | 77.4% | 0.0% |
| GPQA Diamond, first 99 | 7/99 | 2/99 |
| RULER 32k (25/task) | 100 / 100 / 92 / 100 | 20 / 0 / 0 / 16 |

The 4B collapses at 3.0 bpw (RCO's KL vs h0: 1.17). Details and caveats in [`LOG.md`](LOG.md).

**Chosen instead: ~7.15 bpw** (MLP gate/up 6-bit, rest 8-bit): HumanEval+ 76.2%, GPQA 11/99, RULER 100/100/96/100. Near-lossless.

## Stage 4: QAT SFT (harwyn-4b-sft, served at ~7.15 bpw with int4 KV)
0.65 epoch of `sft_v1` (212 steps, ~20M tokens). Train loss by quarter: 0.511 → 0.460 → 0.448 → 0.434.

| Benchmark (8k budget) | h0 bf16 | h0 q7 (bf16 KV) | **harwyn-4b-sft q7 + int4 KV** |
|---|---|---|---|
| HumanEval+ | 77.4% | 76.2% | **80.5%** |
| HumanEval (plain) | — | 80.5% | **88.4%** |
| GPQA Diamond, first 99 | 7/99 | 11/99 | **18/99** (36% hit the 8k limit) |
| RULER 32k, 25/task (mk1/mk2/mk3/mv) | 100/100/92/100 | 100/100/96/100 | **100/100/100/100** |

**Rerun (same data, seed and settings; the first run's weights were lost):** train loss matched run 1 to
3 decimals throughout. HumanEval+ **78.0%** (plain 82.3%), GPQA **18/99**, RULER **100/100/100/100**.
Run-to-run spread on HumanEval+ is ±2.5 pts, mostly from answers hitting the 8k budget (12.8% vs 5.5%).
Run 2 is the one kept (`results/stage4_run2/`).

## Stage 6: GRPO (harwyn-4b-grpo, q7 + int4 KV, **32k budget**)
| Benchmark | Score |
|---|---|
| HumanEval+ | 75.0% (plain 79.9%) |
| GPQA Diamond, first 99 | 32/99 |
| RULER 32k (mk1/mk2/mk3/mv) | 100/100/100/100 |
Training reward, first → last 4 iterations: coding 0.69 → 0.73, webdev 0.27 → 0.44. Not comparable with the 8k rows above.
