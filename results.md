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

## Stage 2: GSQ + RCO at 3.0 bpw (in progress)

GPTQ database and GSQ 2/3-bit MLPs done; the bit-allocation search is paused (GPU time ran out).
RCO's sensitivity-only allocation is degenerate (all MLPs at 2 bits; KL 4.65 vs bf16) and was not
evaluated. Details: [`LOG.md`](LOG.md).
