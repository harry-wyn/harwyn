"""Generate answers for MATH-500, GPQA Diamond, HumanEval+ and IFEval from a running vLLM server.

usage: python gen.py OUT_DIR math500,gpqa,humaneval,ifeval   (LIMIT=3 for a smoke test)
Writes OUT_DIR/<bench>.jsonl, one line per item, resumable. Scoring is CPU-only: score.py.
All benches run concurrently against one server so the GPU stays saturated.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI

# Qwen3.5 model-card sampling for thinking mode
GENERAL = dict(temperature=1.0, top_p=0.95, presence_penalty=1.5, extra_body={"top_k": 20})
CODING = dict(temperature=0.6, top_p=0.95, presence_penalty=0.0, extra_body={"top_k": 20})
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MC_SUFFIX = '\nPlease show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.'
# Output budget: the Qwen3.5 card's recommended 32,768 tokens (fits --max-model-len 40960 with the prompt).
# Results before 2026-10-15 used a fixed 8,192-token budget; MAX_TOKENS=8192 reproduces them.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 32768))
CODE_PROMPT = ("Complete the following Python function. Return the full function (with imports) "
               "in a single ```python code block.\n\n```python\n{}```")


def math500():
    for r in load_dataset("HuggingFaceH4/MATH-500", split="test"):
        yield r["unique_id"], r["problem"] + MATH_SUFFIX, GENERAL, MAX_TOKENS


def gpqa():
    for i, r in enumerate(load_dataset("fingertap/GPQA-Diamond", split="test")):
        yield str(i), r["question"] + MC_SUFFIX, GENERAL, MAX_TOKENS


def humaneval():
    from evalplus.data import get_human_eval_plus
    for tid, p in get_human_eval_plus().items():
        yield tid, CODE_PROMPT.format(p["prompt"]), CODING, MAX_TOKENS


def ifeval():
    for r in load_dataset("google/IFEval", split="train"):
        yield str(r["key"]), r["prompt"], GENERAL, MAX_TOKENS


BENCHES = dict(math500=math500, gpqa=gpqa, humaneval=humaneval, ifeval=ifeval)


async def run_bench(client, sem, name, out_dir, limit):
    path = out_dir / f"{name}.jsonl"
    done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
    items = [it for it in BENCHES[name]() if it[0] not in done][:limit]
    f = path.open("a")

    async def one(id_, prompt, params, max_tokens):
        async with sem:
            r = await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, **params)
        c = r.choices[0]
        f.write(json.dumps(dict(id=id_, content=c.message.content or "", finish=c.finish_reason,
                                tokens=r.usage.completion_tokens)) + "\n")
        f.flush()

    await asyncio.gather(*(one(*it) for it in items))
    print(f"{name}: {len(items)} generated, {len(done)} already done", flush=True)


async def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(os.getenv("LIMIT", "0")) or None
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x", timeout=7200, max_retries=2)
    sem = asyncio.Semaphore(int(os.getenv("CONCURRENCY", "128")))
    await asyncio.gather(*(run_bench(client, sem, b, out_dir, limit) for b in sys.argv[2].split(",")))


if __name__ == "__main__":
    asyncio.run(main())
