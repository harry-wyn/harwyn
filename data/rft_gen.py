"""Stage 5 data: rejection sampling. The current model (served by vLLM at localhost:8000 as "m", in its
deployment format) answers N_PROMPTS KodCode problems K times; each answer's code runs against the
problem's pytest file; per problem we keep the SHORTEST answer that finished and passed. Output is the
same {source, messages, tools} JSONL as build_sft.py, plus replay rows from REPLAY (tool use etc.).
usage: python rft_gen.py OUT.jsonl REPLAY.jsonl [N_PROMPTS=2400] [K=4]
Writes OUT.raw.jsonl (every sample with tokens/pass) so the selection can be redone offline.
"""
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import tempfile

from datasets import load_dataset
from openai import AsyncOpenAI

out_path, replay_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2400
K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
MAX_TOKENS = 8192   # answers that need more than this are not the behaviour we want to reinforce
REPLAY_N = 600
CODING = dict(temperature=0.6, top_p=0.95, extra_body={"top_k": 20})   # Qwen3.5 card, thinking + code
PROMPT = ("{q}\n\nWrite a Python solution. The function must be `{decl}`. "
          "Return the full code (with imports) in a single ```python code block.")


def ngrams(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def code_of(answer):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.S)
    return blocks[-1] if blocks else None


def passes(code, test):
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/solution.py", "w").write(code)
        open(f"{d}/test_solution.py", "w").write(test)
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", d],
                               cwd=d, capture_output=True, timeout=30)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


async def main():
    banned = set().union(*(ngrams(r["prompt"]) for r in load_dataset("evalplus/humanevalplus", split="test")))
    rows = [r for r in load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
            if r["test_info"] and not (ngrams(r["question"]) & banned)]
    random.Random(0).shuffle(rows)
    rows = rows[:N]
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x", timeout=3600)
    sem = asyncio.Semaphore(128)
    raw = open(out_path.replace(".jsonl", ".raw.jsonl"), "w")

    async def one(r):
        prompt = PROMPT.format(q=r["question"], decl=r["test_info"][0]["function_declaration"])
        async with sem:
            resp = await client.chat.completions.create(model="m", messages=[{"role": "user", "content": prompt}],
                                                        max_tokens=MAX_TOKENS, n=K, **CODING)
        best = None
        for c in resp.choices:
            content = c.message.content or ""
            reasoning = getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None) or ""
            code = code_of(content)
            ok = c.finish_reason == "stop" and code is not None and await asyncio.to_thread(passes, code, r["test"])
            toks = len(reasoning) + len(content)   # characters as a length proxy
            raw.write(json.dumps({"id": r["question_id"], "ok": ok, "finish": c.finish_reason, "chars": toks}) + "\n")
            if ok and (best is None or toks < best[0]):
                best = (toks, reasoning, content)
        if best:
            answer = (f"<think>\n{best[1].strip()}\n</think>\n\n" if best[1] else "") + best[2].strip()
            return {"source": "rft_kodcode", "messages": [{"role": "user", "content": prompt},
                                                          {"role": "assistant", "content": answer}], "tools": []}

    kept = [x for x in await asyncio.gather(*(one(r) for r in rows)) if x]
    replay = [json.loads(l) for l in open(replay_path)]
    replay = [x for x in replay if x["source"] in ("tool_use", "code_agent", "swe_traces", "swe_qa", "webdev")]
    random.Random(1).shuffle(replay)
    with open(out_path, "w") as f:
        for x in kept + replay[:REPLAY_N]:
            f.write(json.dumps(x) + "\n")
    print(json.dumps({"prompts": len(rows), "kept": len(kept), "replay": min(REPLAY_N, len(replay))}))


asyncio.run(main())
