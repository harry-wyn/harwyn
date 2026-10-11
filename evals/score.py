"""Score gen.py outputs on CPU. usage: python score.py OUT_DIR  -> prints a markdown table, writes OUT_DIR/scores.json"""
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

from datasets import load_dataset

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
load = lambda b: [json.loads(l) for l in (out / f"{b}.jsonl").open()] if (out / f"{b}.jsonl").exists() else []


def math500(rows):
    from math_verify import parse, verify
    gold = {r["unique_id"]: r["answer"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    return mean(bool(verify(parse(f"${gold[r['id']]}$"), parse(r["content"]))) for r in rows)


def gpqa_letter(text):
    m = re.findall(r'"answer"\s*:\s*"?\(?([A-D])\b', text) or re.findall(r"answer(?: is)?[:\s]*\**\(?([A-D])\b", text, re.I)
    return m[-1].upper() if m else None


def gpqa(rows):
    gold = [r["answer"].strip().upper() for r in load_dataset("fingertap/GPQA-Diamond", split="test")]
    return mean(gpqa_letter(r["content"]) == gold[int(r["id"])] for r in rows)


def humaneval(rows):
    samples = out / "humaneval_samples.jsonl"
    with samples.open("w") as f:
        for r in rows:
            blocks = re.findall(r"```(?:python)?\n(.*?)```", r["content"], re.S)
            f.write(json.dumps(dict(task_id=r["id"], solution=blocks[-1] if blocks else r["content"])) + "\n")
    log = subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", str(samples),
                          "--i-just-wanna-run"], capture_output=True, text=True).stdout
    base, plus = re.findall(r"pass@1:\s*([\d.]+)", log)[:2]
    return float(plus), float(base)


def ifeval(rows):
    from lm_eval.tasks.ifeval.utils import process_results
    docs = {str(d["key"]): d for d in load_dataset("google/IFEval", split="train")}
    res = [process_results(docs[r["id"]], [r["content"]]) for r in rows]
    return mean(x["prompt_level_strict_acc"] for x in res)


def check():
    assert gpqa_letter('blah "answer": "C"') == "C"
    assert gpqa_letter("The answer is (b).") == "B"
    assert gpqa_letter("no letter here") is None


if __name__ == "__main__":
    check()
    scores, lines = {}, ["| Benchmark | Score | n | Mean output tokens | Hit max_tokens |", "|---|---|---|---|---|"]
    for name, fn in dict(math500=math500, gpqa=gpqa, humaneval=humaneval, ifeval=ifeval).items():
        rows = load(name)
        if not rows:
            continue
        s = fn(rows)
        extra = f" (plain HumanEval {s[1]:.1%})" if isinstance(s, tuple) else ""
        s = s[0] if isinstance(s, tuple) else s
        trunc = mean(r["finish"] == "length" for r in rows)
        toks = round(mean(r["tokens"] for r in rows))
        scores[name] = dict(score=s, n=len(rows), mean_tokens=toks, truncated=trunc)
        lines.append(f"| {name} | {s:.1%}{extra} | {len(rows)} | {toks} | {trunc:.1%} |")
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    print("\n".join(lines))
