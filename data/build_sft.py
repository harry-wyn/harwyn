"""Build the stage 4 SFT mix (agentic coding + tool calling focus) as JSONL of {"messages", "tools"}.
Streams each source, normalises to OpenAI-style messages with tool_calls (arguments as dicts, which the
Qwen3.5 chat template expects), renders with Qwen3.5's own template to keep examples <= MAX_TOKENS,
drops refusals and 13-gram overlaps with the eval sets (HumanEval+, GPQA Diamond).
usage: python build_sft.py MODEL_DIR OUT.jsonl
"""
import ast
import json
import random
import re
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

MAX_TOKENS = 16384  # agentic traces are long; trained at 16k context
MIX = [  # (name, dataset, config, split, target count)
    ("swe_traces", "nvidia/Open-SWE-Traces", "v1.1", "openhands", 1500),
    ("code_agent", "openbmb/UltraData-SFT-Agent-2609", "Code-Agent", "train", 1000),
    ("tool_use", "openbmb/UltraData-SFT-Agent-2609", "Tool-Use", "train", 2000),
    ("swe_qa", "TIGER-Lab/SWE-QA-Pro-SFT-Trajectories", "default", "train", 500),
    ("webdev", "mindchain/mimo-v2.6-distill-qwen9b-webdev-full", "default", "train", 800),
    ("code_reasoning", "nvidia/OpenCodeReasoning", "split_0", "split_0", 700),
    ("replay", "HuggingFaceTB/smoltalk", "all", "train", 1000),
]
REFUSAL = re.compile(r"\b(I can(?:no|')t (?:help|assist)|I'm (?:sorry|unable)|as an AI)\b", re.I)


def parse(x):
    if isinstance(x, (list, dict)) or x is None:
        return x
    try:
        return json.loads(x)
    except Exception:
        return ast.literal_eval(x)


def norm_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        role = {"human": "user", "gpt": "assistant", "function": "tool", "observation": "tool"}.get(m.get("role") or m.get("from"), m.get("role") or m.get("from"))
        content = m.get("content", m.get("value", "")) or ""
        if role == "tool_call":  # SWE-QA-Pro: each call is its own message; attach it to the assistant turn
            c = parse(content)
            if not (out and out[-1]["role"] == "assistant"):
                out.append({"role": "assistant", "content": ""})
            out[-1].setdefault("tool_calls", []).append({"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments", {})}})
            continue
        if role == "tool_response":
            role = "tool"
        nm = {"role": role, "content": content if isinstance(content, str) else json.dumps(content)}
        if m.get("tool_calls"):
            calls = []
            for c in parse(m["tool_calls"]):
                f = c.get("function", c)
                args = f.get("arguments", {})
                args = parse(args) if isinstance(args, str) and args.strip()[:1] in "{[" else ({} if args in ("", None) else args)
                calls.append({"type": "function", "function": {"name": f["name"], "arguments": args}})
            nm["tool_calls"] = calls
        if role == "tool" and m.get("tool_call_id"):
            nm["tool_call_id"] = m["tool_call_id"]
        out.append(nm)
    return out


def norm_tools(tools):
    tools = parse(tools) or []
    return [parse(t) if isinstance(t, str) else t for t in tools]


def from_row(name, r):
    if name == "webdev":
        if float(r.get("score") or 0) < 0.85:
            return None
        return [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}], []
    if name == "code_reasoning":
        return [{"role": "user", "content": r["input"]}, {"role": "assistant", "content": r["output"]}], []
    if name == "swe_traces" and str(r.get("resolved")) not in ("1", "True", "true"):
        return None
    if name == "replay" and r.get("source", "").startswith("apigen"):
        return None
    return norm_messages(parse(r["messages"])), norm_tools(r.get("tools"))


def rows(name, ds, cfg, split):
    if name == "tool_use":  # its tool schemas vary row to row, so `datasets` can't cast it: read the raw JSONL
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        for f in ("Tool_Use_part-5-of-9.jsonl", "Tool_Use_part-4-of-9.jsonl"):
            with fs.open(f"datasets/{ds}/data/Tool_Use/{f}") as fh:
                for line in fh:
                    yield json.loads(line)
        return
    yield from load_dataset(ds, cfg, split=split, streaming=True).shuffle(seed=0, buffer_size=2000)


def ngrams(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    eval_text = [r["prompt"] for r in load_dataset("evalplus/humanevalplus", split="test")]
    eval_text += [r["question"] for r in load_dataset("fingertap/GPQA-Diamond", split="test")]
    banned = set().union(*(ngrams(t) for t in eval_text))
    rng = random.Random(0)
    done = {}  # resume: sources already in OUT are kept as they are
    try:
        for l in open(sys.argv[2]):
            s = json.loads(l)["source"]; done[s] = done.get(s, 0) + 1
    except FileNotFoundError:
        pass
    out, stats = open(sys.argv[2], "a"), {}
    for name, ds, cfg, split, want in MIX:
        if name in done:
            stats[name] = (done[name], None); continue
        kept = seen = 0
        for r in rows(name, ds, cfg, split):
            seen += 1
            if kept >= want or seen > want * 10:
                break
            try:
                ex = from_row(name, r)
                if not ex:
                    continue
                msgs, tools = ex
                if sum(len(m["content"]) for m in msgs) > MAX_TOKENS * 5:  # cheap pre-filter (~3-4 chars/token)
                    continue
                if not any(m["role"] == "assistant" for m in msgs):
                    continue
                if any(REFUSAL.search(m["content"]) for m in msgs if m["role"] == "assistant"):
                    continue
                if ngrams(" ".join(m["content"] for m in msgs)) & banned:
                    continue
                n = len(tok(tok.apply_chat_template(msgs, tools=tools or None, tokenize=False)).input_ids)
                if n > MAX_TOKENS:
                    continue
            except Exception:
                continue
            out.write(json.dumps({"source": name, "messages": msgs, "tools": tools}) + "\n")
            kept += 1
        stats[name] = (kept, seen)
        print(name, kept, "kept of", seen, flush=True)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
