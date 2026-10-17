"""Stage 7 data: a short coding top-up with concise reasoning, plus replay so earlier skills stay.
  code     KodCode/KodCode-V1-SFT-R1 `train` (R1 solutions that passed the problem's unit tests), reasoning
           <= MAX_THINK tokens, whole example <= MAX_TOKENS, 13-gram decontaminated against HumanEval+.
  general  HuggingFaceTB/smoltalk2 SFT reasoning chats (everyday + multi-turn instruction following), every
           assistant reasoning block <= MAX_THINK_GENERAL tokens.
  agentic  tool use across several harness styles, so the model isn't tied to one agent framework:
           UltraData Tool-Use (many tool schemas), UltraData Code-Agent, OpenHands traces (Open-SWE-Traces),
           SWE-QA-Pro (read-only codebase agent) via build_sft.py's loaders, each <= MAX_AGENT_TOKENS; plus a
           smaller share of our own stage-6 GRPO episodes that earned full reward.
usage: python build_sft2.py TOKENIZER_DIR STAGE6_RESULTS_DIR OUT.jsonl [N_CODE=2400] [N_GEN=500] [N_AGENT=200]
"""
import glob, gzip, json, random, re, sys

from datasets import load_dataset
from transformers import AutoTokenizer

tok_dir, s6, out = sys.argv[1:4]
N_CODE, N_GEN, N_AGENT = (int(x) for x in (sys.argv[4:7] + ["2400", "500", "200"][len(sys.argv[4:7]):]))
MAX_THINK, MAX_THINK_GENERAL, MAX_TOKENS, MAX_AGENT_TOKENS = 3000, 2000, 6000, 8000
HARNESSES = [("tool_use", 250), ("code_agent", 150), ("swe_qa", 100), ("swe_traces", 100)]   # build_sft.py sources
tok = AutoTokenizer.from_pretrained(tok_dir)
ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)


def ngrams(t, n=13):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


banned = set().union(*(ngrams(r["prompt"]) for r in load_dataset("evalplus/humanevalplus", split="test")))


def think_len(content):
    m = re.search(r"<think>(.*?)</think>", content, re.S)
    return ntok(m.group(1)) if m else 0


rows, stats = [], {}
# code
kept = seen = 0
for r in load_dataset("KodCode/KodCode-V1-SFT-R1", split="train", streaming=True).shuffle(seed=0, buffer_size=20000):
    seen += 1
    if kept >= N_CODE or seen > N_CODE * 40:
        break
    if str(r.get("r1_correctness")) != "True":
        continue
    conv = r["conversations"]
    if len(conv) != 2 or "</think>" not in conv[1]["value"]:
        continue
    q, a = conv[0]["value"], conv[1]["value"]
    if ngrams(q) & banned or think_len(a) > MAX_THINK or ntok(q) + ntok(a) > MAX_TOKENS:
        continue
    rows.append({"source": "kodcode_r1", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "tools": []})
    kept += 1
stats["code"] = (kept, seen)
# general
kept = 0
for split in ("smoltalk_everyday_convs_reasoning_Qwen3_32B_think", "multi_turn_reasoning_if_think"):
    for r in load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split, streaming=True).shuffle(seed=0, buffer_size=5000):
        if kept >= N_GEN:
            break
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"] if m["role"] in ("system", "user", "assistant")]
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        if not asst or any(think_len(c) > MAX_THINK_GENERAL for c in asst) or sum(ntok(m["content"]) for m in msgs) > MAX_TOKENS:
            continue
        rows.append({"source": "smoltalk2_" + split[:20], "messages": msgs, "tools": []})
        kept += 1
stats["general"] = kept
# agentic replay: best stage-6 episodes, later iterations first
eps = []
for f in sorted(glob.glob(f"{s6}/rollouts_it*.jsonl.gz"), key=lambda p: -int(re.search(r"it(\d+)", p).group(1))):
    for l in gzip.open(f):
        e = json.loads(l)
        if "error" in e or e.get("reward") is None:
            continue
        if (e["kind"] == "code" and e["reward"] == 1.0) or (e["kind"] == "webdev" and e["reward"] >= 0.6):
            eps.append(e)
seen_tasks, kept = set(), 0
for e in eps:  # one episode per task, shortest first within the order above
    if kept >= N_AGENT or e["task_id"] in seen_tasks:
        continue
    seen_tasks.add(e["task_id"])
    rows.append({"source": "grpo_" + e["kind"], "messages": e["messages"], "tools": e["tools"]})
    kept += 1
stats["agentic"] = (kept, len(eps))
# multi-harness agentic replay
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import build_sft as B
for name, want in HARNESSES:
    ds, cfg, split = next((d, c, sp) for n, d, c, sp, _ in B.MIX if n == name)
    kept = seen = 0
    for r in B.rows(name, ds, cfg, split):
        seen += 1
        if kept >= want or seen > want * 40:
            break
        try:
            ex = B.from_row(name, r)
            if not ex:
                continue
            msgs, tools = ex
            if not any(m["role"] == "assistant" for m in msgs) or sum(len(m["content"]) for m in msgs) > MAX_AGENT_TOKENS * 5:
                continue
            if any(B.REFUSAL.search(m["content"]) for m in msgs if m["role"] == "assistant"):
                continue
            if ntok(tok.apply_chat_template(msgs, tools=tools or None, tokenize=False)) > MAX_AGENT_TOKENS:
                continue
        except Exception:
            continue
        rows.append({"source": "harness_" + name, "messages": msgs, "tools": tools})
        kept += 1
    stats["harness_" + name] = (kept, seen)
random.Random(0).shuffle(rows)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps(stats), len(rows), "rows")
