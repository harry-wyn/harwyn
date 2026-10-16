"""GRPO advantages: reward minus its group's mean, divided by the group's std (groups with no reward
spread carry no signal and are dropped). Writes the trainer's input rows with an `advantage` field.
usage: python advantages.py ROLLOUTS.jsonl TRAIN.jsonl
"""
import json, statistics, sys, collections
g = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l)
    if "error" not in r and r.get("reward") is not None:
        g[(r["task_id"], r["group"])].append(r)
kept = 0
with open(sys.argv[2], "w") as f:
    for rs in g.values():
        rw = [r["reward"] for r in rs]
        if len(rs) < 2 or max(rw) - min(rw) < 1e-6:
            continue
        mu, sd = statistics.mean(rw), statistics.pstdev(rw)
        for r in rs:
            f.write(json.dumps({"source": r["kind"], "messages": r["messages"], "tools": r["tools"],
                                "advantage": (r["reward"] - mu) / (sd + 1e-6)}) + "\n"); kept += 1
print(f"groups={len(g)} informative={sum(1 for rs in g.values() if len(rs)>1 and max(r['reward'] for r in rs)-min(r['reward'] for r in rs)>1e-6)} rows={kept}")
