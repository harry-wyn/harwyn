"""Build the stage-6 task pool: rl/tasks.jsonl.
  code    KodCode-Light-RL-10K problems NOT used in stage 5 (stage 5 took the first 2,400 of the same
          seed-0 shuffle), 13-gram decontaminated against HumanEval+.
  webdev  XiaomiMiMo/MiMo-V2.6-RL-oss `webdev` briefs.
usage: python tasks.py OUT.jsonl [N_CODE=1200] [N_WEB=800]
"""
import json, random, re, sys
from datasets import load_dataset

out = sys.argv[1]
n_code = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
n_web = int(sys.argv[3]) if len(sys.argv) > 3 else 800

def ngrams(t, n=13):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}

banned = set().union(*(ngrams(r["prompt"]) for r in load_dataset("evalplus/humanevalplus", split="test")))
rows = [r for r in load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
        if r["test_info"] and not (ngrams(r["question"]) & banned)]
random.Random(0).shuffle(rows)
code = [{"task_id": f"kc_{r['question_id']}", "kind": "code", "question": r["question"],
         "decl": r["test_info"][0]["function_declaration"], "test": r["test"]} for r in rows[2400:2400 + n_code]]
web = []
for r in load_dataset("XiaomiMiMo/MiMo-V2.6-RL-oss", "webdev", split="train"):
    ij = json.loads(r["extra_info"]["instance_json"])
    web.append({"task_id": f"web_{ij['instance_id']}", "kind": "webdev", "question": ij["problem_statement"]})
random.Random(1).shuffle(web)
with open(out, "w") as f:
    for t in code + web[:n_web]:
        f.write(json.dumps(t) + "\n")
print(len(code), "code,", min(n_web, len(web)), "webdev")
