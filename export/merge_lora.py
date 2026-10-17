"""Merge a train_qat.py LoRA into h0 and fake-quantise each merged weight to its width, exactly as training and
the evals saw it (W + 2*B@A, symmetric, group 128). Streams one tensor at a time and writes <= 1 GB shards, so it
runs on a laptop with a few GB of free RAM.
usage: python merge_lora.py H0_DIR LORA.pt ASSIGNMENT.json OUT_DIR
"""
import json, os, shutil, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

h0, lora_path, assignment, out = sys.argv[1:5]
GROUP, SCALE, SHARD = 128, 2.0, 1 << 30   # alpha = 2r as in train_qat.py


def fq(w, bits):
    o, i = w.shape
    g = w.float().reshape(o, i // GROUP, GROUP)
    qmax = 2 ** (bits - 1) - 1
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return ((g / s).round().clamp(-qmax - 1, qmax) * s).reshape(o, i)


lora = torch.load(lora_path, map_location="cpu")
bits = json.load(open(assignment))["assignment"]
os.makedirs(out, exist_ok=True)
for f in os.listdir(h0):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
        shutil.copy(os.path.join(h0, f), out)

src = json.load(open(os.path.join(h0, "model.safetensors.index.json")))["weight_map"]
weight_map, buf, size, n_shard, merged = {}, {}, 0, 0, 0


def flush():
    global buf, size, n_shard
    if buf:
        name = f"model-{n_shard:05d}.safetensors"
        save_file(buf, os.path.join(out, name), metadata={"format": "pt"})
        for k in buf:
            weight_map[k] = name
        n_shard += 1; buf, size = {}, 0


for shard in sorted(set(src.values())):
    with safe_open(os.path.join(h0, shard), "pt") as fh:
        for k in fh.keys():
            t = fh.get_tensor(k)
            name = k.replace("model.language_model.", "model.").removesuffix(".weight")
            if k.endswith(".weight") and name in bits:
                w = t.float() + SCALE * (lora[name + ".B"] @ lora[name + ".A"])
                t = fq(w, bits[name]).to(t.dtype)
                merged += 1
            buf[k] = t.contiguous(); size += t.numel() * t.element_size()
            if size >= SHARD:
                flush()
flush()
# shard names were provisional; rename to the -of-N convention
total = n_shard
for i in range(total):
    os.rename(os.path.join(out, f"model-{i:05d}.safetensors"), os.path.join(out, f"model-{i + 1:05d}-of-{total:05d}.safetensors"))
weight_map = {k: f"model-{int(v[6:11]) + 1:05d}-of-{total:05d}.safetensors" for k, v in weight_map.items()}
json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
assert merged == len(bits), f"merged {merged} of {len(bits)}"
print(f"merged {merged} tensors into {total} shards")
