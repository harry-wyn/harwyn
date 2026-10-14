"""Build a fake-quant checkpoint: an exact copy of SRC (same config, file layout and key names, so vLLM
serves it like the bf16 model) whose quantised Linears are replaced by the RCO database's dequantised
weights at the width ASSIGNMENT gives each tensor. Numerically identical to the packed checkpoint
(which dequantises to bf16 for the matmul anyway); doubles as the QAT reference.
usage: python fakequant_from_db.py SRC LAYER_DIR ASSIGNMENT_JSON OUT
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

src, layer_dir, assignment, out = map(Path, sys.argv[1:5])
a = json.load(open(assignment))["assignment"]   # "model.layers.N.x.y" -> bits
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.suffix != ".safetensors":
        shutil.copy(f, out / f.name)

done = 0
for shard in sorted(src.glob("*.safetensors")):
    t = load_file(shard)
    for k in t:
        name = k.replace("model.language_model.", "model.").removesuffix(".weight")
        if k.endswith(".weight") and name in a:
            w = torch.load(layer_dir / name / f"{a[name]}.pth", map_location="cpu")
            assert w.shape == t[k].shape, (k, w.shape, t[k].shape)
            t[k] = w.to(t[k].dtype).contiguous()
            done += 1
    save_file(t, out / shard.name, metadata={"format": "pt"})
assert done == len(a), f"replaced {done} of {len(a)} tensors"
print(f"replaced {done} tensors")
