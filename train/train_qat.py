"""Stage 4: SFT with QAT everywhere.
- Weights: LoRA on every quantised Linear; each forward fake-quantises the *merged* W + BA to that tensor's
  width from ASSIGNMENT (symmetric, group 128, straight-through estimator), so training sees exactly the
  weights that ship.
- KV cache: in full-attention layers, K (after RoPE) and V are fake-quantised the way vLLM's
  int4_per_token_head stores them: Hadamard rotation, then asymmetric int4 with one scale/zero-point per
  (token, head), then rotated back (STE). Rotation signs are a fixed random choice, not vLLM's exact ones.
- Loss on assistant tokens only (including tool calls).
usage: python train_qat.py MODEL_DIR DATA.jsonl ASSIGNMENT.json OUT_DIR [--max-steps N] [--max-len L]
Resumes from OUT_DIR/lora.pt. Writes OUT_DIR/lora.pt every SAVE_EVERY optimiser steps.
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as qmod

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("data"); p.add_argument("assignment"); p.add_argument("out")
p.add_argument("--max-steps", type=int, default=0); p.add_argument("--max-len", type=int, default=8192)
p.add_argument("--rank", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--accum", type=int, default=16); p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--no-kv-qat", action="store_true")
p.add_argument("--init", help="start from a LoRA state dict (e.g. the stage-4 final lora_step212.pt)")
p.add_argument("--export-only", action="store_true", help="with --init: skip training, just write merged_fq.pt")
args = p.parse_args()
SAVE_EVERY, GROUP = 50, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"


def ste(x, xq):
    return x + (xq - x).detach()


def fq_weight(w, bits):
    """Symmetric group-wise fake quantisation along the input dim (group 128)."""
    o, i = w.shape
    g = w.float().reshape(o, i // GROUP, GROUP)
    qmax = 2 ** (bits - 1) - 1
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = (g / s).round().clamp(-qmax - 1, qmax) * s
    return ste(w, q.reshape(o, i).to(w.dtype))


class QATLoRA(nn.Module):
    def __init__(self, base, bits, r):
        super().__init__()
        self.base, self.bits, self.scale = base, bits, 2.0  # alpha = 2r
        base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device, dtype=torch.float32) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=torch.float32))

    def merged(self):
        return self.base.weight + (self.scale * (self.B @ self.A)).to(self.base.weight.dtype)

    def forward(self, x):
        return F.linear(x, fq_weight(self.merged(), self.bits), self.base.bias)


# --- KV-cache QAT (vLLM int4_per_token_head style) ---
_H = {}


def hadamard(d, device):
    if (d, device) not in _H:
        h = torch.ones(1, 1)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        signs = (torch.randint(0, 2, (d,), generator=torch.Generator().manual_seed(0x9E3779B9)) * 2 - 1).float()
        _H[(d, device)] = ((h * signs) / math.sqrt(d)).to(device)   # orthogonal: (H D) / sqrt(d)
    return _H[(d, device)]


def fq_kv(x):
    """x [..., head_dim] -> rotate, asym int4 per vector, rotate back (STE)."""
    H = hadamard(x.shape[-1], x.device)
    y = x.float() @ H
    mn, mx = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    s = ((mx - mn) / 15).clamp_min(1e-6)
    zp = (-mn / s).round().clamp(0, 15)
    yq = ((y / s).round() + zp).clamp(0, 15)
    deq = ((yq - zp) * s) @ H.T
    return ste(x, deq.to(x.dtype))


if not args.no_kv_qat:
    _orig_rope = qmod.apply_rotary_pos_emb

    def rope_kv_qat(q, k, *a, **kw):
        q, k = _orig_rope(q, k, *a, **kw)
        return q, fq_kv(k)
    qmod.apply_rotary_pos_emb = rope_kv_qat

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
model.config.use_cache = False
for prm in model.parameters():
    prm.requires_grad_(False)
assign = json.load(open(args.assignment))["assignment"]
for name, bits in assign.items():
    parent, child = name.rsplit(".", 1)
    pm = model.get_submodule(parent)
    setattr(pm, child, QATLoRA(getattr(pm, child), bits, args.rank))
if not args.no_kv_qat:  # V: fake-quantise v_proj outputs per head in full-attention layers
    for layer in model.model.layers:
        if getattr(layer, "layer_type", "") == "full_attention":
            hd = layer.self_attn.head_dim
            layer.self_attn.v_proj.register_forward_hook(
                lambda m, i, o, hd=hd: fq_kv(o.view(*o.shape[:-1], -1, hd)).view(o.shape))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
params = [p for p in model.parameters() if p.requires_grad]
print(f"trainable {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)

# --- data: full chat-template render, labels = tokens inside assistant blocks ---
IM_START, IM_END = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
ASSIST = tok("assistant", add_special_tokens=False).input_ids


def encode(ex):
    text = tok.apply_chat_template(ex["messages"], tools=ex.get("tools") or None, tokenize=False)
    ids = tok(text, add_special_tokens=False).input_ids[: args.max_len]
    labels, inside, i = [-100] * len(ids), False, 0
    while i < len(ids):
        if ids[i] == IM_START and ids[i + 1:i + 1 + len(ASSIST)] == ASSIST:
            i += 1 + len(ASSIST); inside = True; continue
        if inside:
            labels[i] = ids[i]
            if ids[i] == IM_END:
                inside = False
        i += 1
    return ids, labels


rows = [json.loads(l) for l in open(args.data)] if not args.export_only else [None]
random.Random(0).shuffle(rows)
total_micro = int(len(rows) * args.epochs)
total_steps = args.max_steps or max(1, total_micro // args.accum)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
os.makedirs(args.out, exist_ok=True)
ck = os.path.join(args.out, "lora.pt")
step = 0
lora_named = {n: p for n, p in model.named_parameters() if p.requires_grad}
if args.init and not os.path.exists(ck):
    init = torch.load(args.init, map_location=dev)
    for n, p in lora_named.items():
        p.data.copy_(init[n])
    print("initialised from", args.init, flush=True)
if args.export_only:
    total_steps = step
if os.path.exists(ck):
    st = torch.load(ck, map_location=dev)
    for n, p in lora_named.items():
        p.data.copy_(st["lora"][n])
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
    print("resumed at step", step, flush=True)

def _ce_sum(h, y):
    return F.cross_entropy(model.lm_head(h).float(), y, ignore_index=-100, reduction="sum")


def chunked_loss(h, y, chunk=2048):
    """Mean next-token CE without materialising [len, 248k] logits (16k tokens would be ~16 GB in fp32)."""
    n = (y != -100).sum().clamp_min(1)
    return sum(torch.utils.checkpoint.checkpoint(_ce_sum, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
               for i in range(0, len(y), chunk)) / n


model.train()
t0, toks, micro = time.time(), 0, step * args.accum
while step < total_steps:
    loss_acc = 0.0
    for _ in range(args.accum):
        ids, labels = encode(rows[micro % len(rows)]); micro += 1
        if all(l == -100 for l in labels):
            continue
        x = torch.tensor([ids], device=dev); y = torch.tensor(labels[1:] + [-100], device=dev)
        loss = chunked_loss(model.model(input_ids=x).last_hidden_state[0], y) / args.accum
        loss.backward(); loss_acc += loss.item(); toks += len(ids)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
    el = time.time() - t0
    print(f"step {step}/{total_steps} loss {loss_acc:.4f} lr {sched.get_last_lr()[0]:.2e} tok/s {toks/el:.0f} elapsed {el/60:.1f}m", flush=True)
    if step % SAVE_EVERY == 0 or step == total_steps:
        torch.save({"lora": {n: p.detach() for n, p in lora_named.items()}, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

# --- export: merged weights fake-quantised to their grid, as {module name: tensor} ---
with torch.no_grad():
    merged = {n: fq_weight(m.merged(), m.bits).cpu() for n, m in model.named_modules() if isinstance(m, QATLoRA)}
torch.save(merged, os.path.join(args.out, "merged_fq.pt"))
print("exported", len(merged), "merged fake-quant tensors", flush=True)
