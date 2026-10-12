# Check patches/rco-streamed-deltas.patch against plain autograd (double precision, CPU).
# usage: CUDA_VISIBLE_DEVICES="" python patches/test_rco_streamed_deltas.py <RCO checkout with the patch>
import sys, torch
sys.path.insert(0, sys.argv[1] + "/src")
from search.quant import WeightInterpolation
torch.manual_seed(0)
W = torch.randn(64, 48, dtype=torch.float64)
deltas = [torch.randn(64, 48, dtype=torch.float64) * 0.1 for _ in range(4)]  # n_widths - 1: the reference width has no delta
R = torch.randn(64, 48, dtype=torch.float64)
grads = []
for cpu in (False, True):
    alpha = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
    torch.manual_seed(1); a0 = torch.randn(1, 5, dtype=torch.float64)
    with torch.no_grad(): alpha.copy_(a0)
    m = WeightInterpolation(deltas, alpha, 0, cpu_deltas=cpu)
    loss = (m(W) * R).sum(); loss.backward()
    grads.append((loss.item(), alpha.grad.clone()))
print("loss diff", abs(grads[0][0] - grads[1][0]), "grad max diff", (grads[0][1] - grads[1][1]).abs().max().item())
ga, gb = grads[0][1], grads[1][1]
assert abs(grads[0][0] - grads[1][0]) < 1e-9 and ((ga - gb).abs().max() / ga.abs().max()) < 1e-5  # backward accumulates in fp32
print("PASS")
