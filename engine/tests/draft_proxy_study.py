"""Draft lm_head two-stage study (one GPU, this rank's vocab half): can a cheap proxy of lm_head pick the candidates the
draft needs? The draft samples from q̃ = the head's logits truncated to top_k 20 / top_p 0.95, so it only needs the top of
the logits; a proxy (rank-r factorisation W ≈ (U·S)·Vᵀ from a randomized SVD) scores all rows cheaply, the top-C rows by
proxy score are rescored exactly, and q̃ is built from those. Draft-side only: the accept test uses the q̃ actually drawn
from, so the output distribution does not change — only acceptance can.
Inputs: lm_head inputs of the golden dump (language_model.logits_processor#0.in1: target hidden rows after the final
norm — the MTP head's rows go through the same lm_head). Per (r, C): p̃-mass recall = Σ over the exact q̃'s kept tokens
of q̃(v)·[v among the C candidates], and exact-top-1 recall; plus the proxy's bytes per call relative to NVFP4 lm_head.
  python3 engine/tests/draft_proxy_study.py --rank 0 --dump /work/results/raw/dump/20261111-1632/rank0"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools")]
import qlog  # noqa: E402

import ext  # noqa: E402
from golden import Golden  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--dump", required=True)
ap.add_argument("--model", default="/models/Qwen3.8-Flash-Next-FP4-Local")
ap.add_argument("--table", default="/work/results/tensor_table.tsv")
ap.add_argument("--ranks", default="128,256,384,512")
ap.add_argument("--cands", default="64,128,256,512,1024")
a = ap.parse_args()
w = ext.load("loader", ["src/core/weights.cc"])
w.open(a.model, a.table, a.rank)
b, shape, dt, _ = w.slice("lm_head.weight")
N, K = shape
W = b.view(torch.bfloat16).view(N, K).cuda()
g = Golden(a.dump, os.path.join(os.path.dirname(a.dump), "tokens.json"))
xs = []
for s in g.steps:
    try:
        xs.append(g.get(s["step"], "language_model.logits_processor#0.in1").reshape(-1, K))
    except KeyError:
        pass
X = torch.cat(xs).cuda().bfloat16()
z = (X.float() @ W.float().t())  # exact logits [rows, N] for this half
T, TOPK, TOPP = 1.0, 20, 0.95


def trunc(lg):
    v, i = torch.topk(lg / T, TOPK, dim=-1)
    p = torch.softmax(v.double(), -1)
    keep = (torch.cumsum(p, -1) - p) < TOPP
    p = torch.where(keep, p, torch.zeros_like(p))
    return i, p / p.sum(-1, keepdim=True)


ti, tp = trunc(z)
say = qlog.ok
say("proxy", f"lm_head half {N}×{K} · input {X.shape[0]} rows (golden)")
Wf = W.float()
for r in [int(x) for x in a.ranks.split(",")]:
    U, S, V = torch.svd_lowrank(Wf, q=r, niter=4)  # W ≈ U diag(S) Vᵀ
    A = (U * S).bfloat16()  # [N, r]
    pz = (X.float() @ V) @ A.float().t()
    parts = []
    for C in [int(x) for x in a.cands.split(",")]:
        cand = torch.topk(pz, C, dim=-1).indices
        hit = (ti.unsqueeze(-1) == cand.unsqueeze(1)).any(-1)  # [rows, 20]
        mass = (tp * hit.double()).sum(-1)
        top1 = hit[:, 0].float().mean().item()
        parts.append(f"C{C} mass {100 * mass.mean().item():.2f}% (min {100 * mass.min().item():.1f}%) "
                     f"top1 {100 * top1:.1f}%")
    rel = (N * r * 1 + 0) / (N * K * 0.5 + N * K / 16)  # fp8 proxy bytes vs NVFP4 lm_head (values + scales)
    say(f"proxy r={r}", f"proxy bytes vs NVFP4 {100 * rel:.0f}% · " + " · ".join(parts))
sys.stdout.flush()
os._exit(0)
