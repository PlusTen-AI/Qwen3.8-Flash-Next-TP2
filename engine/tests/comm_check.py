"""TP2 communicator check (run through tools/tp2.sh): correctness of a bf16 all-reduce and small-message latency
(decode all-reduce = [M, 2560] bf16), eager vs inside a CUDA graph. Writes /work/results/comm.json (rank 0)."""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
import qlog  # noqa: E402

import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
a = ap.parse_args()
os.makedirs("/work/results/tests/comm", exist_ok=True)
c = ext.load("tp2comm", ["src/core/comm.cc"], log=f"/work/results/tests/comm/build{a.rank}.log", ldflags=["-lnccl"])
c.init(a.rank, os.environ["Q38_HEAD_IP"], 29611)

x = torch.full((5120,), float(a.rank + 1), dtype=torch.bfloat16, device="cuda")
c.all_reduce(x)
good = bool((x == 3).all())
(qlog.ok if good else qlog.fail)("comm sum", f"rank{a.rank} · 1+2 = {x[0].item():g}")

res = {}
for M in (1, 4, 16, 64, 2048, 8192):
    it = 50 if M <= 64 else 5
    eb, em = c.time_ar(M * 2560, it, False)
    gb, gm = c.time_ar(M * 2560, it, True)
    res[M] = {"eager_best": eb, "eager_med": em, "graph_best": gb, "graph_med": gm}
    if a.rank == 0:
        bw = f" · {M * 5120 / eb / 1e3:.1f} GB/s" if M > 64 else ""
        qlog.ok("comm AR", f"[{M},2560] bf16 · eager {eb:.1f} µs (median {em:.1f}) · "
                           f"graph {gb:.1f} µs (median {gm:.1f}){bw}")
if a.rank == 0:
    json.dump(res, open("/work/results/comm.json", "w"), indent=1)
sys.stdout.flush()
os._exit(0)  # skip NCCL teardown at interpreter exit (it can block once the peer is gone)
