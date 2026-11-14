"""Stress loop (run through tools/tp2.sh) for intermittent faults: --iters rounds of reset → prefill --ctx tokens in
--chunk rows (P002 two-lane mode for ≥ 2048-row chunks) → --decode plain steps, with the logits of every round compared
to the first round's (bitwise; a race that corrupts values without faulting shows up here). Progress goes to the log
file; the screen gets one line at the end (or the CUDA error of a fault). Patches via Q38_PATCHES as usual.
  Q38_PATCHES=-rdma_ar tools/tp2.sh stress python3 engine/tests/stress.py --iters 40"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
import qlog  # noqa: E402

import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--decode", type=int, default=8)
a = ap.parse_args()
OUT = "/work/results/tests/stress"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], 1, a.ctx + a.decode + 64, a.chunk)
toks = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))["long"]["prompt_ids"]
log = open(f"{OUT}/stress{a.rank}.log", "w")
ref, bad, t0 = None, 0, time.time()
for it in range(a.iters):
    eng.reset(0)
    p = 0
    while p < a.ctx:
        n = min(a.chunk, a.ctx - p)
        lg = eng.step([toks[(p + j) % len(toks)] for j in range(n)], list(range(p, p + n)), [0], [0, n], False)
        p += n
    out = [lg[-1].clone()]
    for k in range(a.decode):
        lg = eng.step([int(out[-1].argmax())], [a.ctx + k], [0], [0, 1])
        out.append(lg[0].clone())
    torch.cuda.synchronize()
    if ref is None:
        ref = out
    else:
        bad += any(not torch.equal(x, y) for x, y in zip(ref, out))
    print(f"iter {it + 1}/{a.iters} · {time.time() - t0:.0f} s · "
          f"logits {'equal' if not bad else f'differ, total {bad}'}", file=log, flush=True)
if a.rank == 0:
    (qlog.ok if not bad else qlog.warn)(
        "stress", f"{a.iters} runs · prefill {a.ctx // 1024}K (chunk {a.chunk}) + decode {a.decode} · "
                  f"logits {'bitwise equal every run' if not bad else f'{bad} runs differ'} · "
                  f"{time.time() - t0:.0f} s · patches {os.environ.get('Q38_PATCHES', 'default')}")
sys.stdout.flush()
os._exit(0)
