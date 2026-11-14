"""Patch check (run through tools/tp2.sh): the same prompt with patches off and on, alternately (off, on, off, on … --reps
rounds, so first-run costs such as lazy kernel loading do not fall on one side) — logits of the last prompt row and of
--decode greedy steps must be bitwise equal across every run; prefill tok/s (best of the rounds) and decode ms/step
(median of the steps, best of the rounds) for both. One line per check (rank 0).
  tools/tp2.sh pcheck python3 engine/tests/patch_check.py --patches ar_overlap --chunk-patch 16384
  Q38_RDMA_INIT=1 tools/tp2.sh pcheck python3 engine/tests/patch_check.py --patches rdma_ar   (P001 needs its transport)"""
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
ap.add_argument("--patches", required=True, help="comma list of patch names to compare (off vs on)")
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--chunk-base", type=int, default=8192)
ap.add_argument("--chunk-patch", type=int, default=8192)
ap.add_argument("--decode", type=int, default=16)
ap.add_argument("--reps", type=int, default=2)
a = ap.parse_args()
OUT = "/work/results/tests/patch"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], 1, a.ctx + a.decode + 64, max(a.chunk_base, a.chunk_patch))
toks = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))["long"]["prompt_ids"]
names = a.patches.split(",")


def run(on, chunk):
    for n in names:
        eng.patch(n, on)
    eng.reset(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    p = 0
    while p < a.ctx:
        n = min(chunk, a.ctx - p)
        lg = eng.step([toks[(p + j) % len(toks)] for j in range(n)], list(range(p, p + n)), [0], [0, n], False)
        p += n
    torch.cuda.synchronize()
    tps = a.ctx / (time.perf_counter() - t0)
    out, ts = [lg[-1].clone()], []
    nxt = int(out[0].argmax())
    for k in range(a.decode):
        t0 = time.perf_counter()
        lg = eng.step([nxt], [a.ctx + k], [0], [0, 1])
        nxt = int(lg[0].argmax())  # waits for the step
        ts.append(time.perf_counter() - t0)
        out.append(lg[0].clone())
    ts = sorted(ts[2:]) or [0.0]
    return out, tps, ts[len(ts) // 2] * 1e3


runs = {False: [], True: []}
for _ in range(a.reps):
    for on in (False, True):
        runs[on].append(run(on, a.chunk_patch if on else a.chunk_base))
ref = runs[False][0][0]
nd = sum(int((x != y).sum()) for r in runs[False][1:] + runs[True] for x, y in zip(ref, r[0]))
tb, tp = max(r[1] for r in runs[False]), max(r[1] for r in runs[True])
db, dp = min(r[2] for r in runs[False]), min(r[2] for r in runs[True])
if a.rank == 0:
    (qlog.ok if nd == 0 else qlog.warn)(
        f"patch {a.patches}", f"logits {'bitwise equal' if nd == 0 else f'{nd} values differ'} "
                              f"({1 + a.decode} steps × {2 * a.reps} runs) · "
                              f"prefill {a.ctx // 1024}K {tb:,.0f} → {tp:,.0f} tok/s ({(tp / tb - 1) * 100:+.1f}%) · "
                              f"decode {db:.1f} → {dp:.1f} ms/step ({(dp / db - 1) * 100:+.1f}%) · "
                              f"chunk {a.chunk_base}→{a.chunk_patch}")
sys.stdout.flush()
os._exit(0)
