"""Engine speed (run through tools/tp2.sh): prefill tok/s and decode ms/step (B sequences × 1 token) at a few small
contexts, plus a scaling check — costs that grow with context (QSA indexer ∝ L, prefill indexer ∝ L², KV) must stay
near-linear: decode step +10 % or prefill tok/s −15 % from the smallest to the largest context → warning.
Default contexts 2K, 8K, 32K (128K only at milestones). Screen: one line per case (rank 0); JSON → results/tests/speed."""
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
ap.add_argument("--ctx", default="2048,8192,32768")
ap.add_argument("--batch", default="1,4")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--chunk", type=int, default=64, help="prefill rows per step")
ap.add_argument("--prof", action="store_true", help="per-group time shares (CUDA events) for prefill and decode")
ap.add_argument("--warm", type=int, default=2048, help="tokens of an untimed fill first (first-run costs; 0 = none)")
a = ap.parse_args()
OUT = "/work/results/tests/speed"
os.makedirs(OUT, exist_ok=True)
srcs = ext.engine_sources()
eng = ext.load("engine", srcs, log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
ctxs, bs = [int(x) for x in a.ctx.split(",")], [int(x) for x in a.batch.split(",")]
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], max(bs), (max(ctxs) + a.steps + 127) // 64 * 64, a.chunk)
toks = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))["long"]["prompt_ids"]
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)


def fill(slot, L):
    eng.reset(slot)
    p = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while p < L:
        n = min(a.chunk, L - p)
        eng.step([toks[(p + j) % len(toks)] for j in range(n)], list(range(p, p + n)), [slot], [0, n], False)
        p += n
    torch.cuda.synchronize()
    return L / (time.perf_counter() - t0)


if a.warm:  # first-run costs (lazy kernel loading, first touches) out of the timed fills
    fill(0, min(a.warm, max(ctxs)))
res = []
for L in ctxs:
    pf = None
    for B in bs:
        for s in range(B):  # every case starts from freshly filled slots (decode advances them)
            if pf is None and a.prof:
                eng.set_prof(True)
            r = fill(s, L)
            if pf is None:
                pf = r
                say(qlog.ok, "speed prefill", f"context {L:,} · {a.chunk} rows per step · {pf:,.0f} tok/s")
                if a.prof:
                    say(qlog.info, "speed prefill split", eng.prof_report(True))
                    eng.set_prof(False)
        nxt = [int(toks[L % len(toks)])] * B
        ts = []
        eng.set_prof(a.prof)
        for k in range(a.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lg = eng.step(nxt, [L + k] * B, list(range(B)), list(range(B + 1)))
            ts.append(time.perf_counter() - t0)
            nxt = lg.argmax(-1).tolist()
        ts = sorted(ts[5:])
        ms = ts[len(ts) // 2] * 1e3
        res.append({"ctx": L, "B": B, "prefill_tok_s": pf, "ms_med": ms, "ms_best": ts[0] * 1e3, "tok_s": B / (ms / 1e3)})
        say(qlog.ok, "speed decode", f"context {L:,} · B={B} · step {ms:.1f} ms → {B / (ms / 1e3):.1f} tok/s (no MTP)")
        if a.prof:
            say(qlog.info, "speed decode split", eng.prof_report(True))
            eng.set_prof(False)

if a.rank == 0 and len(ctxs) > 1:
    lo, hi = min(ctxs), max(ctxs)
    for B in bs:
        r0 = next(r for r in res if r["ctx"] == lo and r["B"] == B)
        r1 = next(r for r in res if r["ctx"] == hi and r["B"] == B)
        g = r1["ms_med"] / r0["ms_med"] - 1
        (qlog.warn if g > 0.10 else qlog.ok)("speed length",
                                             f"B={B} decode step {lo // 1024}K→{hi // 1024}K {g * 100:+.1f}%")
    p0 = next(r for r in res if r["ctx"] == lo)["prefill_tok_s"]
    p1 = next(r for r in res if r["ctx"] == hi)["prefill_tok_s"]
    d = p1 / p0 - 1
    (qlog.warn if d < -0.15 else qlog.ok)("speed length", f"prefill tok/s {lo // 1024}K→{hi // 1024}K {d * 100:+.1f}%")
    json.dump(res, open(f"{OUT}/speed.json", "w"), indent=1)
sys.stdout.flush()
os._exit(0)
