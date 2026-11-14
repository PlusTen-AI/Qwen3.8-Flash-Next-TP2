"""Speculative round profile (run through tools/tp2.sh): where the time of one MTP round goes, B = 1 and B = 4.

Each slot gets the same --ctx-token prompt (golden "long" prompt, cut) with the head run over it, then spec rounds with
the server's default sampling (T 1.0 · top_k 20 · top_p 0.95, a seed per slot). Untimed warm rounds first, then
--rounds rounds in profile mode. Screen (rank 0), per B:
  wall ms per round and phase (verify · accept · commit · head over kept rows · draft chain), tokens per round, tok/s
  GPU time shares by op group (CUDA events; target groups and the head's mtp.* groups)
Profile mode syncs at phase boundaries, so its tok/s is a little below the plain loop's (also printed).
  tools/tp2.sh sprof python3 engine/tests/spec_prof.py [--ctx 8192 --batch 1,4 --rounds 40 --depth 4 | 1,2,3]"""
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
ap.add_argument("--ctx", default="8192", help="context, or a comma list (one run per context)")
ap.add_argument("--batch", default="1,4")
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--prof-rounds", type=int, default=0, help="profile-mode rounds (0 = --rounds)")
ap.add_argument("--warm", type=int, default=6)
ap.add_argument("--depth", default="4", help="draft depth, or a comma list (one run per depth)")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=20)
ap.add_argument("--top-p", type=float, default=0.95)
a = ap.parse_args()
OUT = "/work/results/tests/sprof"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
gold = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))["long"]["prompt_ids"]
cs = [int(x) for x in a.ctx.split(",")]
bs = [int(x) for x in a.batch.split(",")]
ds = [int(x) for x in a.depth.split(",")]
per_seq = (a.warm + a.rounds) * 2 * (max(ds) + 1) + 64
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], max(bs), (max(cs) + per_seq + 127) // 64 * 64, a.chunk, mtp=True)
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)
sync = torch.cuda.synchronize


def start(slot, seed, depth, toks):
    """prompt + head over it, then spec_begin (first token, first drafts)"""
    L = len(toks)
    eng.reset(slot)
    for c0 in range(0, L, a.chunk):
        n = min(a.chunk, L - c0)
        eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [slot], [0, n], False)
        if c0 + n < L:
            eng.mtp_step(toks[c0 + 1:c0 + 1 + n], list(range(c0, c0 + n)), [slot], [0, n], list(range(n)), True, False)
    eng.spec_begin(slot, L, depth, a.temp, seed, 0, 0, toks[c0 + 1:L], a.top_k, a.top_p)


res = []
for ctx, B, depth in [(c, B, d) for c in cs for B in bs for d in ds]:
    toks = (gold * (ctx // len(gold) + 1))[:ctx]
    slots = list(range(B))
    for s in slots:
        start(s, 1000 + s, depth, toks)
    for _ in range(a.warm):
        eng.spec_round(slots)
    # plain loop (no profile syncs)
    sync()
    t0 = time.perf_counter()
    emitted, tr = 0, [t0]
    for _ in range(a.rounds):  # spec_round ends with a host sync (drafts back), so per-round wall times are real
        emitted += sum(len(r) for r in eng.spec_round(slots))
        tr.append(time.perf_counter())
    sync()
    dt = time.perf_counter() - t0
    rt = [1e3 * (y - x) for x, y in zip(tr, tr[1:])]
    w = min(30, len(rt))
    plain = {"tok_s": emitted / dt, "ms_round": 1e3 * dt / a.rounds, "tok_round": emitted / a.rounds / B,
             "ms_first": sum(rt[:w]) / w, "ms_last": sum(rt[-w:]) / w}
    # profile mode
    eng.set_prof(True)
    eng.prof_report(True), eng.spec_report(True)
    emitted = 0
    for _ in range(a.prof_rounds or a.rounds):
        emitted += sum(len(r) for r in eng.spec_round(slots))
    walls, gpu = eng.spec_report(True), eng.prof_report(True)
    eng.set_prof(False)
    res.append({"B": B, "depth": depth, "ctx": ctx, "plain": plain, "prof_tok_round": emitted / (a.prof_rounds or a.rounds) / B,
                "walls": walls, "gpu": gpu})
    say(qlog.ok, f"sprof B={B} k={depth}",
        f"context {ctx:,} · round {plain['ms_round']:.1f} ms (first 30 {plain['ms_first']:.1f} · "
        f"last 30 {plain['ms_last']:.1f}) · per sequence {plain['tok_round']:.2f} tokens/round · "
        f"total {plain['tok_s']:.1f} tok/s (T {a.temp} · top_k {a.top_k} · top_p {a.top_p} · depth {depth})")
    say(qlog.info, f"sprof B={B} k={depth} stages", walls)
    say(qlog.info, f"sprof B={B} k={depth} GPU", gpu)
if a.rank == 0:
    json.dump(res, open(f"{OUT}/sprof.json", "w"), indent=1)
sys.stdout.flush()
os._exit(0)
