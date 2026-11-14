"""Next-turn profile (run through tools/tp2.sh): the server's path for a follow-up request whose prompt shares the slot's
snapshot — restore, the new rows minus the last as one prefill chunk (target step + MTP head over it), (snapshot,
skipped here), the last row, spec_begin (first token + drafts). As bench_lite's "next turn": --base tokens already in the slot (snapshot
at base), --new tokens added. Each case is timed --reps times after one untimed run (restore makes them identical).
Cases: P002 ar_overlap on / off; with --prof one more run per case for the op-group shares of the big step.
Screen (rank 0): one line per case — ms total and per part.
  tools/tp2.sh turn python3 engine/tests/turn_prof.py [--base 129843 --new 6695 --prof]"""
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
ap.add_argument("--base", type=int, default=129843)
ap.add_argument("--new", type=int, default=6695)
ap.add_argument("--chunk", type=int, default=16384)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--prof", action="store_true")
ap.add_argument("--p002", default="on,off", help="cases: on, off (off fails for chunks > 8192 rows)")
a = ap.parse_args()
OUT = "/work/results/tests/turn"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
gold = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))["long"]["prompt_ids"]
L = a.base + a.new
toks = (gold * (L // len(gold) + 1))[:L]
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], 1, (L + 64 + 127) // 64 * 64, a.chunk, mtp=True)
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)
sync = torch.cuda.synchronize


def rows(c0, c1):
    """prefill rows c0 … c1−1 in chunks (target + head), as the scheduler's _prefill_chunk"""
    for s in range(c0, c1, a.chunk):
        e = min(s + a.chunk, c1)
        n = e - s
        eng.step(toks[s:e], list(range(s, e)), [0], [0, n], False)
        eng.mtp_step(toks[s + 1:e + 1], list(range(s, e)), [0], [0, n], list(range(n)), True, False)


eng.reset(0)
sync()
t0 = time.perf_counter()
rows(0, a.base)
eng.snapshot(0)
sync()
say(qlog.ok, "turn fill", f"{a.base:,} tokens · {time.perf_counter() - t0:.1f} s · snapshot {a.base:,}")


def turn(prof=False):
    """restore → new rows (but the last) → snapshot → last row → spec_begin; ms per part"""
    ms = {}
    assert eng.restore(0) == a.base
    sync()
    t = time.perf_counter()

    def lap(k):
        nonlocal t
        sync()
        now = time.perf_counter()
        ms[k] = ms.get(k, 0) + 1e3 * (now - t)
        t = now
    n = a.new - 1
    if prof:
        eng.set_prof(True)
        eng.prof_report(True)
    eng.step(toks[a.base:L - 1], list(range(a.base, L - 1)), [0], [0, n], False)
    lap("step")
    if prof:
        grp = eng.prof_report(True)
    eng.mtp_step(toks[a.base + 1:L], list(range(a.base, L - 1)), [0], [0, n], list(range(n)), True, False)
    lap("head")
    if prof:
        eng.set_prof(False)
    # (the server snapshots here — a ~60 MB device copy; skipped so every rep restores the same base snapshot)
    eng.step([toks[L - 1]], [L - 1], [0], [0, 1], False)
    lap("last")
    eng.spec_begin(0, L, a.depth, 1.0, 1, 0, 0, [], 20, 0.95)
    lap("begin")
    ms["total"] = sum(ms.values())
    return ms, (grp if prof else None)


for name, p2 in [(n, v) for n, v in (("P002 on", True), ("P002 off", False)) if ("on" if v else "off") in a.p002]:
    eng.patch("ar_overlap", p2)
    turn()  # untimed
    best = None
    for _ in range(a.reps):
        ms, _ = turn()
        best = ms if best is None else {k: min(best[k], ms[k]) for k in ms}
    say(qlog.ok, f"turn {name}", f"context {a.base:,} + {a.new:,} · total {best['total']:.0f} ms = "
                                f"main step {best['step']:.0f} · head {best['head']:.0f} · "
                                f"last row {best['last']:.0f} · spec_begin {best['begin']:.0f} (best of {a.reps})")
    if a.prof:
        _, grp = turn(True)
        say(qlog.info, f"turn {name} main step", grp)
eng.patch("ar_overlap", True)
sys.stdout.flush()
os._exit(0)
