"""Kernel timeline of speculative rounds (run through tools/tp2.sh): every kernel's GPU start/end from CUPTI activity
records (engine/tests/bind/ktrace.cpp — no event marks between launches, so PDL overlap stays as in production).
Same setup as spec_prof.py (golden "long" prompt cut to --ctx, server sampling, a seed per slot, --depth), warm rounds,
then --rounds traced rounds per B. Screen (rank 0), per B: round wall ms; per kernel family the busy time per round
(Σ durations), the idle time in front of it (GPU idle between the end of everything before and its start), and calls;
then the HC module's kernels (k_combine, down, reduce, up) — duration per call and the gap before each.
Details per kernel name → /work/results/tests/strace/strace{B}.txt; raw records per rank → strace{B}_r{rank}.json (on
each rank's node).
  tools/tp2.sh strace python3 engine/tests/step_trace.py [--ctx 8192 --batch 1,4 --rounds 8 --depth 4]"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
import qlog  # noqa: E402

import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--batch", default="1,4")
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--warm", type=int, default=8)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--prompt", default="golden", help="golden: the golden long prompt repeated (all slots alike); bench: the "
                "bench-like document question (model README + chat template, thinking on), a different nonce per slot")
a = ap.parse_args()
OUT = "/work/results/tests/strace"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
kt = ext.load("ktrace", [], log=f"{OUT}/build{a.rank}.log", ldflags=["-lcupti"])
gold = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))["long"]["prompt_ids"]
toks = (gold * (a.ctx // len(gold) + 1))[:a.ctx]
if a.prompt == "bench":  # as server/bench_lite.py: nonce + document cut to ≈ ctx tokens + one question, thinking on
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "server"))
    from chat import Chat  # noqa: E402
    MD = "/models/Qwen3.8-Flash-Next-FP4-Local"
    chat = Chat(MD)
    doc = open(f"{MD}/README.md").read() + open(f"{MD}/chat_template.jinja").read()
    q = "\n\nBased on the document above, explain in detail how its parts fit together and what a reader should check first."

    def bench_toks(nonce):
        n = int(a.ctx * 4.4)  # ≈ 4.4 characters per token on this document (bench_lite)
        while True:
            body = (doc * (n // len(doc) + 1))[:n]
            t = chat.encode(chat.render([{"role": "user", "content": f"[{nonce}]\n" + body + q}]))
            if len(t) <= a.ctx:
                return t
            n -= 6 * (len(t) - a.ctx)
bs = [int(x) for x in a.batch.split(",")]
per_seq = (a.warm + a.rounds) * 2 * (a.depth + 1) + 64
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], max(bs), (a.ctx + per_seq + 127) // 64 * 64, a.chunk, mtp=True)
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)

FAMILY = [  # (substring of the demangled name, family) — first match wins
    ("k_combine", "hc.K1 combine"), ("p12::k_down", "hc.K2 down"), ("k_down<", "hc.K2 down"), ("k_down_reduce", "hc.K2b reduce"),
    ("p12::k_reduce", "hc.K2b reduce"), ("p12::k_up", "hc.K3 up"), ("k_up<", "hc.K3 up"), ("k_expand", "hc.expand"),
    ("lin_kernel", "linear"), ("k_route", "moe.route"), ("k_experts", "moe.experts"),
    ("k_ll", "AR.ll"), ("k_fused", "AR.fused"), ("rdma", "AR.other"), ("nccl", "AR.nccl"),
    ("gdn", "gdn"), ("qsa", "qsa"), ("ple", "ple"), ("embed", "embed"), ("accept", "sample.accept"),
    ("sample", "sample"), ("mtp", "mtp"), ("Memcpy", "copy"), ("memcpy", "copy"), ("Memset", "copy"),
]


def demangle(names):
    try:
        out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout.split("\n")
        return dict(zip(names, out))
    except OSError:
        return {n: n for n in names}


def family(dn):
    for sub, f in FAMILY:
        if sub in dn:
            return f
    return "other:" + dn.split("(")[0][-40:]


def start(slot, seed):
    toks = bench_toks(f"{slot:04d}{seed:08x}") if a.prompt == "bench" else globals()["toks"]
    L = len(toks)
    eng.reset(slot)
    for c0 in range(0, L, a.chunk):
        n = min(a.chunk, L - c0)
        eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [slot], [0, n], False)
        if c0 + n < L:
            eng.mtp_step(toks[c0 + 1:c0 + 1 + n], list(range(c0, c0 + n)), [slot], [0, n], list(range(n)), True, False)
    eng.spec_begin(slot, L, a.depth, 1.0, seed, 0, 0, toks[c0 + 1:L], 20, 0.95)


for B in bs:
    slots = list(range(B))
    for s in slots:
        start(s, 1000 + s)
    for _ in range(a.warm):
        eng.spec_round(slots)
    torch.cuda.synchronize()
    kt.start()
    t0 = time.perf_counter()
    for _ in range(a.rounds):
        eng.spec_round(slots)
    torch.cuda.synchronize()
    wall = 1e3 * (time.perf_counter() - t0) / a.rounds
    recs = sorted(kt.stop(), key=lambda r: r[1])
    dm = demangle(sorted({r[0] for r in recs}))
    # both ranks keep their raw records (all-reduce waits: compare the two ranks' durations per call)
    json.dump([(dm[r[0]][:200], r[1], r[2], r[3]) for r in recs], open(f"{OUT}/strace{B}_r{a.rank}.json", "w"))
    if a.rank != 0:
        continue
    busy, idle, calls = defaultdict(float), defaultdict(float), defaultdict(int)
    per_name = defaultdict(lambda: [0, 0.0, 0.0])  # calls, busy, idle before
    hc_seq = []  # (family, dur, gap) of HC kernels in order
    last_end = recs[0][1] if recs else 0
    for name, s0, e0, st, gx, gy, gz in recs:
        f = family(dm[name])
        d = (e0 - s0) / 1e3
        gap = max(0.0, (s0 - last_end) / 1e3)
        busy[f] += d
        idle[f] += gap
        calls[f] += 1
        pn = per_name[dm[name][:110]]
        pn[0] += 1
        pn[1] += d
        pn[2] += gap
        if f.startswith("hc.K"):
            hc_seq.append((f, d, (s0 - last_end) / 1e3))
        last_end = max(last_end, e0)
    span = (recs[-1][2] - recs[0][1]) / 1e6 / a.rounds if recs else 0
    R = a.rounds
    tot_b, tot_i = sum(busy.values()) / R / 1e3, sum(idle.values()) / R / 1e3
    say(qlog.ok, f"strace B={B}", f"round wall {wall:.2f} ms · GPU span {span:.2f} ms · "
                                 f"kernel busy total {tot_b:.2f} ms · idle between kernels {tot_i:.2f} ms "
                                 f"({len(recs) // R} kernels/round)")
    fam = sorted(busy, key=lambda f: -busy[f])
    say(qlog.info, f"strace B={B} kinds", " · ".join(
        f"{f} {busy[f] / R / 1e3:.2f}+{idle[f] / R / 1e3:.2f} idle ({calls[f] // R})" for f in fam))
    hk = defaultdict(lambda: [0, 0.0, 0.0])
    for f, d, g in hc_seq:
        hk[f][0] += 1
        hk[f][1] += d
        hk[f][2] += g
    say(qlog.info, f"strace B={B} HC", " · ".join(f"{f[3:]} {v[1] / v[0]:.1f} µs (gap before {v[2] / v[0]:+.1f})"
                                                for f, v in sorted(hk.items())) +
        f" · total per module {sum(v[1] for v in hk.values()) / max(hk['hc.K1 combine'][0], 1):.1f} µs")
    with open(f"{OUT}/strace{B}.txt", "w") as fo:
        fo.write(f"B={B} rounds {R} wall {wall:.3f} ms/round\n")
        for n, (c, bsum, isum) in sorted(per_name.items(), key=lambda x: -x[1][1]):
            fo.write(f"{bsum / R / 1e3:8.3f} ms busy {isum / R / 1e3:7.3f} ms idle-before {c // R:4d} calls/round "
                     f"{bsum / c:8.1f} µs/call  {n}\n")
sys.stdout.flush()
os._exit(0)
