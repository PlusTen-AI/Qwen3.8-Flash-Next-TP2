"""Penalties and min_p through speculative decoding (run through tools/tp2.sh): vLLM's repetition / frequency / presence
penalties on the target rows of every verify (counts include the drafts before each row) and min_p after temperature.
1. Greedy with penalties: the engine loop (spec_begin / spec_round, P019 / P016 as configured) gives the same tokens
   as plain decoding — one step per token, the penalties applied to the fp32 logits row on the host with vLLM's
   formulas (torch, one rounding per op: l·fl(1/r) if l > 0 else l·r on prompt ∪ generated tokens, then l − f·n − p
   on generated ones) and argmax.
2. Sampling with penalties and min_p: a seed gives the same tokens alone, in a batch and repeated.
One line per case (rank 0).
  tools/tp2.sh pen python3 engine/tests/pen_check.py [--decode 96 --depth 4]"""
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
ap.add_argument("--only", default="short,long")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--decode", type=int, default=96)
ap.add_argument("--depth", type=int, default=4)
a = ap.parse_args()
OUT = "/work/results/tests/pen"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
gold = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))
names = a.only.split(",")
prompts = {n: gold[n]["prompt_ids"] for n in names}
L_max = max(len(t) for t in prompts.values())
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank, os.environ["Q38_HEAD_IP"],
         len(names), (L_max + a.decode + 2 * a.depth + 16 + 127) // 64 * 64, a.chunk, mtp=True)
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)
REP, FREQ, PRES = 1.3, 0.5, 0.3


def prefill(toks, slot, mtp):
    L = len(toks)
    eng.reset(slot)
    for c0 in range(0, L, a.chunk):
        n = min(a.chunk, L - c0)
        lg = eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [slot], [0, n], False)
        m = n if c0 + n < L else 0
        if mtp and m > 0:
            eng.mtp_step(toks[c0 + 1:c0 + 1 + m], list(range(c0, c0 + m)), [slot], [0, m], list(range(m)), True, False)
    return lg[-1].clone(), c0


def penalise(lg, prompt_mask, cnt, r, f, p):
    x = lg.float().clone()
    seen = prompt_mask | (cnt > 0)
    rt = torch.full_like(x, r)
    inv = torch.full_like(x, 1.0) / rt
    y = torch.where(seen, torch.where(x > 0, x * inv, x * rt), x)
    y3 = (y - torch.full_like(x, f) * cnt.float()) - torch.full_like(x, p) * (cnt > 0).float()
    return torch.where(cnt > 0, y3, y)


V = 248320
bad = 0
for name in names:
    toks = prompts[name]
    L = len(toks)
    pm = torch.zeros(V, dtype=torch.bool, device="cuda")
    pm[torch.tensor(toks, device="cuda")] = True
    cnt = torch.zeros(V, dtype=torch.int32, device="cuda")
    lg, _ = prefill(toks, 0, False)
    ref = [int(penalise(lg, pm, cnt, REP, FREQ, PRES).argmax())]
    for i in range(a.decode):
        cnt[ref[-1]] += 1
        lg = eng.step([ref[-1]], [L + i], [0], [0, 1])[0].clone()
        ref.append(int(penalise(lg, pm, cnt, REP, FREQ, PRES).argmax()))
    plain = [int(ref[0])]  # without penalties, to show they change the text
    # engine loop, greedy, penalties
    _, c0 = prefill(toks, 0, True)
    eng.pen_begin(0, toks)
    out = [eng.spec_begin(0, L, a.depth, 0.0, 7, 0, 0, toks[c0 + 1:L], 0, 1.0, 0.0, REP, FREQ, PRES)]
    rounds = 0
    while len(out) <= a.decode:
        out += eng.spec_round([0])[0]
        rounds += 1
    out = out[:a.decode + 1]
    eq = next((i for i in range(len(ref)) if out[i] != ref[i]), len(ref))
    # the same without penalties (first tokens only), to see that they act
    _, c0 = prefill(toks, 0, True)
    nop = [eng.spec_begin(0, L, a.depth, 0.0, 7, 0, 0, toks[c0 + 1:L], 0, 1.0)]
    while len(nop) <= a.decode:
        nop += eng.spec_round([0])[0]
    differs = sum(x != y for x, y in zip(nop[:a.decode + 1], out))
    ok = eq == len(ref) and differs > 0
    bad += not ok
    say(qlog.ok if ok else qlog.warn, f"pen greedy {name}",
        f"r {REP} f {FREQ} p {PRES}: engine == plain+penalties {eq}/{len(ref)} tokens · {rounds} rounds · "
        f"{differs} of {len(out)} tokens differ from no penalties")

# 2. sampling: alone == batch == repeated
T, TK, TP, MP = 0.8, 20, 0.95, 0.05


def run(group, seeds):
    res = {}
    for s, name in enumerate(group):
        toks = prompts[name]
        _, c0 = prefill(toks, s, True)
        eng.pen_begin(s, toks)
        res[name] = [eng.spec_begin(s, len(toks), a.depth, T, seeds[name], 0, 0, toks[c0 + 1:], TK, TP, MP, 1.1, 0.4,
                                    0.2)]
    live = list(group)
    while live:
        for name, r in zip(live, eng.spec_round([live.index(n) if False else group.index(n) for n in live])):
            res[name] += r
        live = [n for n in live if len(res[n]) <= a.decode]
    return {n: v[:a.decode + 1] for n, v in res.items()}


seeds = {n: 99 + i for i, n in enumerate(names)}
alone = {}
for n in names:
    alone.update(run([n], seeds))
both = run(names, seeds)
again = {}
for n in names:
    again.update(run([n], seeds))
same = all(alone[n] == both[n] == again[n] for n in names)
bad += not same
say(qlog.ok if same else qlog.warn, "pen sample",
    f"T {T} top_k {TK} top_p {TP} min_p {MP} r 1.1 f 0.4 p 0.2: same seed alone == batch == repeated "
    f"{'yes' if same else 'NO'} ({', '.join(f'{n} {len(alone[n])}' for n in names)} tokens)")
say(qlog.ok if not bad else qlog.warn, "pen", "all cases pass" if not bad else f"{bad} case(s) failed")
