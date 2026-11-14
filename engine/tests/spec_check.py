"""Speculative decoding check (run through tools/tp2.sh): MTP draft → verify → accept → commit, against plain decoding.

1. Python loop, greedy (--py-depth): every emitted token and the fp32 logits row that produced it bitwise equal to
   plain greedy decoding (verify + commit leave GDN/PLE/QSA states exactly as plain steps would).
2. Engine loop (Engine::spec_begin / spec_round, accept and drafts sampled on the GPU), greedy, B = 1 per prompt and
   all prompts together: tokens equal to plain greedy; ms per token.
3. Engine loop, sampling (--temp): a sequence's seed gives the same tokens alone and inside the batch (random keys
   depend on (seed, position) only, kernels are batch-invariant), twice in a row; acceptance per round.
4. Depth schedule (--cases dsched): every sequence changes its depth each round by a fixed per-sequence schedule (1 … 7,
   up and down, so batches mix depths and drop pending drafts): greedy tokens equal plain greedy (B = 1 and together);
   sampling: same seed + same schedule → same tokens alone, batched and repeated (the draft chain runs to the batch's
   largest depth, so alone and batched keep different numbers of extra drafts — the q rows must still line up).
One line per case (rank 0).
  tools/tp2.sh spec python3 engine/tests/spec_check.py [--depth 4 --decode 128]"""
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
ap.add_argument("--only", default="short,long,short2")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--decode", type=int, default=128)
ap.add_argument("--py-depth", default="4", help="depths of the Python-loop logits check ('' = skip)")
ap.add_argument("--depth", type=int, default=4, help="depth of the engine-loop cases")
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--top-k", type=int, default=0, help="sampling cases: top_k (0 = off; the server default is 20)")
ap.add_argument("--top-p", type=float, default=1.0, help="sampling cases: top_p (1 = off; the server default is 0.95)")
ap.add_argument("--cases", default="py,engine,sample")
ap.add_argument("--patch-after", default="", help="switch this patch on after the plain references (eager vs patch, bitwise)")
a = ap.parse_args()
OUT = "/work/results/tests/spec"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
gold = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))
names = a.only.split(",")
prompts = {n: gold[n]["prompt_ids"] for n in names}
L_max = max(len(t) for t in prompts.values())
NS = len(names)
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], NS, (L_max + a.decode + 2 * a.depth + 16 + 127) // 64 * 64, a.chunk,
         mtp=True)
log = open(f"{OUT}/spec{a.rank}.log", "w")
sync = torch.cuda.synchronize
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)


def prefill(toks, slot, mtp, last_mtp=True):
    """Prompt in chunks; the head over every row whose next token is known (the last chunk's rows except the last if
    last_mtp, none of the last chunk otherwise — spec_begin does it). Returns last-row logits, chunk start, rows."""
    L = len(toks)
    eng.reset(slot)
    for c0 in range(0, L, a.chunk):
        n = min(a.chunk, L - c0)
        lg = eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [slot], [0, n], False)
        m = n if c0 + n < L else (n - 1 if last_mtp else 0)
        if mtp and m > 0:
            eng.mtp_step(toks[c0 + 1:c0 + 1 + m], list(range(c0, c0 + m)), [slot], [0, m], list(range(m)), True, False)
    return lg[-1].clone(), c0, n


# plain greedy references: ref[name][i] = token at position L + i, rl = the logits row that produced it
ref, rl, ms_plain = {}, {}, {}
for name in names:
    toks = prompts[name]
    L = len(toks)
    lg, _, _ = prefill(toks, 0, False)
    r, l = [int(lg.argmax())], [lg]
    sync()
    t0 = time.perf_counter()
    for i in range(a.decode):
        lg = eng.step([r[-1]], [L + i], [0], [0, 1])[0].clone()
        r.append(int(lg.argmax()))
        l.append(lg)
    sync()
    ms_plain[name] = 1e3 * (time.perf_counter() - t0) / a.decode
    ref[name], rl[name] = r, l

if a.patch_after:  # the references above ran without it; everything below compares against them
    eng.patch(a.patch_after, True)
    say(qlog.info, "spec", f"{a.patch_after} on (references computed with it off)")

# 1. Python loop, greedy, logits bitwise
for k in [int(x) for x in a.py_depth.split(",") if x and "py" in a.cases]:
    for name in names:
        toks = prompts[name]
        L = len(toks)
        lg, _, last = prefill(toks, 0, True)
        out, ol = [int(lg.argmax())], [lg]
        p, src, ids, pos = L, [last - 1], [out[-1]], [L - 1]
        while len(out) <= a.decode:
            dl = eng.mtp_step(ids, pos, [0], [0, len(ids)], src, True, False)
            d = [int(dl[-1].argmax())]
            for i in range(1, k):  # continue from the head's last row (row len(ids) − 1 of the first pass, then 0)
                r = len(ids) - 1 if i == 1 else 0
                d.append(int(eng.mtp_step([d[-1]], [p + i - 1], [0], [0, 1], [r], False, False)[-1].argmax()))
            vl = eng.verify([out[-1]] + d, list(range(p, p + k + 1)), [0], [0, k + 1])
            am = vl.argmax(-1).tolist()
            j = 0
            while j < k and am[j] == d[j]:
                j += 1
            eng.commit([j + 1])
            new = d[:j] + [am[j]]
            out += new
            ol += [vl[r].clone() for r in range(j + 1)]
            src, ids, pos = list(range(j + 1)), new, list(range(p, p + j + 1))
            p += j + 1
        n = a.decode + 1
        tok_eq = next((i for i in range(n) if out[i] != ref[name][i]), n)
        lg_eq = sum(1 for i in range(tok_eq) if torch.equal(ol[i], rl[name][i]))
        ok = tok_eq == n and lg_eq == n
        say(qlog.ok if ok else qlog.warn, f"spec py {name} k={k}",
            "tokens·logits bitwise equal" if ok
            else f"differ: first {tok_eq}/{n} tokens equal · {lg_eq} logits rows equal")


# 2./3. engine loop
def sched(name, r):
    """depth of round r of a sequence (depth schedule case): fixed per sequence, independent of the batch"""
    h = (sum(map(ord, name)) * 7919 + r * 104729 + r * r * 31) % 1000003
    return 1 + h % 7


def engine_run(group, k, temp, seeds, dsched=False):
    """Prefill each prompt of `group` in its own slot, spec_begin, then spec_round over the group; a sequence leaves it
    once it has ≥ decode + 1 tokens. dsched: depths from sched() (k ignored). Returns tokens per name, rounds (per
    sequence), accepted drafts, ms per token."""
    out, slots, cnt = {}, {}, {}
    for s, name in enumerate(group):
        toks = prompts[name]
        L = len(toks)
        _, c0, n = prefill(toks, s, True, last_mtp=False)
        out[name] = [eng.spec_begin(s, L, sched(name, 0) if dsched else k, temp, seeds[name], 0, 0, toks[c0 + 1:L],
                                    a.top_k if temp > 0 else 0, a.top_p if temp > 0 else 1.0)]
        slots[name], cnt[name] = s, 1
    rounds = acc = emitted = 0
    sync()
    t0 = time.perf_counter()
    live = list(group)  # a sequence leaves the batch once it has decode + 1 tokens (kernels are batch-invariant)
    while live:
        if dsched:
            for nm in live:
                eng.spec_depth(slots[nm], sched(nm, cnt[nm]))
                cnt[nm] += 1
        res = eng.spec_round([slots[n] for n in live])
        for name, r in zip(live, res):
            out[name] += r
            acc += len(r) - 1
            emitted += len(r)
        rounds += len(live) / len(group)
        live = [n for n in live if len(out[n]) <= a.decode]
    sync()
    ms = 1e3 * (time.perf_counter() - t0) / emitted
    return out, rounds, acc, ms


k = a.depth
n = a.decode + 1
seeds = {nm: 1234 + i for i, nm in enumerate(names)}
for name in names if "engine" in a.cases else []:
    out, rounds, acc, ms = engine_run([name], k, 0.0, seeds)
    eq = next((i for i in range(n) if out[name][i] != ref[name][i]), n)
    say(qlog.ok if eq == n else qlog.warn, f"spec engine {name} k={k}",
        f"{'tokens equal' if eq == n else f'differ: first {eq}/{n} equal'} · accepted per round {acc / rounds:.2f}+1 · "
        f"{ms:.1f} ms/token → {1e3 / ms:.1f} tok/s (plain {1e3 / ms_plain[name]:.1f})")
if "engine" in a.cases:
    out, rounds, acc, ms = engine_run(names, k, 0.0, seeds)
    bad = [nm for nm in names if out[nm][:n] != ref[nm][:n]]
    say(qlog.ok if not bad else qlog.warn, f"spec engine B={NS} k={k}",
        f"{'tokens equal plain greedy per sequence' if not bad else 'differ: ' + ','.join(bad)} · accepted per round "
        f"{acc / rounds / NS:.2f}+1 · {ms:.1f} ms/token → total {1e3 / ms:.1f} tok/s")
# sampling: alone vs batched vs repeated, each sequence with its own seed
if "sample" in a.cases:
    def first_diff(x, y):
        return next((i for i in range(min(len(x), len(y), n)) if x[i] != y[i]), None)

    solo = {nm: engine_run([nm], k, a.temp, seeds)[0][nm] for nm in names}
    solo2 = {nm: engine_run([nm], k, a.temp, seeds)[0][nm] for nm in names}
    b1, r1, acc1, ms1 = engine_run(names, k, a.temp, seeds)
    b2 = engine_run(names, k, a.temp, seeds)[0]
    bad = []
    for nm in names:
        d = {"solo repeat": first_diff(solo[nm], solo2[nm]), "solo-batch": first_diff(solo[nm], b1[nm]),
             "batch repeat": first_diff(b1[nm], b2[nm])}
        print(f"temp {a.temp} {nm}: diff {d} solo {solo[nm][:24]} b1 {b1[nm][:24]}", file=log, flush=True)
        bad += [f"{nm} {c}@{v}" for c, v in d.items() if v is not None]
    json.dump({nm: solo[nm] for nm in names}, open(f"{OUT}/sample_tokens{a.rank}.json", "w"))  # for runs to compare
    say(qlog.ok if not bad else qlog.warn, f"spec sampling T={a.temp} top_k {a.top_k} top_p {a.top_p} k={k}",
        f"{'same seed → same tokens (solo = batched = repeated)' if not bad else 'differ: ' + ' · '.join(bad)} · "
        f"accepted per round {acc1 / r1 / NS:.2f}+1 · B={NS} {ms1:.1f} ms/token → total {1e3 / ms1:.1f} tok/s")
# 4. depth schedule: greedy = plain; sampling alone = batched = repeated
if "dsched" in a.cases:
    bad = []
    for name in names:
        o = engine_run([name], 0, 0.0, seeds, dsched=True)[0][name]
        if o[:n] != ref[name][:n]:
            bad.append(f"{name}@{next(i for i in range(n) if o[i] != ref[name][i])}")
    ob, rb, accb, msb = engine_run(names, 0, 0.0, seeds, dsched=True)
    bad += [f"B={NS} {nm}" for nm in names if ob[nm][:n] != ref[nm][:n]]
    say(qlog.ok if not bad else qlog.warn, "spec depth schedule greedy",
        f"{'tokens = plain greedy, depths 1…7 (B=1 · B=' + str(NS) + ')' if not bad else 'differ: ' + ' · '.join(bad)}"
        f" · accepted per round {accb / rb / NS:.2f}+1")

    def first_diff(x, y):
        return next((i for i in range(min(len(x), len(y), n)) if x[i] != y[i]), None)

    solo = {nm: engine_run([nm], 0, a.temp, seeds, dsched=True)[0][nm] for nm in names}
    b1, r1, acc1, ms1 = engine_run(names, 0, a.temp, seeds, dsched=True)
    b2 = engine_run(names, 0, a.temp, seeds, dsched=True)[0]
    bad = []
    for nm in names:
        d = {"solo-batch": first_diff(solo[nm], b1[nm]), "batch repeat": first_diff(b1[nm], b2[nm])}
        bad += [f"{nm} {c}@{v}" for c, v in d.items() if v is not None]
    say(qlog.ok if not bad else qlog.warn, f"spec depth schedule sampling T={a.temp}",
        f"{'same seed·schedule → same tokens (solo = batched = repeated)' if not bad else 'differ: ' + ' · '.join(bad)}"
        f" · accepted per round {acc1 / r1 / NS:.2f}+1")
sys.stdout.flush()
os._exit(0)
