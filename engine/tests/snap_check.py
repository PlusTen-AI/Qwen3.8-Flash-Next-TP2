"""Snapshot inside a step (P011 last_row_in_chunk; run through tools/tp2.sh): for prompts of several lengths (one lane,
P008 two attention lanes, P002 two lanes) and snapshot rows at the start, the lane boundary and the last row —
  old: step(rows before r) → snapshot → step(rows r …)          new: step(all rows, snap_row = r)
→ the logits of the last row, then of 4 more decode steps, are bitwise equal; restore() gives the same token count on
both slots and replaying rows r … plus the decode rows from the restored states gives bitwise equal logits again.
Also the time of the last prompt token both ways (the old path's extra one-row step).
Checkpoint pool (step(..., snaps)): one step with checkpoints at several rows at once (start, inside each lane, the lane
boundary, the last row) into pool buffers → the same logits; loading each checkpoint and replaying the rest gives the
same logits and decode rows again. --pool: the same with the shared KV pool, where the pool buffers are pool chunks
mapped when a checkpoint goes in and given back by ckpt_drop / reset (free chunks counted at each point).
  tools/tp2.sh snap python3 engine/tests/snap_check.py [--pool]"""
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
ap.add_argument("--lens", default="900,6000,12000")
ap.add_argument("--pool", action="store_true", help="shared KV pool: checkpoint buffers mapped from it")
ap.add_argument("--share-lens", default="900,6000,12000,40000", help="--pool: lengths of the ckpt_share check (read in "
                "16K chunks; past 32,768 the compressed indexer keys share a chunk too)")
a = ap.parse_args()
OUT = "/work/results/tests/snap"
os.makedirs(OUT, exist_ok=True)
LENS = [int(x) for x in a.lens.split(",")]
SHARE = [int(x) for x in a.share_lens.split(",")] if a.pool else []
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], 2, max(LENS + SHARE) + 64, 16384, checkpoints=6, kv_pool=a.pool)
if a.pool:
    eng.kv_pool_init(15.0, 0.0, 8.0)
CK = eng.ckpt_chunks()
toks = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))["long"]["prompt_ids"]
DEC = 4


def prompt(n):
    return [toks[j % len(toks)] for j in range(n)]


def tail(slot, L, first):
    """4 greedy decode steps after position L − 1 from token `first` → logits rows"""
    out, nxt = [], first
    for k in range(DEC):
        lg = eng.step([nxt], [L + k], [slot], [0, 1])
        out.append(lg[0].clone())
        nxt = int(lg[0].argmax())
    return out


bad, lines = [], []
for L in LENS:
    P = prompt(L)
    for r in sorted({0, L // 2 - 1, L // 2, L - 1}):
        # old: two steps with snapshot() between
        eng.reset(0)
        if r > 0:
            eng.step(P[:r], list(range(r)), [0], [0, r], False)
        eng.snapshot(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lg_old = eng.step(P[r:], list(range(r, L)), [0], [0, L - r], False)[0].clone()
        torch.cuda.synchronize()
        t_old = time.perf_counter() - t0
        # new: one step, snapshot at row r inside it
        eng.reset(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lg_new = eng.step(P, list(range(L)), [1], [0, L], False, r)[0].clone()
        torch.cuda.synchronize()
        t_new = time.perf_counter() - t0
        first = int(lg_old.argmax())
        ok = torch.equal(lg_old, lg_new)
        d_old, d_new = tail(0, L, first), tail(1, L, first)
        ok &= all(torch.equal(x, y) for x, y in zip(d_old, d_new))
        # restore both and replay rows r … L − 1 and the decode rows
        n0, n1 = eng.restore(0), eng.restore(1)
        ok &= n0 == r and n1 == r
        rep = []
        for s in (0, 1):
            lg = eng.step(P[r:], list(range(r, L)), [s], [0, L - r], False)[0].clone()
            rep.append([lg] + tail(s, L, first))
        ok &= all(torch.equal(x, y) for x, y in zip(rep[0], rep[1])) and torch.equal(rep[0][0], lg_old)
        if not ok:
            bad.append(f"L{L} r{r}")
        if r == L - 1 and L > 1:
            # the server's old last-token path: chunk without the last row (timed above as part of t_old only when r = 0)
            eng.reset(0)
            eng.step(P[:L - 1], list(range(L - 1)), [0], [0, L - 1], False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng.snapshot(0)
            eng.step([P[L - 1]], [L - 1], [0], [0, 1], False)
            torch.cuda.synchronize()
            t_last = time.perf_counter() - t0
            lines.append(f"{L} rows: last token alone {t_last * 1e3:.1f} ms · "
                         f"one step with the in-chunk snapshot {t_new * 1e3:.0f} ms")
    # several checkpoints in one step (pool buffers 2 …), then each loaded and the rest replayed
    rows = sorted({0, L // 4, L // 2 - 1, L // 2, (3 * L) // 4, L - 1})
    eng.reset(0)
    ref = eng.step(P, list(range(L)), [0], [0, L], False)[0].clone()
    ref_tail = tail(0, L, int(ref.argmax()))
    eng.reset(1)
    f0 = eng.kv_free()
    got = eng.step(P, list(range(L)), [1], [0, L], False, -1, [(r, 2 + k) for k, r in enumerate(rows)])[0].clone()
    if a.pool:  # the slot's caches and one checkpoint's chunks per buffer were mapped
        if f0 - eng.kv_free() != eng.kv_held(1) + CK * len(rows):
            bad.append(f"L{L} pool chunks {f0} → {eng.kv_free()} (slot {eng.kv_held(1)}, {len(rows)} × {CK})")
    ok = torch.equal(got, ref) and all(torch.equal(x, y) for x, y in zip(tail(1, L, int(ref.argmax())), ref_tail))
    for k, r in enumerate(rows):
        n = eng.ckpt_load(1, 2 + k)
        lg = eng.step(P[r:], list(range(r, L)), [1], [0, L - r], False)[0].clone()
        ok &= n == r and torch.equal(lg, ref) and all(torch.equal(x, y) for x, y in zip(tail(1, L, int(ref.argmax())), ref_tail))
    ok &= eng.ckpt_load(0, 2) == -1  # another slot's buffer
    if a.pool:  # dropped buffers give their chunks back; saved again they hold the same state; reset gives back the rest
        f1 = eng.kv_free()
        eng.ckpt_drop(2)
        ok &= eng.kv_free() == f1 + CK and eng.ckpt_load(1, 2) == -1
        eng.reset(1)
        eng.step(P[:rows[1]], list(range(rows[1])), [1], [0, rows[1]], False)
        eng.ckpt_save(1, 2)
        n = eng.ckpt_load(1, 2)
        lg = eng.step(P[rows[1]:], list(range(rows[1], L)), [1], [0, L - rows[1]], False)[0].clone()
        ok &= n == rows[1] and torch.equal(lg, ref)
        eng.reset(1)
        ok &= eng.kv_free() == f0 + eng.kv_held(1)
        if not ok:
            lines.append(f"pool L{L}: free {f0} → {eng.kv_free()}")
    if not ok:
        bad.append(f"L{L} pool")


def run_tail(slot, p, first, k):
    """k greedy decode steps at positions p … from token `first` → logits rows"""
    out, nxt = [], first
    for j in range(k):
        lg = eng.step([nxt], [p + j], [slot], [0, 1])
        out.append(lg[0].clone())
        nxt = int(lg[0].argmax())
    return out


def same(xs, ys):
    return len(xs) == len(ys) and all(torch.equal(x, y) for x, y in zip(xs, ys))


def read(slot, P, p0, snaps=()):
    """positions p0 … len(P) − 1 of P in 16K-row steps, checkpoints (position, buffer) taken inside them → the last
    row's logits"""
    lg = None
    for c0 in range(p0, len(P), 16384):
        c1 = min(c0 + 16384, len(P))
        sn = [(q - c0, b) for q, b in snaps if c0 <= q < c1]
        lg = eng.step(P[c0:c1], list(range(c0, c1)), [slot], [0, c1 - c0], False, -1, sn)[0].clone()
    return lg


# Another slot's checkpoint (ckpt_share, pool only): slot 1 takes slot 0's checkpoints — the caches below them are slot
# 0's chunks mapped again, the chunk each falls in copied — and reads the rest: the same logits and decode rows as the
# cold read, taking only the copies from the pool. Then the holder goes back below the shared part (ckpt_load) and
# writes other tokens there, and is emptied (reset): the sharer's next decode rows are still the cold ones, and every
# chunk comes back once both are emptied.
if a.pool:
    NREF = 8
    for L in SHARE:
        P = prompt(L)
        rows = sorted({max(1, L // 4), L // 2, L - 1})
        ids = {r: 2 + k for k, r in enumerate(rows)}
        eng.reset(0)
        eng.reset(1)
        for b in range(2, 2 + len(rows)):
            eng.ckpt_drop(b)
        f_start = eng.kv_free()
        ref = read(0, P, 0, [(r, ids[r]) for r in rows])
        first = int(ref.argmax())
        ref_tail = run_tail(0, L, first, NREF)  # slot 0 now holds L + NREF positions
        ok, notes = True, []
        for r in rows:
            eng.reset(1)
            f0 = eng.kv_free()
            n = eng.ckpt_share(1, ids[r])
            copies = f0 - eng.kv_free()
            lg = read(1, P, r)
            good = n == r and torch.equal(lg, ref) and same(run_tail(1, L, first, NREF), ref_tail)
            good &= copies <= eng.kv_need(0) and eng.history(1) == L + NREF
            ok &= good
            notes.append(f"r{r}: {copies} copied, {eng.kv_held(1) - eng.kv_sole(1)} shared"
                         + ("" if good else " DIFFER"))
        # isolation: slot 1 shares up to L // 2 and has read to L + 4; slot 0 goes back to L // 4 and writes other
        # tokens over its positions L // 4 …, then is emptied — slot 1's rows L + 4 … stay the cold ones
        eng.reset(1)
        n = eng.ckpt_share(1, ids[L // 2])
        read(1, P, n)
        iso = [same(run_tail(1, L, first, 4), ref_tail[:4])]
        back = eng.ckpt_load(0, ids[rows[0]])
        Q = P[:back] + [toks[(7 * j + 3) % len(toks)] for j in range(back, L)]
        read(0, Q, back)
        run_tail(0, L, Q[-1], 4)
        iso.append(same(run_tail(1, L + 4, int(ref_tail[3].argmax()), 2), ref_tail[4:6]))
        f1, sole0 = eng.kv_free(), eng.kv_sole(0)
        eng.reset(0)  # its own chunks and its pool checkpoints come back; the shared ones stay with slot 1
        freed = eng.kv_free() - f1
        iso.append(freed == sole0 + CK * len(rows))
        iso.append(same(run_tail(1, L + 6, int(ref_tail[5].argmax()), 2), ref_tail[6:8]))
        ok &= all(iso)
        if not all(iso):
            notes.append("isolation " + "".join("+" if x else "-" for x in iso))
        eng.reset(1)
        for b in range(2, 2 + len(rows)):
            eng.ckpt_drop(b)
        ok &= eng.kv_free() == f_start
        lines.append(f"share L{L}: " + ", ".join(notes) + f" · holder back to {back} + other tokens, then emptied "
                     f"(gave back {freed} = its own) · pool {f_start} → {eng.kv_free()} after")
        if not ok:
            bad.append(f"L{L} share")
if a.rank == 0:
    (qlog.ok if not bad else qlog.warn)(
        "snap", ("bitwise equal (logits·decode·replay after restore)" if not bad else "differ: " + ", ".join(bad))
        + f" · lengths {','.join(map(str, LENS))} × 4 snapshot rows"
          f"{' · shared ' + ','.join(map(str, SHARE)) if SHARE else ''}"
          f"{' · KV pool (checkpoint ' + str(CK) + ' chunks)' if a.pool else ''} · "
        + " · ".join(lines))
sys.stdout.flush()
os._exit(0)
