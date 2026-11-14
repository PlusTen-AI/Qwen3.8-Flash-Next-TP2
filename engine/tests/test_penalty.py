"""min_p and the penalties (src/kernels/sample: sample.cu / p016 / p019 min_p, penalty.cu; contract sample.h).

min_p: the premise (ex2 probe: weights non-decreasing in the logit over every fp32 argument); the kernels' cut and W'
(sample_cut) == a reference of sample.h's definition that decides M by each token's own weight (no prefix assumption),
min_p alone and with top-k / top-p, every path (cap 16, cluster sizes); M vs vLLM's float rule (differences only at
fp32 boundaries); sample_rows frequencies vs the truncated distribution; spec_accept (base rule and P016) == the
Python-integer models of test_sample on rows truncated with min_p (target) and q_min_p (draft); spec_accept output
distribution (first token, first two tokens) with min_p; the sparse kernels == the full kernels bitwise with min_p;
invariance (alone == batch == cluster sizes); min_p = 0 / negative / NaN == no min_p bitwise.
Penalties: the seen-token state after reset / prompt / add (spec_accept's layout, −1 / kSparseDense skipped) == a
Python reference; penalize_rows == a CPU fp32 reference of sample.h's formulas bitwise (full rows and vocab halves,
extras = drafts with repeats / prompt / output / new / out-of-range ids, r < 1 and > 1, f, p of both signs, sequences
off or without state untouched), == vLLM's formulas up to the sign of zero; timing.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sample as ts  # noqa: E402

V = ts.V
VH = V // 2
INF = float("inf")
NAN = float("nan")
P_MIN = ts.P_MIN
DENSE = ts.DENSE
_mod = ts._mod
F32 = torch.float32


def _f32(x):
    return float(torch.tensor(x, dtype=F32))


def _mp_thr(mp):
    """min_p's weight threshold (sample.h): ⌈mp·2^44⌉ (0 < mp < 1, mp's fp32 value), 2^44 (mp ≥ 1), 0 (off)."""
    mp = _f32(mp)
    if not mp > 0:
        return 0
    if mp >= 1:
        return 1 << 44
    num, den = mp.as_integer_ratio()
    return -(-num * (1 << 44) // den)


def _mpt(n, mp):
    return torch.full((n,), mp, dtype=F32, device="cuda")


# ---- reference truncation with min_p ------------------------------------------------------------------------------
def _ref_cut_mp(order, w, k, p, mp):
    """sample.h's truncation of one row from the definition: order = its tokens by Z descending, w = the kernels'
    integer weights (int64 [V]). M = {v : w_v ≥ t_mp} token by token (the top weight below t_mp: min_p ignored), K =
    the min(k, |M|) highest-ranked tokens of M, then top-p's prefix (≥ ⌈p·W_K⌉) → kept mask [V], W'."""
    n = order.numel()
    thr = _mp_thr(mp)
    if thr and int(w.max()) < thr:
        thr = 0
    o = order[w[order] >= thr] if thr else order
    p = _f32(p)
    kk = k if 0 < k < n else n
    if p <= 0:
        kk, p = 1, 1.0
    K = o[:kk]
    cum = torch.cumsum(w[K], 0)
    j = K.numel() - 1
    if p < 1:
        num, den = p.as_integer_ratio()
        j = int(torch.searchsorted(cum, torch.tensor([-(-num * int(cum[-1]) // den)], device=cum.device)))
    keep = torch.zeros(n, dtype=torch.bool, device=w.device)
    keep[K[:j + 1]] = True
    return keep, int(cum[j])


def _trunc_probs_mp(m, x, T, tk, tp, mp):
    """Truncated distribution w'/W' (float64 [V]) of one row with min_p, its kept mask, the weights and W'."""
    w = m.weights(x[None].contiguous(), torch.tensor([T], device="cuda"))[0]
    order = torch.sort(ts._zkeys(x), descending=True).indices
    keep, W = _ref_cut_mp(order, w, tk, tp, mp)
    return torch.where(keep, w, 0).double() / W, keep, w, W


def _boundary_rows(gen):
    """Rows whose weights sit exactly at min_p boundaries: the top at 0, groups of tokens at fl(−j·ln 2·s) (weights
    ≈ 2^44·2^(−j·s)), and their neighbouring floats, rest N(−20, 1)."""
    rows = []
    for s in (0.5, 1.0, 2.0):
        r = torch.randn(V, generator=gen) - 20
        pos = torch.randperm(V, generator=gen)[:400]
        r[pos[0]] = 0.0
        for j in range(1, 8):
            v = torch.tensor(-j * 0.6931472 * s, dtype=F32)
            g = pos[1 + 50 * (j - 1):1 + 50 * j]
            r[g[:20]] = v
            r[g[20:35]] = torch.nextafter(v, torch.tensor(0.0))
            r[g[35:]] = torch.nextafter(v, torch.tensor(-1.0))
        rows.append(r)
    return torch.stack(rows).cuda()


def _cut_rows(ctx, gen):
    real = ts._real_rows(ctx)
    return torch.cat([real[:20], (torch.randn(6, V, generator=gen) * 3).cuda(),
                      torch.randint(-6, 6, (6, V), generator=gen).float().cuda(),
                      torch.stack([ts._support_row(gen, "inf"), ts._support_row(gen, "neg")]),
                      torch.zeros(1, V, device="cuda"), _boundary_rows(gen)])


MPS = [1e-30, 1e-4, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0, 1.5]
MP_TKP = [(0, 1.0), (20, 0.95), (-1, 0.95), (50, 0.8), (1000, 0.99), (5000, 1.0), (3, 1e-9), (-1, 0.5), (2, 1.0)]


def test_minp_premise(ctx):
    """The ex2 probe: weight(y, 0, 1) = round(ex2.approx.ftz(y)·2^44) is non-decreasing over every fp32 y ≤ 0 (2^31
    arguments, −0 … −inf) — min_p's M is then a rank-order prefix (sample.h); coverage: the summed decreases telescope
    to w(−0) − w(−inf) = 2^44 exactly only if every argument was walked."""
    m = _mod(ctx)
    bad, desc = m.weight_monotone()
    ctx.detail(f"penalty minp premise: violations {bad}, total descent {desc} (2^44 = {1 << 44})")
    return {"ok": bad == 0 and desc == 1 << 44,
            "msg": f"ex2 weights monotone: 2^31 fp32 arguments (−0…−inf) violations {bad} · summed decrease == 2^44 "
                   f"{desc == 1 << 44} (whole range walked)"}


def test_minp_cut(ctx):
    """Cut and W' (sample_cut with min_p) == _ref_cut_mp on reference / N(0, 3²) / small-integer ties / support / flat
    / boundary rows at T = 0.6 / 1 / 1.5, min_p MPS × (top_k, top_p) MP_TKP; default cap and cap 16 (refinement
    paths), clusters automatic / 1 / 5 (cap 16). Kept masks compared token by token (the per-token rule: an M that were
    not a rank prefix would differ)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2100)
    x = _cut_rows(ctx, gen)
    R = x.shape[0]
    temp = torch.tensor([0.6, 1.0, 1.5], device="cuda").repeat(R)[:R].contiguous()
    w, z = m.weights(x, temp), ts._zkeys(x)
    order = [torch.sort(z[r], descending=True).indices for r in range(R)]
    bad, n, sizes = [], 0, {}
    prev_c, prev_cap = m.force_cluster(0), m.force_cap(0)
    try:
        for mp in MPS:
            for tk, tp in MP_TKP:
                ref = [_ref_cut_mp(order[r], w[r], tk, tp, mp) for r in range(R)]
                sizes.setdefault(mp, []).extend(int(k.sum()) for k, _ in ref)
                for c, cap in ((0, 0), (0, 16), (1, 16), (5, 0)):
                    m.force_cluster(c), m.force_cap(cap)
                    cut = m.cut(x, temp, **ts._tkp(R, tk, tp), min_p=_mpt(R, mp)).cpu()
                    for r in range(R):
                        n += 1
                        keep = z[r] >= int(cut[r, 0])
                        if not torch.equal(keep, ref[r][0]) or int(cut[r, 1]) != ref[r][1]:
                            bad.append(f"r{r} mp{mp} ({tk},{tp}) C{c} cap{cap}: {int(keep.sum())}/{int(cut[r, 1])} vs "
                                       f"{int(ref[r][0].sum())}/{ref[r][1]}")
    finally:
        m.force_cluster(prev_c), m.force_cap(prev_cap)
    rng = " · ".join(f"{mp:g}: {min(v)}…{max(v)}" for mp, v in sizes.items())
    ctx.detail(f"penalty minp cut: {n} cuts, bad {bad[:20]}; kept sizes per min_p {rng}")
    return {"ok": not bad and n > 0,
            "msg": f"min_p cut·W' == definition (per-token rule) {n - len(bad)}/{n} ({R} rows: reference·random·ties·"
                   f"support·flat·boundary, min_p {len(MPS)} settings × (k,p) {len(MP_TKP)} settings, cap default·16, "
                   f"cluster auto·1·5)" if not bad else "differs: " + ", ".join(bad[:3])}


def test_minp_vllm(ctx):
    """M vs vLLM's rule on every reference decode row at T = 0.7 / 1 / 1.3 for min_p 0.01 / 0.05 / 0.1 / 0.3: vLLM
    keeps logits/T ≥ max + ln(min_p) (fp32). A difference passes only at an fp32 boundary: |(x_v − m)/T − ln(min_p)| ≤
    1e-5 (float64)."""
    m = _mod(ctx)
    real = ts._real_rows(ctx)
    R = real.shape[0]
    same = bnd = 0
    other = []
    z = ts._zkeys(real)
    for T in (0.7, 1.0, 1.3):
        temp = torch.full((R,), T, device="cuda")
        for mp in (0.01, 0.05, 0.1, 0.3):
            ours = z >= m.cut(real, temp, min_p=_mpt(R, mp))[:, :1]
            lg = real / temp[:, None]
            ref = lg >= lg.max(1, keepdim=True).values + math.log(mp)
            for r in range(R):
                if torch.equal(ours[r], ref[r]):
                    same += 1
                    continue
                d = ours[r] ^ ref[r]
                gap = ((real[r][d].double() - real[r].max().double()) / T - math.log(mp)).abs().max().item()
                if gap <= 1e-5:
                    bnd += 1
                else:
                    other.append(f"T{T} mp{mp} r{r}: ours {int(ours[r].sum())} vLLM {int(ref[r].sum())} gap {gap:.2e}")
    tot = same + bnd + len(other)
    ctx.detail(f"penalty minp vllm: same {same}/{tot} boundary {bnd} other {other[:10]}")
    return {"ok": not other, "msg": f"same set as vLLM min_p {same}/{tot} (reference {R} rows × 3 T × min_p "
            f".01·.05·.1·.3) · fp32 boundary {bnd} · other {len(other)}"}


def test_minp_dist(ctx):
    """sample_rows frequencies vs w'/W' (reference cut), 2^20 draws per case (16 identical rows × 65536 counters):
    min_p alone on reference / N(0, 3²) (thousands kept: masked pick) / tied rows, with top-p only (general top-p
    path), with top-k / top-p, cap 8. Chi-square over the top-15 kept + rest; no draw outside the kept set; prob ==
    w/W' bitwise."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2101)
    real = ts._real_rows(ctx)
    R, N = 16, 65536
    rnd = (torch.randn(V, generator=gen) * 3).cuda()
    cases = [("ref", real[3], 2.5, 0, 1.0, 0.02, 0), ("ref", real[11], 2.5, 20, 0.95, 0.01, 0),
             ("randn", rnd, 1.0, -1, 0.9, 0.01, 0), ("randn", rnd, 1.0, 0, 1.0, 1e-4, 0),
             ("ties", torch.randint(-6, 6, (V,), generator=gen).float().cuda(), 1.0, 0, 1.0, 0.3, 0),
             ("support", ts._support_row(gen, "neg"), 1.0, 5, 1.0, 0.1, 0),
             ("ref cap8", real[17], 0.8, 50, 0.8, 0.01, 8)]
    pmin, ok, out = 1.0, True, []
    prev = m.force_cap(0)
    try:
        for i, (name, row, T, tk, tp, mp, cap) in enumerate(cases):
            m.force_cap(cap)
            p, keep, w, W = _trunc_probs_mp(m, row, T, tk, tp, mp)
            ctr = torch.arange(N * R, device="cuda").view(N, R) + ((i + 64) << 40)
            tok, prob = m.sample(row.expand(R, V).contiguous(), torch.full((R,), T, device="cuda"), ts._i64(gen, R), ctr,
                                 **ts._tkp(R, tk, tp), min_p=_mpt(R, mp))
            tok = tok.flatten().long()
            nk = int(keep.sum())
            pos = torch.topk(p, min(15, nk)).indices.tolist()
            pv, tv, noise = ts._gof(torch.bincount(ts._bins(tok, pos), minlength=len(pos) + 1), ts._bin_probs(p, pos))
            outside = int((~keep[tok]).sum())
            pbad = int((prob.flatten() != (w[tok].double() / W).float()).sum())
            ctx.detail(f"penalty minp dist {name} T={T} ({tk},{tp}) mp {mp} cap {cap}: kept {nk} p={pv:.3g} TV={tv:.2e} "
                       f"(noise {noise:.1e}) outside {outside} prob mismatches {pbad}")
            pmin = min(pmin, pv)
            ok &= pv >= P_MIN and outside == 0 and pbad == 0
            out.append(f"{name} mp{mp:g} ({tk},{tp}) {nk} kept")
    finally:
        m.force_cap(prev)
    return {"ok": ok, "msg": f"2^20 draws×{len(cases)}: min p {pmin:.3f} · outside the kept set 0 · prob == w/W' "
            "bitwise (" + " · ".join(out) + ")"}


# ---- spec_accept with min_p: the integer models of test_sample -----------------------------------------------------
def _wrows_mp(m, rows, temp, tk, tp, mp):
    """w' (int64 [n, V], CPU: 0 off the kept set and on degenerate rows), W' list, argmax list — the rows truncated as
    spec_accept truncates them (sample_cut with top_k / top_p / min_p)."""
    w = m.weights(rows, temp)
    am = torch.argmax(rows, -1)
    deg = ~torch.isfinite(rows.gather(1, am[:, None])[:, 0])
    w[deg] = 0
    cut = m.cut(rows, temp, top_k=tk, top_p=tp, min_p=mp)
    w = torch.where(ts._zkeys(rows) >= cut[:, :1], w, torch.zeros_like(w)).cpu()
    return w, w.sum(1).tolist(), am.tolist()


def test_minp_accept_exact(ctx):
    """spec_accept with min_p (target) and q_min_p (draft: 0, or = min_p) == the Python-integer models of sample.h —
    the base rule (P016 off) and the block rule (P016 on, k > 15 the base) — on rows truncated by sample_cut. V = 256
    'rand' / 'ties' rows (q = p on some rows, degenerate rows), k = 0 … 16, and V = 248320 reference rows, k = 1 … 8;
    T 0.7 / 1 / 1.3; min_p 0.001 / 0.01 / 0.1 / 1 per sequence with (k, p) none / (20, 0.95) / (−1, 0.8) / (3, 1);
    drafts from q (+ random / out-of-range ones)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2102)
    real = ts._real_rows(ctx)
    trs = [(None, None), (20, 0.95), (-1, 0.8), (3, 1.0)]
    cases = [(256, kind, tr, 24, 17) for kind in ("rand", "ties") for tr in trs for _ in range(2)]
    cases += [(V, "ref", tr, 6, 9) for tr in trs[:3]]
    n = 0
    bad = {False: 0, True: 0}
    fails, taus = [], {}
    for ci, (Vr, kind, (tk, tp), B, kmax) in enumerate(cases):
        ks = [int(x) for x in torch.randint(0 if kmax > 9 else 1, kmax, (B,), generator=gen)]
        temps = [[0.7, 1.0, 1.3][int(torch.randint(0, 3, (1,), generator=gen))] for _ in range(B)]
        seqs = [(*ts._seq(gen, Vr, k, kind, real), T, tk, tp) for k, T in zip(ks, temps)]
        args, ks = ts._bv_call(m, gen, seqs, (ci + 300) << 32)
        P, ps, Q, qs, drafts, temp, seed, ctr, tkt, tpt = args
        mps = torch.tensor([[0.001, 0.01, 0.1, 1.0][int(torch.randint(0, 4, (1,), generator=gen))] for _ in range(B)],
                           dtype=F32, device="cuda")
        same_q = ci % 2 == 1
        qmps = mps.clone() if same_q else torch.zeros(B, dtype=F32, device="cuda")
        rr = lambda t, e: torch.cat([t[b:b + 1].repeat(k + e) for b, k in enumerate(ks)])  # noqa: E731
        wp = _wrows_mp(m, P, rr(temp, 1), rr(tkt, 1), rr(tpt, 1), rr(mps, 1))
        wq = (_wrows_mp(m, Q[:sum(ks)].contiguous(), rr(temp, 0), rr(tkt, 0), rr(tpt, 0), rr(qmps, 0))
              if sum(ks) else None)
        dl, sl, cl = drafts.tolist(), seed.tolist(), ctr.tolist()
        for on in (False, True):
            with ts._Block(m, on):
                na, nx = m.accept(*args, min_p=mps, q_min_p=qmps)
            for b, k in enumerate(ks):
                p0, q0 = int(ps[b]), int(qs[b])
                Pb = (wp[0][p0:p0 + k + 1], wp[1][p0:p0 + k + 1], wp[2][p0:p0 + k + 1])
                Qb = (wq[0][q0:q0 + k], wq[1][q0:q0 + k], wq[2][q0:q0 + k]) if k else (None, [], [])
                ref = (ts._model_bv if on else ts._model_tbt)(Pb, Qb, dl[q0:q0 + k], sl[b], cl[b])
                got = (int(na[0, b]), int(nx[0, b]))
                n += 1
                if float(temp[b]) > 0 and got != ref:
                    bad[on] += 1
                    if len(fails) < 10:
                        fails.append(f"case{ci} b{b} k{k} {kind} P016 {on} mp {float(mps[b])} q {float(qmps[b])}: "
                                     f"{got} vs {ref}")
                if on:
                    taus[(k, got[0])] = taus.get((k, got[0]), 0) + 1
    ks_hit = sorted({k for k, _ in taus})
    full = sum(1 for k in ks_hit if all((k, t) in taus for t in range(k + 1)))
    ctx.detail(f"penalty minp accept exact: {n} (seq, rule) pairs, mismatches base {bad[False]} P016 {bad[True]} "
               f"{fails}; (k, tau) {sorted(taus.items())}")
    ok = bad[False] == 0 and bad[True] == 0
    return {"ok": ok, "msg": f"spec_accept with min_p == integer model {n - bad[False] - bad[True]}/{n} (base "
            f"rule·P016, V=256 random·ties·degenerate k=0…16, V=248320 reference k=1…8, min_p .001·.01·.1·1, "
            f"q_min_p 0·same, (k,p) 4 settings) · k with every τ 0…k seen {full}/{len(ks_hit)}"}


def _rounds_mp(m, gen, pl, ql, T, B, n_iter, ctr0, tk, tp, mp, qmp):
    """test_sample._rounds with min_p: drafts drawn from q with (tk, tp, qmp), spec_accept with min_p = mp on the
    target and q_min_p = qmp, x1 from p_1 with (tk, tp, mp) → first two emitted tokens, n_acc."""
    k = ql.shape[0]
    Vr = pl.shape[1]
    P, Q = pl.repeat(B, 1), ql.repeat(B, 1)
    kw = lambda n: ts._tkp(n, tk if tk is not None else 0, tp if tp is not None else 1.0)  # noqa: E731
    d, _ = m.sample(Q, torch.full((B * k,), T, device="cuda"), ts._i64(gen, B * k),
                    torch.arange(n_iter * B * k, device="cuda").view(n_iter, B * k) + ctr0, **kw(B * k),
                    min_p=_mpt(B * k, qmp))
    n, nx = m.accept(P, ts._starts([k] * B, 1), Q, ts._starts([k] * B, 0), d, torch.full((B,), T, device="cuda"),
                     ts._i64(gen, B), torch.arange(n_iter * B, device="cuda").view(n_iter, B) + ctr0, **kw(B),
                     min_p=_mpt(B, mp), q_min_p=_mpt(B, qmp))
    x1, _ = m.sample(pl[1:2].expand(B, Vr).contiguous(), torch.full((B,), T, device="cuda"), ts._i64(gen, B),
                     torch.arange(n_iter * B, device="cuda").view(n_iter, B) + ctr0 + (1 << 39), **kw(B),
                     min_p=_mpt(B, mp))
    d = d.view(n_iter, B, k)
    e0 = torch.where(n >= 1, d[..., 0], nx)
    e1 = torch.where(n == 1, nx, x1)
    if k >= 2:
        e1 = torch.where(n >= 2, d[..., 1], e1)
    return e0.flatten(), e1.flatten(), n.flatten()


def test_minp_accept_dist(ctx):
    """spec_accept's output distribution with min_p: V = 256 (support 8), 2^20 rounds per case, P016 off and on: q
    random (+2 tokens outside p) or q = p, min_p on the target only (drafts drawn without it) or on both, alone or with
    (−1, 0.9) / (5, 1): first token ~ p₀', first two ~ p₀'⊗p₁', no token outside p's kept sets; V = 248320 reference
    rows k = 1 / 3, drafts = targets + N(0, 0.7²), (20, 0.95) + min_p 0.05 and min_p 0.1 alone, 2^15 rounds: first token
    ~ p₀'."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2103)
    pmin, n, outs, ok = 1.0, 0, 0, True
    cases = [("rand", 1, 0.8, None, None, 0.2, 0.0), ("rand", 3, 1.2, -1, 0.9, 0.1, 0.0),
             ("rand", 2, 1.0, 5, 1.0, 0.3, 0.3), ("q=p", 3, 1.0, None, None, 0.25, 0.0),
             ("q=p", 2, 1.0, None, None, 0.25, 0.25)]
    for i, (kind, k, T, tk, tp, mp, qmp) in enumerate(cases):
        pl, ql = ts._small_rows(gen, kind, k)
        tkx, tpx = (tk if tk is not None else 0), (tp if tp is not None else 1.0)
        p0, k0, _, _ = _trunc_probs_mp(m, pl[0], T, tkx, tpx, mp)
        p1, k1, _, _ = _trunc_probs_mp(m, pl[1], T, tkx, tpx, mp)
        q0, _, _, _ = _trunc_probs_mp(m, ql[0], T, tkx, tpx, qmp)
        for on in (False, True):
            with ts._Block(m, on):
                e0, e1, na = _rounds_mp(m, gen, pl, ql, T, 16384, 64, (i * 2 + on + 80) << 40, tk, tp, mp, qmp)
            pv, z = ts._check_round(f"minp {kind} k={k} T={T} ({tk},{tp}) mp {mp} q {qmp} P016 {on}", e0, e1, na, p0, p1,
                                    q0, ts.S_POS, True, ctx)
            out = int((~k0[e0.long()]).sum() + (~k1[e1.long()]).sum())
            pmin, n, outs = min(pmin, pv), n + 1, outs + out
            ok &= pv >= P_MIN and z <= 5 and out == 0
    real = ts._real_rows(ctx)
    for i, (k, tk, tp, mp) in enumerate(((1, 20, 0.95, 0.05), (3, 0, 1.0, 0.1))):
        pl = real[30 + 4 * i:30 + 4 * i + k + 1].contiguous()
        ql = (pl[:k] + (torch.randn(k, V, generator=gen) * 0.7).cuda()).contiguous()
        p0, k0, _, _ = _trunc_probs_mp(m, pl[0], 1.0, tk, tp, mp)
        p1, _, _, _ = _trunc_probs_mp(m, pl[1], 1.0, tk, tp, mp)
        q0, _, _, _ = _trunc_probs_mp(m, ql[0], 1.0, tk, tp, 0.0)
        with ts._Block(m, True):
            e0, e1, na = _rounds_mp(m, gen, pl, ql, 1.0, 32, 1024, (i + 96) << 40, tk, tp, mp, 0.0)
        pos = torch.topk(p0, min(15, int(k0.sum()))).indices.tolist()
        pv, z = ts._check_round(f"minp ref k={k} ({tk},{tp}) mp {mp}", e0, e1, na, p0, p1, q0, pos, False, ctx)
        out = int((~k0[e0.long()]).sum())
        pmin, n, outs = min(pmin, pv), n + 1, outs + out
        ok &= pv >= P_MIN and z <= 5 and out == 0
    return {"ok": ok, "msg": f"min_p truncated p₀'·p₀'⊗p₁' distributions: {n} cases min p {pmin:.3f} · outside the "
            f"kept set {outs} (V=256 2^20 rounds P016 off·on: q random·q=p, min_p target only·both, "
            f"alone·(−1,.9)·(5,1); V=248320 reference 2^15 rounds)"}


def test_minp_sparse(ctx):
    """sample_rows_sparse / spec_accept_sparse == sample_rows / spec_accept bitwise with min_p: the P019 rows
    (reference, ties, flat, sorted, 64 ties, ±0, extremes) at T 0 / 0.7 / 1 / 1.5 × (k, p) from 1 / 2 / 20 / 64 × 0.5
    / 0.95 / 1 × min_p 0.01 / 0.1 / 0.5 / 1 (and per-row mixed), 8 counters; accept: 48 calls of B = 1 … 4 sequences
    (k = 0 … 16, reference / random / ties, q = p rows, degenerate rows → dense markers) with min_p and q_min_p per
    sequence, P016 off / on, 8 counters."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2104)
    x = torch.cat([r for _, r in ts._p019_rows(ctx, gen)])
    rc = ts._recs(m, x)
    R = x.shape[0]
    mism = n = wrong = 0
    for T in ts.P019_T:
        for tk, tp in ts.P019_KP + ts.P019_EDGE[:4]:
            for mp in (0.01, 0.1, 0.5, 1.0, None):
                mpt = _mpt(R, mp) if mp is not None else torch.tensor(
                    [[0.0, 0.01, 0.1, 0.5, 1.0, NAN, -1.0][r % 7] for r in range(R)], dtype=F32, device="cuda")
                temp = torch.full((R,), T, device="cuda")
                tkt = torch.full((R,), tk, dtype=torch.int32, device="cuda")
                tpt = torch.full((R,), tp, dtype=F32, device="cuda")
                seed, ctr = ts._i64(gen, R), ts._i64(gen, 8, R)
                tok, prob = m.sample(x, temp, seed, ctr, tkt, tpt, min_p=mpt)
                st, sp = m.sample_sparse(rc, V, temp, seed, ctr, tkt, tpt, min_p=mpt)
                flag = ~torch.isfinite(x).all(1)
                okp = torch.tensor([m.sparse_ok(T, tk, tp, V)] * R, device="cuda")
                dn = (flag | ~okp)[None].expand(8, R)
                mark = st == DENSE
                mism += int(((st != tok) | (sp.view(torch.int32) != prob.view(torch.int32)))[~dn].sum())
                n += int((~dn).sum())
                wrong += int((mark != dn).sum())
    n_s, mism_s = n, mism
    st_a = {"n": 0, "bad": 0, "wrong": 0}
    kps = ts.P019_KP + [(0, 0.95), (65, 1.0)]
    for ci in range(48):
        kind = ("ref", "rand", "ties")[ci % 3]
        seqs = []
        for b in range(1 + ci % 4):
            k = int(torch.randint(0, 17, (1,), generator=gen)) if ci % 6 else int(torch.randint(1, 5, (1,), generator=gen))
            T = ts.P019_T[int(torch.randint(0, 4, (1,), generator=gen))]
            tk, tp = kps[int(torch.randint(0, len(kps), (1,), generator=gen))]
            seqs.append((*ts._seq(gen, V, k, kind, ts._real_rows(ctx)), T, tk, tp))
        args, pc, qc, ks = ts._p019_seq_call(m, gen, seqs, (ci + 400) << 36, 8)
        P, ps, Q, qs, dN, temp, seed, ctrN, tk, tp = args
        B = len(ks)
        mps = torch.tensor([[0.0, 0.01, 0.1, 0.5, 1.0][int(torch.randint(0, 5, (1,), generator=gen))] for _ in range(B)],
                           dtype=F32, device="cuda")
        qmps = torch.where(torch.rand(B, generator=gen).cuda() < 0.5, mps, torch.zeros_like(mps))
        dense = ts._seq_dense(m, P, Q, ks, temp, tk, tp)
        dn = dense[None].expand(8, -1)
        for on in (False, True):
            with ts._Block(m, on):
                na, nx = m.accept(*args, min_p=mps, q_min_p=qmps)
                sa, sx = m.accept_sparse(pc, ps, qc, qs, dN, V, temp, seed, ctrN, tk, tp, min_p=mps, q_min_p=qmps)
            mk = (sa == DENSE) & (sx == DENSE)
            st_a["wrong"] += int((mk != dn).sum())
            st_a["n"] += int((~dn).sum())
            st_a["bad"] += int((((sa != na) | (sx != nx)) & ~dn).sum())
    ok = mism == 0 and wrong == 0 and st_a["bad"] == 0 and st_a["wrong"] == 0 and n > 0 and st_a["n"] > 0
    ctx.detail(f"penalty minp sparse: sample {n_s} compared, mismatches {mism_s}, wrong markers {wrong}; accept "
               f"{st_a}")
    return {"ok": ok, "msg": f"min_p sparse == full bitwise: sample {n_s - mism_s:,}/{n_s:,} draws · accept "
            f"{st_a['n'] - st_a['bad']:,}/{st_a['n']:,} (P016 off·on) · wrong dense markers {wrong + st_a['wrong']}"}


def test_minp_invariance(ctx):
    """Bitwise with per-row min_p mixed (0 / 0.01 / 0.1 / 0.5 / 1 / NaN / −1) and (k, p) mixed: every row alone ==
    permuted batch == cluster sizes 1, 3, 8, 16 == cap 8 (sample_rows), and spec_accept (k = 0 … 4, P016 off / on) alone
    == permuted batch == cluster sizes; min_p 0 / negative / NaN == no min_p (sample_rows, sample_cut, spec_accept,
    sparse kernels) on the same inputs."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2105)
    real = ts._real_rows(ctx)
    rows = torch.cat([real[:6], (torch.randn(4, V, generator=gen) * 3).cuda(), torch.zeros(1, V, device="cuda"),
                      torch.randint(-5, 5, (2, V), generator=gen).float().cuda(), ts._support_row(gen, "neg")[None]])
    R = rows.shape[0]
    MPX = [0.0, 0.01, 0.1, 0.5, 1.0, NAN, -1.0]
    temp = torch.tensor([0, 0.5, 1, 1.3] * 4, device="cuda")[:R]
    tk = torch.tensor([ts.TK_MIX[r % len(ts.TK_MIX)][0] for r in range(R)], dtype=torch.int32, device="cuda")
    tp = torch.tensor([ts.TK_MIX[r % len(ts.TK_MIX)][1] for r in range(R)], dtype=F32, device="cuda")
    mp = torch.tensor([MPX[(3 * r) % len(MPX)] for r in range(R)], dtype=F32, device="cuda")
    seed, ctr = ts._i64(gen, R), ts._i64(gen, R)
    alone = [m.sample(rows[r:r + 1], temp[r:r + 1], seed[r:r + 1], ctr[r:r + 1], tk[r:r + 1], tp[r:r + 1],
                      min_p=mp[r:r + 1]) for r in range(R)]
    at, ap = torch.cat([a[0] for a in alone], 1), torch.cat([a[1] for a in alone], 1)
    perm = torch.randperm(R, generator=gen).cuda()
    bad = []
    prev_c, prev_cap = m.force_cluster(0), m.force_cap(0)
    try:
        for c, cap in [(0, 0), (1, 0), (3, 0), (8, 0), (16, 0), (0, 8), (3, 8)]:
            m.force_cluster(c), m.force_cap(cap)
            t, p = m.sample(rows[perm], temp[perm], seed[perm], ctr[perm], tk[perm], tp[perm], min_p=mp[perm])
            if not (torch.equal(t[0], at[0, perm]) and torch.equal(p[0], ap[0, perm])):
                bad.append(f"sample C{c} cap{cap}")
        m.force_cluster(0), m.force_cap(0)
        # min_p off in every spelling == no min_p
        for off in (0.0, -1.0, NAN):
            for kw in ({}, {"top_k": tk, "top_p": tp}):
                a = m.sample(rows, temp, seed, ctr, **kw)
                b = m.sample(rows, temp, seed, ctr, **kw, min_p=_mpt(R, off))
                if not (torch.equal(a[0], b[0]) and torch.equal(a[1].view(torch.int32), b[1].view(torch.int32))):
                    bad.append(f"sample off {off} {bool(kw)}")
                tc = torch.full((R,), 0.9, device="cuda")
                if not torch.equal(m.cut(rows, tc, **kw), m.cut(rows, tc, **kw, min_p=_mpt(R, off))):
                    bad.append(f"cut off {off} {bool(kw)}")
        ks = [0, 1, 2, 3, 4, 4, 3, 2, 1, 4]
        B = len(ks)
        P = torch.cat([real[(3 * b + torch.arange(k + 1)) % real.shape[0]] for b, k in enumerate(ks)])
        Q = torch.cat([real[(3 * b + torch.arange(k) + 7 * (b % 2)) % real.shape[0]] for b, k in enumerate(ks) if k])
        Q = Q + (torch.randn(Q.shape, generator=gen) * 1.5).cuda()
        ts_ = torch.tensor([0.0 if b % 4 == 0 else 0.8 + 0.1 * b for b in range(B)], device="cuda")
        bk = torch.tensor([ts.TK_MIX[(b + 1) % len(ts.TK_MIX)][0] for b in range(B)], dtype=torch.int32, device="cuda")
        bp = torch.tensor([ts.TK_MIX[(b + 1) % len(ts.TK_MIX)][1] for b in range(B)], dtype=F32, device="cuda")
        bm = torch.tensor([MPX[(2 * b + 1) % len(MPX)] for b in range(B)], dtype=F32, device="cuda")
        bq = torch.where(torch.arange(B, device="cuda") % 3 == 0, bm, torch.zeros_like(bm))
        rep = lambda t: torch.cat([t[b].repeat(k) for b, k in enumerate(ks)])  # noqa: E731
        drafts = m.sample(Q, rep(ts_), ts._i64(gen, Q.shape[0]), ts._i64(gen, Q.shape[0]), rep(bk), rep(bp),
                          min_p=rep(bq))[0][0]
        seed2, ctr2 = ts._i64(gen, B), ts._i64(gen, B)
        qs, ps = ts._starts(ks, 0), ts._starts(ks, 1)
        order = torch.argsort(torch.rand(B, generator=gen)).tolist()
        ks2 = [ks[b] for b in order]
        P2 = torch.cat([P[int(ps[b]):int(ps[b + 1])] for b in order])
        Q2 = torch.cat([Q[int(qs[b]):int(qs[b + 1])] for b in order if ks[b]])
        d2 = torch.cat([drafts[int(qs[b]):int(qs[b + 1])] for b in order])
        o = torch.tensor(order, device="cuda")
        for on in (False, True):
            with ts._Block(m, on):
                one = []
                for b in range(B):
                    q0, q1 = int(qs[b]), int(qs[b + 1])
                    one.append(m.accept(P[int(ps[b]):int(ps[b + 1])], ts._starts([ks[b]], 1),
                                        Q[q0:q1] if q1 > q0 else Q[:1], ts._starts([ks[b]], 0),
                                        drafts[q0:q1].contiguous() if q1 > q0 else drafts[:0], ts_[b:b + 1],
                                        seed2[b:b + 1], ctr2[b:b + 1], bk[b:b + 1], bp[b:b + 1], min_p=bm[b:b + 1],
                                        q_min_p=bq[b:b + 1]))
                ref = torch.cat([x[0] for x in one], 1), torch.cat([x[1] for x in one], 1)
                for c in (0, 1, 3, 8, 16):
                    m.force_cluster(c)
                    na, nx = m.accept(P2, ts._starts(ks2, 1), Q2, ts._starts(ks2, 0), d2, ts_[o], seed2[o], ctr2[o],
                                      bk[o], bp[o], min_p=bm[o], q_min_p=bq[o])
                    if not (torch.equal(na[0], ref[0][0, o]) and torch.equal(nx[0], ref[1][0, o])):
                        bad.append(f"accept P016 {on} C{c}")
                m.force_cluster(0)
                z0 = torch.zeros(B, dtype=F32, device="cuda")
                a = m.accept(P, ps, Q, qs, drafts, ts_, seed2, ctr2, bk, bp)
                b_ = m.accept(P, ps, Q, qs, drafts, ts_, seed2, ctr2, bk, bp, min_p=z0, q_min_p=_mpt(B, NAN))
                if not (torch.equal(a[0], b_[0]) and torch.equal(a[1], b_[1])):
                    bad.append(f"accept off P016 {on}")
    finally:
        m.force_cluster(prev_c), m.force_cap(prev_cap)
    ctx.detail(f"penalty minp invariance: bad {bad}")
    return {"ok": not bad, "msg": "min_p mixed per row: alone == permuted batch == cluster 1·3·8·16 == cap 8 bitwise "
            f"(sample {R} rows, accept {len(ks)} sequences P016 off·on) · min_p 0·negative·NaN == no min_p" if not bad
            else "mismatch: " + ", ".join(bad[:4])}


# ---- penalties ------------------------------------------------------------------------------------------------------
class _State:
    """S sequences' seen-token state on the GPU and its Python mirror (output counts, prompt sets)."""

    def __init__(self, m, S):
        self.m = m
        self.cnt = torch.zeros(S, V, dtype=torch.int32, device="cuda")
        self.ids = torch.zeros(S, V, dtype=torch.int32, device="cuda")
        self.n = torch.zeros(S, dtype=torch.int32, device="cuda")
        self.out = [torch.zeros(V, dtype=torch.int64) for _ in range(S)]
        self.prompt = [torch.zeros(V, dtype=torch.bool) for _ in range(S)]
        for s in range(S):
            m.pen_reset(self.cnt, self.ids, self.n, s)

    def add_prompt(self, s, tok):
        self.m.pen_prompt(self.cnt, self.ids, self.n, s, tok.int().cuda())
        t = tok[(tok >= 0) & (tok < V)].long()
        self.prompt[s][t] = True

    def add(self, slots, tok=None, t_start=None, n_tok=None, last=None):
        """pen_add with the given device-layout arguments (CPU int lists / tensors), mirrored."""
        cu = lambda t: None if t is None else torch.as_tensor(t, dtype=torch.int32).cuda()  # noqa: E731
        self.m.pen_add(self.cnt, self.ids, self.n, slots, tok=cu(tok), t_start=cu(t_start), n_tok=cu(n_tok),
                       last=cu(last))
        for b, s in enumerate(slots):
            if s < 0:
                continue
            nb = 0
            if tok is not None:
                span = int(t_start[b + 1]) - int(t_start[b])
                nb = min(int(n_tok[b]), span) if n_tok is not None else span
                if nb < 0:
                    continue
            ts_ = [int(tok[int(t_start[b]) + j]) for j in range(nb)]
            if last is not None:
                ts_.append(int(last[b]))
            for v in ts_:
                if 0 <= v < V:
                    self.out[s][v] += 1

    def check(self):
        """GPU state == mirror: cnt bits, ids[:n] = the nonzero tokens, each once → list of problems."""
        bad = []
        cnt = self.cnt.cpu().long() & 0xFFFFFFFF
        for s in range(len(self.out)):
            ref = self.out[s] | (self.prompt[s].long() << 31)
            if not torch.equal(cnt[s], ref):
                bad.append(f"slot {s} cnt ({int((cnt[s] != ref).sum())} tokens)")
            nz = (ref != 0).nonzero().flatten()
            k = int(self.n[s])
            got = torch.sort(self.ids[s, :k].cpu().long()).values
            if k != nz.numel() or not torch.equal(got, nz):
                bad.append(f"slot {s} ids ({k} vs {nz.numel()})")
        return bad


def _ref_pen(x, out, prompt, extras, r, f, p, literal=False):
    """sample.h's penalties on one fp32 row (CPU) x [V]: n = output counts + the extras' multiplicities; repetition on
    prompt ∪ {n > 0}, then frequency and presence on {n > 0} (literal: on every token, as vLLM) — every operand a full
    tensor (no scalar reciprocal shortcut), one rounding per op."""
    ex = torch.tensor([e for e in extras if 0 <= e < V], dtype=torch.long)
    n = out + (torch.bincount(ex, minlength=V) if ex.numel() else 0)
    seen = prompt | (n > 0)
    if r == 1.0 and f == 0.0 and p == 0.0:
        return x.clone()
    rt = torch.full_like(x, r)
    inv = torch.full_like(x, 1.0) / rt  # vLLM: logits *= 1.0 / scale for positive logits (fp32 reciprocal, then product)
    y = torch.where(seen, torch.where(x > 0, x * inv, x * rt), x)
    y2 = y - torch.full_like(x, f) * n.float()
    y3 = y2 - torch.full_like(x, p) * (n > 0).float()
    return y3 if literal else torch.where(n > 0, y3, y)


def _pen_setup(m, gen):
    """State of 6 slots and 8 sequences: (slot, r, f, p, k) — prompt and output mixes, an empty state, an all-off
    sequence, one without state, two sequences on one state."""
    st = _State(m, 6)
    sub = torch.randperm(V, generator=gen)
    st.add_prompt(0, sub[torch.randint(0, 2000, (5000,), generator=gen)])
    st.add_prompt(1, sub[torch.randint(1000, 4000, (3000,), generator=gen)])
    st.add_prompt(3, sub[torch.randint(0, 20000, (130000,), generator=gen)])
    st.add_prompt(3, torch.tensor([-1, V, V + 7, 5, 5]))  # out-of-range ids ignored, repeats harmless
    st.add_prompt(5, sub[:100])
    # outputs: chunks through the spec_accept layout (n_acc, next) and last-only adds
    for _ in range(40):
        slots = [0, 2, 3, 5]
        ks = [int(torch.randint(0, 6, (1,), generator=gen)) for _ in slots]
        t_start = [0]
        for k in ks:
            t_start.append(t_start[-1] + k)
        tok = sub[torch.randint(1500, 2500, (max(t_start[-1], 1),), generator=gen)].tolist()[:t_start[-1]]
        n_acc = [int(torch.randint(-2, k + 1, (1,), generator=gen)) for k in ks]  # −2 / −1: skipped
        nxt = sub[torch.randint(0, 30000, (len(slots),), generator=gen)].tolist()
        st.add(slots, tok or None, t_start if tok else None, n_acc if tok else None, nxt)
    st.add([2, 3], last=[V + 1, -1])  # out-of-range next tokens: nothing
    st.add([3], tok=sub[torch.randint(0, 25000, (2000,), generator=gen)].tolist(), t_start=[0, 2000])
    seqs = [(0, 1.3, 0.5, 0.7, 3), (1, 0.7, 0.0, 0.0, 0), (2, 1.0, -0.8, 1.5, 5), (3, 2.0, 2.0, -2.0, 15),
            (4, 1.2, 0.3, 0.1, 2), (5, 1.0, 0.0, 0.0, 4), (-1, 1.5, 0.5, 0.5, 1), (0, 1.1, 0.0, 0.4, 7)]
    return st, sub, seqs


def _pen_rows(gen, seqs, sub):
    """Rows [R, V] (CPU fp32: N(0, 3²) with ±0, ±inf, huge and tiny values), p_start, q_start, drafts — the drafts of
    a sequence mix prompt / output / new tokens, repeats and out-of-range ids."""
    ks = [s[4] for s in seqs]
    R = sum(k + 1 for k in ks)
    x = torch.randn(R, V, generator=gen) * 3
    x[:, sub[:50]] = 0.0
    x[:, sub[50:100]] = -0.0
    x[:, sub[100:110]] = INF
    x[:, sub[110:120]] = -INF
    x[:, sub[120:130]] = 3e38
    x[:, sub[130:140]] = 1e-30
    drafts = []
    for k in ks:
        d = sub[torch.randint(0, 3000, (k,), generator=gen)].tolist()
        for i in range(k):
            u = float(torch.rand(1, generator=gen))
            if u < 0.2 and i:
                d[i] = d[int(torch.randint(0, i, (1,), generator=gen))]  # repeat
            elif u < 0.25:
                d[i] = [-1, V, V + 3][i % 3]
        drafts += d
    return x, ts._starts([s[4] for s in seqs], 1), ts._starts(ks, 0), torch.tensor(drafts or [0], dtype=torch.int32)


def test_pen_state(ctx):
    """The seen-token state after penalty_reset / penalty_prompt (130K-token prompt with repeats, out-of-range ids) /
    penalty_add (spec_accept's layout: accepted drafts n_acc ≤ k with −1 / kSparseDense skipped, next; last-only adds;
    out-of-range tokens) == a Python mirror: cnt bits exact, ids[:n] = the nonzero tokens each once; reset clears."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2200)
    st, _, _ = _pen_setup(m, gen)
    bad = st.check()
    nn = st.n.tolist()
    m.pen_reset(st.cnt, st.ids, st.n, 3)
    st.out[3].zero_(), st.prompt[3].zero_()
    bad += [f"after reset: {b}" for b in st.check()]
    ctx.detail(f"penalty state: n per slot {nn}, bad {bad}")
    return {"ok": not bad, "msg": f"state == Python mirror (cnt bits·ids list without duplicates; distinct tokens per "
            f"slot {nn}) · 0 after reset" if not bad else "differs: " + ", ".join(bad[:4])}


def test_pen_exact(ctx):
    """penalize_rows == _ref_pen bitwise on 8 sequences × their verify rows (k = 0 … 15: row i counts d_1 … d_i), full
    rows and each vocab half separately (off 0 / Vh, rows of Vh): repetition r 0.7 … 2, frequency / presence ±, prompt
    only / output only / both / empty state, drafts repeated / in the prompt / generated / new / out of range; an all-off
    sequence and one without state untouched; vs vLLM's formulas (subtractions on every token) equal up to ±0."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2201)
    st, sub, seqs = _pen_setup(m, gen)
    x, ps, qs, drafts = _pen_rows(gen, seqs, sub)
    x = torch.cat([x, torch.randn(2, V, generator=gen)])  # rows past p_start[B]: untouched
    slots = [s[0] for s in seqs]
    rep, freq, pres = [s[1] for s in seqs], [s[2] for s in seqs], [s[3] for s in seqs]
    ref, lit = x.clone(), x.clone()
    dl = drafts.tolist()
    for b, (s, r, f, p, k) in enumerate(seqs):
        for i in range(k + 1):
            row = int(ps[b]) + i
            ex = dl[int(qs[b]):int(qs[b]) + i]
            if s < 0:
                ref[row] = lit[row] = x[row]
                continue
            ref[row] = _ref_pen(x[row], st.out[s], st.prompt[s], ex, r, f, p)
            lit[row] = _ref_pen(x[row], st.out[s], st.prompt[s], ex, r, f, p, literal=True)
    dd = drafts.cuda()
    full = x.cuda()
    m.penalize(full, 0, ps, qs, dd, st.cnt, st.ids, st.n, slots, rep, freq, pres)
    halves = [x[:, q * VH:(q + 1) * VH].contiguous().cuda() for q in range(2)]
    for q in range(2):
        m.penalize(halves[q], q * VH, ps, qs, dd, st.cnt, st.ids, st.n, slots, rep, freq, pres)
    got_f, got_h = full.cpu(), torch.cat([h.cpu() for h in halves], 1)
    bits = lambda t: t.view(torch.int32)  # noqa: E731
    bad_f = int((bits(got_f) != bits(ref)).sum())
    bad_h = int((bits(got_h) != bits(ref)).sum())
    zero_eq = (got_f == lit) | (torch.isnan(got_f) & torch.isnan(lit))
    bad_l = int((~zero_eq).sum())
    sign0 = int(((bits(got_f) != bits(lit)) & zero_eq).sum())
    changed = int((bits(got_f) != bits(x)).sum())
    off_rows = [int(ps[b]) + i for b, sq in enumerate(seqs) if sq[0] < 0 or sq[1:4] == (1.0, 0.0, 0.0)
                for i in range(sq[4] + 1)] + [x.shape[0] - 2, x.shape[0] - 1]
    off_bad = int((bits(got_f[off_rows]) != bits(x[off_rows])).sum())
    # no penalties anywhere: nothing changes
    st2 = x[:4].cuda()
    m.penalize(st2, 0, ts._starts([0, 0, 0, 0], 1), None, None, st.cnt, st.ids, st.n, [0, 1, 2, 3], [1.0] * 4, [0.0] * 4,
               [0.0] * 4)
    none_bad = int((bits(st2.cpu()) != bits(x[:4])).sum())
    ok = bad_f == 0 and bad_h == 0 and bad_l == 0 and off_bad == 0 and none_bad == 0 and changed > 0
    ctx.detail(f"penalty exact: {x.shape[0]} rows, logits changed {changed}, mismatches full {bad_f} halves {bad_h}, "
               f"vs vLLM formulas {bad_l} (sign of zero only: {sign0}), off rows changed {off_bad}, all-off call {none_bad}")
    return {"ok": ok, "msg": f"penalize_rows == fp32 reference bitwise: full rows·half blocks {x.shape[0]} rows "
            f"(logits changed {changed:,}, wrong {bad_f}/{bad_h}) · vLLM formulas == (only ±0 differs {sign0}) · "
            f"off sequences·sequences without state untouched"}


def test_pen_perf(ctx):
    """µs per penalize_rows call (a graph of 50 calls, best of 5): 8 sequences, each a 130K-token prompt over ~20K
    distinct tokens + 2000 generated tokens; R = 20 rows (k = 1 / 2, drafts as extras) and R = 128 (k = 15), full rows
    (V) and one vocab half (Vh); all penalties off (launch only)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(2202)
    S = 8
    st = _State(m, S)
    for s in range(S):
        sub = torch.randperm(V, generator=gen)[:20000]
        st.m.pen_prompt(st.cnt, st.ids, st.n, s, sub[torch.randint(0, 20000, (130000,), generator=gen)].int().cuda())
        tok = sub[torch.randint(0, 20000, (2000,), generator=gen)].int().cuda()
        st.m.pen_add(st.cnt, st.ids, st.n, [s], tok=tok, t_start=torch.tensor([0, 2000], dtype=torch.int32).cuda())
    uniq = st.n.tolist()
    res = {}
    for name, ks in (("R20", [1, 1, 1, 1, 2, 2, 2, 2]), ("R128", [15] * S)):
        ps, qs = ts._starts(ks, 1), ts._starts(ks, 0)
        R = sum(k + 1 for k in ks)
        dr = torch.randint(0, V, (sum(ks),), generator=gen, dtype=torch.int32).cuda()
        for w, off in (("V", 0), ("Vh", VH)):
            x = torch.randn(R, V if w == "V" else VH, generator=gen).cuda()
            res[(name, w)] = m.pen_bench(x, off, ps, qs, dr, st.cnt, st.ids, st.n, list(range(S)), [1.1] * S, [0.2] * S,
                                         [0.3] * S, 50, 5)
        x = torch.randn(R, V, generator=gen).cuda()
        res[(name, "off")] = m.pen_bench(x, 0, ps, qs, dr, st.cnt, st.ids, st.n, list(range(S)), [1.0] * S, [0.0] * S,
                                         [0.0] * S, 50, 5)
    ctx.detail("penalty perf (us per call): " + ", ".join(f"{k}: {v:.2f}" for k, v in res.items())
               + f"; distinct seen tokens per sequence {uniq}")
    return {"ok": True, "msg": f"penalize_rows µs (8 sequences × ~{min(uniq) // 1000}K distinct tokens): R=20 full "
            f"rows {res[('R20', 'V')]:.1f} · half {res[('R20', 'Vh')]:.1f} · R=128 {res[('R128', 'V')]:.1f} / "
            f"{res[('R128', 'Vh')]:.1f} · all off {res[('R20', 'off')]:.1f} (graph of 50 calls, best)"}
