"""sample family (src/kernels/sample/sample.cu, contract sample.h): greedy argmax vs torch.argmax; the Philox counter
layout and the exact integer arithmetic (flat rows, reproduced with Python integers); sample_rows frequencies vs
softmax(logits / T); determinism and invariance to batch composition and cluster size (bitwise); spec_accept exactness
(first emitted token ~ p_0, first two ~ p_0 · p_1) for q random / = p / disjoint / one-hot; the T = 0 accept path vs a
torch reference; timing. Truncation (top-k / top-p): the kernels' cut and W' vs a torch reference of sample.h's
definition (from the kernels' integer weights), kept sets vs vLLM's apply_top_k_top_p, draw frequencies, spec_accept
exactness on truncated rows, invariance with mixed parameters, timing. The base accept tests (acceptance rate
Σ min(p₀, q₀), timing) run with P016 off; P016 block_verify (test_bv_*): decisions == a Python-integer model of the
block rule, output distribution, enumeration identity in fractions, gain on reference rows, invariance, timing.
P019 sparse_logits (test_p019_*): sample_cands == a sorted reference, sample_rows_sparse / spec_accept_sparse == the
full kernels bitwise on halved rows (dense markers exactly where the sparse conditions fail), the dense flag, timing.

Statistics: Pearson chi-square over the bins expecting ≥ 5 (the rest pooled into one bin, or — if the pool expects
< 5 — a Poisson upper-tail test of its count), p-value from the regularised incomplete gamma; a case passes at
p ≥ 1e-4 (seeds are fixed, so every run draws the same numbers). TV is shown next to its noise level √(bins / n).
"""
import math
import os

import torch
from safetensors import safe_open

V = 248320
INF = float("inf")
F64 = torch.float64
LOGITS = "language_model.logits_processor#0.out"
P_MIN = 1e-4
_m = None
_real = None


def _mod(ctx):
    global _m
    if _m is None:
        _m = ctx.ext("sample", ["src/kernels/sample/sample.cu", "src/kernels/sample/penalty.cu", "src/model/patches.cc"])
    return _m


class _Block:
    """P016 block_verify on / off inside a with-block (the previous state restored after)."""
    def __init__(self, m, on):
        self.m, self.on = m, on

    def __enter__(self):
        self.prev = self.m.block(self.on)

    def __exit__(self, *a):
        self.m.block(self.prev)


def _real_rows(ctx):
    """fp32 logits rows of the reference's decode steps (golden dump), [n, V] on the GPU."""
    global _real
    if _real is None:
        rows = []
        for k in ctx.golden.decode_steps():
            with safe_open(os.path.join(ctx.golden.dir, f"step{k:04d}.safetensors"), "pt", device="cpu") as f:
                if LOGITS in f.keys():
                    rows.append(f.get_tensor(LOGITS).float().reshape(-1, V)[-1])
        _real = torch.stack(rows).cuda()
    return _real


# ---- Philox4x32-10 and the kernels' uniform (sample.h) ----------------------------------------------------------
_M32 = 0xFFFFFFFF
_KAT = [  # Random123 known-answer vectors: counter, key → output
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((_M32,) * 4, (_M32,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


def _philox4(c, k):
    c0, c1, c2, c3 = c
    k0, k1 = k
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & _M32, p1 & _M32, ((p0 >> 32) ^ c3 ^ k1) & _M32, p0 & _M32
        k0, k1 = (k0 + 0x9E3779B9) & _M32, (k1 + 0xBB67AE85) & _M32
    return c0, c1, c2, c3


def _u64(seed, ctr, j, tag):
    o = _philox4((ctr & _M32, ctr >> 32, j, tag), (seed & _M32, seed >> 32))
    return o[0] | o[1] << 32


def _fix44(e):
    """The kernels' round(e·2^44) by fp32 magic adds (sample.cu weight()), emulated with IEEE fp32 torch ops on the CPU."""
    f = lambda v: torch.tensor(v, dtype=torch.float32)  # noqa: E731
    y = e * f(2.0 ** 22)
    h = y + f(2.0 ** 23)
    r = y - (h - f(2.0 ** 23))
    lo = (r * f(2.0 ** 22) + f(1.5 * 2.0 ** 23)).view(torch.int32).long() - 0x4B400000
    return ((h.view(torch.int32).long() - 0x4B000000) << 22) + lo


# ---- statistics ------------------------------------------------------------------------------------------------
def _gof(counts, probs):
    """Goodness of fit of bin counts to bin probabilities (Σ = 1) → (p-value, TV, noise √(bins / n))."""
    counts, probs = counts.double().cpu(), probs.double().cpu()
    n = counts.sum().item()
    e = probs * n
    big = e >= 5
    c, ee = counts[big], e[big]
    rc, re = counts[~big].sum().item(), e[~big].sum().item()
    p_tail = 1.0
    if re >= 5:
        c, ee = torch.cat([c, torch.tensor([rc], dtype=F64)]), torch.cat([ee, torch.tensor([re], dtype=F64)])
    elif rc > 0:  # P(Poisson(re) ≥ rc)
        p_tail = torch.special.gammainc(torch.tensor(rc, dtype=F64), torch.tensor(re, dtype=F64)).item() if re > 0 else 0.0
    df = len(c) - 1
    x2 = ((c - ee) ** 2 / ee).sum().item()
    p = torch.special.gammaincc(torch.tensor(df / 2, dtype=F64), torch.tensor(x2 / 2, dtype=F64)).item() if df > 0 else 1.0
    return min(p, p_tail), 0.5 * (counts / n - probs).abs().sum().item(), math.sqrt(len(counts) / n)


def _softmax(row, T):
    return torch.softmax(row.double() / T, -1)


def _bins(tokens, pos):
    """Bin index per token: position in `pos`, or len(pos) for every other token."""
    lut = torch.full((V,), len(pos), dtype=torch.long, device=tokens.device)
    lut[torch.as_tensor(pos, device=tokens.device)] = torch.arange(len(pos), device=tokens.device)
    return lut[tokens.long()]


def _bin_probs(p, pos):
    s = p[torch.as_tensor(pos, device=p.device)]
    return torch.cat([s, (1 - s.sum()).clamp_min(0).view(1)])


def _starts(ks, extra):
    s = [0]
    for k in ks:
        s.append(s[-1] + k + extra)
    return torch.tensor(s, dtype=torch.int32, device="cuda")


def _i64(gen, *shape):
    return torch.randint(0, 2 ** 62, shape, generator=gen).cuda()


# ---- greedy ----------------------------------------------------------------------------------------------------
def _argmax_rows(ctx, gen):
    """Rows with known torch.argmax behaviour: random, reference logits, ties (across chunk edges), ±inf, NaN, flat."""
    rnd = torch.randn(8, V, generator=gen) * 4
    tie = torch.randint(-40, 40, (8, V), generator=gen).float()  # the max repeats many times
    for i, (a, b) in enumerate([(15519, 15520), (0, V - 1), (124159, 124160), (31040, 3)]):
        tie[i, [a, b]] = 50.0
    spec = torch.randn(8, V, generator=gen)
    spec[0, [100, 7]] = INF
    spec[1] = -INF
    spec[2, :] = -INF
    spec[2, 200000] = -1e30
    spec[3, [9000, 300, 248000]] = float("nan")
    spec[4, 1234] = float("nan")
    spec[4, 1233] = INF
    spec[5] = 0.0
    spec[6, V - 1] = 100.0
    spec[7, 0] = 100.0
    spec[7, 1] = 100.0
    return torch.cat([rnd, tie, spec]).cuda(), _real_rows(ctx)[:24]


def test_argmax(ctx):
    """sample_rows at T = 0 == torch.argmax (first maximum, NaN = maximum) on random, tied, ±inf/NaN, flat and
    reference rows, at V = 248320 and small V, for cluster sizes 1, 3, 8, 16 and automatic; prob == 1."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(1)
    syn, real = _argmax_rows(ctx, gen)
    small = torch.randint(-3, 3, (64, 1000), generator=gen).float().cuda()
    bad, n = [], 0
    prev = m.force_cluster(0)
    try:
        for c in (0, 1, 3, 8, 16):
            m.force_cluster(c)
            for name, x in (("syn", syn), ("real", real), ("small", small)):
                R = x.shape[0]
                tok, prob = m.sample(x, torch.zeros(R, device="cuda"), torch.zeros(R, dtype=torch.long, device="cuda"),
                                     torch.zeros(R, dtype=torch.long, device="cuda"))
                ref = torch.argmax(x, -1).int()
                n += R
                if not torch.equal(tok[0], ref) or not bool((prob == 1).all()):
                    bad.append(f"C{c} {name} {(tok[0] != ref).sum().item()} rows")
    finally:
        m.force_cluster(prev)
    ctx.detail(f"sample argmax: {n} rows, bad {bad}")
    return {"ok": not bad, "msg": f"== torch.argmax {n} rows (random·ties·±inf·NaN·flat·reference logits "
            f"{real.shape[0]} rows, V=248320·1000, cluster auto·1·3·8·16)" if not bad
            else "differs: " + ", ".join(bad[:4])}


# ---- RNG layout and exact integer arithmetic ---------------------------------------------------------------------
def test_rng(ctx):
    """Philox KAT; the weight integerisation (fp32 magic adds) == round-half-even(e·2^44); flat rows: sample_rows token
    == ⌊U·V / 2^64⌋ with U(seed, ctr, 0, 0) and prob == fp32(1/V); spec_accept (k = 1) == the Python-integer model of
    sample.h: p flat over V, q flat over the first V/4 (accept iff ⌊U₁·B / 2^64⌋ < A, residual / bonus from
    U(seed, ctr, 0, 1)), and p = q flat over the first V/4 with drafts outside it (rejected, Σr = 0 → drawn from p)."""
    m = _mod(ctx)
    kat = sum(_philox4(c, k) == o for c, k, o in _KAT)
    gen = torch.Generator().manual_seed(2)
    # weight integerisation == round-half-even(e·2^44): uniform e, e = m·2^-k (k ≤ 60), exact ties (2n+1)·2^-45, 0, 1
    ties = (torch.randint(0, 1 << 20, (1 << 16,), generator=gen) * 2 + 1).double() * 2.0 ** -45
    e = torch.cat([torch.rand(1 << 20, generator=gen), torch.rand(1 << 20, generator=gen) *
                   torch.pow(2.0, -torch.randint(0, 61, (1 << 20,), generator=gen).float()),
                   ties.float(), torch.tensor([0.0, 1.0, 2.0 ** -45, 3 * 2.0 ** -46])])
    fix_bad = int((_fix44(e) != torch.round(e.double() * 2.0 ** 44).long()).sum())
    R, N = 8, 128
    seed, ctr = _i64(gen, R), _i64(gen, N, R)
    tok, prob = m.sample(torch.zeros(R, V, device="cuda"), torch.full((R,), 0.7, device="cuda"), seed, ctr)
    sl, cl, tl = seed.tolist(), ctr.tolist(), tok.tolist()
    bad_s = sum(tl[n][r] != (_u64(sl[r], cl[n][r], 0, 0) * V) >> 64 for n in range(N) for r in range(R))
    p_ok = bool((prob == torch.tensor(1 / V, dtype=torch.float32)).all())
    # spec_accept, B sequences of k = 1: q0 = 0 on [0, V/4), −inf elsewhere; p0 = p1 = 0 (flat) for b < B/2, = q0 for
    # the others, whose drafts lie outside [0, V/4)
    B, N2, H = 16, 64, V // 4
    pl = torch.zeros(2 * B, V, device="cuda")
    pl[B:, H:] = -INF
    ql = torch.full((B, V), -INF, device="cuda")
    ql[:, :H] = 0
    drafts = torch.randint(0, V, (N2, B), generator=gen, dtype=torch.int32)
    drafts[:, B // 2:] = torch.randint(H, V, (N2, B - B // 2), generator=gen, dtype=torch.int32)
    drafts = drafts.cuda()
    seed2, ctr2 = _i64(gen, B), _i64(gen, N2, B)
    n_acc, nxt = m.accept(pl, _starts([1] * B, 1), ql, _starts([1] * B, 0), drafts,
                          torch.full((B,), 1.3, device="cuda"), seed2, ctr2)
    one = 1 << 44
    Wq = H * one
    bad_a, acc, fb = 0, 0, 0
    for n, (dl, cl2, al, xl) in enumerate(zip(drafts.tolist(), ctr2.tolist(), n_acc.tolist(), nxt.tolist())):
        for b in range(B):
            s, c, d = seed2[b].item(), cl2[b], dl[b]
            u0 = _u64(s, c, 0, 1)
            if b >= B // 2:  # A = B = 0 → rejected, residual 0 → drawn from p (flat over [0, H))
                ref = (0, (u0 * Wq >> 64) // one)
                fb += 1
            else:
                Wp = V * one
                A, Bq = one * Wq, (one if d < H else 0) * Wp
                if (_u64(s, c, 1, 1) * Bq) >> 64 < A:
                    ref = (1, (u0 * Wp >> 64) // one)
                else:  # residual: one·Wq on [H, V), 0 below
                    ref = (0, H + (u0 * ((V - H) * one * Wq) >> 64) // (one * Wq))
            acc += ref[0]
            bad_a += (al[b], xl[b]) != ref
    ok = kat == 3 and bad_s == 0 and p_ok and bad_a == 0 and fix_bad == 0
    ctx.detail(f"sample rng: KAT {kat}/3, flat sample mismatches {bad_s}/{N * R}, prob ok {p_ok}, "
               f"accept mismatches {bad_a}/{N2 * B} (accepted {acc}), fix44 mismatches {fix_bad}/{e.numel()}")
    return {"ok": ok, "msg": f"Philox KAT {kat}/3 · flat-row token == ⌊U·V/2^64⌋ {N * R - bad_s}/{N * R} · "
            f"spec_accept == integer model {N2 * B - bad_a}/{N2 * B} (accepted {acc}, residual·bonus·Σr=0 {fb}) · "
            f"integerisation == round(e·2^44) {e.numel() - fix_bad}/{e.numel()}"}


# ---- sample_rows distribution -------------------------------------------------------------------------------
# across the CTA chunks of a 12-CTA cluster (edges 20692·c), warp runs, the vocab halves and both ends
SUPPORT = [0, 3, 20691, 20692, 41383, 62080, 99999, 124159, 124160, 186240, 248000, V - 1]


def _support_row(gen, rest):
    """Logits: N(0, 2²) on SUPPORT; elsewhere −inf ('inf') or U(−25, −12) ('neg', ~0.1–1 % of the mass)."""
    row = torch.full((V,), -INF) if rest == "inf" else torch.rand(V, generator=gen) * 13 - 25
    row[SUPPORT] = torch.randn(len(SUPPORT), generator=gen) * 2
    return row.cuda()


def test_dist(ctx):
    """sample_rows frequencies vs softmax(x / T) (float64) for a 12-token support inside V = 248320, rest −inf or
    U(−25, −12): 2^20 draws per case (16 identical rows × 65536 counters), T = 0.6 / 1.8 (−inf) and 1.0 / 1.4 (rest);
    chi-square over the support + rest bin; prob vs the float64 probability of the drawn support token."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(3)
    R, N = 16, 65536
    out, pmin, worst_rel, ok = [], 1.0, 0.0, True
    for i, (rest, T) in enumerate((("inf", 0.6), ("inf", 1.8), ("neg", 1.0), ("neg", 1.4))):
        row = _support_row(gen, rest)
        ctr = torch.arange(N * R, device="cuda").view(N, R) + (i << 40)
        tok, prob = m.sample(row.expand(R, V).contiguous(), torch.full((R,), T, device="cuda"), _i64(gen, R), ctr)
        p = _softmax(row, T)
        bins = _bins(tok.flatten(), SUPPORT)
        pv, tv, noise = _gof(torch.bincount(bins, minlength=len(SUPPORT) + 1), _bin_probs(p, SUPPORT))
        sup = bins < len(SUPPORT)
        rel = ((prob.flatten()[sup].double() - p[tok.flatten()[sup].long()]) / p[tok.flatten()[sup].long()]).abs().max().item()
        rest_mass = 1 - p[SUPPORT].sum().item()
        ctx.detail(f"sample dist rest={rest} T={T}: p={pv:.3g} TV={tv:.2e} (noise {noise:.1e}) rest mass {rest_mass:.2e} "
                   f"drawn {(~sup).sum().item()} prob rel {rel:.2e}")
        pmin, worst_rel = min(pmin, pv), max(worst_rel, rel)
        ok &= pv >= P_MIN and rel <= 1e-5
        out.append(f"T={T} TV {tv:.1e}")
    return {"ok": ok, "msg": f"2^20 draws×4: min p {pmin:.3f} · " + " · ".join(out) + f" (noise ~{noise:.0e}) · "
            f"prob rel err ≤ {worst_rel:.1e} (12-token support + rest −inf / −25…−12)"}


# ---- determinism and invariance ---------------------------------------------------------------------------------
def test_invariance(ctx):
    """(The base rule: P016 off; test_bv_invariance covers P016.) See _invariance."""
    with _Block(_mod(ctx), False):
        return _invariance(ctx)


def _invariance(ctx):
    """Bitwise: every row / sequence alone == in a permuted batch with others == repeated == cluster sizes 1, 2, 3, 5,
    8, 16. Rows: reference, random, flat, tied, small support; T ∈ {0, 0.5, 1, 1.3}; spec_accept with k = 0…4, T = 0
    and > 0, drafts from perturbed target rows (accepted) or from other steps' rows (rejected: residual path)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(4)
    real = _real_rows(ctx)
    rows = torch.cat([real[:6], (torch.randn(4, V, generator=gen) * 3).cuda(), torch.zeros(1, V, device="cuda"),
                      torch.randint(-5, 5, (2, V), generator=gen).float().cuda(), _support_row(gen, "neg")[None]])
    R = rows.shape[0]
    temp = torch.tensor([0, 0.5, 1, 1.3] * 4, device="cuda")[:R]
    seed, ctr = _i64(gen, R), _i64(gen, R)
    alone = [m.sample(rows[r:r + 1], temp[r:r + 1], seed[r:r + 1], ctr[r:r + 1]) for r in range(R)]
    at, ap = torch.cat([a[0] for a in alone], 1), torch.cat([a[1] for a in alone], 1)
    perm = torch.randperm(R, generator=gen).cuda()
    bad = []
    prev = m.force_cluster(0)
    try:
        for c in (0, 1, 2, 3, 5, 8, 16):
            m.force_cluster(c)
            for _ in range(2):
                t, p = m.sample(rows[perm], temp[perm], seed[perm], ctr[perm])
                if not (torch.equal(t[0], at[0, perm]) and torch.equal(p[0], ap[0, perm])):
                    bad.append(f"sample C{c}")
        # spec_accept: sequences of k = 0…4 drafts, reference rows as targets, perturbed copies as drafts (rejections
        # occur)
        ks = [0, 1, 2, 3, 4, 4, 3, 2, 1, 4]
        B = len(ks)
        P = torch.cat([real[(3 * b + torch.arange(k + 1)) % real.shape[0]] for b, k in enumerate(ks)])
        Q = torch.cat([real[(3 * b + torch.arange(k) + 7 * (b % 2)) % real.shape[0]] for b, k in enumerate(ks) if k])
        Q = Q + (torch.randn(Q.shape, generator=gen) * 1.5).cuda()  # odd b: another step's logits → rejections
        ts = torch.tensor([0.0 if b % 3 == 0 else 0.8 + 0.1 * b for b in range(B)], device="cuda")
        qt = torch.cat([ts[b].repeat(k) for b, k in enumerate(ks)])
        drafts = m.sample(Q, qt, _i64(gen, Q.shape[0]), _i64(gen, Q.shape[0]))[0][0]
        seed2, ctr2 = _i64(gen, B), _i64(gen, B)
        m.force_cluster(0)
        qs = _starts(ks, 0)
        ps = _starts(ks, 1)
        one = []
        for b in range(B):
            q0, q1 = int(qs[b]), int(qs[b + 1])
            one.append(m.accept(P[int(ps[b]):int(ps[b + 1])], _starts([ks[b]], 1), Q[q0:q1] if q1 > q0 else Q[:1],
                                _starts([ks[b]], 0), drafts[q0:q1].contiguous() if q1 > q0 else drafts[:0],
                                ts[b:b + 1], seed2[b:b + 1], ctr2[b:b + 1]))
        ref = torch.cat([o[0] for o in one], 1), torch.cat([o[1] for o in one], 1)
        order = torch.argsort(torch.rand(B, generator=gen)).tolist()
        ks2 = [ks[b] for b in order]
        P2 = torch.cat([P[int(ps[b]):int(ps[b + 1])] for b in order])
        Q2 = torch.cat([Q[int(qs[b]):int(qs[b + 1])] for b in order if ks[b]])
        d2 = torch.cat([drafts[int(qs[b]):int(qs[b + 1])] for b in order])
        o = torch.tensor(order, device="cuda")
        for c in (0, 1, 3, 8, 16):
            m.force_cluster(c)
            na, nx = m.accept(P2, _starts(ks2, 1), Q2, _starts(ks2, 0), d2, ts[o], seed2[o], ctr2[o])
            if not (torch.equal(na[0], ref[0][0, o]) and torch.equal(nx[0], ref[1][0, o])):
                bad.append(f"accept C{c}")
    finally:
        m.force_cluster(prev)
    acc = ref[0][0].tolist()
    ctx.detail(f"sample invariance: bad {bad}; accept n_acc {acc} k {ks}")
    return {"ok": not bad, "msg": f"alone == permuted batch == repeated == cluster 1·2·3·5·8·16, bitwise (sample {R} "
            f"rows T=0·0.5·1·1.3, accept {B} sequences k=0…4 accepted {'·'.join(map(str, acc))})" if not bad
            else "mismatch: " + ", ".join(bad[:4])}


# ---- spec_accept exactness ----------------------------------------------------------------------------------------
def _tkp(n, tk, tp):
    """Per-row truncation tensors (None when tk is None: no truncation)."""
    if tk is None:
        return {}
    return {"top_k": torch.full((n,), tk, dtype=torch.int32, device="cuda"),
            "top_p": torch.full((n,), tp, dtype=torch.float32, device="cuda")}


def _rounds(m, gen, pl, ql, T, B, n_iter, ctr0, tk=None, tp=None):
    """n_iter rounds of B sequences sharing target rows pl [k+1, V] and draft rows ql [k, V]: drafts drawn from q by
    sample_rows, then spec_accept; x1 = an independent draw from p_1 (second token when nothing was accepted); every
    draw and the accept truncated by (tk, tp) if given. Returns e0, e1 (first two emitted tokens) and n_acc, flattened
    over rounds × B."""
    k = ql.shape[0]
    Vr = pl.shape[1]
    P, Q = pl.repeat(B, 1), ql.repeat(B, 1)
    d, _ = m.sample(Q, torch.full((B * k,), T, device="cuda"), _i64(gen, B * k),
                    torch.arange(n_iter * B * k, device="cuda").view(n_iter, B * k) + ctr0, **_tkp(B * k, tk, tp))
    n, nx = m.accept(P, _starts([k] * B, 1), Q, _starts([k] * B, 0), d, torch.full((B,), T, device="cuda"),
                     _i64(gen, B), torch.arange(n_iter * B, device="cuda").view(n_iter, B) + ctr0, **_tkp(B, tk, tp))
    x1, _ = m.sample(pl[1:2].expand(B, Vr).contiguous(), torch.full((B,), T, device="cuda"), _i64(gen, B),
                     torch.arange(n_iter * B, device="cuda").view(n_iter, B) + ctr0 + (1 << 39), **_tkp(B, tk, tp))
    d = d.view(n_iter, B, k)
    e0 = torch.where(n >= 1, d[..., 0], nx)
    e1 = torch.where(n == 1, nx, x1)
    if k >= 2:
        e1 = torch.where(n >= 2, d[..., 1], e1)
    return e0.flatten(), e1.flatten(), n.flatten()


def _check_round(name, e0, e1, n_acc, p0, p1, q0, pos, joint, ctx):
    """p-values of e0 ~ p0 and (if joint) (e0, e1) ~ p0 ⊗ p1 over the bins `pos` + rest; acceptance rate of d_0 vs
    Σ min(p0, q0) (z ≤ 5)."""
    K = len(pos)
    b0 = _bins(e0, pos)
    pv, tv, noise = _gof(torch.bincount(b0, minlength=K + 1), _bin_probs(p0, pos))
    pj = 1.0
    if joint:
        bj = b0 * (K + 1) + _bins(e1, pos)
        pj, _, _ = _gof(torch.bincount(bj, minlength=(K + 1) ** 2),
                        torch.outer(_bin_probs(p0, pos), _bin_probs(p1, pos)).flatten())
    nt = n_acc.numel()
    rate, exp = (n_acc >= 1).double().mean().item(), torch.minimum(p0, q0).sum().item()
    z = abs(rate - exp) / math.sqrt(max(exp * (1 - exp), 1e-12) / nt)
    ctx.detail(f"sample accept {name}: n={nt} p(e0)={pv:.3g} TV={tv:.2e} (noise {noise:.1e}) p(e0,e1)={pj:.3g} "
               f"accept d0 {rate:.5f} vs {exp:.5f} (z {z:.2f})")
    return min(pv, pj), z


SMALL_V = 256
S_POS = [3, 40, 41, 100, 127, 128, 200, 255]
D_POS = [1, 2, 50, 51, 77, 180, 181, 254]


def _small_rows(gen, kind, k):
    """Target rows [k+1, 256] (N(0, 1.5²) on S_POS) and draft rows [k, 256] by kind."""
    def on(pos, vals):
        r = torch.full((SMALL_V,), -INF)
        r[pos] = vals
        return r
    pl = torch.stack([on(S_POS, torch.randn(8, generator=gen) * 1.5) for _ in range(k + 1)])
    if kind == "rand":  # also two tokens p never produces
        ql = torch.stack([on(S_POS + [10, 150], torch.randn(10, generator=gen) * 1.5) for _ in range(k)])
    elif kind == "q=p":
        ql = pl[:k].clone()
    elif kind == "disjoint":
        ql = torch.stack([on(D_POS, torch.randn(8, generator=gen) * 1.5) for _ in range(k)])
    else:  # one-hot on a token of p's support
        ql = torch.stack([on([S_POS[(3 * i + 1) % 8]], torch.zeros(1)) for i in range(k)])
    return pl.cuda(), ql.cuda()


def test_accept(ctx):
    """spec_accept exactness at V = 256 (support 8): 2^20 rounds per case, q random (+2 tokens outside p) / = p /
    disjoint / one-hot, k = 1 and 3 (T = 0.8; random also T = 1.5): first emitted token ~ p_0 and first two ~ p_0 ⊗ p_1
    (chi-square), d_0 acceptance rate vs Σ min(p_0, q_0)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(5)
    B, n_iter = 16384, 64
    pmin, zmax, n, ok = 1.0, 0.0, 0, True
    cases = [(kind, k, 0.8) for kind in ("rand", "q=p", "disjoint", "onehot") for k in (1, 3)] + [("rand", 3, 1.5)]
    for i, (kind, k, T) in enumerate(cases):
        pl, ql = _small_rows(gen, kind, k)
        with _Block(m, False):
            e0, e1, na = _rounds(m, gen, pl, ql, T, B, n_iter, i << 40)
        pv, z = _check_round(f"{kind} k={k} T={T}", e0, e1, na, _softmax(pl[0], T), _softmax(pl[1], T),
                             _softmax(ql[0], T), S_POS, True, ctx)
        pmin, zmax, n = min(pmin, pv), max(zmax, z), n + 1
        ok &= pv >= P_MIN and z <= 5
    return {"ok": ok, "msg": f"first token ~ p₀, first two ~ p₀⊗p₁: {n} cases × 2^20 rounds min p {pmin:.3f} · d₀ "
            f"acceptance rate |z| ≤ {zmax:.1f} (q random·=p·disjoint·one-hot, k=1·3, V=256)"}


def test_accept_full(ctx):
    """spec_accept exactness at V = 248320: (a) 12-token support, rest U(−25, −12), k = 3; (b) reference rows as
    targets, drafts = targets + N(0, 0.7²), k = 1 and 3; 2^15 rounds each (B = 32), first token over the top-15 /
    support bins + rest, d_0 acceptance rate."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(6)
    real = _real_rows(ctx)
    B, n_iter = 32, 1024
    pmin, zmax, out, ok = 1.0, 0.0, [], True
    for i, (kind, k) in enumerate((("support", 3), ("ref", 1), ("ref", 3))):
        if kind == "support":
            pl = torch.stack([_support_row(gen, "neg") for _ in range(k + 1)])
            ql = torch.stack([_support_row(gen, "neg") for _ in range(k)])
            pos = SUPPORT
        else:
            pl = real[2 * i:2 * i + k + 1].contiguous()
            ql = (pl[:k] + (torch.randn(k, V, generator=gen) * 0.7).cuda()).contiguous()
            pos = torch.topk(pl[0], 15).indices.tolist()
        with _Block(m, False):
            e0, e1, na = _rounds(m, gen, pl, ql, 1.0, B, n_iter, (i + 16) << 40)
        pv, z = _check_round(f"full {kind} k={k}", e0, e1, na, _softmax(pl[0], 1.0), _softmax(pl[1], 1.0),
                             _softmax(ql[0], 1.0), pos, False, ctx)
        pmin, zmax = min(pmin, pv), max(zmax, z)
        ok &= pv >= P_MIN and z <= 5
        out.append(f"{kind} k={k} acceptance {(na >= 1).float().mean().item():.2f}")
    return {"ok": ok, "msg": f"V=248320 first token ~ p₀: 3 cases × 2^15 rounds min p {pmin:.3f} · |z| ≤ {zmax:.1f} · "
            + " · ".join(out)}


def test_greedy(ctx):
    """T = 0: n_acc / next == torch reference (accept while d_i == argmax p_i, next = argmax of the first mismatching
    row or row k): V = 248320 with ties (draft = the later of two equal maxima → rejected) and V = 64 × 4096 sequences."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(7)
    bad, n = 0, 0
    for Vr, B, kmax, scale in ((V, 12, 4, 3.0), (64, 4096, 6, 0.0)):
        ks = torch.randint(0, kmax + 1, (B,), generator=gen).tolist()
        P = (torch.randn(sum(ks) + B, Vr, generator=gen) * scale if scale else
             torch.randint(-3, 3, (sum(ks) + B, Vr), generator=gen).float())
        if Vr == V:  # every 3rd target row: the max twice
            for r in range(0, P.shape[0], 3):
                a, b = sorted(torch.randint(0, V, (2,), generator=gen).tolist())
                P[r, [a, b]] = 50.0
        am = P.argmax(-1)
        drafts, ref = [], []
        r = 0
        for b, k in enumerate(ks):
            j = int(torch.randint(0, k + 1, (1,), generator=gen))  # accepted prefix
            ds = []
            for i in range(k):
                tw = (P[r + i] == P[r + i].max()).nonzero().flatten()  # tied maxima: draft the later one → rejected
                ds.append(int(am[r + i]) if i < j else int(tw[-1]) if i == j and len(tw) > 1 else
                          int(torch.randint(0, Vr, (1,), generator=gen)))
            nacc = next((i for i in range(k) if ds[i] != int(am[r + i])), k)
            ref.append((nacc, int(am[r + nacc])))
            drafts += ds
            r += k + 1
        Q = torch.zeros(max(sum(ks), 1), Vr)
        dt = torch.tensor(drafts or [0], dtype=torch.int32).cuda()[:sum(ks)]
        na, nx = m.accept(P.cuda(), _starts(ks, 1), Q.cuda(), _starts(ks, 0), dt.contiguous(),
                          torch.zeros(B, device="cuda"), _i64(gen, B), _i64(gen, B))
        got = list(zip(na[0].tolist(), nx[0].tolist()))
        bad += sum(g != e for g, e in zip(got, ref))
        n += B
    return {"ok": bad == 0, "msg": f"T=0 n_acc·next == torch reference {n - bad}/{n} (V=248320 12 sequences with ties, "
            f"V=64 4096 sequences, k=0…6)"}


def test_perf(ctx):
    """µs per call at V = 248320, inputs cycled over sets ≫ L2 (rows from DRAM), best of 5. argmax and T = 1 sampling at
    R = 1 and 16 (µs per row): a CUDA graph of 100 back-to-back calls. spec_accept B = 1, 4 × k = 4 with every draft
    accepted (2k + 1 rows read, then the bonus draw): one call alone (graph launch included — in a decode step it never
    runs back to back), and B = 1 with its 9 rows L2-resident (the same rows every call: the verify rows have just been
    written when it runs). Pass: R = 1 argmax ≤ 11 µs, sampling ≤ 30 µs."""
    import time
    m = _mod(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench(0, V, 16, 0, 1.0, 4, 50, 1)
    res = {}
    for name, R in (("argmax", 1), ("argmax", 16), ("sample", 1), ("sample", 16)):
        res[(name, R)] = m.bench(0, V, R, 0, 0.0 if name == "argmax" else 1.0, max(2, 96 // R), 100, 5)
    with _Block(m, False):
        for B in (1, 4):
            sets = max(2, -(-96 // (9 * B)))
            res[("accept", B)] = m.bench(1, V, B, 4, 1.0, sets, 1, 5)
        res[("accept", "l2")] = m.bench(1, V, 1, 4, 1.0, 1, 100, 5)
    ctx.detail("sample perf (us per call): " + ", ".join(f"{k[0]} {k[1]}: {v:.2f}" for k, v in res.items()))
    ok = res[("argmax", 1)] <= 11 and res[("sample", 1)] <= 30
    return {"ok": ok, "msg": f"argmax {res[('argmax', 1)]:.1f} µs/row (R=16 {res[('argmax', 16)] / 16:.1f}) · sample "
            f"{res[('sample', 1)]:.1f} µs/row (R=16 {res[('sample', 16)] / 16:.1f}) · accept k=4 B=1 "
            f"{res[('accept', 1)]:.0f} µs (in L2 {res[('accept', 'l2')]:.0f}) · B=4 {res[('accept', 4)]:.0f} µs "
            f"(all accepted, DRAM rows, graph best)"}


# ---- truncation (sample.h): rank keys, the reference cut, vLLM's top-k / top-p -------------------------------------
Z_TOP = 1 << 52
# (top_k, top_p) settings: defaults, k = 1, k ≥ V, p tiny / ≤ 0 / NaN, large k (general path), top-p only
TK_SETS = [(20, 0.95), (-1, 0.95), (50, 0.8), (1, 1.0), (V, 1.0), (V + 5, 0.95), (3, 1e-9), (-1, 1e-30), (-1, 0.0),
           (1000, 0.99), (5000, 1.0), (-1, 0.5), (2, 0.5), (0, 1.0), (7, float("nan"))]


def _zkeys(x):
    """Rank keys Z_v = key(x_v)·2^20 + 2^20 − 1 − v of fp32 rows x [..., V] (int64; key: sign set → ~bits, else
    bits | 2^31; −0 as +0)."""
    b = x.contiguous().view(torch.int32).long() & _M32
    b = torch.where(x == 0, torch.zeros_like(b), b)
    k = torch.where(b >= 1 << 31, b ^ _M32, b | 1 << 31)
    return k << 20 | (0xFFFFF - torch.arange(x.shape[-1], device=x.device))


def _ref_cut(z, w, k, p):
    """sample.h's truncation of one row straight from the definition, given its keys z and the kernels' integer weights
    w (int64 [V]): sort by Z, the first k, the first prefix ≥ ⌈p·W_K⌉ (Python integers) → (cut z, W', kept count)."""
    n = z.numel()
    zs, order = torch.sort(z, descending=True)
    p = float(torch.tensor(p, dtype=torch.float32))
    kk = k if 0 < k < n else n
    if p <= 0:
        kk, p = 1, 1.0
    cum = torch.cumsum(w[order[:kk]], 0)
    j = kk - 1
    if p < 1:
        num, den = p.as_integer_ratio()
        j = int(torch.searchsorted(cum, torch.tensor([-(-num * int(cum[-1]) // den)], device=cum.device)))
    return int(zs[j]), int(cum[j]), j + 1


def _trunc_probs(m, x, T, tk, tp):
    """Truncated distribution w'/W' of one row (float64 [V]) from the kernels' weights and the reference cut; its kept
    mask, the weights and W'."""
    w = m.weights(x[None].contiguous(), torch.tensor([T], device="cuda"))[0]
    z = _zkeys(x)
    zc, W, _ = _ref_cut(z, w, tk, tp)
    keep = z >= zc
    return torch.where(keep, w, 0).double() / W, keep, w, W


def _vllm_keep(x, T, k, p):
    """vLLM v1's apply_top_k_top_p (vllm/v1/sample/ops/topk_topp_sampler.py) on fp32 logits / T → kept mask [R, V]:
    ascending sort, mask below the k-th largest, softmax, cumsum, mask cumsum ≤ 1 − p, always keep the last."""
    lg = x / T[:, None]
    srt, idx = lg.sort(dim=-1, descending=False)
    if k is not None:
        kth = srt.gather(1, (srt.shape[1] - k.long()).unsqueeze(1))
        srt = srt.masked_fill(srt < kth, -INF)
    if p is not None:
        cs = srt.softmax(dim=-1).cumsum(dim=-1)
        mask = cs <= 1 - p.unsqueeze(1)
        mask[:, -1] = False
        srt = srt.masked_fill(mask, -INF)
    return srt.scatter(-1, idx, srt) > -INF


def test_trunc(ctx):
    """Cut and W' (sample_cut) == the reference of sample.h's definition on reference rows, N(0, 3²), small-integer
    ties, a 12-token support (rest −inf / U(−25, −12)) and flat rows at T = 0.6 / 1 / 1.5, for TK_SETS (k = 1, k ≥ V,
    p tiny, p ≤ 0, p = 1, NaN, large k …), also at V = 4096; default cap and cap 16 (every row through refinement
    passes), clusters automatic / 1 / 5. Degenerate rows (+inf, NaN, all −inf): cut 2^52, W' = 0, the draw = argmax."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(8)
    real = _real_rows(ctx)
    rows = torch.cat([real[:24], (torch.randn(8, V, generator=gen) * 3).cuda(),
                      torch.randint(-6, 6, (8, V), generator=gen).float().cuda(),
                      torch.stack([_support_row(gen, "inf"), _support_row(gen, "neg")]), torch.zeros(2, V, device="cuda")])
    small = (torch.randn(8, 4096, generator=gen) * 2).cuda()
    bad, n = [], 0
    prev_c, prev_cap = m.force_cluster(0), m.force_cap(0)
    try:
        for x in (rows, small):
            R, Vx = x.shape
            temp = torch.tensor([0.6, 1.0, 1.5], device="cuda").repeat(R)[:R].contiguous()
            w, z = m.weights(x, temp), _zkeys(x)
            for tk, tp in TK_SETS:
                ref = [_ref_cut(z[r], w[r], tk, tp) for r in range(R)]
                for c, cap in ((0, 0), (1, 0), (5, 16), (0, 16)):
                    m.force_cluster(c), m.force_cap(cap)
                    cut = m.cut(x, temp, **_tkp(R, tk, tp))
                    cnt = (z >= cut[:, :1]).sum(1).tolist()
                    for r, (cr, (zc, W)) in enumerate(zip(cnt, cut.tolist())):
                        n += 1
                        if (cr, W) != (ref[r][2], ref[r][1]):
                            bad.append(f"V{Vx} r{r} ({tk},{tp}) C{c} cap{cap}: {cr}/{W} vs {ref[r][2]}/{ref[r][1]}")
        m.force_cluster(0), m.force_cap(0)
        deg = torch.randn(3, V, generator=gen).cuda()
        deg[0, 5], deg[1, 7], deg[2] = INF, float("nan"), -INF
        one = torch.ones(3, device="cuda")
        for tk, tp in ((20, 0.95), (-1, 0.9), (0, 1.0)):
            cut = m.cut(deg, one, **_tkp(3, tk, tp)).tolist()
            tok, prob = m.sample(deg, one, torch.zeros(3, dtype=torch.long, device="cuda"),
                                 torch.zeros(3, dtype=torch.long, device="cuda"), **_tkp(3, tk, tp))
            if any(c != [Z_TOP, 0] for c in cut) or tok[0].tolist() != [5, 7, 0] or not bool((prob == 1).all()):
                bad.append(f"degenerate ({tk},{tp}): {cut} {tok[0].tolist()}")
    finally:
        m.force_cluster(prev_c), m.force_cap(prev_cap)
    ctx.detail(f"sample trunc: {n} cuts, bad {bad[:20]}")
    return {"ok": not bad, "msg": f"cut·W' == definition {n - len(bad)}/{n} (reference·random·ties·support·flat rows, "
            f"V=248320·4096, T=0.6·1·1.5, (k,p) {len(TK_SETS)} settings, cap default·16, cluster auto·1·5) · "
            f"degenerate rows cut 2^52·argmax" if not bad else "differs: " + ", ".join(bad[:3])}


def test_trunc_vllm(ctx):
    """Kept sets == vLLM's apply_top_k_top_p (reimplemented from its source) on every reference decode row at T = 0.7 /
    1 / 1.3 / 2 for (20, 0.95), (−1, 0.95), (50, 0.8). A difference is a tie if every token in only one of the two sets
    has logit / T equal to our last kept one (vLLM keeps every token tied with the k-th and sorts ties arbitrarily), an
    fp32 boundary if the float64 cumulative probability at our last or second-to-last kept token is within 1e-5 of p
    (vLLM's fp32 cumsum decides); any other difference fails."""
    m = _mod(ctx)
    real = _real_rows(ctx)
    R = real.shape[0]
    same = ties = bnd = 0
    other, sizes, tie_at = [], {}, {}
    z = _zkeys(real)
    for T in (0.7, 1.0, 1.3, 2.0):
        temp = torch.full((R,), T, device="cuda")
        for tk, tp in ((20, 0.95), (-1, 0.95), (50, 0.8)):
            ours = z >= m.cut(real, temp, **_tkp(R, tk, tp))[:, :1]
            ref = _vllm_keep(real, temp, torch.full((R,), tk, device="cuda") if tk > 0 else None,
                             torch.full((R,), tp, device="cuda"))
            n = ours.sum(1)
            sizes[(tk, tp)] = sizes.get((tk, tp), []) + n.tolist()
            for r in range(R):
                if torch.equal(ours[r], ref[r]):
                    same += 1
                    continue
                lg = real[r] / temp[r]
                if bool((lg[ours[r] ^ ref[r]] == lg[ours[r]].min()).all()):
                    ties += 1
                    tie_at[(T, tk, tp)] = tie_at.get((T, tk, tp), 0) + 1
                    continue
                topk = torch.zeros(V, dtype=torch.bool, device="cuda")
                topk[torch.sort(z[r], descending=True).indices[:tk if tk > 0 else V]] = True
                pr = torch.softmax(torch.where(topk, lg.double(), -INF), -1)
                cum = torch.cumsum(torch.sort(pr, descending=True).values, 0)
                j = int(n[r]) - 1
                if abs(cum[j].item() - tp) < 1e-5 or (j > 0 and abs(cum[j - 1].item() - tp) < 1e-5):
                    bnd += 1
                else:
                    other.append(f"T{T} ({tk},{tp}) r{r}: ours {int(n[r])} vLLM {int(ref[r].sum())}")
    tot = same + ties + bnd + len(other)
    rng = " · ".join(f"({k},{p}) {min(v)}…{max(v)}" for (k, p), v in sizes.items())
    ctx.detail(f"sample trunc vllm: same {same}/{tot} ties {ties} {tie_at} boundary {bnd} other {other[:20]}; "
               f"kept sizes {rng}")
    return {"ok": not other, "msg": f"same set as vLLM apply_top_k_top_p {same}/{tot} (reference {R} rows × "
            f"T=0.7·1·1.3·2 × (20,0.95)·(−1,0.95)·(50,0.8)) · ties {ties} · fp32 boundary {bnd} · other {len(other)} · "
            f"kept {rng}"}


def test_trunc_dist(ctx):
    """sample_rows frequencies vs the truncated distribution w'/W' (reference cut on the kernels' weights), 2^20 draws per
    case (16 identical rows × 65536 counters): reference row (20, 0.95) and (−1, 0.95), N(0, 3²) row (−1, 0.9)
    (thousands kept: general path), support row (5, 0.9), reference row at T = 0.8 with (50, 0.8) and cap 8 (refinement
    passes, masked pick). Chi-square over the top-15 kept + the rest; no draw outside the kept set; prob ==
    fp32(w_v / W') bitwise."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(9)
    real = _real_rows(ctx)
    R, N = 16, 65536
    cases = [("ref", real[3], 1.0, 20, 0.95, 0), ("ref", real[11], 1.0, -1, 0.95, 0),
             ("randn", (torch.randn(V, generator=gen) * 3).cuda(), 1.0, -1, 0.9, 0),
             ("support", _support_row(gen, "neg"), 1.0, 5, 0.9, 0), ("ref cap8", real[17], 0.8, 50, 0.8, 8)]
    pmin, ok, out = 1.0, True, []
    prev = m.force_cap(0)
    try:
        for i, (name, row, T, tk, tp, cap) in enumerate(cases):
            m.force_cap(cap)
            p, keep, w, W = _trunc_probs(m, row, T, tk, tp)
            ctr = torch.arange(N * R, device="cuda").view(N, R) + ((i + 32) << 40)
            tok, prob = m.sample(row.expand(R, V).contiguous(), torch.full((R,), T, device="cuda"), _i64(gen, R), ctr,
                                 **_tkp(R, tk, tp))
            tok = tok.flatten().long()
            nk = int(keep.sum())
            pos = torch.topk(p, min(15, nk)).indices.tolist()
            pv, tv, noise = _gof(torch.bincount(_bins(tok, pos), minlength=len(pos) + 1), _bin_probs(p, pos))
            outside = int((~keep[tok]).sum())
            pbad = int((prob.flatten() != (w[tok].double() / W).float()).sum())
            ctx.detail(f"sample trunc dist {name} T={T} ({tk},{tp}) cap {cap}: kept {nk} p={pv:.3g} TV={tv:.2e} "
                       f"(noise {noise:.1e}) outside {outside} prob mismatches {pbad}")
            pmin = min(pmin, pv)
            ok &= pv >= P_MIN and outside == 0 and pbad == 0
            out.append(f"{name} ({tk},{tp}) {nk} kept")
    finally:
        m.force_cap(prev)
    return {"ok": ok, "msg": f"2^20 draws×{len(cases)}: min p {pmin:.3f} · outside the kept set 0 · prob == w/W' "
            "bitwise (" + " · ".join(out) + ")"}


def test_trunc_accept(ctx):
    """spec_accept exactness with truncation (p and q truncated with the same parameters): V = 256 (support 8), 2^20
    rounds each — q random (+2 tokens outside p) with (3, 1) / (−1, 0.7) / (5, 0.8), q one-hot on p's 4th token with
    (2, 1) (every draft outside p's kept set: never accepted), q on p's ranks 1, 5, 7 with (4, 0.9): first token ~ p₀',
    first two ~ p₀'⊗p₁', no token outside the kept sets, d₀ acceptance rate vs Σ min(p₀', q₀'); V = 248320, reference
    targets, drafts = targets + N(0, 0.7²), (20, 0.95) and (−1, 0.9), k = 1, 2^15 rounds: first token ~ p₀'."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(10)
    pmin, zmax, n, ok, outs = 1.0, 0.0, 0, True, 0
    cases = [("rand", 1, 0.8, 3, 1.0), ("rand", 3, 1.2, -1, 0.7), ("rand", 2, 1.0, 5, 0.8), ("4th", 1, 1.0, 2, 1.0),
             ("mix", 2, 0.9, 4, 0.9)]
    for i, (kind, k, T, tk, tp) in enumerate(cases):
        pl, ql = _small_rows(gen, "rand", k)
        if kind != "rand":
            ranks = torch.argsort(pl, -1, descending=True)
            ql = torch.full((k, SMALL_V), -INF, device="cuda")
            for j in range(k):
                pick = ranks[j, [3]] if kind == "4th" else ranks[j, [0, 4, 6]]
                ql[j, pick] = torch.randn(len(pick), generator=gen).cuda() if kind == "mix" else 0.0
        with _Block(m, False):
            e0, e1, na = _rounds(m, gen, pl, ql, T, 16384, 64, (i + 48) << 40, tk, tp)
        p0, k0, _, _ = _trunc_probs(m, pl[0], T, tk, tp)
        p1, k1, _, _ = _trunc_probs(m, pl[1], T, tk, tp)
        q0, _, _, _ = _trunc_probs(m, ql[0], T, tk, tp)
        pv, z = _check_round(f"trunc {kind} k={k} T={T} ({tk},{tp})", e0, e1, na, p0, p1, q0, S_POS, True, ctx)
        out = int((~k0[e0.long()]).sum() + (~k1[e1.long()]).sum())
        pmin, zmax, n, outs = min(pmin, pv), max(zmax, z), n + 1, outs + out
        ok &= pv >= P_MIN and z <= 5 and out == 0
    real = _real_rows(ctx)
    for i, (tk, tp) in enumerate(((20, 0.95), (-1, 0.9))):
        pl = real[5 + 2 * i:7 + 2 * i].contiguous()
        ql = (pl[:1] + (torch.randn(1, V, generator=gen) * 0.7).cuda()).contiguous()
        with _Block(m, False):
            e0, e1, na = _rounds(m, gen, pl, ql, 1.0, 32, 1024, (i + 56) << 40, tk, tp)
        p0, k0, _, _ = _trunc_probs(m, pl[0], 1.0, tk, tp)
        p1, _, _, _ = _trunc_probs(m, pl[1], 1.0, tk, tp)
        q0, _, _, _ = _trunc_probs(m, ql[0], 1.0, tk, tp)
        pos = torch.topk(p0, min(15, int(k0.sum()))).indices.tolist()
        pv, z = _check_round(f"trunc full ({tk},{tp})", e0, e1, na, p0, p1, q0, pos, False, ctx)
        out = int((~k0[e0.long()]).sum())
        pmin, zmax, n, outs = min(pmin, pv), max(zmax, z), n + 1, outs + out
        ok &= pv >= P_MIN and z <= 5 and out == 0
    return {"ok": ok, "msg": f"truncated p₀'·p₀'⊗p₁' distributions: {n} cases min p {pmin:.3f} · d₀ acceptance rate "
            f"|z| ≤ {zmax:.1f} · outside the kept set {outs} (V=256 2^20 rounds: q random·one-hot on p's 4th (always "
            f"rejected)·mixed, (3,1)·(−1,.7)·(5,.8)·(2,1)·(4,.9); V=248320 reference 2^15 rounds (20,.95)·(−1,.9))"}


# mixed per-row parameters for the invariance test (T = 0 rows ignore them)
TK_MIX = [(20, 0.95), (-1, 0.95), (50, 0.8), (1, 1.0), (0, 1.0), (V, 0.5), (3, 0.0), (1000, 0.99), (-1, 0.3), (7, 1.0)]


def test_trunc_invariance(ctx):
    """(The base rule: P016 off; test_bv_invariance covers P016.) See _trunc_invariance."""
    with _Block(_mod(ctx), False):
        return _trunc_invariance(ctx)


def _trunc_invariance(ctx):
    """Bitwise with per-row top_k / top_p mixed (TK_MIX): every row alone == in a permuted batch == repeated == cluster
    sizes 1, 2, 3, 5, 8, 16 == cap 8 (general path) — sample_rows over reference / random / flat / tied / support rows
    at T ∈ {0, 0.5, 1, 1.3} (T = 0: torch.argmax whatever the parameters); spec_accept with k = 0…4 and mixed
    parameters, drafts drawn with them from perturbed (accepted) or other steps' (rejected) rows."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(11)
    real = _real_rows(ctx)
    rows = torch.cat([real[:6], (torch.randn(4, V, generator=gen) * 3).cuda(), torch.zeros(1, V, device="cuda"),
                      torch.randint(-5, 5, (2, V), generator=gen).float().cuda(), _support_row(gen, "neg")[None]])
    R = rows.shape[0]
    temp = torch.tensor([0, 0.5, 1, 1.3] * 4, device="cuda")[:R]
    tk = torch.tensor([TK_MIX[r % len(TK_MIX)][0] for r in range(R)], dtype=torch.int32, device="cuda")
    tp = torch.tensor([TK_MIX[r % len(TK_MIX)][1] for r in range(R)], dtype=torch.float32, device="cuda")
    seed, ctr = _i64(gen, R), _i64(gen, R)
    alone = [m.sample(rows[r:r + 1], temp[r:r + 1], seed[r:r + 1], ctr[r:r + 1], tk[r:r + 1], tp[r:r + 1])
             for r in range(R)]
    at, ap = torch.cat([a[0] for a in alone], 1), torch.cat([a[1] for a in alone], 1)
    g0 = temp == 0
    bad = [] if torch.equal(at[0][g0], torch.argmax(rows[g0], -1).int()) else ["greedy"]
    perm = torch.randperm(R, generator=gen).cuda()
    prev_c, prev_cap = m.force_cluster(0), m.force_cap(0)
    try:
        for c, cap in [(c, 0) for c in (0, 1, 2, 3, 5, 8, 16)] + [(0, 8), (3, 8)]:
            m.force_cluster(c), m.force_cap(cap)
            for _ in range(2):
                t, p = m.sample(rows[perm], temp[perm], seed[perm], ctr[perm], tk[perm], tp[perm])
                if not (torch.equal(t[0], at[0, perm]) and torch.equal(p[0], ap[0, perm])):
                    bad.append(f"sample C{c} cap{cap}")
        m.force_cluster(0), m.force_cap(0)
        ks = [0, 1, 2, 3, 4, 4, 3, 2, 1, 4]
        B = len(ks)
        P = torch.cat([real[(3 * b + torch.arange(k + 1)) % real.shape[0]] for b, k in enumerate(ks)])
        Q = torch.cat([real[(3 * b + torch.arange(k) + 7 * (b % 2)) % real.shape[0]] for b, k in enumerate(ks) if k])
        Q = Q + (torch.randn(Q.shape, generator=gen) * 1.5).cuda()
        ts = torch.tensor([0.0 if b % 4 == 0 else 0.8 + 0.1 * b for b in range(B)], device="cuda")
        bk = torch.tensor([TK_MIX[(b + 1) % len(TK_MIX)][0] for b in range(B)], dtype=torch.int32, device="cuda")
        bp = torch.tensor([TK_MIX[(b + 1) % len(TK_MIX)][1] for b in range(B)], dtype=torch.float32, device="cuda")
        rep = lambda t: torch.cat([t[b].repeat(k) for b, k in enumerate(ks)])  # noqa: E731
        drafts = m.sample(Q, rep(ts), _i64(gen, Q.shape[0]), _i64(gen, Q.shape[0]), rep(bk), rep(bp))[0][0]
        seed2, ctr2 = _i64(gen, B), _i64(gen, B)
        qs, ps = _starts(ks, 0), _starts(ks, 1)
        one = []
        for b in range(B):
            q0, q1 = int(qs[b]), int(qs[b + 1])
            one.append(m.accept(P[int(ps[b]):int(ps[b + 1])], _starts([ks[b]], 1), Q[q0:q1] if q1 > q0 else Q[:1],
                                _starts([ks[b]], 0), drafts[q0:q1].contiguous() if q1 > q0 else drafts[:0],
                                ts[b:b + 1], seed2[b:b + 1], ctr2[b:b + 1], bk[b:b + 1], bp[b:b + 1]))
        ref = torch.cat([o[0] for o in one], 1), torch.cat([o[1] for o in one], 1)
        order = torch.argsort(torch.rand(B, generator=gen)).tolist()
        ks2 = [ks[b] for b in order]
        P2 = torch.cat([P[int(ps[b]):int(ps[b + 1])] for b in order])
        Q2 = torch.cat([Q[int(qs[b]):int(qs[b + 1])] for b in order if ks[b]])
        d2 = torch.cat([drafts[int(qs[b]):int(qs[b + 1])] for b in order])
        o = torch.tensor(order, device="cuda")
        for c, cap in [(c, 0) for c in (0, 1, 3, 8, 16)] + [(0, 8)]:
            m.force_cluster(c), m.force_cap(cap)
            na, nx = m.accept(P2, _starts(ks2, 1), Q2, _starts(ks2, 0), d2, ts[o], seed2[o], ctr2[o], bk[o], bp[o])
            if not (torch.equal(na[0], ref[0][0, o]) and torch.equal(nx[0], ref[1][0, o])):
                bad.append(f"accept C{c} cap{cap}")
    finally:
        m.force_cluster(prev_c), m.force_cap(prev_cap)
    acc = ref[0][0].tolist()
    ctx.detail(f"sample trunc invariance: bad {bad}; accept n_acc {acc} k {ks}")
    return {"ok": not bad, "msg": f"(k,p) mixed per row: alone == permuted batch == repeated == cluster 1·2·3·5·8·16 "
            f"== cap 8, bitwise (sample {R} rows T=0·0.5·1·1.3, accept {B} sequences k=0…4 accepted "
            f"{'·'.join(map(str, acc))})" if not bad else "mismatch: " + ", ".join(bad[:4])}


def test_trunc_perf(ctx):
    """µs per call at V = 248320 as test_perf (a CUDA graph of 100 calls, rows cycled over sets ≫ L2, best of 5): R = 1
    sampling with (20, 0.95) on N(0, 3²) and reference rows, (20, 1) and (−1, 0.95) on reference rows, (−1, 0.95) on
    N(0, 3²) rows (≈ 2·10^4 kept: general path); R = 16 (20, 0.95) reference (µs per row); spec_accept B = 1, k = 4,
    every draft accepted, (20, 0.95), reference rows, one call alone. Pass: (20, 0.95) ≤ 30 µs, (−1, 0.95) ≤ 150 µs,
    accept ≤ 200 µs."""
    import time
    m = _mod(ctx)
    real = _real_rows(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench(0, V, 16, 0, 1.0, 4, 50, 1, 20, 0.95, real)
    res = {}
    for name, tk, tp, src in (("k20p95", 20, 0.95, None), ("k20p95 ref", 20, 0.95, real), ("k20 ref", 20, 1.0, real),
                              ("p95 ref", 0, 0.95, real), ("p95", 0, 0.95, None)):
        res[name] = m.bench(0, V, 1, 0, 1.0, 96, 100, 5, tk, tp, src)
    res["R16"] = m.bench(0, V, 16, 0, 1.0, 6, 100, 5, 20, 0.95, real) / 16
    with _Block(m, False):
        res["accept"] = m.bench(1, V, 1, 4, 1.0, 11, 1, 5, 20, 0.95, real)
    ctx.detail("sample trunc perf (us per call): " + ", ".join(f"{k}: {v:.2f}" for k, v in res.items()))
    ok = max(res["k20p95"], res["k20p95 ref"]) <= 30 and max(res["p95"], res["p95 ref"]) <= 150 and res["accept"] <= 200
    return {"ok": ok, "msg": f"(20,.95) {res['k20p95 ref']:.1f} µs/row (random {res['k20p95']:.1f}, R=16 "
            f"{res['R16']:.1f}) · (20,1) {res['k20 ref']:.1f} · (−1,.95) {res['p95 ref']:.1f} (random "
            f"{res['p95']:.1f}) · accept k=4 B=1 (20,.95) {res['accept']:.0f} µs (reference rows, DRAM, graph best)"}


# ---- P016 block_verify: the block rule in Python integers (sample.h) ----------------------------------------------
BV_MAX = 15


def _wrows(m, rows, temp, tk=None, tp=None):
    """spec_accept's integer weights of rows [n, V] (temp [n]; tk / tp [n] tensors or None): w' int64 [n, V] on the CPU
    (0 off the kept set, and on a degenerate row — its maximum in torch.argmax order not finite), W' (ints), argmax."""
    w = m.weights(rows, temp)
    am = torch.argmax(rows, -1)
    deg = ~torch.isfinite(rows.gather(1, am[:, None])[:, 0])
    w[deg] = 0
    if tk is not None:
        cut = m.cut(rows, temp, top_k=tk, top_p=tp)
        w = torch.where(_zkeys(rows) >= cut[:, :1], w, torch.zeros_like(w))
    w = w.cpu()
    return w, w.sum(1).tolist(), am.tolist()


def _draw_w(w, W, am, U):
    """a row's draw (sample.h): t = ⌊U·W / 2^64⌋, inverse CDF of w in index order; W = 0 → the argmax am"""
    if W == 0:
        return am
    return int(torch.searchsorted(torch.cumsum(w, 0), torch.tensor([(U * W) >> 64]), right=True))


def _pos(al, be, wp, wq):
    """r_v = al·wp_v − be·wq_v > 0 (bool [V]): float64 with the near-ties settled in Python integers"""
    if al == 0:
        return torch.zeros(wp.shape, dtype=torch.bool)
    if be == 0:
        return wp > 0
    tf = al / be
    a, b = wp.double(), wq.double()
    d = a * tf - b
    pos = (d > 0) | ((wq == 0) & (wp > 0))
    for j in ((d.abs() <= 1e-9 * (a * tf + b)) & (wp > 0) & (wq > 0)).nonzero().flatten().tolist():
        pos[j] = al * int(wp[j]) - be * int(wq[j]) > 0
    return pos


def _resid(al, be, wp, wq):
    """num = Σ_v max(al·wp_v − be·wq_v, 0) and the prefix sums of wp, wq over the tokens with r > 0"""
    pos = _pos(al, be, wp, wq)
    z = torch.zeros_like(wp)
    cp, cq = torch.cumsum(torch.where(pos, wp, z), 0), torch.cumsum(torch.where(pos, wq, z), 0)
    return al * int(cp[-1]) - be * int(cq[-1]), cp, cq


def _search(al, be, cp, cq, t):
    """the first v with al·cp[v] − be·cq[v] > t"""
    lo, hi = 0, cp.numel() - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if al * int(cp[mid]) - be * int(cq[mid]) > t:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _ratios(P, Q, d):
    """a_i = wp_i(d_i)·Wq_i, b_i = wq_i(d_i)·Wp_i (0 for a draft outside [0, V))"""
    (wp, Wp, _), (wq, Wq, _) = P, Q
    V = wp.shape[1]
    a, b = [], []
    for i, x in enumerate(d):
        ok = 0 <= x < V
        a.append((int(wp[i, x]) if ok else 0) * Wq[i])
        b.append((int(wq[i, x]) if ok else 0) * Wp[i])
    return a, b


def _chain(a, b):
    """c_i = N_i / D_i, i = 0 … k"""
    N, D = [1], [1]
    for x, y in zip(a, b):
        if N[-1] == 0 or x == 0:
            N.append(0), D.append(1)
        elif N[-1] * x >= D[-1] * y:
            N.append(1), D.append(1)
        else:
            N.append(N[-1] * x), D.append(D[-1] * y)
    return N, D


def _h_row(P, Q, i, N, D):
    """row i: α, β, (num, cp, cq), den — h_i = num / den"""
    (wp, Wp, _), (wq, Wq, _) = P, Q
    al, be = N[i] * Wq[i], D[i] * Wp[i]
    rr = _resid(al, be, wp[i], wq[i])
    return al, be, rr, rr[0] + (D[i] - N[i]) * Wp[i] * Wq[i]


def _res_draw(P, i, al, be, rr, U0):
    """next from row i's residual (Σr = 0: from p_i)"""
    num, cp, cq = rr
    if num == 0:
        return _draw_w(P[0][i], P[1][i], P[2][i], U0)
    return _search(al, be, cp, cq, (U0 * num) >> 64)


def _model_tbt(P, Q, d, seed, ctr):
    """the base rule (sample.h) → (n_acc, next)"""
    k = len(d)
    a, b = _ratios(P, Q, d)
    U0 = _u64(seed, ctr, 0, 1)
    for i in range(k):
        if (_u64(seed, ctr, 1 + i, 1) * b[i]) >> 64 < a[i]:
            continue
        al, be = Q[1][i], P[1][i]
        return i, _res_draw(P, i, al, be, _resid(al, be, P[0][i], Q[0][i]), U0)
    return k, _draw_w(P[0][k], P[1][k], P[2][k], U0)


def _model_bv(P, Q, d, seed, ctr):
    """P016's block rule in integers (sample.h) → (n_acc, next); k = 0 or k > 15: the base rule. P = (w' [k+1, V],
    W' list, argmax list) of the target rows, Q = the same of the draft rows."""
    k = len(d)
    if k == 0 or k > BV_MAX:
        return _model_tbt(P, Q, d, seed, ctr)
    N, D = _chain(*_ratios(P, Q, d))
    U0 = _u64(seed, ctr, 0, 1)
    if (_u64(seed, ctr, k, 1) * D[k]) >> 64 < N[k]:
        return k, _draw_w(P[0][k], P[1][k], P[2][k], U0)
    for i in range(k - 1, 0, -1):
        if N[i] == 0:
            continue
        al, be, rr, den = _h_row(P, Q, i, N, D)
        if den == 0 or (_u64(seed, ctr, i, 1) * den) >> 64 < rr[0]:
            return i, _res_draw(P, i, al, be, rr, U0)
    al, be, rr, _ = _h_row(P, Q, 0, N, D)
    return 0, _res_draw(P, 0, al, be, rr, U0)


def _expect(P, Q, d):
    """E[tokens per round | drafts] (uniforms integrated) of the base rule and of the block rule"""
    k = len(d)
    a, b = _ratios(P, Q, d)
    e_t, pr = 1.0, 1.0
    for x, y in zip(a, b):
        pr *= 0.0 if x == 0 else 1.0 if x >= y else x / y
        e_t += pr
    N, D = _chain(a, b)
    h = [0.0] * (k + 1)
    h[k] = N[k] / D[k]
    for i in range(1, k):
        if N[i]:
            _, _, rr, den = _h_row(P, Q, i, N, D)
            h[i] = 1.0 if den == 0 else rr[0] / den
    e_b, tail = 1.0, 1.0
    for i in range(k, 0, -1):
        e_b += i * h[i] * tail
        tail *= 1 - h[i]
    return e_t, e_b


def _seq(gen, V, k, kind, real=None):
    """target rows [k+1, V] and draft rows [k, V] (CPU) of one sequence: 'rand' N(0, 2²) and q = p + N(0, σ²), σ 0.1 /
    0.5 / 1.5; 'ties' small integers, q = p with 20 % of the entries moved by ±1; 'ref' consecutive reference rows,
    q = p + N(0, 0.7²). Then q = p bitwise on ~15 % of the rows and (not ref) ~4 % degenerate rows (all −inf, +inf,
    NaN)."""
    r = lambda: float(torch.rand(1, generator=gen))  # noqa: E731
    if kind == "ref":
        n = real.shape[0]
        P = real[(int(torch.randint(0, n, (1,), generator=gen)) + torch.arange(k + 1)) % n].cpu()
        Q = P[:k] + torch.randn(k, V, generator=gen) * 0.7
    elif kind == "ties":
        P = torch.randint(-3, 3, (k + 1, V), generator=gen).float()
        Q = P[:k] + torch.randint(-1, 2, (k, V), generator=gen).float() * (torch.rand(k, V, generator=gen) < 0.2)
    else:
        P = torch.randn(k + 1, V, generator=gen) * 2
        Q = P[:k] + torch.randn(k, V, generator=gen) * [0.1, 0.5, 1.5][int(torch.randint(0, 3, (1,), generator=gen))]
    for i in range(k):
        if r() < 0.15:
            Q[i] = P[i]
    if kind != "ref":
        for X in (P, Q):
            for i in range(X.shape[0]):
                if r() < 0.04:
                    j = int(torch.randint(0, 3, (1,), generator=gen))
                    if j == 0:
                        X[i] = -INF
                    else:
                        X[i, int(torch.randint(0, V, (1,), generator=gen))] = INF if j == 1 else float("nan")
    return P, Q


def _bv_call(m, gen, seqs, ctr0):
    """One spec_accept call's inputs for sequences [(P, Q, T, tk, tp)] (tk None: untruncated): drafts drawn from q with
    the sequence's parameters by sample_rows, ~10 % then replaced by a random token and ~2 % by −1 / V + 3."""
    ks = [s[1].shape[0] for s in seqs]
    V = seqs[0][0].shape[1]
    P = torch.cat([s[0] for s in seqs]).cuda()
    Q = torch.cat([s[1] for s in seqs if s[1].shape[0]]).cuda() if sum(ks) else torch.zeros(1, V, device="cuda")
    tk = torch.tensor([s[3] if s[3] is not None else 0 for s in seqs], dtype=torch.int32, device="cuda")
    tp = torch.tensor([s[4] if s[3] is not None else 1.0 for s in seqs], dtype=torch.float32, device="cuda")
    temp = torch.tensor([s[2] for s in seqs], device="cuda")
    nq = Q.shape[0]
    rep = lambda t: torch.cat([t[b:b + 1].repeat(k) for b, k in enumerate(ks)]) if sum(ks) else t[:1]  # noqa: E731
    d = m.sample(Q, rep(temp)[:nq].contiguous(), _i64(gen, nq), torch.arange(nq, device="cuda") + ctr0,
                 rep(tk)[:nq].contiguous(), rep(tp)[:nq].contiguous())[0][0].cpu()
    u = torch.rand(nq, generator=gen)
    d = torch.where(u < 0.1, torch.randint(0, V, (nq,), generator=gen, dtype=torch.int32), d)
    d = torch.where(u > 0.98, torch.where(u > 0.99, torch.tensor(-1, dtype=torch.int32),
                                          torch.tensor(V + 3, dtype=torch.int32)), d).int()
    drafts = d[:sum(ks)].contiguous().cuda()
    seed, ctr = _i64(gen, len(seqs)), torch.arange(len(seqs), device="cuda") + ctr0 + (1 << 38)
    return (P, _starts(ks, 1), Q, _starts(ks, 0), drafts, temp, seed, ctr, tk, tp), ks


def test_bv_exact(ctx):
    """P016 (patch on) == the Python-integer model of sample.h's block rule, (n_acc, next) per sequence. V = 256: rows
    'rand' / 'ties' (_seq: q = p bitwise on some rows, degenerate rows), drafts from q (+ random / out-of-range ones),
    k = 0 … 16 mixed in calls of 24 sequences (k > 15: the base rule), T 0.7 / 1 / 1.3, untruncated and (20, 0.95),
    (−1, 0.8), (3, 1), (1, 1); V = 248320 reference rows, k = 1 … 8, untruncated / (20, 0.95) / (−1, 0.9). Every call
    again
    with every classification through the exact big-integer path (force_exact): bitwise the same; sequences with k ≤ 1
    also == the base kernel (P016 off)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(16)
    real = _real_rows(ctx)
    trs = [(None, None), (20, 0.95), (-1, 0.8), (3, 1.0), (1, 1.0)]
    cases = [(256, kind, tr, 24, 17) for kind in ("rand", "ties") for tr in trs for _ in range(3)]
    cases += [(V, "ref", tr, 8, 9) for tr in ((None, None), (20, 0.95), (-1, 0.9)) for _ in range(2)]
    n = bad = bad_x = bad_k1 = n_k1 = 0
    taus, fails = {}, []
    with _Block(m, True):
        for ci, (Vr, kind, (tk, tp), B, kmax) in enumerate(cases):
            ks = [int(x) for x in torch.randint(0 if kmax > 9 else 1, kmax, (B,), generator=gen)]
            temps = [[0.7, 1.0, 1.3][int(torch.randint(0, 3, (1,), generator=gen))] for _ in range(B)]
            seqs = [(*_seq(gen, Vr, k, kind, real), T, tk, tp) for k, T in zip(ks, temps)]
            args, ks = _bv_call(m, gen, seqs, ci << 32)
            na, nx = m.accept(*args)
            prev = m.force_exact(1)
            na2, nx2 = m.accept(*args)
            m.force_exact(prev)
            bad_x += int((na2 != na).sum() + (nx2 != nx).sum())
            with _Block(m, False):
                nb, xb = m.accept(*args)
            P, ps, Q, qs, drafts, temp, seed, ctr, tkt, tpt = args
            rr = lambda t, e: torch.cat([t[b:b + 1].repeat(k + e) for b, k in enumerate(ks)])  # noqa: E731
            wp = _wrows(m, P, rr(temp, 1), *((rr(tkt, 1), rr(tpt, 1)) if tk is not None else ()))
            wq = (_wrows(m, Q[:sum(ks)].contiguous(), rr(temp, 0), *((rr(tkt, 0), rr(tpt, 0)) if tk is not None else ()))
                  if sum(ks) else None)
            dl, sl, cl = drafts.tolist(), seed.tolist(), ctr.tolist()
            for b, k in enumerate(ks):
                p0, q0 = int(ps[b]), int(qs[b])
                Pb = (wp[0][p0:p0 + k + 1], wp[1][p0:p0 + k + 1], wp[2][p0:p0 + k + 1])
                Qb = (wq[0][q0:q0 + k], wq[1][q0:q0 + k], wq[2][q0:q0 + k]) if k else (None, [], [])
                ref = _model_bv(Pb, Qb, dl[q0:q0 + k], sl[b], cl[b])
                got = (int(na[0, b]), int(nx[0, b]))
                n += 1
                if got != ref:
                    bad += 1
                    fails.append(f"case{ci} b{b} k{k} {kind} ({tk},{tp}): {got} vs {ref}")
                taus[(k, got[0])] = taus.get((k, got[0]), 0) + 1
                if k <= 1:
                    n_k1 += 1
                    bad_k1 += (int(nb[0, b]), int(xb[0, b])) != got
    ks_hit = sorted({k for k, _ in taus})
    full = sum(1 for k in ks_hit if all((k, t) in taus for t in range(k + 1)))
    ctx.detail(f"sample bv exact: {n} seqs, model mismatches {bad} {fails[:10]}, force_exact mismatches {bad_x}, "
               f"k<=1 vs base {bad_k1}/{n_k1}; (k, tau) counts {sorted(taus.items())}")
    ok = bad == 0 and bad_x == 0 and bad_k1 == 0
    return {"ok": ok, "msg": f"integer model == kernel {n - bad}/{n} sequences (V=256 random·ties·degenerate rows, "
            f"k=0…16, 5 truncations; V=248320 reference k=1…8, 3 truncations) · forced exact path == default bitwise "
            f"(diff {bad_x}) · k≤1 == P016 off {n_k1 - bad_k1}/{n_k1} · k with every τ 0…k seen {full}/{len(ks_hit)}"}


def _tau_small(pw, Pw, qw, Qw, x):
    """P(τ = i | drafts x), i = 0 … k, in fractions from integer weights (lists)"""
    from fractions import Fraction as Fr
    k = len(x)
    a = [pw[i][x[i]] * Qw[i] for i in range(k)]
    b = [qw[i][x[i]] * Pw[i] for i in range(k)]
    N, D = _chain(a, b)
    h = [Fr(0)] * (k + 1)
    h[k] = Fr(N[k], D[k])
    for i in range(1, k):
        if N[i]:
            al, be = N[i] * Qw[i], D[i] * Pw[i]
            num = sum(max(al * u - be * v, 0) for u, v in zip(pw[i], qw[i]))
            den = num + (D[i] - N[i]) * Pw[i] * Qw[i]
            h[i] = Fr(1) if den == 0 else Fr(num, den)
    out, tail = [Fr(0)] * (k + 1), Fr(1)
    for i in range(k, 0, -1):
        out[i] = h[i] * tail
        tail *= 1 - h[i]
    out[0] = tail
    return out


def _enum_identity(seed, cases):
    """The block rule in fractions with prefix-dependent random rows (integer weights, zeros, q = p sometimes):
    enumerating every draft chain, τ and next (and target continuations up to k+1 tokens) gives exactly the target's
    joint of the first k+1 tokens. Returns (checked cases, mismatching sequences)."""
    import itertools
    import random
    from fractions import Fraction as Fr
    rng = random.Random(seed)

    def row(V):
        w = [0 if rng.random() < 0.3 else rng.randint(1, 9) for _ in range(V)]
        if not any(w):
            w[rng.randrange(V)] = 1
        return [Fr(x, sum(w)) for x in w]

    n = bad = 0
    for V, k, reps in cases:
        for _ in range(reps):
            P, Q = {}, {}
            for L in range(k + 2):
                for pre in itertools.product(range(V), repeat=L):
                    P[pre] = row(V)
                    if L < k:
                        Q[pre] = list(P[pre]) if rng.random() < 0.3 else row(V)
            out = {}
            for x in itertools.product(range(V), repeat=k):
                pq = Fr(1)
                for i in range(k):
                    pq *= Q[x[:i]][x[i]]
                if pq == 0:
                    continue
                c = [Fr(1)]
                for i in range(1, k + 1):
                    c.append(min(Fr(1), c[-1] * P[x[:i - 1]][x[i - 1]] / Q[x[:i - 1]][x[i - 1]]))
                h, res = {k: c[k]}, {}
                for i in range(k):
                    r = [max(c[i] * P[x[:i]][v] - Q[x[:i]][v], Fr(0)) for v in range(V)]
                    R = sum(r)
                    res[i] = [rv / R for rv in r] if R > 0 else list(P[x[:i]])
                    if i >= 1:
                        h[i] = R / (R + 1 - c[i]) if R + 1 - c[i] != 0 else Fr(1)
                tail, pt = Fr(1), {}
                for i in range(k, 0, -1):
                    pt[i] = h[i] * tail
                    tail *= 1 - h[i]
                pt[0] = tail
                for tau, ptau in pt.items():
                    if ptau == 0:
                        continue
                    for y, py in enumerate(list(P[x[:k]]) if tau == k else res[tau]):
                        if py == 0:
                            continue
                        em = x[:tau] + (y,)
                        for cont in itertools.product(range(V), repeat=k + 1 - len(em)):
                            sq, pc = em + cont, Fr(1)
                            for j in range(len(em), k + 1):
                                pc *= P[sq[:j]][sq[j]]
                            if pc:
                                out[sq] = out.get(sq, Fr(0)) + pq * ptau * py * pc
            n += 1
            for sq in itertools.product(range(V), repeat=k + 1):
                t = Fr(1)
                for j in range(k + 1):
                    t *= P[sq[:j]][sq[j]]
                bad += out.get(sq, Fr(0)) != t
    return n, bad


def test_bv_dist(ctx):
    """P016's output distribution: V = 8 (the whole row), target rows N(0, 1.5²) with −inf entries, draft rows random
    (−inf entries, q = p on one row in some cases), 1,056,000 rounds per case (drafts drawn from q by sample_rows; tokens past the
    emitted ones drawn from the target rows): the first min(k+1, 4) output tokens' joint vs the product of the target
    rows (chi-square over 8^L bins) and n_acc's histogram vs the exact P(τ = i) (fractions over all 8^k chains), k = 2,
    3, 4, untruncated and (5, 0.9) / (3, 1); and the rule itself in fractions: every chain enumerated with
    prefix-dependent random rows (V 2–4, k 1–3) gives exactly the target's joint of k+1 tokens."""
    import itertools
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(17)
    Vs, B, n_iter = 8, 16000, 66  # B·k ≤ 65535 rows per sample_rows call
    cases = [(2, 0.8, None, None, []), (3, 1.2, None, None, [1]), (4, 1.0, None, None, [2]), (3, 1.0, 5, 0.9, []),
             (4, 0.9, 3, 1.0, [0])]  # (k, T, top_k, top_p, rows with q = p)
    pmin, ok, out = 1.0, True, []
    with _Block(m, True):
        for ci, (k, T, tk, tp, same) in enumerate(cases):
            def rows(n):
                x = torch.randn(n, Vs, generator=gen) * 1.5
                x[torch.rand(n, Vs, generator=gen) < 0.15] = -INF
                x[:, int(torch.randint(0, Vs, (1,), generator=gen))] = torch.randn(n, generator=gen)  # a finite one
                return x
            pl, ql = rows(k + 1), rows(k)
            for i in same:
                ql[i] = pl[i]
            pl, ql = pl.cuda(), ql.cuda()
            kw = (lambda n: _tkp(n, tk, tp))  # noqa: E731
            c0 = (ci + 96) << 40
            P, Q = pl.repeat(B, 1), ql.repeat(B, 1)
            d, _ = m.sample(Q, torch.full((B * k,), T, device="cuda"), _i64(gen, B * k),
                            torch.arange(n_iter * B * k, device="cuda").view(n_iter, B * k) + c0, **kw(B * k))
            na, nx = m.accept(P, _starts([k] * B, 1), Q, _starts([k] * B, 0), d, torch.full((B,), T, device="cuda"),
                              _i64(gen, B), torch.arange(n_iter * B, device="cuda").view(n_iter, B) + c0, **kw(B))
            cont = torch.stack([m.sample(pl[j:j + 1].expand(B, Vs).contiguous(), torch.full((B,), T, device="cuda"),
                                         _i64(gen, B), torch.arange(n_iter * B, device="cuda").view(n_iter, B) + c0
                                         + ((j + 1) << 36), **kw(B))[0] for j in range(k + 1)], -1)
            d = d.view(n_iter, B, k)
            o = cont.clone()
            for j in range(k + 1):
                if j < k:
                    o[..., j] = torch.where(na > j, d[..., j], o[..., j])
                o[..., j] = torch.where(na == j, nx, o[..., j])
            L = min(k + 1, 4)
            temp = torch.full((k + 1,), T, device="cuda")
            tw = lambda n: (kw(n)["top_k"], kw(n)["top_p"]) if tk is not None else ()  # noqa: E731
            wp = _wrows(m, pl, temp, *tw(k + 1))
            wq = _wrows(m, ql, temp[:k].contiguous(), *tw(k))
            pr = [wp[0][j].double() / wp[1][j] for j in range(L)]
            joint = pr[0]
            for j in range(1, L):
                joint = (pr[j][:, None] * joint[None, :]).flatten()
            bins = sum(o[..., j].long() * Vs ** j for j in range(L)).flatten()
            pj, tv, noise = _gof(torch.bincount(bins, minlength=Vs ** L), joint)
            pwl, qwl = wp[0].tolist(), wq[0].tolist()
            pt = [0.0] * (k + 1)
            for x in itertools.product(range(Vs), repeat=k):
                pq = 1.0
                for i in range(k):
                    pq *= qwl[i][x[i]] / wq[1][i]
                if pq:
                    for i, f in enumerate(_tau_small(pwl, wp[1], qwl, wq[1], x)):
                        pt[i] += pq * float(f)
            ptau, _, _ = _gof(torch.bincount(na.flatten().long(), minlength=k + 1), torch.tensor(pt, dtype=F64))
            mean_t = sum(i * p for i, p in enumerate(pt))
            ctx.detail(f"sample bv dist k={k} T={T} ({tk},{tp}): joint of {L} p={pj:.3g} TV={tv:.2e} (noise {noise:.1e}) "
                       f"tau p={ptau:.3g} E[tau] {mean_t:.4f} got {na.double().mean().item():.4f} P(tau) {pt}")
            pmin = min(pmin, pj, ptau)
            ok &= pj >= P_MIN and ptau >= P_MIN
            out.append(f"k={k}{'' if tk is None else f' ({tk},{tp})'}")
    n_en, bad_en = _enum_identity(18, ((2, 1, 30), (3, 1, 30), (2, 2, 30), (3, 2, 20), (2, 3, 15), (3, 3, 4), (4, 2, 4)))
    ctx.detail(f"sample bv enum: {n_en} rule checks, {bad_en} mismatching sequences")
    ok &= bad_en == 0
    return {"ok": ok, "msg": f"V=8 {B * n_iter:,} rounds × {len(cases)} cases ({' · '.join(out)}): joint of the first "
            f"tokens == target product, n_acc distribution == exact P(τ) (fractions, every chain) — min p {pmin:.3f} · "
            f"rule enumerated in fractions {n_en} cases, mismatches {bad_en}"}


def test_bv_gain(ctx):
    """P016 vs the base on reference rows (V = 248320): targets = consecutive decode-step rows (per sequence its own
    start), drafts drawn from q = targets + N(0, σ²), T = 1, (k, σ) = (4, 1.5), (4, 3), (8, 1.5), untruncated and
    (20, 0.95), 2^13 rounds (32 sequences × 256): tokens per round (n_acc + 1) with the patch off and on over the same
    drafts and uniforms; and the exact expectations given the drafts (Python model, uniforms integrated) over the first
    64 chains."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(19)
    real = _real_rows(ctx)
    nr = real.shape[0]
    B, n_iter = 32, 256
    out, ok = [], True
    cases = [(k, sig, tk, tp) for k, sig in ((4, 1.5), (4, 3.0), (8, 1.5)) for tk, tp in ((None, None), (20, 0.95))]
    for ci, (k, sig, tk, tp) in enumerate(cases):
        st = torch.randint(0, nr, (B,), generator=gen)
        P = torch.cat([real[(int(s) + torch.arange(k + 1)) % nr] for s in st])
        Q = torch.cat([P[b * (k + 1):b * (k + 1) + k] for b in range(B)])
        Q = (Q + (torch.randn(Q.shape, generator=gen) * sig).cuda()).contiguous()
        kw = (lambda n: _tkp(n, tk, tp))  # noqa: E731
        tw = lambda n: (kw(n)["top_k"], kw(n)["top_p"]) if tk is not None else ()  # noqa: E731
        c0 = (ci + 128) << 40
        d, _ = m.sample(Q, torch.ones(B * k, device="cuda"), _i64(gen, B * k),
                        torch.arange(n_iter * B * k, device="cuda").view(n_iter, B * k) + c0, **kw(B * k))
        seed, ctr = _i64(gen, B), torch.arange(n_iter * B, device="cuda").view(n_iter, B) + c0
        res = {}
        for on in (False, True):
            with _Block(m, on):
                res[on] = m.accept(P, _starts([k] * B, 1), Q, _starts([k] * B, 0), d, torch.ones(B, device="cuda"),
                                   seed, ctr, **kw(B))[0].double()
        t0, t1 = res[False].mean().item() + 1, res[True].mean().item() + 1
        diff = res[True] - res[False]
        se = max(diff.std().item() / math.sqrt(diff.numel()), 1e-12)
        et = eb = 0.0
        nch = 0
        for b in range(B):
            one = torch.ones(k + 1, device="cuda")
            wp = _wrows(m, P[b * (k + 1):(b + 1) * (k + 1)].contiguous(), one, *tw(k + 1))
            wq = _wrows(m, Q[b * k:(b + 1) * k].contiguous(), one[:k].contiguous(), *tw(k))
            for it in range(2):
                e = _expect(wp, wq, d[it, b * k:(b + 1) * k].tolist())
                et, eb, nch = et + e[0], eb + e[1], nch + 1
        et, eb = et / nch, eb / nch
        ctx.detail(f"sample bv gain k={k} ({tk},{tp}) sigma {sig}: kernel tokens/round off {t0:.4f} on {t1:.4f} "
                   f"({100 * (t1 / t0 - 1):+.2f}%, paired z {(t1 - t0) / se:.1f}); exact over {nch} chains off {et:.4f} "
                   f"on {eb:.4f} ({100 * (eb / et - 1):+.2f}%)")
        ok &= t1 >= t0 - 3 * se  # (per chain the block rule can be below the base; on average it is not)
        out.append(f"k={k} σ{sig}{'' if tk is None else ' (20,.95)'} {t0:.2f}→{t1:.2f} ({100 * (t1 / t0 - 1):+.1f}%, "
                   f"exact {100 * (eb / et - 1):+.1f}%)")
    return {"ok": ok, "msg": "tokens/round base→block (reference rows, q = p + N(0, σ²), 2^13 rounds; exact "
            "expectation 64 chains): " + " · ".join(out)}


def test_bv_invariance(ctx):
    """P016 bitwise: every sequence alone == in a permuted batch == repeated == cluster sizes 1, 2, 3, 5, 8, 16 == cap 8
    == force_exact; k = 0 … 16 (16: the base rule), T = 0 / 0.7 / 1 / 1.3, per-sequence top_k / top_p mixed (TK_MIX),
    reference targets, drafts drawn with the sequence's parameters from targets + N(0, σ²), σ 0.2 … 2 (every τ)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(20)
    real = _real_rows(ctx)
    nr = real.shape[0]
    ks = list(range(17))
    B = len(ks)
    ts = [0.0 if b % 5 == 0 else [0.7, 1.0, 1.3][b % 3] for b in range(B)]
    P = torch.cat([real[(3 * b + torch.arange(k + 1)) % nr] for b, k in enumerate(ks)])
    Q = torch.cat([real[(3 * b + torch.arange(k)) % nr] + (torch.randn(k, V, generator=gen) * (0.2 + 0.12 * b)).cuda()
                   for b, k in enumerate(ks) if k]).contiguous()
    tkt = torch.tensor([TK_MIX[b % len(TK_MIX)][0] for b in range(B)], dtype=torch.int32, device="cuda")
    tpt = torch.tensor([TK_MIX[b % len(TK_MIX)][1] for b in range(B)], dtype=torch.float32, device="cuda")
    temp = torch.tensor(ts, device="cuda")
    rep = lambda t: torch.cat([t[b:b + 1].repeat(k) for b, k in enumerate(ks)])  # noqa: E731
    drafts = m.sample(Q, rep(temp), _i64(gen, Q.shape[0]), _i64(gen, Q.shape[0]), rep(tkt), rep(tpt))[0][0]
    seed, ctr = _i64(gen, B), _i64(gen, B)
    qs, ps = _starts(ks, 0), _starts(ks, 1)
    bad = []
    prev_c, prev_cap = m.force_cluster(0), m.force_cap(0)
    try:
        with _Block(m, True):
            one = []
            for b in range(B):
                q0, q1 = int(qs[b]), int(qs[b + 1])
                one.append(m.accept(P[int(ps[b]):int(ps[b + 1])], _starts([ks[b]], 1), Q[q0:q1] if q1 > q0 else Q[:1],
                                    _starts([ks[b]], 0), drafts[q0:q1].contiguous() if q1 > q0 else drafts[:0],
                                    temp[b:b + 1], seed[b:b + 1], ctr[b:b + 1], tkt[b:b + 1], tpt[b:b + 1]))
            ref = torch.cat([o[0] for o in one], 1), torch.cat([o[1] for o in one], 1)
            order = torch.argsort(torch.rand(B, generator=gen)).tolist()
            ks2 = [ks[b] for b in order]
            P2 = torch.cat([P[int(ps[b]):int(ps[b + 1])] for b in order])
            Q2 = torch.cat([Q[int(qs[b]):int(qs[b + 1])] for b in order if ks[b]])
            d2 = torch.cat([drafts[int(qs[b]):int(qs[b + 1])] for b in order])
            o = torch.tensor(order, device="cuda")
            for c, cap, fx in [(c, 0, 0) for c in (0, 1, 2, 3, 5, 8, 16)] + [(0, 8, 0), (0, 0, 1), (3, 8, 1)]:
                m.force_cluster(c), m.force_cap(cap)
                pf = m.force_exact(fx)
                for _ in range(2 if c == 0 and not fx else 1):
                    na, nx = m.accept(P2, _starts(ks2, 1), Q2, _starts(ks2, 0), d2, temp[o], seed[o], ctr[o], tkt[o],
                                      tpt[o])
                    if not (torch.equal(na[0], ref[0][0, o]) and torch.equal(nx[0], ref[1][0, o])):
                        bad.append(f"C{c} cap{cap} exact{fx}")
                m.force_exact(pf)
    finally:
        m.force_cluster(prev_c), m.force_cap(prev_cap)
    acc = ref[0][0].tolist()
    ctx.detail(f"sample bv invariance: bad {bad}; n_acc {acc} k {ks}")
    return {"ok": not bad, "msg": f"alone == permuted batch == repeated == cluster 1·2·3·5·8·16 == cap 8 == forced "
            f"exact path, bitwise ({B} sequences k=0…16, T=0·0.7·1·1.3, (k,p) mixed, accepted "
            f"{'·'.join(map(str, acc))})" if not bad else "mismatch: " + ", ".join(bad[:4])}


def test_bv_perf(ctx):
    """µs per spec_accept call, base (P016 off) → P016 on, V = 248320, one call alone (graph launch included), rows from
    DRAM (input sets cycled): k = 4 at B = 1 and 4 — every draft accepted (q = p: the best of 5 per set, min), rejected
    from the first draft (q = p shifted by a token), mixed (q = p + N(0, 0.7²), drafts drawn; mean over the sets of each
    set's best) — untruncated (N(0, 3²) rows) and (20, 0.95) (reference rows); B = 1, k = 15 mixed (the longest
    chains)."""
    import time
    m = _mod(ctx)
    real = _real_rows(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench(1, V, 4, 4, 1.0, 2, 1, 1)
    res = {}
    cases = [(w, B, 4, tk, tp) for tk, tp in ((0, 1.0), (20, 0.95)) for B in (1, 4) for w in (1, 2, 3)]
    cases.append((3, 1, 15, 0, 1.0))
    for what, B, k, tk, tp in cases:
        sets = max(4 if what == 1 else 12, -(-160 // ((2 * k + 1) * B)))
        for on in (False, True):
            with _Block(m, on):
                res[(what, B, k, tk, on)] = m.bench(what, V, B, k, 1.0, sets, 1, 5 if what == 1 else 3, tk, tp,
                                                    real if tk else None)
    ctx.detail("sample bv perf (us per call, off -> on): " + ", ".join(
        f"what{w} B{B} k{k} tk{tk}: {res[(w, B, k, tk, False)]:.1f} -> {res[(w, B, k, tk, True)]:.1f}"
        for w, B, k, tk, _ in cases))
    name = {1: "all accepted", 2: "first draft rejected", 3: "mixed"}
    msg = []
    for tk in (0, 20):
        for B in (1, 4):
            msg.append(f"{'truncated ' if tk else ''}B={B} " + " / ".join(
                f"{name[w]} {res[(w, B, 4, tk, False)]:.0f}→{res[(w, B, 4, tk, True)]:.0f}" for w in (1, 2, 3)))
    msg.append(f"k=15 B=1 mixed {res[(3, 1, 15, 0, False)]:.0f}→{res[(3, 1, 15, 0, True)]:.0f}")
    return {"ok": True, "msg": "µs/call base→P016, k=4: " + " · ".join(msg)}


# ---- P019 sparse_logits: sampling / spec_accept from each vocab half's top-64 candidates ------------------------------
VH = V // 2
KC = 64  # kSparseK
DENSE = -2  # kSparseDense


def _recs(m, x):
    """Records [2, R, 65, 2] (int32: id, logit bits; entry 64 = flag) of full rows x [R, Vx]: each half through
    sample_cands with its id offset — what the two ranks would send."""
    h = x.shape[1] // 2
    return torch.stack([m.cands(x[:, :h].contiguous(), 0), m.cands(x[:, h:].contiguous(), h)])


def _ref_cands(x):
    """Per half q of rows x [R, Vx]: the 64 highest-ranked tokens by Z (global ids) → ids, values [R, 64]; the flag
    (a NaN or ±inf in the half) [R]."""
    z = _zkeys(x)
    h = x.shape[1] // 2
    out = []
    for q in range(2):
        idx = torch.sort(z[:, q * h:(q + 1) * h], dim=1, descending=True).indices[:, :KC] + q * h
        out.append((idx, x.gather(1, idx), (~torch.isfinite(x[:, q * h:(q + 1) * h])).any(1)))
    return out


def _cands_bad(rec, ref):
    """Rows whose records differ from the reference (ids, values as fp32 ==, flag; candidates of flagged halves not
    compared)."""
    bad = torch.zeros(rec.shape[1], dtype=torch.bool, device=rec.device)
    for q, (ids, vals, flag) in enumerate(ref):
        fl = rec[q, :, KC, 0] != 0
        same = (rec[q, :, :KC, 0] == ids).all(1) & (rec[q, :, :KC, 1].contiguous().view(torch.float32) == vals).all(1)
        bad |= (fl != flag) | (~flag & ~same)
    return bad


def _p019_rows(ctx, gen):
    """[(name, rows [n, V])] covering the sparse path's corners: reference decode rows; N(0, 3²); small-integer ties
    (thousands of equal logits within and across the halves); flat (all equal); ascending / descending / sawtooth (the
    top tokens packed at one end); the top 64 tied at 10.0 over both halves, some at half the weight (exact top-p
    boundaries); ±0 ties; subnormal and huge (±1e37) finite logits; the top tokens packed into 64 threads' float4s of a
    1-CTA chunk (sample_cands' slow path at cluster 1)."""
    real = _real_rows(ctx)
    ar = torch.arange(V, dtype=torch.float32)
    tie64 = torch.randn(4, V, generator=gen)
    for r in range(4):
        pos = torch.randperm(V, generator=gen)[:64]
        tie64[r, pos] = 10.0
        tie64[r, pos[:8 * r]] = 10.0 - 0.6931472 * (r + 1) / 4
    zero = torch.full((3, V), -1.0)
    for r in range(3):
        zero[r, torch.randperm(V, generator=gen)[:100]] = 0.0
        zero[r, torch.randperm(V, generator=gen)[:100]] = -0.0
    pack = torch.randn(2, V, generator=gen)
    sel = torch.repeat_interleave((torch.arange(VH // 4) % 512) < 64, 4)
    n_sel = int(sel.sum())
    pack[0, :VH][sel] = 100 + torch.rand(n_sel, generator=gen)
    pack[1, VH:][sel] = 100 + torch.rand(n_sel, generator=gen)
    return [("ref", real), ("randn", (torch.randn(8, V, generator=gen) * 3).cuda()),
            ("ties", torch.randint(-6, 6, (6, V), generator=gen).float().cuda()),
            ("flat", torch.stack([torch.zeros(V), torch.full((V,), 5.0)]).cuda()),
            ("sorted", torch.stack([ar / 1e4, -ar / 1e4, (ar % 4096) / 64]).cuda()),
            ("tie64", tie64.cuda()), ("zeros", zero.cuda()),
            ("tiny", (torch.randn(2, V, generator=gen) * 1e-39).cuda()),
            ("huge", (torch.randn(2, V, generator=gen) * 1e37).cuda()), ("packed", pack.cuda())]


def test_p019_cands(ctx):
    """sample_cands == reference (each half's top 64 by Z, global ids; the flag) on _p019_rows plus rows with a NaN /
    +inf / −inf in one half, at R = 1, 4, 5, 20 and all rows, cluster sizes automatic / 1 / 2 / 5 / 12 / 16; the packed
    rows at cluster 1 take the slow path (counted)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(1900)
    sets = _p019_rows(ctx, gen)
    nf = torch.randn(9, V, generator=gen)
    for r in range(9):
        pos = (VH if r % 2 else 0) + torch.randint(0, VH, (1 + r,), generator=gen)
        nf[r, pos] = [float("nan"), INF, -INF][r % 3]
    allr = torch.cat([x for _, x in sets] + [nf.cuda()])
    bad, n = [], 0
    m.cands_slow(True)
    prev = m.force_cluster(0)
    try:
        for c in (0, 1, 2, 5, 12, 16):
            m.force_cluster(c)
            for R in (1, 4, 5, 20, allr.shape[0]):
                for s0 in range(0, allr.shape[0] - R + 1, R) if R < 20 else (0,):
                    x = allr[s0:s0 + R]
                    b = _cands_bad(_recs(m, x), _ref_cands(x))
                    n += x.shape[0]
                    if bool(b.any()):
                        bad.append(f"C{c} R{R} rows {(s0 + b.nonzero().flatten()).tolist()[:4]}")
        slow0 = m.cands_slow(True)
        slow_ref = {}
        for name in ("ref", "randn"):  # logits-like rows: the fast path only (per cluster size)
            for c in (0, 1, 2, 5, 12, 16):
                m.force_cluster(c)
                xs = dict(sets)[name]
                for R in (1, 4, xs.shape[0]):
                    _recs(m, xs[:R])
                slow_ref[(name, c)] = m.cands_slow(True)
        if any(slow_ref.values()):
            bad.append(f"slow path on logits-like rows {slow_ref}")
        m.force_cluster(1)
        pk = dict(sets)["packed"]
        b = _cands_bad(_recs(m, pk), _ref_cands(pk))
        slow = m.cands_slow(True)
        if bool(b.any()) or slow < 2:
            bad.append(f"packed C1 bad {b.tolist()} slow {slow}")
    finally:
        m.force_cluster(prev)
    ctx.detail(f"sample p019 cands: {n} rows, slow-path CTAs {slow0} (all calls) {slow} (packed, C1), on reference / "
               f"N(0, 3²) rows {slow_ref}, bad {bad[:10]}")
    return {"ok": not bad, "msg": f"64 candidates == torch sort {n} rows (reference·random·ties·flat·sorted·64 ties·±0·"
            f"subnormal·huge·packed, NaN/±inf flag on that half only, R=1·4·5·20·{allr.shape[0]}, cluster "
            f"auto·1·2·5·12·16) · slow path {slow} CTAs match" if not bad else "differs: " + ", ".join(bad[:4])}


P019_T = (0.0, 0.7, 1.0, 1.5)
P019_KP = [(k, p) for k in (1, 2, 20, 64) for p in (0.5, 0.95, 1.0)]
# outside / at the edge of the sparse conditions: top_k 0 / −1 / 65 / V (no top-k → dense unless top_p ≤ 0), top_p
# ≤ 0 (top-1, any top_k), NaN (no top-p), tiny
P019_EDGE = [(0, 0.95), (-1, 1.0), (65, 0.95), (V, 0.5), (0, 0.0), (300, -1.0), (7, float("nan")), (64, 1e-30)]


def _sparse_vs_full(m, x, rc, T, tk, tp, gen, N):
    """sample_rows on rows x vs sample_rows_sparse on their records rc, per-row T / top_k / top_p (lists), N counters
    → (mismatching draws, draws compared, dense markers where expected, wrong markers)."""
    R = x.shape[0]
    temp = torch.tensor(T, dtype=torch.float32, device="cuda")
    tkt = torch.tensor(tk, dtype=torch.int32, device="cuda")
    tpt = torch.tensor(tp, dtype=torch.float32, device="cuda")
    seed, ctr = _i64(gen, R), _i64(gen, N, R)
    tok, prob = m.sample(x, temp, seed, ctr, tkt, tpt)
    st, sp = m.sample_sparse(rc, V, temp, seed, ctr, tkt, tpt)
    flag = ~torch.isfinite(x).all(1)
    ok = torch.tensor([m.sparse_ok(float(temp[r]), tk[r], float(tpt[r]), V) for r in range(R)], device="cuda")
    dn = (flag | ~ok)[None].expand(N, R)
    mark = st == DENSE
    cmp = ~dn
    mism = int(((st != tok) | (sp.view(torch.int32) != prob.view(torch.int32)))[cmp].sum())
    return mism, int(cmp.sum()), int((mark & dn).sum()), int((mark != dn).sum())


def test_p019_sample(ctx):
    """sample_rows_sparse == sample_rows bitwise (token and prob) on _p019_rows split into halves: T 0 / 0.7 / 1 / 1.5
    × top_k 1 / 2 / 20 / 64 × top_p 0.5 / 0.95 / 1, 24 counters per row and setting; edge settings (top_k 0 / −1 / 65 /
    V, top_p ≤ 0 / NaN / 1e-30) and T = +inf, 1e30, 1e-30 (finite c), 1e-45 (c = inf → dense): sparse rows equal, the
    others marked kSparseDense (exactly the rows sparse_ok rejects)."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(1901)
    x = torch.cat([r for _, r in _p019_rows(ctx, gen)])
    rc = _recs(m, x)
    R = x.shape[0]
    mism = n = marks = wrong = 0
    for T in P019_T:
        for tk, tp in P019_KP + P019_EDGE:
            a = _sparse_vs_full(m, x, rc, [T] * R, [tk] * R, [tp] * R, gen, 24)
            mism, n, marks, wrong = mism + a[0], n + a[1], marks + a[2], wrong + a[3]
    kp = [P019_KP[r % len(P019_KP)] for r in range(R)]
    for T in (INF, 1e30, 1e-30, 1e-45):  # extreme temperatures, settings mixed per row
        a = _sparse_vs_full(m, x, rc, [T] * R, [k for k, _ in kp], [p for _, p in kp], gen, 24)
        mism, n, marks, wrong = mism + a[0], n + a[1], marks + a[2], wrong + a[3]
    ok = mism == 0 and wrong == 0 and n > 0
    ctx.detail(f"sample p019 sample: {n} draws compared, mismatches {mism}, dense markers {marks}, wrong markers {wrong}")
    return {"ok": ok, "msg": f"sparse == full rows bitwise (token·prob) {n - mism:,}/{n:,} draws ({R} rows × 4 T × "
            f"(k,p) {len(P019_KP)}+{len(P019_EDGE)} settings × 24 counters, 4 extreme T) · dense markers {marks:,} "
            f"correct (wrong {wrong})"}


def _p019_seq_call(m, gen, seqs, ctr0, N):
    """spec_accept / spec_accept_sparse inputs for sequences [(P, Q, T, tk, tp)] (CPU rows): _bv_call's drafts, N accept
    counters per sequence → (full args with ctr [N, B] and drafts [N, Q], records of P and Q, k_b)."""
    args, ks = _bv_call(m, gen, seqs, ctr0)
    P, ps, Q, qs, drafts, temp, seed, ctr, tk, tp = args
    ctrN = ctr[None] + (torch.arange(N, device="cuda")[:, None] << 44)
    dN = drafts[None].expand(N, drafts.numel()).contiguous()
    return (P, ps, Q, qs, dN, temp, seed, ctrN, tk, tp), _recs(m, P), _recs(m, Q), ks


def _seq_dense(m, P, Q, ks, temp, tk, tp):
    """Expected kSparseDense per sequence: parameters outside the sparse conditions or k > 15 at T > 0, a non-finite
    logit in a target row (any T) or a draft row (T > 0)."""
    out, p0, q0 = [], 0, 0
    pf, qf = ~torch.isfinite(P).all(1), ~torch.isfinite(Q).all(1)
    for b, k in enumerate(ks):
        T = float(temp[b])
        d = bool(pf[p0:p0 + k + 1].any())
        if T > 0:
            d |= not m.sparse_ok(T, int(tk[b]), float(tp[b]), V) or k > 15 or bool(qf[q0:q0 + k].any())
        out.append(d)
        p0, q0 = p0 + k + 1, q0 + k
    return torch.tensor(out, device="cuda")


def test_p019_accept(ctx):
    """spec_accept_sparse == spec_accept bitwise (n_acc, next). Part 1, P016 off, on, and on with force_exact: 72 calls
    of B = 1 … 4 sequences with k_b = 0 … 16 mixed (16 > kSparseMaxK: dense at T > 0, greedy sparse), T 0 / 0.7 / 1 /
    1.5, (top_k, top_p) from 1 / 2 / 20 / 64 × 0.5 / 0.95 / 1 plus edge settings (dense); rows (_seq): reference
    (consecutive decode rows, drafts = targets + N(0, 0.7²)), N(0, 2²) with q = p + N(0, σ²), small-integer ties; q = p
    bitwise on some rows (Σr = 0), drafts from q plus random and out-of-range ones; degenerate rows (NaN / ±inf, not
    reference) → kSparseDense; 16 accept counters per call. Part 2, P016 off and on: 24 calls of B = 4 reference
    sequences, k = 1 … 8, drafts drawn from q = targets + N(0, σ²), σ 0.3 / 0.7 / 1.5 / 3 (every τ), T 0.7 / 1,
    (20, 0.95) / (64, 1) / (2, 0.5) / (5, 0.8), 64 counters per call."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(1902)
    real = _real_rows(ctx)
    nr = real.shape[0]
    st = {"n": 0, "bad": 0, "marks": 0, "wrong": 0}
    taus, fails = {}, []

    def run(seqs, ci, N, modes):
        args, pc, qc, ks = _p019_seq_call(m, gen, seqs, (ci + 200) << 36, N)
        P, ps, Q, qs, dN, temp, seed, ctrN, tk, tp = args
        dense = _seq_dense(m, P, Q, ks, temp, tk, tp)
        dn = dense[None].expand(N, -1)
        for on, fx in modes:
            with _Block(m, on):
                pf = m.force_exact(fx)
                na, nx = m.accept(*args)
                sa, sx = m.accept_sparse(pc, ps, qc, qs, dN, V, temp, seed, ctrN, tk, tp)
                m.force_exact(pf)
            mk = (sa == DENSE) & (sx == DENSE)
            st["wrong"] += int((mk != dn).sum())
            st["marks"] += int((mk & dn).sum())
            diff = ((sa != na) | (sx != nx)) & ~dn
            st["n"] += int((~dn).sum())
            st["bad"] += int(diff.sum())
            if bool(diff.any()) and len(fails) < 10:
                j, b = diff.nonzero()[0].tolist()
                fails.append(f"case{ci} P016 {on} fx{fx} b{b} k{ks[b]} T{float(temp[b])} ({int(tk[b])},{float(tp[b])}) "
                             f"full ({int(na[j, b])},{int(nx[j, b])}) sparse ({int(sa[j, b])},{int(sx[j, b])})")
            if on and not fx:
                for b, k in enumerate(ks):
                    if not bool(dense[b]):
                        for t in na[:, b].tolist():
                            taus[(k, t)] = taus.get((k, t), 0) + 1

    kps = P019_KP + [(0, 0.95), (65, 1.0), (0, 0.0), (-1, -1.0)]
    for ci in range(72):
        kind = ("ref", "rand", "ties")[ci % 3]
        seqs = []
        for b in range(1 + ci % 4):
            k = int(torch.randint(0, 17, (1,), generator=gen)) if ci % 6 else int(torch.randint(1, 5, (1,), generator=gen))
            T = P019_T[int(torch.randint(0, 4, (1,), generator=gen))]
            tk, tp = kps[int(torch.randint(0, len(kps), (1,), generator=gen))]
            seqs.append((*_seq(gen, V, k, kind, real), T, tk, tp))
        run(seqs, ci, 16, ((False, 0), (True, 0), (True, 1)))
    n1 = st["n"]
    for ci in range(24):
        sig = (0.3, 0.7, 1.5, 3.0)[ci % 4]
        tk, tp = ((20, 0.95), (64, 1.0), (2, 0.5), (5, 0.8))[(ci // 4) % 4]
        seqs = []
        for b in range(4):
            k = 1 + (ci + 3 * b) % 8
            P = real[(int(torch.randint(0, nr, (1,), generator=gen)) + torch.arange(k + 1)) % nr].cpu()
            seqs.append((P, P[:k] + torch.randn(k, V, generator=gen) * sig, (0.7, 1.0)[b % 2], tk, tp))
        run(seqs, 100 + ci, 64, ((False, 0), (True, 0)))
    ks_hit = sorted({k for k, _ in taus})
    full = sum(1 for k in ks_hit if all((k, t) in taus for t in range(k + 1)))
    n, bad, marks, wrong = st["n"], st["bad"], st["marks"], st["wrong"]
    ctx.detail(f"sample p019 accept: {n} compared ({n1} part 1), mismatches {bad} {fails}, dense markers {marks}, wrong "
               f"{wrong}; (k, tau) {sorted(taus.items())}")
    return {"ok": bad == 0 and wrong == 0 and n > 0,
            "msg": f"sparse == full rows (n_acc, next) bitwise {n - bad:,}/{n:,} (P016 off·on·forced exact path; "
                   f"72 calls B=1…4, k=0…16, 4 T, (k,p) {len(kps)} settings, reference·random·ties·q=p·degenerate "
                   f"rows, 16 counters; reference σ 0.3…3 24 calls k=1…8 64 counters) · dense markers {marks:,} "
                   f"(wrong {wrong}) · k with every τ 0…k seen {full}/{len(ks_hit)}"}


def test_p019_flag(ctx):
    """The dense flag: set exactly in the half holding a NaN / +inf / −inf (one value, either half, at both ends of
    both halves or random); never for finite rows, extreme ones included (±3.4e38, subnormals, ±0); flagged rows →
    kSparseDense from sample_rows_sparse at every T and from spec_accept_sparse for the sequences that read them (a
    greedy sequence ignores its draft rows: equal to spec_accept). The flag never misses: extreme finite rows, sparse ==
    full bitwise at T 0 / 0.7 / 1 / 1.5 × the (k, p) settings."""
    m = _mod(ctx)
    gen = torch.Generator().manual_seed(1903)
    R = 24
    x = torch.randn(R, V, generator=gen) * 2
    where = []
    for r in range(R):
        pos = [0, VH - 1, VH, V - 1][r % 4] if r < 8 else int(torch.randint(0, VH, (1,), generator=gen)) + (r % 2) * VH
        x[r, pos] = [float("nan"), INF, -INF][r % 3]
        where.append(pos // VH)
    x = x.cuda()
    rc = _recs(m, x)
    fl = (rc[:, :, KC, 0] != 0).cpu()
    flag_ok = all(bool(fl[where[r], r]) and not bool(fl[1 - where[r], r]) for r in range(R))
    ext = torch.cat([torch.full((1, V), 3.4e38), torch.full((1, V), -3.4e38), torch.randn(2, V, generator=gen) * 3e37,
                     torch.randn(2, V, generator=gen) * 1e-42, torch.zeros(1, V), -torch.zeros(1, V)])
    ext[0, 7] = -3.4e38
    ext[1, 123456] = 3.4e38
    ext = ext.cuda()
    re = _recs(m, ext)
    fin_ok = bool((re[:, :, KC, 0] == 0).all())
    marks = 0
    for T in P019_T:
        st, _ = m.sample_sparse(rc, V, torch.full((R,), T, device="cuda"), _i64(gen, R), _i64(gen, R),
                                torch.full((R,), 20, dtype=torch.int32, device="cuda"),
                                torch.full((R,), 0.95, device="cuda"))
        marks += int((st == DENSE).sum())
    # spec_accept, B = 6 × k = 2: b0 / b1 / b4 a flagged target row (T 1 / 0 / 0), b2 / b3 a flagged draft row (T 1),
    # b5 greedy with flagged draft rows (not read: equal to spec_accept)
    B, k = 6, 2
    rnd = lambda n: torch.randn(n, V, generator=gen).cuda()  # noqa: E731
    P = torch.cat([x[0:3], x[3:6], rnd(3), rnd(3), x[6:9], rnd(3)])
    Q = torch.cat([rnd(4), x[9:11], x[11:13], rnd(2), x[13:15]])
    temp = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0], device="cuda")
    tkt, tpt = torch.full((B,), 20, dtype=torch.int32, device="cuda"), torch.full((B,), 0.95, device="cuda")
    dr = torch.randint(0, V, (B * k,), generator=gen, dtype=torch.int32).cuda()
    seed, ctr = _i64(gen, B), _i64(gen, B)
    sa, sx = m.accept_sparse(_recs(m, P), _starts([k] * B, 1), _recs(m, Q), _starts([k] * B, 0), dr, V, temp, seed, ctr,
                             tkt, tpt)
    na, nx = m.accept(P, _starts([k] * B, 1), Q, _starts([k] * B, 0), dr, temp, seed, ctr, tkt, tpt)
    exp = [True, True, True, True, True, False]
    acc_ok = [v == DENSE for v in sa[0].tolist()] == exp and (int(sa[0, 5]), int(sx[0, 5])) == (int(na[0, 5]), int(nx[0, 5]))
    mism = n = wrong = 0
    for T in P019_T:
        for tk, tp in P019_KP:
            Rx = ext.shape[0]
            a = _sparse_vs_full(m, ext, re, [T] * Rx, [tk] * Rx, [tp] * Rx, gen, 8)
            mism, n, wrong = mism + a[0], n + a[1], wrong + a[3]
    ok = flag_ok and fin_ok and marks == 4 * R and acc_ok and mism == 0 and wrong == 0
    ctx.detail(f"sample p019 flag: flag halves ok {flag_ok}, finite extremes unflagged {fin_ok}, sample markers {marks}/"
               f"{4 * R}, accept {sa[0].tolist()} {sx[0].tolist()} vs full {na[0].tolist()} {nx[0].tolist()} (dense "
               f"expected {exp}), extremes {n} compared, mismatches {mism}, wrong markers {wrong}")
    return {"ok": ok, "msg": f"NaN/±inf → flag on that half only {R} rows · finite extremes (±3.4e38·subnormal·±0) "
            f"unflagged · sample markers {marks}/{4 * R} · accept markers {'correct' if acc_ok else 'wrong'} · extreme "
            f"finite rows sparse == full {n - mism}/{n}"}


def test_p019_perf(ctx):
    """µs per call, V = 248320, reference rows tiled: sample_cands (one rank's half) at R = 1, 4, 5, 20 — L2-resident
    (the same rows every call, as right after lm_head) and from DRAM (input sets ≫ L2); sample_rows_sparse vs
    sample_rows (full rows from DRAM) at R = 1 … 5, T = 1, (20, 0.95), a graph of 100 calls; spec_accept_sparse vs
    spec_accept (P016 on) at B = 1 and 4, k = 4, (20, 0.95), every draft accepted and mixed (q = p + N(0, 0.7²)), one
    call alone per set."""
    import time
    m = _mod(ctx)
    real = _real_rows(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench_sparse(0, 4, 0, 4, 50, 1, 20, 0.95, real)
    res = {}
    cl = {R: m.cands_cluster(VH, R) for R in (1, 4, 5, 20)}
    for R in (1, 4, 5, 20):
        res[("cands", R, "l2")] = m.bench_sparse(0, R, 0, 1, 100, 5, 20, 0.95, real)
        res[("cands", R, "dram")] = m.bench_sparse(0, R, 0, max(2, 192 // R), 100, 5, 20, 0.95, real)
    for R in (1, 2, 3, 4, 5):
        res[("sparse", R)] = m.bench_sparse(1, R, 0, 4, 100, 5, 20, 0.95, real)
        res[("full", R)] = m.bench(0, V, R, 0, 1.0, max(2, 96 // R), 100, 5, 20, 0.95, real)
    with _Block(m, True):
        for B in (1, 4):
            for w, name in ((3, "all"), (2, "mixed")):
                res[("acc_s", B, name)] = m.bench_sparse(w, B, 4, 12, 1, 5, 20, 0.95, real)
                res[("acc_f", B, name)] = m.bench(1 if w == 3 else 3, V, B, 4, 1.0, max(12, -(-160 // (9 * B))), 1,
                                                  5 if w == 3 else 3, 20, 0.95, real)
    ctx.detail("sample p019 perf (us per call): " + ", ".join(f"{k}: {v:.2f}" for k, v in res.items())
               + f"; sample_cands cluster sizes {cl}")
    c = " · ".join(f"R{R} (C{cl[R]}) {res[('cands', R, 'l2')]:.1f}/{res[('cands', R, 'dram')]:.1f}" for R in (1, 4, 5, 20))
    s = " · ".join(f"R{R} {res[('full', R)]:.1f}→{res[('sparse', R)]:.1f}" for R in (1, 2, 3, 4, 5))
    a = " · ".join(f"B{B} {nm} {res[('acc_f', B, nm)]:.0f}→{res[('acc_s', B, nm)]:.1f}" for B in (1, 4)
                   for nm in ("all", "mixed"))
    return {"ok": True, "msg": f"µs: sample_cands (L2/DRAM) {c} · sample full→sparse {s} · accept k=4 (20,.95) P016 "
            f"full→sparse {a}"}
