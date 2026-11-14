"""QSA core (src/kernels/qsa.cu): RoPE table vs vLLM, prepare vs a torch op-by-op reference, top-k ties, golden dump per
request (teacher forcing), T>1 and batch invariance (bitwise), timing, positions up to 1M (long_pos).

Golden per request: empty caches, every prompt row from the full-row key `qkv_proj#0.out.0` fed in 64-row calls (the
engine's max rows), then each decode row in order. Compared at the kept prefill rows (last 64) and every decode row:
Q after prepare (`fn.forward_qsa#0.in.query`), selection (`indexer#0.out.0`, as sets — vLLM lists blocks in its top-k
kernel's order) and the gated output (`fn.forward_qsa#0.post.output` = o_proj input). Golden prefill ran the
reference's prefill configs (61 rows: 8 splits; 8,603 rows: one split, running max across tiles), decode the 64-split
kernel (one 32-token tile per split, as here) → reported separately. Selection differences are explained with our own
scores (rank and gap of the blocks that differ).

Q38_QSA_WINDOW=N (default 262,144) runs every test with an N-row RoPE table / workspace (qsa_rope_set, default YaRN
parameters): with N > 262,144, calls without a max_pos hint take the wide selection tier (qsa.h).
"""
import math
import os
import re

import torch
from safetensors import safe_open

from golden import cmp

QSA = [i for i in range(48) if i % 4 == 3]
KEY = "language_model.model.layers.{}.self_attn."
BF = torch.bfloat16
CHUNK = 64
EPS = 1e-6
WINDOW = int(os.environ.get("Q38_QSA_WINDOW", "262144"))
_m = None


def _mod(ctx):
    global _m
    if _m is None:
        _m = ctx.ext("qsa", ["src/kernels/qsa/qsa.cu", "src/kernels/qsa/qsa_prefill.cu", "src/core/weights.cc",
                             "src/model/patches.cc"])
        _m.open(ctx.model_dir, ctx.table, ctx.rank)
        for i in QSA:
            _m.upload(i)
        if WINDOW != 262144:
            _rope_default(_m)
    return _m


def _rope_default(m):
    """The suite's table: default YaRN parameters, WINDOW rows (after a test changed it)."""
    m.rope_set(1.5625, 262144, 262144, WINDOW)


def _cache(cap):
    cap = (cap + 3) // 4 * 4
    return [torch.zeros(cap, 512, dtype=BF, device="cuda"), torch.zeros(cap, 128, dtype=BF, device="cuda"),
            torch.zeros(cap // 4, 128, dtype=BF, device="cuda")]


def _call(m, i, seqs, q=False, hint=False):
    """One qsa_core call over sequences [(cache, qkv [T,7296] cuda, first pos)] → attn, sel (, Q, qi). hint: pass the
    call's largest position as qsa_core's max_pos (else unknown)."""
    qkv = torch.cat([s[1] for s in seqs]).contiguous()
    pos = torch.cat([torch.arange(s[2], s[2] + s[1].shape[0], dtype=torch.int32, device="cuda") for s in seqs])
    seq = [0]
    for s in seqs:
        seq.append(seq[-1] + s[1].shape[0])
    a = m.args([s[0] for s in seqs], seq)
    M = qkv.shape[0]
    attn = torch.empty(M, 3072, dtype=BF, device="cuda")
    sel = torch.empty(M, 2052, dtype=torch.int32, device="cuda")
    m.run(i, qkv, pos, attn, a[1], a[2], sel, max(s[2] + s[1].shape[0] - 1 for s in seqs) if hint else -1)
    if q:
        Q, qi = m.last_q(M)
        return attn, sel, Q, qi
    torch.cuda.synchronize()
    return attn, sel


def _feed(m, i, cache, qkv, pos0, chunks, q=False, hint=False):
    """Rows of one sequence in calls of the given sizes (cycled) → attn [n,3072], sel [n,2052] (, Q, qi)."""
    n, s, k, outs = qkv.shape[0], 0, 0, []
    while s < n:
        e = min(n, s + chunks[k % len(chunks)])
        outs.append(_call(m, i, [(cache, qkv[s:e], pos0 + s)], q, hint))
        s, k = e, k + 1
    return [torch.cat(x) for x in zip(*outs)]


def _step(ctx, k):
    return safe_open(os.path.join(ctx.golden.dir, f"step{k:04d}.safetensors"), "pt", device="cpu")


def _reqs(ctx):
    return sorted(ctx.golden.requests().values(), key=lambda r: r[0][0])  # short, long, short2


def _kept(ctx, k):
    return next(s["rows_kept"] for s in ctx.golden.steps if s["step"] == k)


# ---- torch references ----------------------------------------------------------------------------------------------
def _rope_vllm(n, device, factor=1.5625, orig=262144):
    """vLLM cos_sin_cache rows [0, n), torch fp32 → bf16: factor > 1 MRotaryEmbedding(YaRN) (YaRNScalingRotaryEmbedding
    code; correction range from `orig` = the max_position it is given), factor ≤ 1 RotaryEmbedding (rope_type
    "default"). → (table, (lo, hi) or None)."""
    rot, base = 64, 10000000
    if factor <= 1:
        inv = 1.0 / (base ** (torch.arange(0, rot, 2, dtype=torch.float, device=device) / rot))
        freqs = torch.einsum("i,j -> ij", torch.arange(n, dtype=torch.float, device=device), inv)
        return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(BF), None

    def corr(r):
        return (rot * math.log(orig / (r * 2 * math.pi))) / (2 * math.log(base))

    low, high = max(math.floor(corr(32)), 0), min(math.ceil(corr(1)), rot - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (torch.arange(0, rot, 2, dtype=torch.float, device=device) / rot)
    inv_extra, inv_inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    mask = 1 - torch.clamp((torch.arange(rot // 2, dtype=torch.float, device=device) - low) / (high - low), 0, 1)
    inv = inv_inter * (1 - mask) + inv_extra * mask
    freqs = torch.einsum("i,j -> ij", torch.arange(n, dtype=torch.float32, device=device), inv)
    ms = 0.1 * 1 * math.log(factor) + 1.0
    return torch.cat((freqs.cos() * ms, freqs.sin() * ms), dim=-1).to(BF), (low, high)


def _ref_main(x, w, cs):
    """vLLM's qsa_prepare main path: fp32 Gemma RMSNorm, bf16 normed, fp32 NeoX rotate of dims 0..63 with Triton's
    contraction o1 = fma(x1, c, −x2·s), o2 = fma(x2, c, x1·s) (fma via fp64)."""
    x = x.float()
    inv = torch.rsqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + EPS)
    y = (x * inv * (w.float() + 1)).to(BF).float()
    c, s = cs[..., :32].float(), cs[..., 32:].float()
    x1, x2 = y[..., :32], y[..., 32:64]
    o1 = (x1.double() * c.double() - (x2 * s).double()).float()
    o2 = (x2.double() * c.double() + (x1 * s).double()).float()
    return torch.cat([o1, o2, y[..., 64:]], -1).to(BF)


def _ref_idx(x, w, cs, comp=False):
    """vLLM's _norm_rope as Triton 3.7 compiles it: fp32 RMSNorm → bf16, o1 = fma_bf16(x1, c, −bf16(x2·s)),
    o2 = fma_bf16(x2, c, bf16(x1·s)) — for the 1-row compressed-key tile fma_bf16(x1, s, bf16(x2·c)) (fma via fp32)."""
    x = x.float()
    inv = torch.rsqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + EPS)
    y = (x * inv * (w.float() + 1)).to(BF)
    c, s = cs[..., :32].float(), cs[..., 32:].float()
    r0, r1 = y[..., :32].float(), y[..., 32:64].float()
    o0 = (r0 * c - (r1 * s).to(BF).float()).to(BF)
    o1 = (r0 * s + (r1 * c).to(BF).float()).to(BF) if comp else (r1 * c + (r0 * s).to(BF).float()).to(BF)
    return torch.cat([o0, o1, y[..., 64:]], -1)


def _ndiff(a, b):
    """(# differing elements, max |Δ| in bf16 ulps of the larger magnitude)."""
    d = a.float() != b.float()
    if not d.any():
        return 0, 0
    x, y = a.float()[d], b.float()[d]
    ulp = torch.ldexp(torch.ones_like(x), torch.frexp(torch.maximum(x.abs(), y.abs()))[1] - 8)
    return int(d.sum()), int(((x - y).abs() / ulp).max().round())


def _sel_sets(sel):
    return [set(r[:r[-1]].tolist()) for r in sel.cpu()]


def _sel_explain(cache, qi, sel_got, ref_row, pos):
    """Blocks that differ: their rank in our scores (torch from our caches/q) and the gap to the 512th score."""
    vis = (pos + 1) // 4
    if vis <= 512:
        return len(set(sel_got) ^ set(ref_row)), -1, float("inf")
    k = cache[2][:vis].float()
    s = torch.relu(k @ qi.float().T).sum(-1)
    order = torch.argsort(-s, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(vis, device=s.device)
    g = set(t // 4 for t in sel_got if t < 4 * vis)
    r = set(t // 4 for t in ref_row if t < 4 * vis)
    diff = sorted((g ^ r))
    thr = s[order[511]].item()
    rel = max(abs(s[b].item() - thr) / max(thr, 1e-30) for b in diff)
    return len(g ^ r), max(rank[b].item() for b in diff), rel


# ---- tests ---------------------------------------------------------------------------------------------------------
ROPE_MODES = (  # (tag, factor, correction-range length, expected (lo, hi))
    ("default YaRN×1.5625 ramp 256K", 1.5625, 262144, (14, 22)),
    ("plain RoPE", 1.0, 262144, None),
    ("YaRN×4 ramp 256K", 4.0, 262144, (14, 22)),
    ("YaRN×4 ramp 1M", 4.0, 1048576, (16, 24)),
    ("YaRN×4 ramp 32K", 4.0, 32768, (10, 17)),  # hi − lo = 7: torch's GPU ÷ scalar is × its fp32 reciprocal
)


def test_rope(ctx):
    """The shared table (default parameters) vs vLLM's formula in torch on the GPU (bit for bit), rows < 262,144; then
    every mode (ROPE_MODES) over all 1,048,576 rows through qsa_rope_fill (the same kernel) vs torch GPU — mismatches
    counted with their max bf16 ulp and smallest row, never tolerated; the CPU computation's differences for info."""
    m = _mod(ctx)
    n = 262144
    got = m.rope(n)
    ref, (lo, hi) = _rope_vllm(n, "cuda")
    cpu, _ = _rope_vllm(n, "cpu")
    ng = int((got != ref).sum())
    nc = int((got.cpu() != cpu).sum())
    ctx.detail(f"qsa rope lo {lo} hi {hi} · vs torch GPU diff {ng} · vs torch CPU diff {nc} / {got.numel()}")
    ok = ng == 0 and (lo, hi) == (14, 22)
    msg = [f"shared table {n} rows == vLLM (torch GPU) bitwise {got.numel() - ng}/{got.numel()} ({nc} values differ "
           f"from the CPU computation)"]
    N = 1 << 20
    del ref, cpu
    for tag, f, ramp, lohi in ROPE_MODES:
        t = m.rope_fill(f, 262144, ramp, N)
        r, lh = _rope_vllm(N, "cuda", f, ramp)
        nd, ulp = _ndiff(t, r)
        row = int((t != r).any(1).nonzero()[0]) if nd else -1
        nc = int((t.cpu() != _rope_vllm(N, "cpu", f, ramp)[0]).sum())
        same = f != 1.5625 or ramp != 262144 or torch.equal(t[:n], got)
        ok &= nd == 0 and lh == lohi and same
        ctx.detail(f"qsa rope {tag}: lo/hi {lh} · {N} rows vs torch GPU: {nd} differ (max {ulp} ulp, first row {row})"
                   f" · vs torch CPU {nc} differ" + ("" if same else " · rows < 262144 ≠ shared table"))
        msg.append(f"{tag}{'' if lh is None else f' ({lh[0]}..{lh[1]})'} {N} rows, differing values {nd}"
                   + (f" (max {ulp}ulp)" if nd else "") + ("" if same else " differs from the shared table"))
        del t, r
    torch.cuda.empty_cache()
    return {"ok": ok, "msg": " · ".join(msg)}


def test_prepare(ctx):
    """Q, K‖V cache, raw/compressed indexer keys, indexer q vs the torch op-by-op reference (short request rows)."""
    m = _mod(ctx)
    steps = _reqs(ctx)[0]
    tab = m.rope(4096)
    out = []
    for i in (3, 27, 47):
        P = KEY.format(i)
        rows = []
        for k, n, _ in steps:
            with _step(ctx, k) as f:
                rows.append(f.get_tensor(P + "qkv_proj#0.out.0"))
        qkv = torch.cat(rows).cuda()
        n = qkv.shape[0]
        c = _cache(n + 4)
        _, _, Q, qi = _feed(m, i, c, qkv, 0, [7, 64, 1, 13], q=True)
        qn, kn, iqn, ikn = m.upload(i)
        cs = tab[:n]
        rq = _ref_main(qkv[:, :6144].view(n, 12, 2, 256)[:, :, 0], qn, cs[:, None])
        rk = _ref_main(qkv[:, 6144:6400], kn, cs)
        riq = _ref_idx(qkv[:, 6656:7168].view(n, 4, 128), iqn, cs[:, None])
        ik = qkv[:, 7168:]
        g = n // 4
        pooled = (ik[:4 * g].float().view(g, 4, 128).sum(1) * 0.25).to(BF)
        rc = _ref_idx(pooled, ikn, tab[0:4 * g:4], comp=True)
        res = {"Q": _ndiff(Q, rq), "K": _ndiff(c[0][:n, :256], rk), "V": _ndiff(c[0][:n, 256:], qkv[:, 6400:6656]),
               "ik": _ndiff(c[1][:n], ik), "ikc": _ndiff(c[2][:g], rc), "iq": _ndiff(qi, riq)}
        ctx.detail(f"qsa prepare L{i} rows {n} " + " ".join(f"{k} {v[0]}({v[1]}ulp)" for k, v in res.items()))
        out.append(res)
    tot = {k: (sum(r[k][0] for r in out), max(r[k][1] for r in out)) for k in out[0]}
    exact = tot["V"][0] == 0 and tot["ik"][0] == 0
    ok = exact and all(v[1] <= 1 for v in tot.values())
    return {"ok": ok, "msg": "bf16 values differing from the torch reference (max ulp): "
            + " ".join(f"{k} {v[0]}({v[1]})" for k, v in tot.items()) + " · layers 3·27·47 · short-request rows"}


def test_topk(ctx):
    """Exact top-512 with ties (scores α_j·S0, α_j = 2^level, many equal) → lowest block index wins; all-zero scores."""
    m = _mod(ctx)
    nb, p = 4096, 4 * 4096 + 1
    c = _cache(p + 4)
    c[0].normal_()
    qkv = torch.randn(1, 7296, device="cuda").to(BF)
    _, _, _, qi = _call(m, 3, [(c, qkv, p)], q=True)
    v = qi[0, 0]
    g = torch.Generator(device="cuda").manual_seed(0)
    bad = []
    for case in ("levels", "zero"):
        if case == "levels":
            lev = torch.randint(0, 8, (nb,), device="cuda", generator=g)
            c[2][:nb] = (v.float()[None] * torch.exp2(lev.float())[:, None]).to(BF)
            key = -lev * nb + torch.arange(nb, device="cuda")  # ascending = (level desc, index asc)
        else:
            c[2][:nb] = 0
            key = torch.arange(nb, device="cuda")
        _, sel = _call(m, 3, [(c, qkv, p)])
        blocks = torch.sort(torch.argsort(key)[:512]).values
        ref = torch.cat([(blocks[:, None] * 4 + torch.arange(4, device="cuda")).flatten(),
                         torch.tensor([4 * nb, 4 * nb + 1, -1, 2050], device="cuda")]).int()
        if not torch.equal(sel[0], ref):
            bad.append(case)
    return {"ok": not bad, "msg": "ties exact (lowest block first) · 2^k-scaled scores 4096 blocks · all-zero scores"
            if not bad else f"differs: {' '.join(bad)}"}


def _golden_request(ctx, which, chunks=(CHUNK,)):
    m = _mod(ctx)
    steps = _reqs(ctx)[which]
    k0, n0, _ = steps[0]
    kept = _kept(ctx, k0)
    caches = {i: _cache(n0 + len(steps) + 4) for i in QSA}
    R = {ph: {"o": [], "q": [], "rows": 0, "same": 0, "diff": 0, "rank": 0, "gap": 0.0, "o_same": [], "ne": 0, "n": 0,
              "ulp": 0} for ph in ("pre", "dec")}

    def compare(ph, i, attn, sel, Q, f, pos, cache, qi):
        P = KEY.format(i)
        ro = f.get_tensor(P + "fn.forward_qsa#0.post.output").reshape(-1, 3072).cuda()
        rq = f.get_tensor(P + "fn.forward_qsa#0.in.query").cuda()
        rs = f.get_tensor(P + "indexer#0.out.0")
        r = R[ph]
        r["o"].append(cmp(f"L{i}", attn, ro))
        r["q"].append(cmp(f"L{i}", Q, rq))
        ne, ulp = _ndiff(attn, ro)
        r["ne"], r["n"], r["ulp"] = r["ne"] + ne, r["n"] + attn.numel(), max(r["ulp"], ulp)
        gs, rss = _sel_sets(sel), _sel_sets(rs)
        same = [a == b for a, b in zip(gs, rss)]
        r["rows"] += len(same)
        r["same"] += sum(same)
        for j, s in enumerate(same):
            if not s:
                nd, rk, gap = _sel_explain(cache, qi[j], gs[j], rss[j], pos[j])
                r["diff"] += nd
                r["rank"], r["gap"] = max(r["rank"], rk), max(r["gap"], gap)
                ctx.detail(f"qsa sel {ph} L{i} pos {pos[j]} differ {nd} blocks · max rank {rk} · rel gap {gap:.2e}")
        if any(same):
            idx = torch.tensor([j for j, s in enumerate(same) if s], device="cuda")
            r["o_same"].append(cmp(f"L{i}", attn[idx], ro[idx]))

    with _step(ctx, k0) as f:
        for i in QSA:
            qkv = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
            assert qkv.shape[0] == n0
            attn, sel, Q, qi = _feed(m, i, caches[i], qkv, 0, list(chunks), q=True)
            compare("pre", i, attn[n0 - kept:], sel[n0 - kept:], Q[n0 - kept:], f, list(range(n0 - kept, n0)),
                    caches[i], qi[n0 - kept:])
    for k, n, comp in steps[1:]:
        with _step(ctx, k) as f:
            for i in QSA:
                qkv = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
                attn, sel, Q, qi = _call(m, i, [(caches[i], qkv, comp)], q=True)
                compare("dec", i, attn, sel, Q, f, [comp], caches[i], qi)
    return steps, R


def _w(rs):
    if not rs:
        return 0, 0, 1
    return max(x["max_abs"] for x in rs), max(x["max_rel"] for x in rs), min(x["cos"] for x in rs)


def _fmt(rs):
    a, r, c = _w(rs)
    return f"abs {a:.2e} rel {r:.2e} cos {c:.6f}"


def _golden_test(ctx, which, chunks=(CHUNK,)):
    steps, R = _golden_request(ctx, which, chunks)
    name = ["short", "long", "short2"][which] + ("" if tuple(chunks) == (CHUNK,) else f" T={chunks[0]}")
    for ph, r in R.items():
        for x, y in zip(r["o"], r["q"]):
            ctx.detail(f"qsa {name} {ph} {x['name']} out abs {x['max_abs']:.3e} rel {x['max_rel']:.3e} cos "
                       f"{x['cos']:.7f} · Q abs {y['max_abs']:.3e} rel {y['max_rel']:.3e} cos {y['cos']:.7f}")
    p, d = R["pre"], R["dec"]

    def sel(r):
        s = f"selection {r['same']}/{r['rows']} rows match"
        return s + (f" ({r['diff']} blocks differ, our rank ≤{r['rank']}, gap to the 512th ≤{r['gap']:.1e})"
                    if r["diff"] else "")

    def same(r):
        return f"bf16 equal {100 * (1 - r['ne'] / max(r['n'], 1)):.1f}% (max {r['ulp']}ulp)"

    msg = (f"{name} decode {len(steps) - 1} steps: {sel(d)} · output {_fmt(d['o'])}, {same(d)} · Q {_fmt(d['q'])} | "
           f"prefill {steps[0][1]} rows (last {p['rows'] // len(QSA)}): {sel(p)} · output {_fmt(p['o'])}, {same(p)}"
           + (f" · same-selection rows only {_fmt(d['o_same'])}" if d["same"] < d["rows"] else "")
           + f" · {len(QSA)} layers")
    ok = (_w(d["o"])[2] >= 0.999 and _w(p["o"])[2] >= 0.999 and _w(d["q"])[1] <= 1e-2
          and (d["same"] == d["rows"] or d["rank"] < 600))
    return {"ok": ok, "msg": msg}


def test_golden_short(ctx):
    return _golden_test(ctx, 0)


def test_golden_long(ctx):
    return _golden_test(ctx, 1)


def test_golden_long_prefill(ctx):
    """Long request through the prefill path: 8,603 prompt rows in calls of 8,192 + 411 rows, then the decode steps
    from those caches (the reference's prefill attention used one split with a running max → tolerance as for 64-row
    calls)."""
    _mod(ctx).reserve(8192)
    return _golden_test(ctx, 1, (8192,))


def test_prefill_bitwise(ctx):
    """Bitwise: the long prompt (8,603 rows, full-row qkv) one row per call (decode path), in 64-row calls (decode path),
    in 2,048- and 8,192-row calls (prefill path) and in a mixed plan that alternates the paths across the dense/sparse
    boundary → identical selections, outputs and caches (layers 3, 27, 47); plus a two-sequence prefill-path call
    (long rows 2000..2999 with the 92-row short prompt, both orders) == each sequence alone."""
    m = _mod(ctx)
    m.reserve(8192)
    k0, n0, _ = _reqs(ctx)[1][0]
    plans = {"T=1": [1], "T=64": [64], "T=2048": [2048], "T=8192": [8192], "mixed": [65, 1, 3000, 7, 2048, 129]}
    bad, nrows = [], 0
    for i in (3, 27, 47):
        with _step(ctx, k0) as f:
            X = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
        assert X.shape[0] == n0
        ref = None
        for name, ch in plans.items():
            c = _cache(n0 + 4)
            attn, sel = _feed(m, i, c, X, 0, ch)
            got = [attn, sel, c[0][:n0], c[1][:n0], c[2][:n0 // 4]]
            if ref is None:
                ref = got
                nrows += attn.shape[0]
                continue
            for tag, a, b in zip(("output", "selection", "KV", "ik", "ikc"), got, ref):
                if not torch.equal(a, b):
                    nd = int((a != b).reshape(a.shape[0], -1).any(1).sum())
                    bad.append(f"L{i} {name} {tag} {nd} rows")
                    ctx.detail(f"qsa prefill bitwise L{i} {name} {tag}: {nd} rows differ")
            del got, attn, sel, c
        if i == 3:  # B=2 through the prefill path: long rows 2000..2999 + the short prompt in one call (both orders)
            ys = []
            for k, n, _ in _reqs(ctx)[0]:
                with _step(ctx, k) as f:
                    ys.append(f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0"))
            Y = torch.cat(ys).cuda()
            cy = _cache(Y.shape[0] + 4)
            ry = _feed(m, i, cy, Y, 0, [1])
            for order in (0, 1):
                cx, cy2 = _cache(n0 + 4), _cache(Y.shape[0] + 4)
                _feed(m, i, cx, X[:2000], 0, [2000])
                seqs = [(cx, X[2000:3000], 2000), (cy2, Y, 0)]
                at, sl = _call(m, i, seqs[::-1] if order else seqs)
                ax, ay = (at[Y.shape[0]:], at[:Y.shape[0]]) if order else (at[:1000], at[1000:])
                sx, sy = (sl[Y.shape[0]:], sl[:Y.shape[0]]) if order else (sl[:1000], sl[1000:])
                if not (torch.equal(ax, ref[0][2000:3000]) and torch.equal(sx, ref[1][2000:3000])
                        and torch.equal(ay, ry[0]) and torch.equal(sy, ry[1]) and torch.equal(cx[0][:3000], ref[2][:3000])
                        and torch.equal(cy2[0], cy[0]) and torch.equal(cy2[2], cy[2])):
                    bad.append(f"B=2 prefill {order}")
        torch.cuda.empty_cache()
    return {"ok": not bad, "msg": f"bitwise: T=1 (decode path) = T=64 = T=2048 = T=8192 (prefill path) = mixed · {n0} "
            "rows × layers 3·27·47 · selection·output·KV·ik·compressed keys · B=2 prefill call (either order)"
            if not bad else "differs: " + ", ".join(bad[:6])}


def test_invariance(ctx):
    """Bitwise: T>1 calls == other chunkings (1 row at a time over the sparse region) and B=2 batches == alone."""
    m = _mod(ctx)
    reqs = _reqs(ctx)
    i = 3
    P = KEY.format(i)
    with _step(ctx, reqs[1][0][0]) as f:
        X = f.get_tensor(P + "qkv_proj#0.out.0")[:2600].cuda()
    ys = []
    for k, n, _ in reqs[0]:
        with _step(ctx, k) as f:
            ys.append(f.get_tensor(P + "qkv_proj#0.out.0"))
    Y = torch.cat(ys).cuda()
    bad = []

    def same(tag, a, b):
        if not all(torch.equal(x, y) for x, y in zip(a, b)):
            bad.append(tag)

    cA = _cache(2604)
    A = _feed(m, i, cA, X, 0, [CHUNK])
    cB = _cache(2604)
    B1 = _feed(m, i, cB, X[:2000], 0, [CHUNK])
    B2 = _feed(m, i, cB, X[2000:], 2000, [1, 2, 3, 5, 7, 11, 13, 17, 31, 64])
    same("T>1 vs split", A, [torch.cat([B1[0], B2[0]]), torch.cat([B1[1], B2[1]])])
    same("cache", cA, cB)
    cY = _cache(Y.shape[0] + 4)
    AY = _feed(m, i, cY, Y, 0, [CHUNK])
    # batched: X in 32-row calls and Y in 29-row calls together; then swapped order
    for order in (0, 1):
        cx, cy = _cache(2604), _cache(Y.shape[0] + 4)
        ox, oy, sx, sy = [], [], [], []
        a = b = 0
        while a < X.shape[0] or b < Y.shape[0]:
            seqs, tags = [], []
            if a < X.shape[0]:
                e = min(X.shape[0], a + 32)
                seqs.append((cx, X[a:e], a)), tags.append(("x", a, e))
                a = e
            if b < Y.shape[0]:
                e = min(Y.shape[0], b + 29)
                seqs.append((cy, Y[b:e], b)), tags.append(("y", b, e))
                b = e
            if order:
                seqs, tags = seqs[::-1], tags[::-1]
            at, sl = _call(m, i, seqs)
            s = 0
            for t, lo, hi in tags:
                (ox if t == "x" else oy).append(at[s:s + hi - lo])
                (sx if t == "x" else sy).append(sl[s:s + hi - lo])
                s += hi - lo
        same(f"B=2 x{order}", A, [torch.cat(ox), torch.cat(sx)])
        same(f"B=2 y{order}", AY, [torch.cat(oy), torch.cat(sy)])
        same(f"cache B=2 {order}", cA + cY, cx + cy)
    return {"ok": not bad, "msg": "bitwise: 64-row calls = 1–64-row splits (sparse region 2000–2600) = B=2 batch "
            "(either order) · output·selection·cache" if not bad else f"differs: {', '.join(bad)}"}


def _bench(m, i, sets, iters=24, reps=7):
    """Best µs per call (another agent shares the GPU), cycling through `sets` (each a list of (cache, qkv, pos0)) so
    the caches are cold in L2 as in a real step (12 layers' caches ≫ 24 MB L2); modelled bytes; per-kernel µs."""
    runs = []
    for seqs in sets:
        qkv = torch.cat([s[1] for s in seqs]).contiguous()
        pos = torch.cat([torch.arange(s[2], s[2] + s[1].shape[0], dtype=torch.int32, device="cuda") for s in seqs])
        seq = [0]
        for s in seqs:
            seq.append(seq[-1] + s[1].shape[0])
        a = m.args([s[0] for s in seqs], seq)
        attn = torch.empty(qkv.shape[0], 3072, dtype=BF, device="cuda")
        runs.append((qkv, pos, attn, a))
    sel = torch.empty(runs[0][0].shape[0], 2052, dtype=torch.int32, device="cuda")
    qkv, pos, attn, a = runs[0]
    m.run(i, qkv, pos, attn, a[1], a[2], sel)

    def go(n):
        for k in range(n):
            qkv, pos, attn, a = runs[k % len(runs)]
            m.run(i, qkv, pos, attn, a[1], a[2])

    go(len(runs))
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        go(iters)
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / iters)
    seqs = sets[0]
    cnt = sel[:, -1].long()
    nvis = [(s[2] + s[1].shape[0]) // 4 for s in seqs]
    by = sum(v * 256 for v in nvis if v > 512) + int(cnt.sum()) * 1024 + qkv.shape[0] * (7296 + 3072) * 2
    prof = ""
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            go(2 * len(runs))
            torch.cuda.synchronize()
        ev = [(e.key, getattr(e, "device_time", getattr(e, "cuda_time", 0))) for e in pr.key_averages()]
        prof = " ".join(f"{re.search(r'k_[a-z]+', k).group(0)} {t:.1f}" for k, t in ev
                        if t > 0 and re.search(r"k_[a-z]+", k))
    except Exception as ex:  # noqa: BLE001
        prof = f"(profiler: {type(ex).__name__})"
    return best, by, prof


def test_perf(ctx):
    m = _mod(ctx)
    m.reserve(64)
    i = 3
    P = KEY.format(i)
    reqs = _reqs(ctx)
    rows = []
    torch.manual_seed(0)

    def rec(tag, sets):
        us, by, prof = _bench(m, i, sets)
        rows.append((tag, us, by / us / 1e3))
        ctx.detail(f"qsa perf {tag}: {us:.1f} us · {by / 1e6:.2f} MB · {by / us / 1e3:.0f} GB/s · kernels(us): {prof}")

    for w, name in ((0, "L=62"), (1, "L=8604")):
        with _step(ctx, reqs[w][0][0]) as f:
            S = f.get_tensor(P + "qkv_proj#0.out.0").cuda()
        with _step(ctx, reqs[w][1][0]) as f:
            s1 = f.get_tensor(P + "qkv_proj#0.out.0").cuda()
        c = _cache(S.shape[0] + 8)
        _feed(m, i, c, S, 0, [CHUNK])
        cs = [c] + [[t.clone() for t in c] for _ in range(11)]  # 12 layers' worth of caches
        rec(name, [[(x, s1, S.shape[0])] for x in cs])
        del cs, c
    for L, ncopy in ((32768, 8), (131072, 6)):
        cs = []
        for _ in range(ncopy + 3):
            c = _cache(L + 8)
            c[0].normal_(), c[1].normal_(), c[2].normal_()
            cs.append(c)
        q = (torch.randn(4, 7296, device="cuda") * 2).to(BF)
        p = L - 2  # count 2051 (3-token tail)
        rec(f"L={L // 1024}K", [[(cs[k], q[:1], p)] for k in range(ncopy)])
        rec(f"L={L // 1024}K T=4", [[(cs[k], q, p - 3)] for k in range(ncopy)])
        rec(f"L={L // 1024}K B=4", [[(cs[(k + b) % len(cs)], q[b:b + 1], p) for b in range(4)] for k in range(ncopy)])
        del cs
    msg = " · ".join(f"{t} {us:.0f}µs {gb:.0f}GB/s" for t, us, gb in rows)
    return {"ok": True, "msg": msg + " (per layer, caches outside L2, best of 7)"}


def _kprof(m, fn, n=3):
    """Per-kernel device µs (average per call) of fn() via torch.profiler."""
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
        out = {}
        for e in pr.key_averages():
            t = getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0))
            k = re.search(r"k_[a-z]+", e.key)
            if t > 0 and k:
                out[k.group(0)] = out.get(k.group(0), 0) + t / n
        return out
    except Exception as ex:  # noqa: BLE001
        return {"profiler": type(ex).__name__}


def test_prefill_perf(ctx):
    """Prefill path speed per layer (layer 3; prompt rows = the long golden prompt repeated, so selections behave like
    real text): T = 2,048 / 8,192 rows ending at contexts 8K and 32K (best of 5, per-kernel split), and a full 128K
    prefill in 8,192-row chunks (every chunk timed) → 12 layers vs the 25 % budget of a 5,000 tok/s prefill."""
    m = _mod(ctx)
    m.reserve(8192)
    i = 3
    k0, n0, _ = _reqs(ctx)[1][0]
    with _step(ctx, k0) as f:
        X = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
    L = 131072
    c = _cache(L + 8)

    def prep(p0, T):
        qkv = X[torch.arange(p0, p0 + T, device="cuda") % n0].contiguous()
        pos = torch.arange(p0, p0 + T, dtype=torch.int32, device="cuda")
        args = m.args([c], [0, T])
        attn = torch.empty(T, 3072, dtype=BF, device="cuda")
        return lambda: m.run(i, qkv, pos, attn, args[1], args[2])

    def best(fn, reps=5):
        fn()
        torch.cuda.synchronize()
        t = float("inf")
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            t = min(t, e0.elapsed_time(e1) * 1e3)
        return t

    rows, chunk_us, pts = [], [], {}
    for k in range(L // 8192):
        p0 = k * 8192
        fn = prep(p0, 8192)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        chunk_us.append(e0.elapsed_time(e1) * 1e3)
        ctxk = (p0 + 8192) // 1024
        if ctxk in (8, 32, 128):
            for T in ((2048, 8192) if ctxk < 128 else (8192,)):
                g = prep(p0 + 8192 - T, T)
                us = best(g)
                kp = _kprof(m, g)
                sc = kp.get("k_pscore", 0) + kp.get("k_tau", 0) + kp.get("k_cscore", 0)  # base or P009 kernels
                se = kp.get("k_select", 0) + kp.get("k_cselect", 0) + kp.get("k_cfall", 0)
                at = kp.get("k_pattn", 0)
                pts[(T, ctxk)] = us
                rows.append(f"T={T} {ctxk}K {us / 1e3:.1f}ms (score {sc / 1e3:.1f} select {se / 1e3:.1f} attention "
                            f"{at / 1e3:.1f} prepare {kp.get('k_prepare', 0) / 1e3:.2f})")
                ctx.detail(f"qsa prefill perf T={T} ctx {ctxk}K: {us:.0f} us/layer · kernels(us): "
                           + " ".join(f"{n} {t:.0f}" for n, t in sorted(kp.items())))
                del g
            chunk_us[-1] = min(chunk_us[-1], pts[(8192, ctxk)])
    tot = sum(chunk_us) * 12 / 1e6
    # projection from the 8K / 32K points (T=8192): per-chunk time linear in the context
    t8, t32 = pts[(8192, 8)], pts[(8192, 32)]
    proj = sum(t8 + (t32 - t8) * (8 * (k + 1) - 8) / 24 for k in range(16)) * 12 / 1e6
    budget = 131072 / 5000 * 0.25
    ctx.detail("qsa prefill perf 128K chunks (us): " + " ".join(f"{t:.0f}" for t in chunk_us))
    ok = tot <= budget
    return {"ok": ok, "msg": " · ".join(rows) + f" | 128K prefill 12 layers measured {tot:.2f}s (estimated from 8K·32K "
            f"{proj:.2f}s) · target ≤{budget:.2f}s (per layer, best)"}


# ---- P009 qsa_fast_index: patch on == off ---------------------------------------------------------------------------
P009 = 9  # patches.def


def _best(fn, reps):
    fn()
    torch.cuda.synchronize()
    t = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        t = min(t, e0.elapsed_time(e1) * 1e3)
    return t


def _p9_stats(m, pos):
    """Candidates of the last P009 call's rows (pos [rows] of its first round): sampled rows (> 3,072 blocks) median /
    min / max total and largest split, every-block rows, fallback rows."""
    cnt, nfb = m.p009_last(pos.shape[0])
    vis = (pos.long() + 1) // 4
    tot = cnt.sum(1)
    sm = vis > 3072
    s = f"fallback {nfb} rows"
    if sm.any():
        t = tot[sm].float()
        s = (f"sampled rows {int(sm.sum())}: candidates median {int(t.median())} min {int(t.min())} max {int(t.max())} "
             f"(largest split {int(cnt[sm].max())}) · " + s)
    dm = (vis > 512) & ~sm
    if dm.any():
        s = f"every-block rows {int(dm.sum())} (= visible blocks {bool(torch.equal(tot[dm], vis[dm].int()))}) · " + s
    return s


def test_p009(ctx):
    """P009 qsa_fast_index == base, bitwise (selection and output), patch off then on per call: a whole 131,072-row
    prefill of layer 3 in 8,192-row calls (the long golden prompt repeated, as prefill_perf), every call timed both ways
    (best of 3) → 12-layer totals, per-kernel µs at 8K / 32K / 128K; 128–2,048-row calls at 8K / 32K / 128K (P009 from
    512 rows, best of 5); a B=2 call whose sequences meet inside a row tile (4,000 + 4,192 rows, both orders); one
    16,384-row call (two P009 rounds); constructed scores in 512-row calls (2^k levels with ties, all-zero blocks:
    every-block candidates, sampled rows, crowded threshold bin, fallback rows)."""
    m = _mod(ctx)
    m.reserve(8192)
    was = bool(m.patch_on(P009))
    i = 3
    k0, n0, _ = _reqs(ctx)[1][0]
    with _step(ctx, k0) as f:
        X = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
    L = 131072
    c = _cache(L + 8)
    bad, ncase = [], 0

    def rows(p0, T):
        return X[torch.arange(p0, p0 + T, device="cuda") % n0].contiguous()

    def prep(seqs):
        qkv = torch.cat([s[1] for s in seqs]).contiguous()
        pos = torch.cat([torch.arange(s[2], s[2] + s[1].shape[0], dtype=torch.int32, device="cuda") for s in seqs])
        st = [0]
        for s in seqs:
            st.append(st[-1] + s[1].shape[0])
        a = m.args([s[0] for s in seqs], st)
        attn = torch.empty(qkv.shape[0], 3072, dtype=BF, device="cuda")
        sel = torch.empty(qkv.shape[0], 2052, dtype=torch.int32, device="cuda")
        return (lambda: m.run(i, qkv, pos, attn, a[1], a[2], sel)), attn, sel, pos

    def both(tag, seqs, reps=0):
        nonlocal ncase
        out = []
        for on in (False, True):
            m.patch_set("qsa_fast_index", on)
            run, attn, sel, pos = prep(seqs)
            run()
            torch.cuda.synchronize()
            out.append((run, attn, sel))
        (r0, a0, s0), (r1, a1, s1) = out
        ncase += 1
        if not (torch.equal(s0, s1) and torch.equal(a0, a1)):
            bad.append(f"{tag} selection {int((s0 != s1).any(1).sum())} rows output "
                       f"{int((a0 != a1).any(1).sum())} rows")
            ctx.detail(f"qsa p009 {tag}: DIFFERENT sel rows {int((s0 != s1).any(1).sum())} attn rows "
                       f"{int((a0 != a1).any(1).sum())}")
        t = [0.0, 0.0]
        if reps:
            m.patch_set("qsa_fast_index", False)
            t[0] = _best(r0, reps)
            m.patch_set("qsa_fast_index", True)
            t[1] = _best(r1, reps)
        return t, pos, r0, r1

    try:
        tot, pts = [0.0, 0.0], {}
        for k in range(L // 8192):
            p0 = k * 8192
            ck = (p0 + 8192) // 1024
            t, pos, r0, r1 = both(f"{ck}K", [(c, rows(p0, 8192), p0)], reps=3)
            tot[0] += t[0]
            tot[1] += t[1]
            if ck in (8, 32, 128):
                st = _p9_stats(m, pos)
                kp = []
                for on, r in ((False, r0), (True, r1)):
                    m.patch_set("qsa_fast_index", on)
                    kp.append(_kprof(m, r))
                pts[ck] = (t, kp)
                ctx.detail(f"qsa p009 T=8192 ctx {ck}K: base {t[0]:.0f} us · P009 {t[1]:.0f} us · {st}")
                for tag, d in zip(("base", "P009"), kp):
                    ctx.detail(f"qsa p009   {tag} kernels(us): " + " ".join(f"{n} {v:.0f}" for n, v in sorted(d.items())))
        for ck in (8, 32, 128):
            for T in (128, 256, 512, 1024, 2048):
                p0 = ck * 1024 - T
                t, _, _, _ = both(f"T={T} {ck}K", [(c, rows(p0, T), p0)], reps=5)
                ctx.detail(f"qsa p009 T={T} ctx {ck}K: base {t[0]:.0f} us · P009 {t[1]:.0f} us")
        c2 = [x.clone() for x in c]
        seqs = [(c, rows(L - 4000, 4000), L - 4000), (c2, rows(65536 - 4192, 4192), 65536 - 4192)]
        both("B=2", seqs)
        both("B=2 reversed", seqs[::-1])
        del c2
        t, _, _, _ = both("T=16384 128K", [(c, rows(L - 16384, 16384), L - 16384)], reps=1)  # two P009 rounds
        ctx.detail(f"qsa p009 T=16384 ctx 128K: base {t[0]:.0f} us · P009 {t[1]:.0f} us")
        torch.cuda.empty_cache()
        gen = torch.Generator(device="cuda").manual_seed(9)
        for nb, case in ((2048, "zero"), (2048, "levels"), (4096, "levels"), (4096, "zero"), (20000, "levels"),
                         (32768, "zero")):
            T = 512  # ≥ kP9MinRows: through P009
            p0 = 4 * nb
            cc = _cache(p0 + T + 8)
            cc[0].normal_(generator=gen)
            cc[1].normal_(generator=gen)
            if case == "levels":
                v = torch.randn(128, device="cuda", generator=gen)
                lev = torch.randint(0, 8, (nb,), device="cuda", generator=gen)
                cc[2][:nb] = (v[None] * torch.exp2(lev.float())[:, None]).to(BF)
            qkv = torch.randn(T, 7296, device="cuda", generator=gen).to(BF)
            t, pos, _, _ = both(f"{case} {nb} blocks", [(cc, qkv, p0)], reps=2)
            ctx.detail(f"qsa p009 {case} {nb} blocks, {T} rows: base {t[0]:.0f} us · P009 {t[1]:.0f} us · "
                       f"{_p9_stats(m, pos)}")
            del cc
    finally:
        m.patch_set("qsa_fast_index", was)
    torch.cuda.empty_cache()

    def ms(ck):
        t, kp = pts[ck]
        idx = [sum(kp[j].get(n, 0) for n in ("k_pscore", "k_select", "k_tau", "k_cscore", "k_cselect", "k_cfall"))
               for j in (0, 1)]
        return f"{ck}K {t[0] / 1e3:.1f}→{t[1] / 1e3:.1f}ms (indexer {idx[0] / 1e3:.2f}→{idx[1] / 1e3:.2f})"

    msg = (f"patch == base bitwise {ncase} cases (selection·output)" if not bad else "differs: " + "; ".join(bad[:4]))
    msg += (f" · T=8192 per layer {ms(8)} · {ms(32)} · {ms(128)} | 128K prefill 12 layers {tot[0] * 12 / 1e6:.2f}→"
            f"{tot[1] * 12 / 1e6:.2f}s (base→patch, best of 3)")
    return {"ok": not bad, "msg": msg}


# ---- P021 qsa_decode_attn: decode attention in one cluster kernel == k_attn + k_merge (bitwise) ----------------------
P21_VARIANTS = ((1, False, "k_pattn"), (2, True, "P021"), (2, False, "P021 no hint"), (3, False, "P021 kernel"))
# attn_stage (variant, pass the max_pos hint, name); variant 0 = k_attn + k_merge


def _p21_cache(cap, gen, kind="rand"):
    """[kv, ik_raw, ik_comp] for the attention stage (it reads kv only). kind: rand | dup (runs of 8 equal K‖V rows →
    equal scores, tile-max ties) | zerok (K = 0: every score 0) | negv (V = −0 on even tokens) | wide (K, V × 2^±k)."""
    kv = torch.randn(cap, 512, device="cuda", generator=gen)
    if kind == "dup":
        kv = kv[torch.arange(cap, device="cuda") // 8 * 8]
    elif kind == "zerok":
        kv[:, :256] = 0
    elif kind == "negv":
        kv[::2, 256:] = -0.0
    elif kind == "wide":
        kv = kv * torch.exp2(torch.randint(-12, 13, (cap, 1), device="cuda", generator=gen).float())
    small = torch.zeros(8, 128, dtype=BF, device="cuda")
    return [kv.to(BF).contiguous(), small, small[:2]]


def _p21_sel(p, gen, base=None, change=0):
    """Selection row of position p (vLLM layout [2051 tokens | count]): dense 0..p if ≤ 512 blocks are visible, else
    512 blocks (`base` with `change` of them swapped) ascending + the open tail. CPU int32 [2052]."""
    s = torch.full((2052,), -1, dtype=torch.int32)
    vis = (p + 1) // 4
    if vis <= 512:
        s[:p + 1] = torch.arange(p + 1, dtype=torch.int32)
        s[-1] = p + 1
        return s
    if base is None:
        base = torch.randperm(vis, generator=gen)[:576]
    b = base[:512].clone()
    change = min(change, base.numel() - 512)
    if change:
        b[torch.randperm(512, generator=gen)[:change]] = base[512:512 + change]
    b = torch.sort(b).values
    t = (b[:, None] * 4 + torch.arange(4)).flatten()
    tail = torch.arange(vis * 4, p + 1)
    s[:2048] = t.int()
    s[2048:2048 + tail.numel()] = tail.int()
    s[-1] = 2048 + tail.numel()
    return s


def _p21_inputs(m, gen, cg, seqs, qscale=1.0, step=4):
    """seqs: [(cache, [positions of its rows], base blocks or None)] → (q, qkv, sel, args); row k of a sequence swaps
    step·k of its base blocks (≤ 64; consecutive rows share most of their selection)."""
    sel, st = [], [0]
    for c, ps, base in seqs:
        for k, p in enumerate(ps):
            sel.append(_p21_sel(p, cg, base, change=min(64, step * k)))
        st.append(st[-1] + len(ps))
    M = st[-1]
    q = (torch.randn(M, 12, 256, device="cuda", generator=gen) * qscale).to(BF).contiguous()
    qkv = torch.randn(M, 7296, device="cuda", generator=gen).to(BF)
    a = m.args([s[0] for s in seqs], st)
    return q, qkv, torch.stack(sel).cuda().contiguous(), a, max(max(ps) for _, ps, _ in seqs)


def _p21_run(m, variant, q, qkv, sel, a, mp=-1):
    attn = torch.full((q.shape[0], 3072), float("nan"), dtype=BF, device="cuda")  # unwritten → never equal
    m.attn_stage(variant, q, qkv, sel, a[1], a[2], attn, mp)
    return attn


def test_p021_bitwise(ctx):
    """P021 attention stage == k_attn + k_merge (and k_pattn == both), bitwise on the gated output: the P021 cluster
    kernel for every M and P021 as qsa_core runs it (k_attn + k_merge for the small calls). M 1, 2, 4, 5, 8, 16, 20, 33,
    64 rows in 1–4 sequences with uneven rows per sequence, long rows (128K positions, 512 spread blocks + tail, 65
    tiles; rows of a sequence share most blocks) and short ones (dense 0..p: every tile count 1..65 and the 17-tile
    group edges, so every helper/owner case of the 17th tiles), Q scale 1 / 4 / 30 / 0, caches with ties (runs of 8
    equal rows), K = 0, V = −0, wide exponents; then real rows of the long golden request (layer 3, from qsa_core's own
    Q and selection, across the dense/sparse boundary and the last 64 rows)."""
    m = _mod(ctx)
    m.reserve(64)
    gen = torch.Generator(device="cuda").manual_seed(21)
    cg = torch.Generator().manual_seed(21)
    L = 131072
    longc = [_p21_cache(L + 8, gen) for _ in range(4)]
    kinds = ["rand", "dup", "zerok", "negv", "wide"]
    shortc = {k: _p21_cache(2200, gen, k) for k in kinds}
    bad, ncase, nrow = [], 0, 0

    def check(tag, q, qkv, sel, a, mp):
        nonlocal ncase, nrow
        ref = _p21_run(m, 0, q, qkv, sel, a)
        for v, hint, name in P21_VARIANTS:
            got = _p21_run(m, v, q, qkv, sel, a, mp if hint else -1)
            if not torch.equal(got, ref):
                nd = int((got != ref).any(1).sum())
                bad.append(f"{tag} {name} {nd} rows")
                ctx.detail(f"qsa p021 {tag}: {name} differs in {nd} rows")
        torch.cuda.synchronize()
        ncase += 1
        nrow += q.shape[0]

    # dense positions: every tile count 1..65 (count p+1 tokens) and the 17-tile group edges
    short_ps = [0, 1, 30, 31, 32, 33, 63, 64, 511, 542, 543, 544, 545, 1087, 1088, 1089, 1631, 1632, 1633, 2047, 2050]
    short_ps += [32 * k + 7 for k in range(65)]
    for M in (1, 2, 4, 5, 8, 16, 20, 33, 64):
        for trial in range(3):
            B = min(M, 1 + (trial + M) % 4)
            cuts = sorted(torch.randperm(M - 1, generator=cg)[:B - 1].add(1).tolist()) if B > 1 else []
            sizes = [y - x for x, y in zip([0] + cuts, cuts + [M])]
            qs = [1.0, 4.0, 30.0][trial]
            seqs = []
            for b, n in enumerate(sizes):
                if (b + trial) % 2 == 0:  # long: consecutive positions, shared base blocks
                    p0 = L - 2 - n - 64 * b
                    seqs.append((longc[b], list(range(p0, p0 + n)), torch.randperm((p0 + 1) // 4, generator=cg)[:576]))
                else:  # short: positions from the tile-count list
                    ps = [short_ps[int(torch.randint(len(short_ps), (1,), generator=cg))] for _ in range(n)]
                    seqs.append((shortc["rand"], ps, None))
            check(f"M={M} B={B} q×{qs:g}", *_p21_inputs(m, gen, cg, seqs, qs))
    # every tile count in one kind of cache per call, special caches, Q = 0
    for kind in kinds:
        for qs in (1.0, 30.0, 0.0):
            for chunk in (short_ps[:43], short_ps[43:]):
                check(f"{kind} q×{qs:g} {len(chunk)} rows", *_p21_inputs(m, gen, cg, [(shortc[kind], chunk, None)], qs))
    del longc
    torch.cuda.empty_cache()
    # real rows: qsa_core's own Q, selection and caches (long golden request, layer 3)
    i = 3
    k0, n0, _ = _reqs(ctx)[1][0]
    with _step(ctx, k0) as f:
        X = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
    c = _cache(n0 + 4)
    for s in range(0, n0, 64):
        e = min(n0, s + 64)
        attn, sel, Q, _ = _call(m, i, [(c, X[s:e], s)], q=True)
        if s % 1024 == 0 or 1900 <= s <= 2700 or e > n0 - 64:
            a = m.args([c], [0, e - s])
            check(f"golden {s}..{e}", Q.contiguous(), X[s:e].contiguous(), sel, a, e - 1)
            if not torch.equal(attn, _p21_run(m, 0, Q.contiguous(), X[s:e].contiguous(), sel, a)):
                bad.append(f"golden {s} stage≠core")
    return {"ok": not bad, "msg": (f"bitwise {ncase} cases {nrow} rows: k_pattn·P021 kernel (every call)·P021 "
                                   "(qsa_core rule, with/without hint) = k_attn+k_merge" if not bad
                                   else "differs: " + "; ".join(bad[:5]))}


def test_p021_core(ctx):
    """qsa_core with P021 on == off, bitwise (output, selection, caches): layers 3, 27, 47 — the long golden prompt in
    64-row calls (P021 on: with the max_pos hint, so short calls keep k_attn + k_merge), in 1–64-row calls over the
    sparse region (as test_invariance; no hint), and B=2 batches of its first 3,000 rows with the short prompt (32 + 29
    rows per call, both orders); with P021 on, B=2 == alone."""
    m = _mod(ctx)
    was = bool(m.patch_on(21))
    reqs = _reqs(ctx)
    bad, n = [], 0
    try:
        for i in (3, 27, 47):
            with _step(ctx, reqs[1][0][0]) as f:
                X = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
            ys = []
            for k, _, _ in reqs[0]:
                with _step(ctx, k) as f:
                    ys.append(f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0"))
            Y = torch.cat(ys).cuda()
            outs = []
            for on in (False, True):
                m.patch_set("qsa_decode_attn", on)
                cA = _cache(X.shape[0] + 4)
                A = _feed(m, i, cA, X, 0, [CHUNK], hint=on)
                cB = _cache(X.shape[0] + 4)
                B1 = _feed(m, i, cB, X[:2000], 0, [CHUNK])
                B2 = _feed(m, i, cB, X[2000:], 2000, [1, 2, 3, 5, 7, 11, 13, 17, 31, 64])
                res = A + [torch.cat([B1[0], B2[0]]), torch.cat([B1[1], B2[1]])] + cA
                for order in (0, 1):
                    cx, cy = _cache(X.shape[0] + 4), _cache(Y.shape[0] + 4)
                    ox, oy = [], []
                    a = b = 0
                    while a < 3000 or b < Y.shape[0]:
                        seqs, tags = [], []
                        if a < 3000:
                            e = min(3000, a + 32)
                            seqs.append((cx, X[a:e], a)), tags.append(("x", e - a))
                            a = e
                        if b < Y.shape[0]:
                            e = min(Y.shape[0], b + 29)
                            seqs.append((cy, Y[b:e], b)), tags.append(("y", e - b))
                            b = e
                        if order:
                            seqs, tags = seqs[::-1], tags[::-1]
                        at, _ = _call(m, i, seqs)
                        s = 0
                        for t, k in tags:
                            (ox if t == "x" else oy).append(at[s:s + k])
                            s += k
                    res += [torch.cat(ox), torch.cat(oy)]
                outs.append(res)
                n += X.shape[0] * 2 + 3000 + Y.shape[0]
            for k, (x, y) in enumerate(zip(*outs)):
                if not torch.equal(x, y):
                    bad.append(f"L{i} #{k}")
            if not all(torch.equal(outs[1][j], outs[1][0][:3000]) for j in (-4, -2)):
                bad.append(f"L{i} B=2≠alone")
            del outs
            torch.cuda.empty_cache()
    finally:
        m.patch_set("qsa_decode_attn", was)
    return {"ok": not bad, "msg": (f"P021 on = off bitwise (output·selection·cache), with it on B=2 = alone · 64-row "
                                   f"calls·1–64-row splits·B=2 both orders · layers 3·27·47 · {n} rows/setting"
                                   if not bad else "differs: " + ", ".join(bad[:6]))}


def _p21_graph_us(m, variant, runs, iters=24, reps=7):
    """Best µs per call: `iters` calls cycling through `runs` (inputs + caches → L2-cold as in a step) captured in one
    CUDA graph, replayed `reps` times."""
    outs = [torch.empty(r[0].shape[0], 3072, dtype=BF, device="cuda") for r in runs]

    def go():
        for k in range(iters):
            q, qkv, sel, a, mp = runs[k % len(runs)]
            m.attn_stage(variant, q, qkv, sel, a[1], a[2], outs[k % len(runs)], mp)

    go()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        go()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / iters)
    del g
    return best


def test_p021_perf(ctx):
    """µs per call of the decode attention stage: base (k_attn + k_merge) · k_pattn · the P021 cluster kernel · P021 as
    qsa_core runs it with the max_pos hint (base kernels while rows × tiles < 280 + 6·rows or one row per sequence
    up to 6). Graph replay of 24 calls cycling 24 input
    sets (other selections in the same caches / other caches → L2-cold as in a step, where 12 layers' caches and the
    weights pass between two calls), best of 7. Long: 128K positions, 512 spread blocks + 3-token tail per row (65
    tiles); rows of a sequence share most blocks (row k swaps 4k of 512; "less sharing": 32k); B×T = 1×1, 2×1, 4×1 (MTP
    draft passes), 1×4, 1×5 (B=1 verify), 2×5, 3×5, 4×5 (B=4 verify, M 20), 4×8, 4×16. Short: dense rows at p ≈ 2,040
    (64 tiles), 1,200 (38), 900 (29), 600 (19) and 300 (10). Real: the long golden request's last 5 rows (8.6K, layer 3) via qsa_core, 24 cache
    copies. Unique K‖V MB per call (cold DRAM floor ≈ 190–200 GB/s gathered on this node)."""
    m = _mod(ctx)
    m.reserve(64)
    gen = torch.Generator(device="cuda").manual_seed(2)
    cg = torch.Generator().manual_seed(2)
    L, NSET = 131072, 24
    caches = [_p21_cache(L + 8, gen) for _ in range(4)]
    rows = []

    def rec(tag, runs):
        us = {v: _p21_graph_us(m, v, runs) for v in (0, 1, 2, 3)}
        sel, st = runs[0][2], runs[0][3][2].tolist()
        uniq = 0
        for b in range(len(st) - 1):
            s = sel[st[b]:st[b + 1], :2051]
            uniq += int(torch.unique(s[s >= 0]).numel())
        mb = uniq * 1024 / 1e6
        ctx.detail(f"qsa p021 perf {tag} M={runs[0][0].shape[0]}: base {us[0]:.1f} · k_pattn {us[1]:.1f} · P021 kernel "
                   f"{us[3]:.1f} · P021 as run {us[2]:.1f} us/call · unique K‖V {mb:.2f} MB")
        rows.append(f"{tag} {us[0]:.0f}/{us[1]:.0f}/{us[3]:.0f}/{us[2]:.0f}")

    for B, T, step in ((1, 1, 4), (2, 1, 4), (4, 1, 4), (1, 4, 4), (1, 5, 4), (2, 5, 4), (3, 5, 4), (4, 5, 4), (4, 5, 32),
                       (1, 5, 32), (4, 8, 4), (4, 16, 4)):
        runs = []
        for k in range(NSET):
            seqs = []
            for b in range(B):
                p0 = L - 2 - T - 16 * b
                seqs.append((caches[b], list(range(p0, p0 + T)), torch.randperm((p0 + 1) // 4, generator=cg)[:576]))
            runs.append(_p21_inputs(m, gen, cg, seqs, step=step))
        rec(f"128K {B}×{T}" + (" less sharing" if step > 4 else ""), runs)
    del caches
    sc = [_p21_cache(4096, gen) for _ in range(NSET * 4)]
    for p, (B, T) in ((2040, (1, 5)), (2040, (4, 5)), (1200, (1, 5)), (1200, (2, 5)), (1200, (4, 5)), (900, (2, 5)),
                      (900, (4, 5)), (600, (1, 5)), (600, (4, 5)), (300, (4, 5))):
        runs = [_p21_inputs(m, gen, cg, [(sc[4 * k + b], list(range(p, p + T)), None) for b in range(B)])
                for k in range(NSET)]
        rec(f"short p={p} {B}×{T}", runs)
    del sc
    torch.cuda.empty_cache()
    # real: long golden request, layer 3, caches from its prompt (24 copies → L2-cold), its last 5 rows
    i = 3
    k0, n0, _ = _reqs(ctx)[1][0]
    with _step(ctx, k0) as f:
        X = f.get_tensor(KEY.format(i) + "qkv_proj#0.out.0").cuda()
    c = _cache(n0 + 4)
    _feed(m, i, c, X[:n0 - 5], 0, [CHUNK])
    runs, keep = [], []
    for k in range(NSET):
        ck = [t.clone() for t in c]
        keep.append(ck)  # args() holds raw pointers
        _, sel, Q, _ = _call(m, i, [(ck, X[n0 - 5:], n0 - 5)], q=True)
        runs.append((Q.contiguous(), X[n0 - 5:].contiguous(), sel, m.args([ck], [0, 5]), n0 - 1))
    rec("real 8.6K 1×5", runs)
    del runs, keep
    torch.cuda.empty_cache()
    return {"ok": True, "msg": "µs/call base/k_pattn/P021 kernel/P021: " + " · ".join(rows)
            + " (graph replay, outside L2, best of 7)"}


# ---- positions beyond 256K (1M window): wide selection tier == base tier == torch reference --------------------------
LONG_W = 1 << 20
LONG_TAG = {131072: "128K", 300000: "300K", 524288: "512K", 1 << 20: "1M"}


def _ref_sel(qi, comp, pos):
    """Selection rows (vLLM layout [2051 tokens | count], cuda int32) for indexer q [n,4,128] and compressed keys, scores
    formed as the kernels do: s = (ReLU(q0·k) + ReLU(q1·k)) + (ReLU(q2·k) + ReLU(q3·k)) in fp32, each dot in fp64
    rounded once to fp32 — the kernels' value whenever a key has one nonzero element (the dot is one exact product);
    top-512 by (score desc, block asc), ascending + open tail. Also returns the scores [vis_max, n]."""
    n, pos = qi.shape[0], [int(p) for p in pos]
    vmax = (max(pos) + 1) // 4
    K = comp[:vmax].double()
    out = torch.full((n, 2052), -1, dtype=torch.int32, device="cuda")
    ar4 = torch.arange(4, device="cuda")
    S = torch.empty(vmax, n, device="cuda")
    for r0 in range(0, n, 16):
        x = (K @ qi[r0:r0 + 16].double().reshape(-1, 128).T).float().view(vmax, -1, 4)
        x = torch.where(x > 0, x, torch.zeros_like(x))
        S[:, r0:r0 + 16] = (x[..., 0] + x[..., 1]) + (x[..., 2] + x[..., 3])
    for j, p in enumerate(pos):
        vis = (p + 1) // 4
        sc = S[:vis, j]
        thr = torch.topk(sc, 512).values[-1]
        gt = torch.nonzero(sc > thr).flatten()
        b = torch.sort(torch.cat([gt, torch.nonzero(sc == thr).flatten()[:512 - gt.numel()]])).values
        out[j, :2048] = (b[:, None] * 4 + ar4).flatten().int()
        nt = p + 1 - 4 * vis
        out[j, 2048:2048 + nt] = torch.arange(4 * vis, p + 1, device="cuda", dtype=torch.int32)
        out[j, -1] = 2048 + nt
    return out, S


def _long_cache(cap, gen, kind):
    """K‖V and raw keys N(0,1); compressed keys: dense N(0,1) | spread: one nonzero element per block (N(0,1) at a
    random dim) | levels: 2^lev at dim 5, lev 0..6, and lev 7 for 1/16 of the blocks ≥ 65,536 (selected blocks above
    the 16-bit range, ties of the top level in a crowded threshold bin)."""
    c = _cache(cap)
    for t in c[:2]:
        for a in range(0, t.shape[0], 1 << 18):
            t[a:a + (1 << 18)] = torch.randn(t[a:a + (1 << 18)].shape, generator=gen, device="cuda").to(BF)
    nb = c[2].shape[0]
    if kind == "dense":
        c[2].copy_(torch.randn(c[2].shape, generator=gen, device="cuda").to(BF))
    elif kind == "spread":
        c[2].zero_()
        d = torch.randint(0, 128, (nb,), generator=gen, device="cuda")
        c[2][torch.arange(nb, device="cuda"), d] = torch.randn(nb, generator=gen, device="cuda").to(BF)
    else:
        lev = torch.randint(0, 7, (nb,), generator=gen, device="cuda")
        top = (torch.arange(nb, device="cuda") >= 65536) & (torch.rand(nb, generator=gen, device="cuda") < 1 / 16)
        lev[top] = 7
        c[2].zero_()
        c[2][:, 5] = torch.exp2(lev.float()).to(BF)
    return c


def test_long_pos(ctx):
    """1M window (YaRN×4, ramp 1M). Synthetic caches of 1M positions (dense / spread / levels compressed keys, see
    _long_cache), 512 rows ending at 300,000, 524,288 and 1,048,576 (dense also at 131,072: base tier with the hint,
    wide tier without): the decode path (64-row calls, with and without the max_pos hint), the base prefill path (P009
    off, one call) and P009 (one call, with and without the hint) → bitwise the same selection, output and cache rows
    the call writes (K‖V, raw and compressed keys of [L − 512, L)); spread / levels (the rows' own compressed keys
    zeroed) → selection == the torch reference top-512 (_ref_sel, exact there); dense → rows equal to the fp64
    reference counted, differing blocks reported with their score gap to the 512th. P009 fallback rows reported."""
    m = _mod(ctx)
    m.reserve(8192)
    was = bool(m.patch_on(P009))
    i, T = 3, 512
    gen = torch.Generator(device="cuda").manual_seed(1)
    bad, info, fbs, nrun = [], [], [], 0
    try:
        m.rope_set(4.0, 262144, LONG_W, LONG_W)
        for kind in ("dense", "spread", "levels"):
            c = _long_cache(LONG_W + 8, gen, kind)
            for L in ((131072,) if kind == "dense" else ()) + (300000, 524288, LONG_W):
                p0 = L - T
                qkv = (torch.randn(T, 7296, generator=gen, device="cuda") * 2).to(BF)
                if kind != "dense":
                    qkv[:, 7168:] = 0  # the rows' own compressed keys = 0 (exact reference)
                runs, cref = {}, None

                def written():  # the cache rows these rows write
                    return [c[0][p0:L].clone(), c[1][p0:L].clone(), c[2][p0 // 4:L // 4].clone()]

                def run(name, fn, p9):
                    nonlocal cref
                    m.patch_set("qsa_fast_index", p9)
                    runs[name] = fn()
                    w = written()
                    if cref is None:
                        cref = w
                    elif not all(torch.equal(x, y) for x, y in zip(w, cref)):
                        bad.append(f"{kind} {L} {name} cache")

                run("decode", lambda: _feed(m, i, c, qkv, p0, [64], q=True, hint=True), True)
                run("decode no hint", lambda: _feed(m, i, c, qkv, p0, [64], hint=False), True)
                run("prefill", lambda: _call(m, i, [(c, qkv, p0)], hint=True), False)
                run("P009", lambda: _call(m, i, [(c, qkv, p0)], hint=True), True)
                pos = torch.arange(p0, L, dtype=torch.int32, device="cuda")
                st = _p9_stats(m, pos)
                fbs.append(f"{kind} {LONG_TAG[L]} {m.p009_last(T)[1]}")
                run("P009 no hint", lambda: _call(m, i, [(c, qkv, p0)], hint=False), True)
                nrun += len(runs)
                ra, rs, _, qi = runs["decode"]
                for name, r in runs.items():
                    if not (torch.equal(r[0], ra) and torch.equal(r[1], rs)):
                        bad.append(f"{kind} {L} {name}")
                        ctx.detail(f"qsa long {kind} L={L}: {name} differs (sel rows {int((r[1] != rs).any(1).sum())}, "
                                   f"attn rows {int((r[0] != ra).any(1).sum())})")
                ref, S = _ref_sel(qi, c[2], pos.tolist())
                eq = (rs == ref).all(1)
                ne = int((~eq).sum())
                line = f"qsa long {kind} L={L}: sel == reference {T - ne}/{T} · P009 {st}"
                if ne:
                    gs, rr = _sel_sets(rs[~eq]), _sel_sets(ref[~eq])
                    worst = 0.0
                    for k, j in enumerate(torch.nonzero(~eq).flatten().tolist()):
                        vis = (int(pos[j]) + 1) // 4
                        sc = S[:vis, j]
                        thr = torch.topk(sc, 512).values[-1].item()
                        for b in set(t // 4 for t in gs[k] ^ rr[k] if t < 4 * vis):
                            worst = max(worst, abs(sc[b].item() - thr) / max(thr, 1e-30))
                    line += f" · {ne} rows differ, max rel gap to the 512th {worst:.1e}"
                    if kind != "dense":
                        bad.append(f"{kind} {L} reference {ne} rows")
                    else:
                        info.append(f"dense {L // 1024}K {ne} rows differ from the reference (score gap ≤{worst:.0e})")
                ctx.detail(line)
                del runs, ra, rs, qi, ref, S, cref
            del c
            torch.cuda.empty_cache()
    finally:
        m.patch_set("qsa_fast_index", was)
        _rope_default(m)
        torch.cuda.empty_cache()
    msg = (f"bitwise {nrun} runs (decode with/without hint·prefill·P009 with/without hint, 128K at the base/wide tier; "
           "selection·output·cache) · spread·levels selection = torch reference" if not bad
           else "differs: " + ", ".join(bad[:6]))
    return {"ok": not bad, "msg": msg + " · " + (" · ".join(info) + " · " if info else "") + "P009 fallback rows/512: "
            + ", ".join(fbs)}

