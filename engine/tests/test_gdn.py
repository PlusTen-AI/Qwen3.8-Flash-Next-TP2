"""GDN core (src/kernels/gdn.cu): upload, torch reference, golden dump (teacher forcing per request), T>1 and batch
invariance (bitwise), timing; P005 gdn_fast_prefill (p005_gdn_fast_prefill.cuh) bitwise vs base and its timing;
speculative rows (gdn_core_spec + gdn_commit) bitwise vs gdn_core on the accepted rows, and their timing.

Golden per request: zero state, then every step's rows in order through the kernel (prefill: all prompt rows from the
full-row keys `fn._forward_core#0.in.{mixed_qkv,b,a}`, z only on the kept rows; decode: `in_proj_qkvz/ba#0.out.0`).
Compared at kept prefill rows and every decode row: core (= out_proj#0.in0) and o (= post.core_attn_out). Golden prefill
ran the chunked FlashInfer kernel (bf16 l2norm, other order), decode the recurrent Triton kernel → reported separately;
decode is also run from the golden state after prefill (`chunk_gated_delta_rule#0.out.1`) to isolate the decode kernel.
"""
import os

import torch
from safetensors import safe_open

from golden import cmp

GDN = [i for i in range(48) if i % 4 != 3]
KEY = "language_model.model.layers.{}.linear_attn."
BF = torch.bfloat16
_m = None


def _mod(ctx):
    global _m
    if _m is None:
        _m = ctx.ext("gdn", ["src/kernels/gdn/gdn.cu", "src/core/weights.cc", "src/model/patches.cc"])
        _m.open(ctx.model_dir, ctx.table, ctx.rank)
        for i in GDN:
            _m.upload(i)
    return _m


def _w(ctx, i):
    return _mod(ctx).upload(i)  # cached in the binding after the first call


def _run(ctx, i, qkvz, ba, S, conv, slots, seq_start, tap=True):
    M = qkvz.shape[0]
    core = torch.empty(M, 3072, dtype=BF, device="cuda")
    o = torch.empty(M, 3072, dtype=BF, device="cuda") if tap else None
    _mod(ctx).run(i, qkvz.contiguous(), ba.contiguous(), S, conv, slots, seq_start, core, o)
    return core, o


def _states(n):
    return torch.zeros(n, 24, 128, 128, device="cuda"), torch.zeros(n, 3, 5120, dtype=BF, device="cuda")


def _ref(w, qkvz, ba, S, conv):
    """Torch fp32 reference for one sequence (vLLM decode numerics); S [24,128,128] f32 and conv [3,5120] updated."""
    cw, A, dt, nw = w
    T = qkvz.shape[0]
    x = torch.cat([conv.float(), qkvz[:, :5120].float()], 0)
    acc = torch.zeros(T, 5120, device=qkvz.device)
    for j in range(4):
        acc = acc + (cw[:, j].float() * x[j:j + T]).to(BF).float()
    y = (acc / (1 + torch.exp(-acc))).to(BF).float()
    conv.copy_(torch.cat([conv, qkvz[:, :5120]], 0)[-3:])
    q, k, v = y[:, :1024].view(T, 8, 128), y[:, 1024:2048].view(T, 8, 128), y[:, 2048:].view(T, 24, 128)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k = q.repeat_interleave(3, 1), k.repeat_interleave(3, 1)
    xa = ba[:, 24:].float() + dt
    eg = torch.exp(A * torch.where(xa <= 20, torch.log(1 + torch.exp(xa)), xa))
    beta = torch.sigmoid(ba[:, :24].float())
    o = torch.empty(T, 24, 128, device=qkvz.device)
    for t in range(T):
        S.mul_(eg[t][:, None, None])
        d = (S * k[t][:, None, :]).sum(-1)
        S.add_(((v[t] - d) * beta[t][:, None])[:, :, None] * k[t][:, None, :])
        o[t] = (S * q[t][:, None, :]).sum(-1)
    ob = o.to(BF).float()
    core = ob * torch.rsqrt(ob.pow(2).mean(-1, keepdim=True) + 1e-6) * nw.float() * torch.sigmoid(
        qkvz[:, 5120:].float().view(T, 24, 128))
    return core.reshape(T, 3072).to(BF), ob.reshape(T, 3072).to(BF)


def _step_file(ctx, k):
    return safe_open(os.path.join(ctx.golden.dir, f"step{k:04d}.safetensors"), "pt", device="cpu")


def _rows(f, i, n):
    """(qkvz [n,8192], ba [n,48], kept) for layer i of a step with n scheduled rows (full rows rebuilt if needed)."""
    P = KEY.format(i)
    qz = f.get_tensor(P + "in_proj_qkvz#0.out.0")
    kept = qz.shape[0]
    if kept == n:
        return qz, f.get_tensor(P + "in_proj_ba#0.out.0"), kept
    mixed = f.get_tensor(P + "fn._forward_core#0.in.mixed_qkv")
    qkvz = torch.zeros(n, 8192, dtype=BF)
    qkvz[:, :5120] = mixed
    qkvz[n - kept:, 5120:] = qz[:, 5120:]
    assert torch.equal(mixed[n - kept:], qz[:, :5120]), "mixed_qkv ≠ qkvz kept rows"
    ba = torch.cat([f.get_tensor(P + "fn._forward_core#0.in.b"), f.get_tensor(P + "fn._forward_core#0.in.a")], 1)
    return qkvz, ba, kept


def _requests(ctx):
    reqs = sorted(ctx.golden.requests().values(), key=lambda r: r[0][0])
    return reqs  # [short, long, short2]


def _golden_request(ctx, steps, name):
    """Run one request over all GDN layers. Decode runs twice: from our own prefill state, and from the golden state
    after prefill (chunk kernel's final S, conv = last 3 prompt rows) — the latter isolates the decode kernel."""
    S, conv = _states(len(GDN))
    Sg, cg = _states(len(GDN))
    pre = {"core": [], "o": []}
    dec = {i: {"core": [], "o": [], "gcore": [], "rc": [], "ro": []} for i in GDN}
    st, from_golden = [], True
    for k, n, _ in steps:
        with _step_file(ctx, k) as f:
            for li, i in enumerate(GDN):
                qkvz, ba, kept = _rows(f, i, n)
                qkvz, ba = qkvz.cuda(), ba.cuda()
                core, o = _run(ctx, i, qkvz, ba, S, conv, [li], [0, n])
                P = KEY.format(i)
                rc = f.get_tensor(P + "out_proj#0.in0").cuda()
                ro = f.get_tensor(P + "fn._forward_core#0.post.core_attn_out").reshape(kept, 3072).cuda()
                if n > 1:
                    pre["core"].append(cmp(f"{name} L{i} core", core[n - kept:], rc))
                    pre["o"].append(cmp(f"{name} L{i} o", o[n - kept:], ro))
                    sk = P + "chunk_gated_delta_rule#0.out.1"
                    if sk in f.keys() and n >= 3:
                        Sg[li] = f.get_tensor(sk)[0].cuda()
                        cg[li] = qkvz[-3:, :5120]
                        st.append(cmp(f"{name} L{i} S", S[li], Sg[li]))
                    else:
                        from_golden = False
                else:
                    d = dec[i]
                    d["core"].append(core), d["o"].append(o), d["rc"].append(rc), d["ro"].append(ro)
                    if from_golden:
                        d["gcore"].append(_run(ctx, i, qkvz, ba, Sg, cg, [li], [0, n], False)[0])
    dc, do, dg = [], [], []
    for i, d in dec.items():
        rc = torch.cat(d["rc"])
        dc.append(cmp(f"{name} L{i} core", torch.cat(d["core"]), rc))
        dc[-1]["eq"] = (torch.cat(d["core"]) == rc).float().mean().item()
        do.append(cmp(f"{name} L{i} o", torch.cat(d["o"]), torch.cat(d["ro"])))
        if from_golden:
            dg.append(cmp(f"{name} L{i} core*", torch.cat(d["gcore"]), rc))
            dg[-1]["eq"] = (torch.cat(d["gcore"]) == rc).float().mean().item()
    return pre, dc, do, dg, st


def _worst(rs):
    return max(r["max_abs"] for r in rs), max(r["max_rel"] for r in rs), min(r["cos"] for r in rs)


def _fmt(rs):
    a, r, c = _worst(rs)
    return f"abs {a:.2e} rel {r:.2e} cos {c:.6f}"


def _golden_test(ctx, which):
    steps = _requests(ctx)[which]
    name = ["short", "long", "short2"][which]
    pre, dc, do, dg, st = _golden_request(ctx, steps, name)
    for tag, rs in (("pre", pre["core"] + pre["o"]), ("dec", dc + do + dg), ("S", st)):
        for r in rs:
            ctx.detail(f"gdn {r['name']:<22} {tag} n={r['n']} abs {r['max_abs']:.3e} rel {r['max_rel']:.3e} "
                       f"cos {r['cos']:.7f}" + (f" eq {r['eq']:.4f}" if "eq" in r else ""))
    eq = lambda rs: min(r["eq"] for r in rs)  # noqa: E731
    ok = _worst(dc)[2] >= 0.9999 and _worst(dc)[1] <= 2e-2 and _worst(pre["core"])[2] >= 0.999
    ok &= not dg or (_worst(dg)[2] >= 0.99999 and _worst(dg)[1] <= 1e-2)
    msg = (f"{name} decode {len(steps) - 1} steps core {_fmt(dc)} equal≥{eq(dc):.1%} · o {_fmt(do)}"
           + (f" · from golden state core {_fmt(dg)} equal≥{eq(dg):.2%}" if dg else "")
           + f" · prefill {steps[0][1]} rows core {_fmt(pre['core'])}" + (f" · S {_fmt(st)}" if st else "")
           + f" · {len(GDN)} layers")
    return {"ok": ok, "msg": msg}


def test_upload(ctx):
    bad = []
    for i in (0, 1, 46):
        cw, A, dt, nw = _w(ctx, i)
        P = f"model.language_model.layers.{i}.linear_attn."
        sl = lambda n: ctx.w.slice(P + n)[0]  # noqa: E731
        if not torch.equal(cw.cpu().view(torch.uint8).flatten(), sl("conv1d.weight")):
            bad.append(f"L{i} conv")
        if not torch.equal(nw.cpu().view(torch.uint8), sl("norm.weight")):
            bad.append(f"L{i} norm")
        al = sl("A_log").view(BF).float()
        if not torch.allclose(A.cpu(), -torch.exp(al), rtol=1e-6, atol=0):
            bad.append(f"L{i} A")
        if not torch.equal(dt.cpu(), sl("dt_bias").view(BF).float()):
            bad.append(f"L{i} dt")
    return {"ok": not bad, "msg": "conv·norm·A=−exp(A_log)·dt_bias match (layers 0·1·46)" if not bad else " ".join(bad)}


def test_ref(ctx):
    """Kernel vs torch fp32 reference on the short request's 92 real rows in one call (layers 0, 21, 46)."""
    steps = _requests(ctx)[0]
    worst, nd, tot = 0.0, 0, 0
    for i in (0, 21, 46):
        rows = []
        for k, n, _ in steps:
            with _step_file(ctx, k) as f:
                q, b, _ = _rows(f, i, n)
            rows.append((q, b))
        qkvz, ba = torch.cat([r[0] for r in rows]).cuda(), torch.cat([r[1] for r in rows]).cuda()
        w = _w(ctx, i)
        Sr, cr = _states(1)
        rc, ro = _ref(w, qkvz, ba, Sr[0], cr[0])
        S, conv = _states(1)
        core, o = _run(ctx, i, qkvz, ba, S, conv, [0], [0, qkvz.shape[0]])
        for g, r in ((core, rc), (o, ro)):
            d = (g.float() - r.float()).abs()
            worst = max(worst, (d.max() / r.float().abs().max()).item())
            nd += int((d > 0).sum())
            tot += d.numel()
        sd = ((S - Sr).abs().max() / Sr.abs().max()).item()
        worst = max(worst, sd)
        if not torch.equal(conv, cr):
            return {"ok": False, "msg": f"L{i} conv state ≠ ref"}
        ctx.detail(f"gdn ref L{i} rows {qkvz.shape[0]} core/o rel {worst:.2e} S rel {sd:.2e}")
    return {"ok": worst < 1e-2, "msg": f"rel vs torch reference {worst:.2e} · bf16 differing values {nd}/{tot} · "
            f"conv state matches"}


def test_golden_short(ctx):
    return _golden_test(ctx, 0)


def test_golden_long(ctx):
    return _golden_test(ctx, 1)


def test_tmulti(ctx):
    """T rows in one call == one row per call, bitwise (core, S, conv), T ∈ {2,3,4,16,all}."""
    steps = _requests(ctx)[0]
    with _step_file(ctx, steps[0][0]) as f:
        rows = {i: _rows(f, i, steps[0][1])[:2] for i in (0, 46)}
    bad = []
    for i, (qkvz, ba) in rows.items():
        qkvz, ba = qkvz.cuda(), ba.cuda()
        M = qkvz.shape[0]
        S1, c1 = _states(1)
        one = torch.cat([_run(ctx, i, qkvz[t:t + 1], ba[t:t + 1], S1, c1, [0], [0, 1], False)[0] for t in range(M)])
        for T in (2, 3, 4, 16, M):
            S, c = _states(1)
            outs = [_run(ctx, i, qkvz[t:t + T], ba[t:t + T], S, c, [0], [0, min(T, M - t)], False)[0]
                    for t in range(0, M, T)]
            if not (torch.equal(torch.cat(outs), one) and torch.equal(S, S1) and torch.equal(c, c1)):
                bad.append(f"L{i} T={T}")
    return {"ok": not bad, "msg": "T=2·3·4·16·61 == T=1 bitwise (core·S·conv, layers 0·46)" if not bad
            else " ".join(bad)}


def test_batch(ctx):
    """B sequences in one call (different T, states, one empty) == each alone, bitwise."""
    reqs = _requests(ctx)
    src = []
    for r in reqs:
        with _step_file(ctx, r[1][0]) as f:  # a decode step of each request
            src.append(_rows(f, 0, 1)[:2])
    with _step_file(ctx, reqs[0][0][0]) as f:
        pq, pb = _rows(f, 0, reqs[0][0][1])[:2]
    i, Ts = 0, [3, 1, 0, 4, 2]
    gen = torch.Generator().manual_seed(1)
    # rows: mix of prompt rows and decode rows
    seqs = []
    for b, T in enumerate(Ts):
        idx = torch.randint(0, pq.shape[0], (T,), generator=gen)
        q, a = pq[idx].clone(), pb[idx].clone()
        if T:
            q[-1], a[-1] = src[b % 3][0][0], src[b % 3][1][0]
        seqs.append((q, a))
    B = len(Ts)
    S0, c0 = _states(B)  # non-zero start states: warm each slot with a few prompt rows
    for b in range(B):
        _run(ctx, i, pq[b * 5:b * 5 + 5].cuda(), pb[b * 5:b * 5 + 5].cuda(), S0, c0, [b], [0, 5], False)
    Sa, ca = S0.clone(), c0.clone()
    alone = [_run(ctx, i, q.cuda(), a.cuda(), Sa, ca, [b], [0, q.shape[0]], False)[0] for b, (q, a) in enumerate(seqs)]
    starts = [0]
    for T in Ts:
        starts.append(starts[-1] + T)
    qkvz = torch.cat([s[0] for s in seqs]).cuda()
    ba = torch.cat([s[1] for s in seqs]).cuda()
    slots = [4, 2, 0, 1, 3]  # sequence b lives in slot slots[b] (slot order ≠ batch order)
    Sq, cq = torch.empty_like(S0), torch.empty_like(c0)
    for b in range(B):
        Sq[slots[b]], cq[slots[b]] = S0[b], c0[b]
    core = _run(ctx, i, qkvz, ba, Sq, cq, slots, starts, False)[0]
    same = torch.equal(core, torch.cat(alone)) and all(
        torch.equal(Sq[slots[b]], Sa[b]) and torch.equal(cq[slots[b]], ca[b]) for b in range(B))
    return {"ok": same, "msg": f"B={B} (T={'·'.join(map(str, Ts))}, slots shuffled) == one at a time, bitwise"
            if same else "batch result differs from running one at a time"}


def test_perf(ctx):
    """µs per call: CUDA graph of 300 back-to-back calls (kernel + inter-kernel gap; best of 5) and eager launches, 40
    cold state sets cycled, after a 0.5 s clock warm-up. GB/s over the bytes a call must move (S r+w dominates).
    Pass: B=4 ≥ 85 % of 241 GB/s (streaming), B=1 ≥ 80 % (includes the ~1 µs launch gap and ~2 µs latency)."""
    import time
    m = _mod(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench(0, 1, 1, 500, 40, 0)
    out, ok = [], True
    for B, T in ((1, 1), (4, 1), (1, 4)):
        g = min(m.bench(0, B, T, 300, 40, 1) for _ in range(5))
        e = min(m.bench(0, B, T, 300, 40, 0) for _ in range(3))
        state = B * (2 * 24 * 128 * 128 * 4 + 2 * 3 * 5120 * 2)
        io = B * T * (8192 + 48 + 3072) * 2 + 5120 * 8 + 256 + 192
        gbs = (state + io) / g / 1e3
        out.append(f"B{B}/T{T} {g:.1f}µs {gbs:.0f}GB/s")
        ctx.detail(f"gdn perf B={B} T={T} graph {g:.2f} us {gbs:.1f} GB/s · eager {e:.2f} us "
                   f"(state {state} B, io {io} B)")
        if (B, T) == (1, 1):
            ok &= gbs >= 0.80 * 241
        if (B, T) == (4, 1):
            ok &= gbs >= 0.85 * 241
    return {"ok": ok, "msg": " · ".join(out) + " (graph, best, GPU shared)"}


def _p005_inputs(ctx, src, Ts, gen):
    """qkvz/ba rows for sequences of lengths Ts: 'real' = slices of the long request's prefill rows (layer 0), sequence b
    starting at row b·(N − T) / max(B − 1, 1); 'rand' = N(0, 0.5²) / N(0, 1) bf16."""
    M = sum(Ts)
    if src == "rand":
        return ((torch.randn(M, 8192, device="cuda", generator=gen) * 0.5).to(BF),
                torch.randn(M, 48, device="cuda", generator=gen).to(BF))
    global _p005_real
    if _p005_real is None:
        k, n, _ = max((st[0] for st in _requests(ctx)), key=lambda st: st[1])  # the longest prefill step
        with _step_file(ctx, k) as f:
            q, b, _ = _rows(f, 0, n)
        _p005_real = q.cuda(), b.cuda()
    q, b = _p005_real
    N, qs, bs = q.shape[0], [], []
    for i, T in enumerate(Ts):
        r0 = (N - T) * i // max(len(Ts) - 1, 1)
        qs.append(q[r0:r0 + T]), bs.append(b[r0:r0 + T])
    return torch.cat(qs).contiguous(), torch.cat(bs).contiguous()


_p005_real = None


def test_p005_bitwise(ctx):
    """P005 gdn_fast_prefill kernel (forced for every T) == base kernel, bitwise: core, o, S, conv state. Layers 0, 46;
    T = 1, 16, 61, 2048, 8192 × B = 1, 2 (+ one mixed batch with an empty sequence); real prefill rows and random rows;
    random non-zero start states; slot order ≠ batch order."""
    m = _mod(ctx)
    gen = torch.Generator(device="cuda").manual_seed(5)
    cases = [[T] * B for T in (1, 16, 61, 2048, 8192) for B in (1, 2)] + [[2048, 1, 0, 61]]
    bad, n = [], 0
    prev = m.set_patch(False, 1)
    try:
        for i in (0, 46):
            for src in ("real", "rand"):
                for Ts in cases:
                    B = len(Ts)
                    qkvz, ba = _p005_inputs(ctx, src, Ts, gen)
                    S0 = torch.randn(B, 24, 128, 128, device="cuda", generator=gen) * 0.1
                    c0 = (torch.randn(B, 3, 5120, device="cuda", generator=gen) * 0.5).to(BF)
                    starts = [0]
                    for T in Ts:
                        starts.append(starts[-1] + T)
                    slots = list(range(B))[::-1]
                    out = []
                    for on in (False, True):
                        m.set_patch(on, 1)
                        S, c = S0.clone(), c0.clone()
                        core, o = _run(ctx, i, qkvz, ba, S, c, slots, starts)
                        out.append((core, o, S, c))
                    eq = [torch.equal(a, b) for a, b in zip(*out)]
                    n += 1
                    if not all(eq):
                        bad.append(f"L{i} {src} T={'·'.join(map(str, Ts))} " +
                                   "/".join(k for k, e in zip(("core", "o", "S", "conv"), eq) if not e))
                    ctx.detail(f"gdn p005 bitwise L{i} {src} T={Ts}: core {eq[0]} o {eq[1]} S {eq[2]} conv {eq[3]}")
    finally:
        m.set_patch(*prev)
    return {"ok": not bad, "msg": f"patch == base bitwise {n} cases (core·o·S·conv, T=1·16·61·2048·8192 × B=1·2 + "
            f"mixed, layers 0·46, real data·random)" if not bad else "mismatch: " + "; ".join(bad[:4])}


def test_p005_perf(ctx):
    """µs per call (µs/layer), base vs P005 (engine dispatch: patch kernel from 32 rows per sequence, decode keeps the
    base kernel): CUDA graph of back-to-back calls, best of 5, cold state sets cycled. Pass: T=8192 ≤ 4000 µs and decode
    B1/T1 not slower than base (+5 % noise)."""
    import time
    m = _mod(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench(0, 1, 1, 500, 40, 0)
    res = {}
    prev = m.set_patch(False, 1)  # → (state before, engine default rows threshold)
    try:
        for B, T, it, sets in ((1, 1, 300, 40), (1, 2048, 20, 4), (1, 8192, 8, 2)):
            for on in (False, True):
                m.set_patch(on, prev[1])
                res[(T, on)] = min(m.bench(0, B, T, it, sets, 1) for _ in range(5))
                ctx.detail(f"gdn p005 perf B={B} T={T} {'patch' if on else 'base '} {res[(T, on)]:.1f} us "
                           f"({res[(T, on)] / T * 1e3:.1f} ns/token)")
    finally:
        m.set_patch(*prev)
    ok = res[(8192, True)] <= 4000 and res[(1, True)] <= res[(1, False)] * 1.05
    msg = " · ".join(f"T={T} {res[(T, False)]:.{1 if T == 1 else 0}f}→{res[(T, True)]:.{1 if T == 1 else 0}f}µs "
                     f"(×{res[(T, False)] / res[(T, True)]:.2f})" for T in (8192, 2048, 1))
    return {"ok": ok, "msg": msg + " (base→patch, B=1, graph, best of 5, GPU shared)"}


# ---- speculative rows (MTP verify): gdn_core_spec on T rows, gdn_commit keeps the first n ---------------------------
SPEC_L = (0, 46)  # layers of the spec tests; one commit launch covers both
SPEC_T = (1, 2, 5, 13, 16)
SPEC_H = 300      # history rows before the first verify step (S, conv non-trivial)
_spec_src_rows = {}


def _spec_src(ctx, i):
    """Real rows of layer i: the long request's prefill (8603 rows). The dump keeps z only for its last 64 rows → those
    are tiled over the others."""
    if i not in _spec_src_rows:
        k, n, _ = max((st[0] for st in _requests(ctx)), key=lambda st: st[1])
        with _step_file(ctx, k) as f:
            q, b, kept = _rows(f, i, n)
        q[:n - kept, 5120:] = q[n - kept:, 5120:].repeat((n - kept) // kept + 1, 1)[:n - kept]
        _spec_src_rows[i] = q.cuda(), b.cuda()
    return _spec_src_rows[i]


def _rows_at(ctx, i, pos, Ts, ns=None):
    """qkvz, ba of B sequences for layer i: sequence b = Ts[b] true rows from source row pos[b]; with ns, rows ≥ ns[b]
    are rejected drafts — real rows from 4000 rows further on, so they differ from the true continuation."""
    q, a = _spec_src(ctx, i)
    idx = []
    for b, (p, T) in enumerate(zip(pos, Ts)):
        r = torch.arange(p, p + T, device="cuda")
        if ns is not None:
            r[ns[b]:] += 4000
        idx.append(r)
    idx = torch.cat(idx)
    return q[idx].contiguous(), a[idx].contiguous()


def _starts(Ts):
    s = [0]
    for T in Ts:
        s.append(s[-1] + T)
    return s


def _slots(li, B):
    """State / record slot of sequence b in layer li (reversed within a layer: slot order ≠ batch order)."""
    return [li * B + B - 1 - b for b in range(B)]


def _all_slots(B):
    return [s for li in range(len(SPEC_L)) for s in _slots(li, B)]  # commit pair p = li·B + b


def _spec(ctx, i, qkvz, ba, S, conv, rec, slots, seq_start):
    core = torch.empty(qkvz.shape[0], 3072, dtype=BF, device="cuda")
    _mod(ctx).run_spec(i, qkvz, ba, S, conv, rec, slots, seq_start, core)
    return core


def _spec_warm(ctx, B, pos):
    """States [len(SPEC_L)·B] after SPEC_H history rows per sequence (source rows pos[b] − SPEC_H …)."""
    S, c = _states(len(SPEC_L) * B)
    for li, i in enumerate(SPEC_L):
        q, a = _rows_at(ctx, i, [p - SPEC_H for p in pos], [SPEC_H] * B)
        _run(ctx, i, q, a, S, c, _slots(li, B), _starts([SPEC_H] * B), False)
    return S, c


def _spec_case(ctx, B, pos, Ts, ns, S0, c0, rec):
    """Path A: spec on Ts rows (both layers) + one commit(ns); path B: gdn_core on the ns true rows. Then both go on
    with 3 more true rows. Returns the list of mismatches."""
    rec.fill_(255)  # NaN floats: a row the commit reads but the spec did not write would show in S
    S, c = S0.clone(), c0.clone()
    spec = [_spec(ctx, i, *_rows_at(ctx, i, pos, Ts, ns), S, c, rec, _slots(li, B), _starts(Ts))
            for li, i in enumerate(SPEC_L)]
    bad = [] if torch.equal(S, S0) and torch.equal(c, c0) else ["spec wrote the state"]
    _mod(ctx).commit(S, c, rec, _all_slots(B), ns)
    S2, c2 = S0.clone(), c0.clone()
    st = _starts(Ts)
    for li, i in enumerate(SPEC_L):
        if sum(ns):
            ref = _run(ctx, i, *_rows_at(ctx, i, pos, ns), S2, c2, _slots(li, B), _starts(ns), False)[0]
            if not torch.equal(torch.cat([spec[li][st[b]:st[b] + ns[b]] for b in range(B)]), ref):
                bad.append(f"L{i} core")
    bad += [k for k, x, y in (("S", S, S2), ("conv", c, c2)) if not torch.equal(x, y)]
    nxt = [p + n for p, n in zip(pos, ns)]
    for li, i in enumerate(SPEC_L):
        q, a = _rows_at(ctx, i, nxt, [3] * B)
        x = _run(ctx, i, q, a, S, c, _slots(li, B), _starts([3] * B), False)[0]
        y = _run(ctx, i, q, a, S2, c2, _slots(li, B), _starts([3] * B), False)[0]
        if not torch.equal(x, y):
            bad.append(f"L{i} next core")
    if not (torch.equal(S, S2) and torch.equal(c, c2)):
        bad.append("next state")
    return bad


def test_spec_bitwise(ctx):
    """gdn_core_spec on T rows + gdn_commit(n) == gdn_core on the first n rows, bitwise (S, conv state, core rows < n);
    spec alone leaves the state untouched; a further gdn_core call of 3 true rows stays equal (catches conv-state
    errors). B = 1 and 3 (mixed T, slot order ≠ batch order); T = 1·2·5·13·16; every n of sequence 0, random n of the
    others; layers 0 and 46 committed in one launch; real rows after 300 rows of history; rejected rows ≠ the true
    continuation; records start as NaN bytes."""
    gen = torch.Generator().manual_seed(7)
    bad, cases = [], 0
    for B in (1, 3):
        pos = [37 + SPEC_H + 1300 * b for b in range(B)]
        S0, c0 = _spec_warm(ctx, B, pos)
        rec = torch.empty(len(SPEC_L) * B, _mod(ctx).rec_bytes(), dtype=torch.uint8, device="cuda")
        for k, T in enumerate(SPEC_T):
            Ts = [T] + [SPEC_T[(k + 2 * b) % len(SPEC_T)] for b in range(1, B)]
            nb = 0
            for j in range(T + 1):
                ns = [j] + [int(torch.randint(0, t + 1, (1,), generator=gen)) for t in Ts[1:]]
                err = _spec_case(ctx, B, pos, Ts, ns, S0, c0, rec)
                cases += 1
                if err:
                    nb += 1
                    bad.append(f"B{B} T={'·'.join(map(str, Ts))} n={'·'.join(map(str, ns))}: {'/'.join(err)}")
            ctx.detail(f"gdn spec bitwise B={B} T={Ts} n=0…{T}: {T + 1 - nb}/{T + 1} equal")
    return {"ok": not bad, "msg": f"spec+commit(n) == n-row gdn_core bitwise {cases} cases (S·conv·core, through the "
            f"next 3 rows, B=1·3, T=1·2·5·13·16, n=0…T, layers 0·46 committed in one launch)" if not bad
            else "mismatch: " + "; ".join(bad[:4])}


def test_spec_chain(ctx):
    """A decode loop of 12 verify rounds, B = 3 (T random 1…16, n random 1…T, every 4th round 0…T; rejected rows ≠
    the true continuation): spec + commit each round == plain gdn_core on the accepted rows each round == one gdn_core
    call over all accepted rows of each sequence (prefill path: P005 if on), bitwise: S, conv, every accepted core
    row."""
    gen = torch.Generator().manual_seed(11)
    B, rounds = 3, 12
    pos0 = [37 + SPEC_H + 1300 * b for b in range(B)]
    S0, c0 = _spec_warm(ctx, B, pos0)
    rec = torch.full((len(SPEC_L) * B, _mod(ctx).rec_bytes()), 255, dtype=torch.uint8, device="cuda")
    Sa, ca, Sb, cb = S0.clone(), c0.clone(), S0.clone(), c0.clone()
    ga = {i: [[] for _ in range(B)] for i in SPEC_L}
    gb = {i: [[] for _ in range(B)] for i in SPEC_L}
    pos, acc = list(pos0), [0] * B
    for r in range(rounds):
        Ts = [int(torch.randint(1, 17, (1,), generator=gen)) for _ in range(B)]
        ns = [int(torch.randint(0 if r % 4 == 3 else 1, t + 1, (1,), generator=gen)) for t in Ts]
        st, sn = _starts(Ts), _starts(ns)
        for li, i in enumerate(SPEC_L):
            core = _spec(ctx, i, *_rows_at(ctx, i, pos, Ts, ns), Sa, ca, rec, _slots(li, B), st)
            for b in range(B):
                ga[i][b].append(core[st[b]:st[b] + ns[b]])
            if sum(ns):
                core = _run(ctx, i, *_rows_at(ctx, i, pos, ns), Sb, cb, _slots(li, B), sn, False)[0]
                for b in range(B):
                    gb[i][b].append(core[sn[b]:sn[b] + ns[b]])
        _mod(ctx).commit(Sa, ca, rec, _all_slots(B), ns)
        pos = [p + n for p, n in zip(pos, ns)]
        acc = [x + n for x, n in zip(acc, ns)]
    Sc, cc = S0.clone(), c0.clone()
    bad = []
    for li, i in enumerate(SPEC_L):
        one = _run(ctx, i, *_rows_at(ctx, i, pos0, acc), Sc, cc, _slots(li, B), _starts(acc), False)[0]
        if not torch.equal(torch.cat([x for b in range(B) for x in ga[i][b]]), one):
            bad.append(f"L{i} core spec≠one-shot")
        if not torch.equal(torch.cat([x for b in range(B) for x in gb[i][b]]), one):
            bad.append(f"L{i} core per-round≠one-shot")
    for k, x, y in (("S", Sa, Sb), ("conv", ca, cb), ("S one-shot", Sa, Sc), ("conv one-shot", ca, cc)):
        if not torch.equal(x, y):
            bad.append(k)
    ctx.detail(f"gdn spec chain B={B} rounds={rounds} accepted rows {acc}")
    return {"ok": not bad, "msg": f"verify {rounds} rounds B={B} (accepted rows {'·'.join(map(str, acc))}): "
            f"spec+commit == per-round gdn_core == one-shot prefill, bitwise (S·conv·core, layers 0·46)" if not bad
            else "mismatch: " + ", ".join(bad)}


def test_spec_perf(ctx):
    """µs: gdn_core_spec vs gdn_core at B=1, T = 13 (verify step with 12 drafts) and T = 1 — CUDA graph of 300
    back-to-back calls, best of 5, 40 cold state sets. gdn_commit of 36 layers at B = 1 and 4, n = 4 and 16 — mean of 20
    launches, best of 3, S cold (2 sets ≫ L2). GB/s over the bytes a commit must move (S r+w, conv, n record rows).
    Pass: spec ≤ base + 10 %; commit (n = 4) B=1 ≥ 80 %, B=4 ≥ 85 % of 241 GB/s."""
    import time
    m = _mod(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench(0, 1, 1, 500, 40, 0)
    out, ok = [], True
    for T in (13, 1):
        base = min(m.bench(0, 1, T, 300, 40, 1) for _ in range(5))
        spec = min(m.bench_spec(0, 1, T, 300, 40, 1) for _ in range(5))
        ctx.detail(f"gdn spec perf B=1 T={T} base {base:.2f} us spec {spec:.2f} us")
        out.append(f"T={T} base {base:.1f}→spec {spec:.1f}µs")
        ok &= spec <= base * 1.10
    for B in (1, 4):
        for n in (4, 16):
            us = min(m.bench_commit(36, B, n, 20) for _ in range(3))
            byt = 36 * B * (2 * 24 * 128 * 128 * 4 + 2 * 3 * 5120 * 2 + n * (24 * 128 + 8 * 128 + 24) * 4
                            + min(n, 3) * 5120 * 2)
            gbs = byt / us / 1e3
            ctx.detail(f"gdn commit perf 36 layers B={B} n={n} {us:.1f} us {gbs:.1f} GB/s ({byt} B)")
            if n == 4:
                out.append(f"commit 36 layers B{B} {us:.0f}µs {gbs:.0f}GB/s")
                ok &= gbs >= (0.80 if B == 1 else 0.85) * 241
    return {"ok": ok, "msg": " · ".join(out) + " (n=4, graph·best, GPU shared)"}


# ---- P020 gdn_fused_commit: the commit folded into the next verify (gdn_core_spec_pending, gdn_flush) ---------------
_REC = ((0, 24 * 4), (1536, 24 * 128 * 4), (198144, 8 * 128 * 4), (263680, 5120 * 2))  # eg | dl | k | x: offset, row B


def _rec_rows(rec, r, T):
    """Bytes of rows 0 … T−1 of record row r (e^g, δ, k, x): everything a spec call writes."""
    return torch.cat([rec[r, o:o + 16 * n].view(16, n)[:T].flatten() for o, n in _REC])


class _P020:
    """Two schemes over the same rounds. A (today): gdn_core_spec + gdn_commit per round, one record per state slot.
    B (P020): gdn_core_spec_pending — the previous round's kept rows are pending and applied by the next verify of that
    sequence — with records double-buffered (record row 2·slot + parity, the verify writes the other one) or in place
    (record row = slot); gdn_flush applies pending rows before any other use. Sequence b lives in state slots
    _slots(li, B)[b]; pending n and parity are per sequence (all layers alike)."""

    _warm = {}

    def __init__(self, ctx, B, pos, inplace):
        self.ctx, self.m, self.B, self.inplace = ctx, _mod(ctx), B, inplace
        key = (B, tuple(pos))
        if key not in self._warm:
            self._warm[key] = _spec_warm(ctx, B, pos)
        self.S0, self.c0 = self._warm[key]
        self.Sa, self.ca, self.Sb, self.cb = self.S0.clone(), self.c0.clone(), self.S0.clone(), self.c0.clone()
        ns, rb = len(SPEC_L) * B, self.m.rec_bytes()
        self.ra = torch.full((ns, rb), 255, dtype=torch.uint8, device="cuda")  # NaN bytes until written
        self.rb = torch.full(((1 if inplace else 2) * ns, rb), 255, dtype=torch.uint8, device="cuda")
        self.pos, self.pos0 = list(pos), list(pos)
        self.par, self.pn = [0] * B, [0] * B  # record parity, pending rows
        self.bad, self.checks = [], 0

    def _row(self, li, b, par):
        s = _slots(li, self.B)[b]
        return s if self.inplace else 2 * s + par

    def _cmp(self, what, x, y):
        self.checks += 1
        if not torch.equal(x, y):
            self.bad.append(what)

    def verify(self, bs, Ts, ns, tag):
        """One round for sequences bs (batch order), Ts rows each, ns kept (rows ≥ n are rejected drafts)."""
        st, B = _starts(Ts), self.B
        pre_b = (self.Sb.clone(), self.cb.clone())
        for li, i in enumerate(SPEC_L):
            q, a = _rows_at(self.ctx, i, [self.pos[b] for b in bs], Ts, ns)
            slots = [_slots(li, B)[b] for b in bs]
            ca = _spec(self.ctx, i, q, a, self.Sa, self.ca, self.ra, slots, st)
            cb = torch.empty_like(ca)
            self.m.run_spec_pending(i, q, a, self.Sb, self.cb, self.rb, slots,
                                    [self._row(li, b, self.par[b] ^ 1) for b in bs],
                                    [self._row(li, b, self.par[b]) for b in bs], [self.pn[b] for b in bs], st, cb)
            self._cmp(f"{tag} L{i} core", ca, cb)
            for j, b in enumerate(bs):
                self._cmp(f"{tag} L{i} rec b{b}", _rec_rows(self.ra, slots[j], Ts[j]),
                          _rec_rows(self.rb, self._row(li, b, self.par[b] ^ 1), Ts[j]))
        # B's verify committed the pending rows: its state of these sequences == A's before A's commit; others untouched
        for li in range(len(SPEC_L)):
            for b in range(B):
                s = _slots(li, B)[b]
                ra, rc = (self.Sa[s], self.ca[s]) if b in bs else (pre_b[0][s], pre_b[1][s])
                self._cmp(f"{tag} state b{b}", self.Sb[s], ra)
                self._cmp(f"{tag} conv b{b}", self.cb[s], rc)
        self.m.commit(self.Sa, self.ca, self.ra, [_slots(li, B)[b] for li in range(len(SPEC_L)) for b in bs], ns)
        for b, n in zip(bs, ns):
            self.par[b] ^= 1
            self.pn[b] = n
            self.pos[b] += n

    def flush(self, bs, tag):
        """gdn_flush of sequences bs (any pending n, 0 included); then their states equal A's."""
        L = len(SPEC_L)
        self.m.flush(self.Sb, self.cb, self.rb, [_slots(li, self.B)[b] for li in range(L) for b in bs],
                     [self._row(li, b, self.par[b]) for li in range(L) for b in bs], [self.pn[b] for b in bs])
        for b in bs:
            self.pn[b] = 0
            for li in range(L):
                s = _slots(li, self.B)[b]
                self._cmp(f"{tag} flush S b{b}", self.Sb[s], self.Sa[s])
                self._cmp(f"{tag} flush conv b{b}", self.cb[s], self.ca[s])

    def plain(self, bs, k, tag):
        """k true rows per sequence through gdn_core (prefill / plain step) in both schemes — after a flush."""
        assert all(self.pn[b] == 0 for b in bs)
        for li, i in enumerate(SPEC_L):
            q, a = _rows_at(self.ctx, i, [self.pos[b] for b in bs], [k] * len(bs))
            slots = [_slots(li, self.B)[b] for b in bs]
            x = _run(self.ctx, i, q, a, self.Sa, self.ca, slots, _starts([k] * len(bs)), False)[0]
            y = _run(self.ctx, i, q, a, self.Sb, self.cb, slots, _starts([k] * len(bs)), False)[0]
            self._cmp(f"{tag} plain L{i} core", x, y)
        for b in bs:
            self.pos[b] += k

    def finish(self, tag):
        """Flush all; S, conv == A, and == one gdn_core call over all rows each sequence kept (from the start)."""
        self.flush(list(range(self.B)), tag)
        self._cmp(f"{tag} final S", self.Sb, self.Sa)
        self._cmp(f"{tag} final conv", self.cb, self.ca)
        acc = [p - p0 for p, p0 in zip(self.pos, self.pos0)]
        Sc, cc = self.S0.clone(), self.c0.clone()
        live = [b for b in range(self.B) if acc[b]]
        for li, i in enumerate(SPEC_L):
            if live:
                _run(self.ctx, i, *_rows_at(self.ctx, i, [self.pos0[b] for b in live], [acc[b] for b in live]), Sc, cc,
                     [_slots(li, self.B)[b] for b in live], _starts([acc[b] for b in live]), False)
        self._cmp(f"{tag} one-shot S", self.Sb, Sc)
        self._cmp(f"{tag} one-shot conv", self.cb, cc)
        return acc


def test_p020_bitwise(ctx):
    """P020: two verify rounds + flush, every case: fused (gdn_core_spec_pending) == today (gdn_core_spec + gdn_commit)
    bitwise — core rows (all rows, kept or not), records, the state each fused verify writes (== today's state before
    its commit), flushed S and conv, 3 plain rows after the flush, one-shot gdn_core over all kept rows. B = 1 and 3
    (slot order ≠ batch order), round-1 T = 1·2·5·13·16 with every n of sequence 0 (0 … T), random n (0 … T) of the
    others, random round-2 T and n (0 included); records double-buffered and in place, NaN-filled; layers 0, 46."""
    gen = torch.Generator().manual_seed(20)
    r = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=gen))  # noqa: E731
    bad, cases, checks = [], 0, 0
    for inplace in (False, True):
        for B in (1, 3):
            pos = [37 + SPEC_H + 1300 * b for b in range(B)]
            for k, T in enumerate(SPEC_T):
                for n0 in range(T + 1):
                    T1 = [T] + [SPEC_T[(k + 2 * b) % len(SPEC_T)] for b in range(1, B)]
                    n1 = [n0] + [r(0, t) for t in T1[1:]]
                    T2 = [r(1, 16) for _ in range(B)]
                    n2 = [r(0, t) for t in T2]
                    c = _P020(ctx, B, pos, inplace)
                    tag = f"{'in place' if inplace else 'double'} B{B} T={T1}/{T2} n={n1}/{n2}"
                    c.verify(list(range(B)), T1, n1, tag + " r1")
                    c.verify(list(range(B))[::-1], T2[::-1], n2[::-1], tag + " r2")
                    c.flush(list(range(B)), tag)
                    c.plain(list(range(B)), 3, tag)
                    c.finish(tag)
                    cases += 1
                    checks += c.checks
                    bad += c.bad
            ctx.detail(f"gdn p020 bitwise {'in place' if inplace else 'double'} B={B}: {cases} cases so far, "
                       f"{len(bad)} mismatches")
    return {"ok": not bad, "msg": f"fused == spec+commit bitwise {cases} cases·{checks} comparisons (core all "
            f"rows·records·states·flush·next 3 rows·one-shot prefill, B=1·3, T=1·2·5·13·16 × n=0…T, double "
            f"buffer·in place)" if not bad else f"mismatch {len(bad)}: " + "; ".join(bad[:4])}


def test_p020_chain(ctx):
    """P020: decode loops of 16 rounds (B = 1 and 4 = the engine's slots): each round a random subset of the sequences
    in random order (absent ones keep their pending rows), T random 1…16, n random 1…T (every 4th round 0…T); every 3rd
    round a random subset is flushed and, half of the time, runs 1…4 plain rows (a prefill step) — fused vs today
    bitwise after every call (core, records, states), final flush == today == one gdn_core over all kept rows. Records
    double-buffered and in place."""
    gen = torch.Generator().manual_seed(2020)
    r = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=gen))  # noqa: E731
    bad, out, checks = [], [], 0
    for inplace in (False, True):
        for B, rounds in ((1, 16), (4, 16)):
            pos = [37 + SPEC_H + 1000 * b for b in range(B)]
            c = _P020(ctx, B, pos, inplace)
            nflush = nplain = 0
            for rd in range(rounds):
                bs = [b for b in torch.randperm(B, generator=gen).tolist() if B == 1 or r(0, 3)] or [r(0, B - 1)]
                Ts = [r(1, 16) for _ in bs]
                ns = [r(0 if rd % 4 == 3 else 1, t) for t in Ts]
                tag = f"{'in place' if inplace else 'double'} B{B} round {rd}"
                c.verify(bs, Ts, ns, tag)
                if rd % 3 == 2:
                    fs = [b for b in range(B) if r(0, 1)] or [r(0, B - 1)]
                    c.flush(fs, tag)
                    nflush += 1
                    if r(0, 1):
                        c.plain(fs, r(1, 4), tag)
                        nplain += 1
            acc = c.finish(f"{'in place' if inplace else 'double'} B{B}")
            checks += c.checks
            bad += c.bad
            out.append(f"{'in place' if inplace else 'double'} B{B} rows {'·'.join(map(str, acc))}")
            ctx.detail(f"gdn p020 chain {'in place' if inplace else 'double'} B={B} rounds={rounds} kept+plain rows "
                       f"{acc}, flushes {nflush}, plain steps {nplain}, checks {c.checks}, mismatches {len(c.bad)}")
    return {"ok": not bad, "msg": f"16-round chain fused == spec+commit bitwise {checks} comparisons "
            f"({' · '.join(out)}; partial batches·shuffled order·mid-chain flush·plain rows·one-shot prefill)"
            if not bad else f"mismatch {len(bad)}: " + "; ".join(bad[:4])}


def test_p020_perf(ctx):
    """P020 µs. Per layer (CUDA graph of 300 calls, 40 cold state sets, best of 5): gdn_core_spec and the fused verify
    at T = 5 with 4 pending rows, B = 1 and 4; gdn_commit of 36 layers (n = 4) per layer. Per round (the GDN part of one
    verify round: 36 layers, T = 5, n = 4 — today 36 spec calls + one commit, P020 36 fused calls with double-buffered
    records; one graph of 20 rounds, best of 5 alternating). Pass: fused round < today's at B = 1 and 4."""
    import time
    m = _mod(ctx)
    t = time.time()
    while time.time() - t < 0.5:
        m.bench(0, 1, 1, 500, 40, 0)
    T, n, out, ok = 5, 4, [], True
    for B in (1, 4):
        spec = min(m.bench_spec(0, B, T, 300, 40, 1) for _ in range(5))
        fused = min(m.bench_spec_pending(0, B, T, n, 300, 40, 1) for _ in range(5))
        com = min(m.bench_commit(36, B, n, 20) for _ in range(3)) / 36
        today, new = [], []
        for _ in range(5):
            today.append(m.bench_round(0, 36, B, T, n, 20, False))
            new.append(m.bench_round(0, 36, B, T, n, 20, True))
        a, f = min(today), min(new)
        runs = lambda v: ", ".join(f"{x:.1f}" for x in v)  # noqa: E731
        ctx.detail(f"gdn p020 perf B={B} T={T} n={n}: per layer spec {spec:.2f} + commit {com:.2f} = {spec + com:.2f} "
                   f"us, fused {fused:.2f} us; round (36 layers) today {a:.1f} us (runs {runs(today)}) fused {f:.1f} us "
                   f"(runs {runs(new)})")
        out.append(f"B{B} per layer {spec:.1f}+{com:.1f}→{fused:.1f}µs · round {a / 1e3:.3f}→{f / 1e3:.3f}ms "
                   f"({f - a:+.0f}µs)")
        ok &= f < a
    return {"ok": ok, "msg": " · ".join(out) + f" (T={T}, n={n}, 36 layers, graph·best)"}
