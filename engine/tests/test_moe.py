"""MoE family (engine/src/kernels/moe.cu) vs the golden dump and vs an fp64 reference from the checkpoint.

golden: router logits / top-10, silu_and_mul, shared expert, rank0 + rank1 partial sums vs `mlp#0.out` (post all-reduce;
`experts#0.out` is the same tensor), decode steps and prefill kept rows separately. The reference deployment quantises
the shared down_proj input to MXFP8 (b12x activation-quantised route, reproduced bit-exactly here) — the engine keeps
bf16 (lossless), so the routed path is also checked with the reference's own shared part substituted.
reference: rank-0 routed and shared parts recomputed in fp64 from the checkpoint with the reference deployment's bf16
rounding points.
Rank 1 is computed on this node with rank-1 weights (second Weights) and rank 1's golden inputs.
large_*: the large-M path (64 < M ≤ 8192): bitwise against the decode path row by row, golden prefill rows embedded in an
M=2048 call (both ranks), timing at M=512/2048/8192.
p014_*: P014 moe_scale4 (4-bit copy of the routed scales in the decode path): decode == expanded bytes for every expert
of this rank, on == off bitwise with real and forced-exception scales, timing on vs off.
p017_*: P017 router_pack12 (12-bit copy of the bf16 router in the decode path): decode == bf16 bits for every router of
this rank, on == off bitwise with the real and forced-exception routers (P014 on and off), timing on vs off.
p022_*: P022 route_fast (k_route22: one barrier stage, faster routing tail): on == off bitwise — outputs, plan, sig, xf —
for every layer of this rank + MTP, forced ties / degenerate routers, batch invariance with the patch on, timing."""
import os
import time

import torch

from golden import Golden, cmp

L = "language_model.model.layers"
CK = "model.language_model.layers"
LAYERS = [0, 3, 5, 23, 46, 47]
FIXED = 2621440 + 5120 + 2534400   # router + shared gate + shared expert bytes per call (rank)
EXPERT = 1382400                   # one routed expert, this rank
_S = {}


def _st(ctx):
    if not _S:
        m = ctx.ext("moe", ["src/kernels/moe/moe.cu", "src/core/weights.cc", "src/model/patches.cc"])
        for r in (0, 1):
            m.open(ctx.model_dir, ctx.table, r)
        _S.update(m=m, h={}, g1=Golden(os.path.join(os.path.dirname(ctx.dump), "rank1")))
    return _S["m"]


def _h(ctx, rank, layer):
    m = _st(ctx)
    if (rank, layer) not in _S["h"]:
        t0 = time.time()
        _S["h"][(rank, layer)] = m.upload(rank, layer)
        ctx.detail(f"upload rank {rank} layer {layer}: {time.time() - t0:.2f} s")
    return _S["h"][(rank, layer)]


def _g(ctx, rank):
    return ctx.golden if rank == 0 else _S["g1"]


def _in(ctx, rank, step, layer):
    return _g(ctx, rank).get(step, f"{L}.{layer}.mlp#0.in0").cuda().contiguous()


def _topk_ref(logits):
    p = torch.softmax(logits.float(), dim=-1)
    return torch.sort(-p, dim=-1, stable=True).indices[:, :10].int()


def _steps(ctx):
    dec = ctx.golden.decode_steps()
    pre = [s["step"] for s in ctx.golden.steps if s["num_tokens"] > 1]
    return dec, pre


def _agg(rows):
    """worst max_abs / max_rel, min cos over cmp() dicts"""
    return (max(r["max_abs"] for r in rows), max(r["max_rel"] for r in rows), min(r["cos"] for r in rows))


def _bf(x):
    return x.to(torch.bfloat16).double()


def _silu_mul(g, u):  # vLLM silu_and_mul on bf16 values (fp32 math, bf16(silu) then bf16 product)
    g32 = g.float()
    return _bf(_bf(g32 / (1.0 + torch.exp(-g32))) * u)


def _w(ctx, name):
    """this rank's slice (rank 0 loader): cuda uint8 [rows, bytes per row]"""
    b, shape, _, _ = ctx.w.slice(name)
    return b.view(shape[0], -1).cuda()


def _fp4(ctx, p):
    """NVFP4 weight p.weight (packed [N, K/2]) → fp64 [N, K] = e2m1 · e4m3 · g"""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64, device="cuda")
    b = _w(ctx, p + ".weight").long()
    codes = torch.stack([b & 15, b >> 4], -1).flatten(1)
    s = _w(ctx, p + ".weight_scale.nvfp4_csf_fixed").view(torch.float8_e4m3fn).double()
    g = ctx.w.slice(p + ".weight_scale_2")[0].view(torch.float32).item()
    return lut[codes] * s.repeat_interleave(16, 1) * g


def _fp8(ctx, p):
    """MXFP8 weight → fp64 = e4m3 · 2^(s-127)"""
    w = _w(ctx, p + ".weight").view(torch.float8_e4m3fn).double()
    s = _w(ctx, p + ".weight_scale").double()
    return w * torch.pow(2.0, s - 127).repeat_interleave(32, 1)


def _mxfp8_act(h):
    """b12x activation quantisation (the reference's shared down_proj input): per 32 values e = ceil(log2(amax/448)),
    RNE e4m3"""
    x = h.view(h.shape[0], -1, 32)
    sc = torch.pow(2.0, torch.ceil(torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 448.0)))
    return ((x / sc).float().to(torch.float8_e4m3fn).double() * sc).view_as(h)


def test_silu(ctx):
    """silu_and_mul rounding: golden shared gate_up out → ours vs golden act_fn out, bitwise."""
    m = _st(ctx)
    n = same = 0
    dec, pre = _steps(ctx)
    for s in dec[:8] + pre[:1]:
        for li in LAYERS:
            gu = ctx.golden.get(s, f"{L}.{li}.mlp.shared_expert.gate_up_proj#0.out.0").cuda()
            ref = ctx.golden.get(s, f"{L}.{li}.mlp.shared_expert.act_fn#0.out").cuda()
            got = m.silu_mul(gu)
            n += ref.numel()
            same += int((got.view(torch.int16) == ref.view(torch.int16)).sum())
    return {"ok": same == n, "msg": f"bitwise {same}/{n} ({same / n * 100:.4f}%)"}


def test_router(ctx):
    """router logits vs golden (bf16); top-10 ids from golden logits vs ours; kernel top-k vs torch on own logits."""
    m = _st(ctx)
    dec, pre = _steps(ctx)
    rows = id_same = id_same_own = bit = tot = 0
    worst = 0.0
    for s in dec + pre:
        for li in LAYERS:
            v = _in(ctx, 0, s, li)
            out, ids, tw, lg, *_ = m.forward(_h(ctx, 0, li), v)
            ref = ctx.golden.get(s, f"{L}.{li}.mlp.gate#0.out.0").cuda()
            tot += ref.numel()
            bit += int((lg.view(torch.int16) == ref.view(torch.int16)).sum())
            worst = max(worst, (lg.float() - ref.float()).abs().max().item())
            rg, ro = _topk_ref(ref), _topk_ref(lg)
            rows += ids.shape[0]
            id_same += int((torch.sort(ids, -1).values == torch.sort(rg, -1).values).all(-1).sum())
            id_same_own += int((ids == ro).all(-1).sum())
            if not torch.allclose(tw.sum(-1), torch.ones_like(tw[:, 0]), atol=1e-5):
                return {"ok": False, "msg": f"step {s} layer {li}: top-k weights do not sum to 1"}
    ctx.detail(f"router (decode + prefill kept rows, layers {LAYERS}): logits bitwise {bit}/{tot} max abs {worst:.3g}; "
               f"top-10 set == golden-logit top-10 {id_same}/{rows}; kernel top-k == torch top-k on own logits {id_same_own}/{rows}")
    ok = id_same_own == rows and id_same >= rows - max(1, rows // 200)
    return {"ok": ok, "msg": f"logits bitwise {bit / tot * 100:.2f}% · max abs {worst:.3g} · top-10 = golden {id_same}/{rows} · "
                             f"own-logit top-k exact {id_same_own}/{rows}"}


def test_shared(ctx):
    """shared part (rank 0, decode): ours vs golden; reference emulation (act-quantised down input) vs golden; h vs
    golden."""
    m = _st(ctx)
    perm = torch.tensor(m.frag_perm(320), device="cuda").long()
    dec, _ = _steps(ctx)
    ours, emu, hb, hn = [], 0, 0, 0
    en = 0
    for li in LAYERS:
        wd = _fp8(ctx, f"{CK}.{li}.mlp.shared_expert.down_proj")
        for s in dec:
            G = lambda k: ctx.golden.get(s, f"{L}.{li}.mlp.{k}").cuda()
            out, ids, tw, lg, routed, shared, hsh, nu = m.forward(_h(ctx, 0, li), _in(ctx, 0, s, li))
            ref = G("shared_expert#0.out")
            ours.append(cmp("shared", shared, ref))
            href = G("shared_expert.act_fn#0.out")
            hb += int((hsh[:, perm].view(torch.int16) == href.view(torch.int16)).sum())
            hn += href.numel()
            sig = _bf(torch.sigmoid(G("shared_expert_gate#0.out.0").float()))
            e = _bf(sig * _bf(_mxfp8_act(href.double()) @ wd.T))
            emu += int((e.bfloat16().view(torch.int16) == ref.view(torch.int16)).sum())
            en += ref.numel()
    a, r, c = _agg(ours)
    ctx.detail(f"shared (decode, {len(ours)} step×layer): ours vs golden max abs {a:.3g} max rel {r:.3g} min cos {c:.6f}; "
               f"h bitwise {hb}/{hn}; reference emulation (MXFP8 down input) vs golden bitwise {emu}/{en}")
    return {"ok": c > 0.999 and hb >= 0.999 * hn and emu >= 0.999 * en,
            "msg": f"ours vs golden max rel {r:.3g} · cos ≥ {c:.6f} · h bitwise {hb / hn * 100:.2f}% · "
                   f"reference act-quant emulation = golden {emu / en * 100:.2f}%"}


def test_reference(ctx):
    """rank 0 routed + shared parts vs fp64 from the checkpoint (own top-k), the reference deployment's rounding
    points: lossless check."""
    m = _st(ctx)
    dec, pre = _steps(ctx)
    cases = [(0, [dec[0]]), (47, [dec[5]]), (23, [pre[0]])]
    lines, worst, bit, tot = [], 0.0, 0, 0
    for li, steps in cases:
        p = f"{CK}.{li}.mlp."
        sw = {k: _fp8(ctx, p + f"shared_expert.{k}") for k in ("gate_proj", "up_proj", "down_proj")}
        sg = _w(ctx, p + "shared_expert_gate.weight").view(torch.bfloat16).double()
        cache = {}
        for s in steps:
            v = _in(ctx, 0, s, li)
            out, ids, tw, lg, routed, shared, hsh, nu = m.forward(_h(ctx, 0, li), v)
            x = v.double()
            M = x.shape[0]
            ref_r = torch.zeros(M, 2560, dtype=torch.float64, device="cuda")
            for e in sorted(set(ids.flatten().tolist())):
                if e not in cache:
                    pe = p + f"experts.{e}."
                    cache[e] = tuple(_fp4(ctx, pe + k) for k in ("gate_proj", "up_proj", "down_proj"))
                wg, wu, wd = cache[e]
                rows, ks = (ids == e).nonzero(as_tuple=True)
                h = _silu_mul(_bf(x[rows] @ wg.T), _bf(x[rows] @ wu.T))
                ref_r[rows] += _bf((h @ wd.T) * tw[rows, ks].double()[:, None])
            ref_r = _bf(ref_r)
            hs = _silu_mul(_bf(x @ sw["gate_proj"].T), _bf(x @ sw["up_proj"].T))
            sig = _bf(torch.sigmoid(_bf(x @ sg.T).float()))
            ref_s = _bf(sig * _bf(hs @ sw["down_proj"].T))
            for name, got, ref in (("routed", routed, ref_r), ("shared", shared, ref_s), ("out", out, _bf(ref_r + ref_s))):
                same = got.view(torch.int16) == ref.bfloat16().view(torch.int16)
                c = cmp(name, got, ref)
                lines.append(f"layer {li} step {s} M={M} {name}: bitwise {same.float().mean() * 100:.3f}% "
                             f"max rel {c['max_rel']:.2e} cos {c['cos']:.8f}")
                if name == "out":
                    worst, bit, tot = max(worst, c["max_rel"]), bit + int(same.sum()), tot + same.numel()
    for x in lines:
        ctx.detail("reference " + x)
    return {"ok": bit >= 0.99 * tot and worst < 5e-3,
            "msg": f"out bitwise {bit / tot * 100:.2f}% · max rel {worst:.2e} (fp64 reference; M=1, 1, prefill rows)"}


def _mlp(ctx, steps, golden_shared):
    """rank0 + rank1 partials (bf16 add, as the all-reduce) vs golden mlp#0.out; golden_shared: the reference's
    shared part replaces ours (routed path only)."""
    m = _st(ctx)
    rs, per_layer = [], {}
    for s in steps:
        for li in LAYERS:
            parts = []
            for r in (0, 1):
                o = m.forward(_h(ctx, r, li), _in(ctx, r, s, li))
                if golden_shared:
                    gs = _g(ctx, r).get(s, f"{L}.{li}.mlp.shared_expert#0.out").cuda()
                    parts.append((o[4].float() + gs.float()).bfloat16())
                else:
                    parts.append(o[0])
            tot = (parts[0].float() + parts[1].float()).bfloat16()
            c = cmp(f"s{s}.l{li}", tot, ctx.golden.get(s, f"{L}.{li}.mlp#0.out"))
            rs.append(c)
            per_layer.setdefault(li, []).append(c)
    for li, v in per_layer.items():
        a, r, c = _agg(v)
        ctx.detail(f"  layer {li}: max abs {a:.3g} max rel {r:.3g} min cos {c:.6f}")
    return _agg(rs) + (len(rs),)


def test_mlp_decode(ctx):
    """all decode steps × sample layers: ours (rank0 + rank1) vs golden mlp#0.out, and routed-only (the reference's
    shared part)."""
    dec, _ = _steps(ctx)
    ctx.detail(f"mlp decode, ours: {len(dec)} steps × layers {LAYERS}")
    a, r, c, n = _mlp(ctx, dec, False)
    ctx.detail("mlp decode, routed path (golden shared part):")
    a2, r2, c2, _ = _mlp(ctx, dec, True)
    return {"ok": c > 0.999 and c2 > 0.99995 and r2 < 0.02,
            "msg": f"{n} step×layer · ours max rel {r:.3g} cos ≥ {c:.6f} · routed-only max abs {a2:.3g} max rel {r2:.3g} "
                   f"cos ≥ {c2:.6f}"}


def test_mlp_prefill(ctx):
    """prefill kept rows (M ≤ 64; the reference ran other kernels there: Marlin larger blocks, MXFP8 act-quantised
    gate_up too)."""
    _, pre = _steps(ctx)
    ctx.detail(f"mlp prefill kept rows, ours: steps {pre} × layers {LAYERS}")
    a, r, c, n = _mlp(ctx, pre, False)
    ctx.detail("mlp prefill kept rows, routed path (golden shared part):")
    a2, r2, c2, _ = _mlp(ctx, pre, True)
    return {"ok": c > 0.999 and c2 > 0.9999,
            "msg": f"{n} step×layer · ours max rel {r:.3g} cos ≥ {c:.6f} · routed-only max abs {a2:.3g} max rel {r2:.3g} "
                   f"cos ≥ {c2:.6f}"}


def test_invariance(ctx):
    """M=64 (prefill kept rows) vs each row alone (M=1) and vs M=4/16/33 slices: bitwise; repeat run bitwise."""
    m = _st(ctx)
    _, pre = _steps(ctx)
    bad = []
    for li in (LAYERS[0], LAYERS[-1]):
        h = _h(ctx, 0, li)
        v = _in(ctx, 0, pre[1] if len(pre) > 1 else pre[0], li)
        full = m.forward(h, v)
        again = m.forward(h, v)
        if not all(torch.equal(x, y) for x, y in zip(full[:7], again[:7])):
            bad.append(f"l{li} rerun")
        for i in range(v.shape[0]):
            one = m.forward(h, v[i:i + 1].contiguous())
            if not all(torch.equal(one[k], full[k][i:i + 1]) for k in (0, 1, 2, 3, 4, 5)):
                bad.append(f"l{li} row{i}")
        for mm in (4, 16, 33):
            part = m.forward(h, v[-mm:].contiguous())
            if not torch.equal(part[0], full[0][-mm:]):
                bad.append(f"l{li} M{mm}")
    return {"ok": not bad, "msg": "M=1/4/16/33 rows == M=64 rows, rerun bitwise" if not bad else "differs: " + " ".join(bad[:8])}


def test_perf(ctx):
    """CUDA-graph timing with real routing (golden inputs), 4 layers per graph so weights do not stay in L2 (24 MB)."""
    m = _st(ctx)
    dec, pre = _steps(ctx)
    lay = LAYERS[:4]
    hs = [_h(ctx, 0, li) for li in lay]

    def rows(step_list, li):
        return torch.cat([_in(ctx, 0, s, li) for s in step_list])

    cases = {  # M=4: 4 consecutive decode tokens; M=16: 4 × 4 consecutive tokens at 4 positions/requests;
        1: lambda li: rows(dec[:1], li),  # M=64: kept rows of the long prefill
        4: lambda li: rows(dec[:4], li),
        16: lambda li: torch.cat([rows(dec[i:i + 4], li) for i in (0, 31, 46, 60)]),
        64: lambda li: _in(ctx, 0, max(pre, key=lambda s: ctx.golden.steps[s]["num_tokens"]), li)[-64:].contiguous(),
    }
    lines = []
    for M, mk in cases.items():
        vs = [mk(li) for li in lay]
        us = [int(m.forward(h, v)[7].item()) for h, v in zip(hs, vs)]
        t = min(m.bench(hs, vs, 20, 5) for _ in range(3)) / len(hs)
        byt = FIXED + sum(us) / len(us) * EXPERT
        lines.append(f"M={M} {t:.1f} µs · {byt / t / 1e3:.0f} GB/s ({byt / t / 1e3 / 241 * 100:.0f}% of 241) · "
                     f"experts {sum(us) / len(us):.1f}")
    for x in lines:
        ctx.detail("perf " + x)
    return {"ok": True, "msg": " | ".join(lines)}


# ---- large-M path (64 < M ≤ 8192, moe_large.cuh) ------------------------------------------------------------------------
_POOL = {}
_STAGES = ["router", "top-k", "scan", "place", "routed up", "routed down + shared up", "shared down + combine"]


def _pool(ctx, rank, layers):
    """golden MoE inputs of this rank per layer: prefill rows (steps 0/32/48) + 8 decode rows → [n, 2560] bf16 cuda"""
    need = [li for li in layers if (rank, li) not in _POOL]
    if need:
        dec, pre = _steps(ctx)
        acc = {li: [] for li in need}
        for s in pre + dec[:8]:
            for li in need:
                acc[li].append(_g(ctx, rank).get(s, f"{L}.{li}.mlp#0.in0"))
        for li in need:
            _POOL[(rank, li)] = torch.cat(acc[li]).cuda()
    return [_POOL[(rank, li)] for li in layers]


def _synth(pool, M, seed):
    """M activation rows = random golden row + per-feature-std Gaussian noise. Through the real router this reproduces
    the golden rows' routing skew (max rows/expert, top-5 % share, cv; checked on layers 0/23/47) while using more
    experts than 263 golden rows can."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    sd = pool.float().std(0, keepdim=True)
    idx = torch.randint(0, pool.shape[0], (M,), device="cuda", generator=gen)
    return (pool.float()[idx] + sd * torch.randn(M, 2560, device="cuda", generator=gen)).bfloat16().contiguous()


def test_large_bitwise(ctx):
    """large-M path vs decode path, bitwise, rows of: M=8192 (golden rows + zero-mean random + resampled golden+noise) ==
    M=64 decode calls covering all 8192 rows == M=1 rows == M=65/777/2048 large calls; rerun. All 7 outputs."""
    m = _st(ctx)
    bad, calls, info = [], 0, []
    for li in (LAYERS[0], LAYERS[-1]):
        h = _h(ctx, 0, li)
        pool = _pool(ctx, 0, [li])[0]
        gen = torch.Generator(device="cuda")
        gen.manual_seed(li)
        rnd = (pool.float().std(0, keepdim=True) * torch.randn(512, 2560, device="cuda", generator=gen)).bfloat16()
        X = torch.cat([pool, rnd, _synth(pool, 8192 - pool.shape[0] - 512, 100 + li)]).contiguous()
        full = m.forward(h, X)
        again = m.forward(h, X)
        if not all(torch.equal(a, b) for a, b in zip(full[:7], again[:7])):
            bad.append(f"l{li} rerun")
        cnt = torch.bincount(full[1].flatten().long(), minlength=512)
        info.append(f"layer {li}: experts {int(full[7].item())}, rows/expert max {int(cnt.max())} mean {cnt.float().mean():.0f}")

        def chk(tag, sub, r0):
            for k in range(7):  # out, ids, weights, logits, routed, shared, h_s
                if not torch.equal(sub[k], full[k][r0:r0 + sub[k].shape[0]]):
                    bad.append(f"l{li} {tag} out{k}")
                    return
        for c0 in range(0, 8192, 64):
            chk(f"M64@{c0}", m.forward(h, X[c0:c0 + 64].contiguous()), c0)
            calls += 1
        for r0 in list(range(8)) + [300, 1000, 5000, 8191]:
            chk(f"M1@{r0}", m.forward(h, X[r0:r0 + 1].contiguous()), r0)
            calls += 1
        for r0, mm in ((0, 2048), (6144, 2048), (100, 65), (4000, 777)):
            chk(f"M{mm}@{r0}", m.forward(h, X[r0:r0 + mm].contiguous()), r0)
            calls += 1
    ctx.detail("large bitwise: " + " · ".join(info))
    return {"ok": not bad, "msg": (f"bitwise ({calls} calls, 2 layers): M=8192 rows == 128×M=64 decode == M=1 == M=65/777/2048 "
                                   f"· 7 outputs · rerun" if not bad else "differs: " + " ".join(bad[:8]))}


def test_large_golden(ctx):
    """prefill steps 0 (61 rows) and 32 (last 64 kept rows of 8603) placed at the end of an M=2048 large-M call (filler:
    resampled golden + noise), ranks 0 and 1: rows == decode-path call on the golden rows alone (bitwise), and rank0 + rank1
    vs golden mlp#0.out; routed path also with the reference's own shared part (it quantises the shared down
    input)."""
    m = _st(ctx)
    _, pre = _steps(ctx)
    steps = [s for s in pre if ctx.golden.steps[s]["num_tokens"] > 1][:2]
    bitbad, ours, routed, n = [], [], [], 0
    for r in (0, 1):
        _pool(ctx, r, LAYERS)
    for s in steps:
        for li in LAYERS:
            parts, parts_r = [], []
            for r in (0, 1):
                h = _h(ctx, r, li)
                v = _in(ctx, r, s, li)
                k = v.shape[0]
                X = torch.cat([_synth(_pool(ctx, r, [li])[0], 2048 - k, 7 * s + li + 1000 * r), v]).contiguous()
                big = m.forward(h, X)
                dec = m.forward(h, v)
                n += 1
                if not all(torch.equal(big[j][-k:], dec[j]) for j in range(7)):
                    bitbad.append(f"r{r} s{s} l{li}")
                gs = _g(ctx, r).get(s, f"{L}.{li}.mlp.shared_expert#0.out").cuda()
                parts.append(big[0][-k:])
                parts_r.append((big[4][-k:].float() + gs.float()).bfloat16())
            ref = ctx.golden.get(s, f"{L}.{li}.mlp#0.out")
            ours.append(cmp(f"s{s}.l{li}", (parts[0].float() + parts[1].float()).bfloat16(), ref))
            routed.append(cmp(f"s{s}.l{li}", (parts_r[0].float() + parts_r[1].float()).bfloat16(), ref))
            ctx.detail(f"large golden step {s} layer {li}: ours max rel {ours[-1]['max_rel']:.3g} cos {ours[-1]['cos']:.6f} · "
                       f"routed-only max rel {routed[-1]['max_rel']:.3g} cos {routed[-1]['cos']:.6f}")
    a, r_, c = _agg(ours)
    a2, r2, c2 = _agg(routed)
    return {"ok": not bitbad and c > 0.999 and c2 > 0.9999,
            "msg": f"steps {steps} × {len(LAYERS)} layers × 2 ranks: rows == decode path {n - len(bitbad)}/{n} · "
                   f"vs golden ours max rel {r_:.3g} cos ≥ {c:.6f} · routed-only max rel {r2:.3g} cos ≥ {c2:.6f}"}


def test_large_perf(ctx):
    """large-M path, rank 0, real router + golden-like routing (_synth): µs per call (CUDA graph, best of 3×3) and per
    stage (events, best of 5); TFLOPS of the routed expert GEMMs (gate/up + down). Shared GPU: best times."""
    m = _st(ctx)
    res = []
    Ms = [int(x) for x in os.environ.get("Q38_MOE_PERF_M", "512,2048,8192").split(",")]  # rows per call
    for li in (LAYERS[0], 23, LAYERS[-1]):
        h = _h(ctx, 0, li)
        pool = _pool(ctx, 0, [li])[0]
        for M in Ms:
            X = _synth(pool, M, 11 + li)
            o = m.forward(h, X)
            cnt = torch.bincount(o[1].flatten().long(), minlength=512)
            best = None
            for _ in range(5):
                ms = m.prof(h, X)
                best = ms if best is None else [min(x, y) for x, y in zip(best, ms)]
            t = min(m.bench([h], [X], 3, 3) for _ in range(3))
            f_up, f_dn, f_su = 2 * M * 10 * 640 * 2560, 2 * M * 10 * 2560 * 320, 2 * M * 640 * 2560
            f_all = f_up + f_dn + 2 * M * 2560 * (512 + 640 + 320)
            tf_r = (f_up + f_dn + f_su) / ((best[4] + best[5]) * 1e9)
            res.append((li, M, t, f_all / t / 1e6, tf_r, f_up / best[4] / 1e9))
            ctx.detail(f"large perf layer {li} M={M}: {t:.0f} µs/call (graph) · all GEMMs {f_all / t / 1e6:.1f} TFLOPS · "
                       f"expert GEMMs (routed up + routed down & shared up) {tf_r:.1f} TFLOPS (routed up "
                       f"{f_up / best[4] / 1e9:.1f}, routed down + shared up {(f_dn + f_su) / best[5] / 1e9:.1f}) · "
                       f"experts {int(o[7].item())}, rows/expert max {int(cnt.max())} · stages µs: " +
                       " · ".join(f"{nm} {x * 1e3:.0f}" for nm, x in zip(_STAGES, best)))
    msg = []
    for M in Ms:
        rows = [x for x in res if x[1] == M]
        msg.append(f"M={M} {min(x[2] for x in rows):.0f}–{max(x[2] for x in rows):.0f} µs · expert GEMMs "
                   f"{min(x[4] for x in rows):.0f}–{max(x[4] for x in rows):.0f} TF (up {min(x[5] for x in rows):.0f}–"
                   f"{max(x[5] for x in rows):.0f})")
    return {"ok": True, "msg": " | ".join(msg) + " · layers 0/23/47"}


# ---- P010 moe_down_tune (p010_moe_down_tune.cuh: routed down + shared down/combine, same bits) ---------------------------
P010 = 10


def _p010_x(ctx, li):
    """the M=8192 rows of test_large_bitwise: golden rows + zero-mean random + resampled golden + noise"""
    pool = _pool(ctx, 0, [li])[0]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(li)
    rnd = (pool.float().std(0, keepdim=True) * torch.randn(512, 2560, device="cuda", generator=gen)).bfloat16()
    return torch.cat([pool, rnd, _synth(pool, 8192 - pool.shape[0] - 512, 100 + li)]).contiguous()


def test_large_p010(ctx):
    """P010 moe_down_tune: large path with the patch on == off, bitwise, all outputs (M=8192 rows of test_large_bitwise and
    slices 2048/2048/1000/777/65 of it, layers 0/47, rerun); stage µs off vs on (rank 0, _synth rows, M = Q38_MOE_PERF_M,
    layers 0/23/47, best of 7 alternating calls)."""
    m = _st(ctx)
    was = m.patch_on(P010)
    bad, n, lines, gain = [], 0, [], {}
    try:
        for li in (0, 47):
            h = _h(ctx, 0, li)
            X = _p010_x(ctx, li)
            for r0, mm in ((0, 8192), (0, 2048), (6144, 2048), (7000, 1000), (4000, 777), (100, 65)):
                x = X[r0:r0 + mm].contiguous()
                m.patch_set(P010, False)
                ref = m.forward(h, x)
                m.patch_set(P010, True)
                got, again = m.forward(h, x), m.forward(h, x)
                n += 1
                for k in range(8):
                    if not (torch.equal(ref[k], got[k]) and torch.equal(got[k], again[k])):
                        bad.append(f"l{li} M{mm}@{r0} out{k}")
                        break
        Ms = [int(x) for x in os.environ.get("Q38_MOE_PERF_M", "512,2048,8192").split(",")]
        for li in (0, 23, 47):
            h = _h(ctx, 0, li)
            pool = _pool(ctx, 0, [li])[0]
            for M in Ms:
                X = _synth(pool, M, 11 + li)
                best = {}
                for _ in range(7):
                    for on in (False, True):
                        m.patch_set(P010, on)
                        ms = m.prof(h, X)
                        best[on] = ms if on not in best else [min(a, b) for a, b in zip(best[on], ms)]
                f, o = best[False], best[True]
                gain.setdefault(M, []).append(((f[5] + f[6]) - (o[5] + o[6])) * 1e3)
                lines.append(f"layer {li} M={M}: routed down + shared up {f[5] * 1e3:.0f} → {o[5] * 1e3:.0f} µs · shared down + "
                             f"combine {f[6] * 1e3:.0f} → {o[6] * 1e3:.0f} µs · MoE call {sum(f) * 1e3:.0f} → {sum(o) * 1e3:.0f} µs")
    finally:
        m.patch_set(P010, was)
    ctx.detail("\n".join(["== moe.large_p010 (off → on)"] + lines + bad))
    g = " · ".join(f"M={M} −{min(v):.0f}…{max(v):.0f} µs" for M, v in gain.items())
    return {"ok": not bad, "msg": (f"on == off bitwise ({n} calls, 8 outputs, rerun)" if not bad else "differs: " + " ".join(bad[:6]))
            + f" · routed down + shared down/combine: {g} (layers 0/23/47)"}


# ---- P014 moe_scale4 (p014_moe_scale4.cuh: decode kernel reads the routed scales from a 4-bit copy, same bits) ---------
P014 = 14
P14_MS = (1, 2, 4, 5, 8, 16, 20, 33, 64)
SC4 = 88320          # 4-bit records per expert (expanded scales: 153,600 B)
SCALES = 153600


def _upload14(ctx, rank, li=None):
    """upload with the patch on (the 4-bit copy is built only then), whatever the run's default"""
    m = _st(ctx)
    was = m.patch_on(P014)
    m.patch_set(P014, True)
    try:
        return m.upload(rank, li) if li is not None else m.upload_mtp(rank)
    finally:
        m.patch_set(P014, was)


def _h14(ctx, rank, li):
    m = _st(ctx)
    h = _h(ctx, rank, li)
    if m.p014_built(h):
        return h
    if ("p14", rank, li) not in _S["h"]:
        _S["h"][("p14", rank, li)] = _upload14(ctx, rank, li)
    return _S["h"][("p14", rank, li)]


def _same(a, b):
    """bitwise (NaN-safe): integer views"""
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def _onoff(m, h, x, bad, tag):
    """forward with the patch off, on, on again: all 8 outputs bitwise"""
    was = m.patch_on(P014)
    try:
        m.patch_set(P014, False)
        ref = m.forward(h, x)
        m.patch_set(P014, True)
        got, again = m.forward(h, x), m.forward(h, x)
    finally:
        m.patch_set(P014, was)
    for k in range(8):
        if not (_same(ref[k], got[k]) and _same(got[k], again[k])):
            bad.append(f"{tag} out{k}")
            break
    return got


def test_p014_roundtrip(ctx):
    """P014: every routed expert's 4-bit scales decoded on the device by the kernel's decode (escaped rows: plain bytes)
    == the expanded e4m3 bytes, all 48 layers × 512 experts of this rank + the MTP layer's experts; upload statistics."""
    m = _st(ctx)
    tot = [0] * 5
    bad, lines = [], []
    t0 = time.time()
    for li in list(range(48)) + [None]:
        h = _upload14(ctx, ctx.rank, li)
        try:
            dec, ref = m.p014_unpack(h)
            nb = int((dec != ref).sum())
            v = m.p014_info(h)
        finally:
            m.free(h)
        name = f"layer {li}" if li is not None else "mtp"
        if nb or v[0] != 512 * SCALES:
            bad.append(f"{name}: {nb} bytes differ")
        if li is not None:
            tot = [a + b for a, b in zip(tot, v)]
        lines.append(f"p014 roundtrip {name}: {nb} bytes differ · exceptions {v[1] / v[0] * 100:.4f}% · items over 8 "
                     f"{v[3]}/{v[2]} ({v[3] / v[2] * 100:.3f}%) · escaped rows {v[4]}")
    for x in lines:
        ctx.detail(x)
    s, e, it, ei, er = tot
    msg = (f"48 layers × 512 experts (rank {ctx.rank}) + mtp: decoded == expanded" if not bad else "differs: " + " · ".join(bad[:4]))
    return {"ok": not bad, "msg": msg + f" · exceptions {e / s * 100:.3f}% ({e / it:.2f}/item) · items over 8 {ei / it * 100:.2f}% · "
                                        f"escaped rows {er / (8 * it) * 100:.3f}% · mtp {lines[-1].split(' · ', 1)[1]} · {time.time() - t0:.0f}s"}


def _p14_rows(ctx, rank, li, M, k):
    """M golden rows of layer li (rank's pool), k-th window"""
    pool = _pool(ctx, rank, [li])[0]
    n = pool.shape[0]
    o = (17 * k + 7 * M) % (n - M + 1)
    return pool[o:o + M].contiguous()


def test_p014_bitwise(ctx):
    """P014 on == off, bitwise, all 8 outputs (+ rerun on), decode path with real routing: M = 1, 2, 4, 5, 8, 16, 20, 33, 64
    from golden MoE inputs (prefill kept rows + decode rows), layers LAYERS × both ranks' weights; with the patch on,
    rows of an M=64 call == the same rows alone (M=1) and in M=5/20/33 slices."""
    m = _st(ctx)
    bad, n = [], 0
    for r in (0, 1):
        for li in LAYERS:
            h = _h14(ctx, r, li)
            for M in P14_MS:
                for k in range(2 if M < 64 else 1):
                    _onoff(m, h, _p14_rows(ctx, r, li, M, k), bad, f"r{r} l{li} M{M}.{k}")
                    n += 1
            x = _p14_rows(ctx, r, li, 64, 0)
            was = m.patch_on(P014)
            m.patch_set(P014, True)
            try:
                full = m.forward(h, x)
                for i in (0, 1, 31, 63):
                    one = m.forward(h, x[i:i + 1].contiguous())
                    if not all(_same(one[j], full[j][i:i + 1]) for j in range(6)):
                        bad.append(f"r{r} l{li} row{i}")
                for a, b in ((0, 5), (7, 27), (31, 64)):
                    part = m.forward(h, x[a:b].contiguous())
                    if not _same(part[0], full[0][a:b]):
                        bad.append(f"r{r} l{li} rows {a}:{b}")
            finally:
                m.patch_set(P014, was)
    return {"ok": not bad, "msg": (f"on == off bitwise ({n} calls × 8 outputs, rerun; M {','.join(map(str, P14_MS))}; "
                                   f"layers {LAYERS} × ranks 0/1) · on: M=1/5/20/33 rows == M=64 rows"
                                   if not bad else "differs: " + " ".join(bad[:8]))}


def _force(gen, t, mode):
    """t: one expert's expanded scales (uint8 cpu [153600]: gate/up [40][8][160][2], down [160][8][20][2]) → forced
    exceptions. mode 0: 0…8 per item (out of window below / above, boundary values); 1: 9…72 per item (row escapes, whole
    rows / items random); 2: edges — rows built around a base: b … b+15 in window, b−1 / b+16 out, 0x00 and 0x7E,
    the last row's last scale = the entry 0xFFFE."""
    gu = t[:102400].view(40, 8, 5, 32, 2)  # i, r, kp, ks, p
    dn = t[102400:].view(160, 8, 20, 2)    # n, r, ks, half
    def rnd(lo, hi, shape):
        return torch.randint(lo, hi + 1, shape, generator=gen, dtype=torch.int64).to(torch.uint8)

    def put(blk, k, vals):  # k random positions of the (strided) view blk ← vals(k)
        flat = blk.reshape(-1).clone()
        flat[torch.randperm(flat.numel(), generator=gen)[:k]] = vals(k)
        blk.copy_(flat.view(blk.shape))
    if mode == 0:
        for i in range(40):
            for kp in range(5):
                put(gu[i, :, kp], (i * 5 + kp) % 9,
                    lambda k: torch.where(torch.rand(k, generator=gen) < 0.5, rnd(0, 0x30, (k,)), rnd(0x7B, 0x7E, (k,))))
        for nn in range(160):
            put(dn[nn], nn % 9, lambda k: rnd(0, 0x30, (k,)))
    elif mode == 1:
        for i in range(40):
            for kp in range(5):
                put(gu[i, :, kp], 9 + (i * 5 + kp) * 7 % 64, lambda k: rnd(0, 0x7E, (k,)))
        gu[3, :, 2] = rnd(0, 0x7E, (8, 32, 2))   # a whole item random
        gu[5, 6, 1] = rnd(0, 0x7E, (32, 2))      # a whole row random
        for nn in range(0, 160, 3):
            put(dn[nn], 9 + nn % 50, lambda k: rnd(0, 0x7E, (k,)))
        dn[100] = rnd(0, 0x7E, (8, 20, 2))
    else:
        for i in range(40):
            for kp in range(5):
                b = int(rnd(0, 0x6F, (1,)))
                blk = (b + rnd(0, 15, (8, 32, 2)).long()).clamp(max=0x7E).to(torch.uint8)
                if kp == 0:
                    blk[0, 0, 0], blk[1, 31, 1] = max(b - 1, 0), min(b + 16, 0x7E)   # just outside (b−1 may be in a lower window)
                if kp == 1:
                    blk[2] = 0                                                      # zero scales, base 0
                    blk[2, 5, 0] = 0x7E
                if kp == 2:
                    blk[7] = rnd(0, 15, (32, 2))
                    blk[7, 31, 1] = 0x7E                                            # r 7, word 15, byte 3, 0x7E: entry 0xFFFE
                gu[i, :, kp] = blk
        for nn in range(160):
            b = int(rnd(0x50, 0x6F, (1,)))
            blk = (b + rnd(0, 15, (8, 20, 2)).long()).to(torch.uint8)
            if nn % 2 == 0:
                blk[7, 19, 1] = 0                                                   # r 7, word 9, byte 3
            dn[nn] = blk
    return t


def test_p014_forced(ctx):
    """P014 with forced exceptions: layer 0 (this rank), every expert the test rows route to gets synthetic scales (_force:
    0…8 exceptions per item, row escapes up to whole items, window edges, zero scales, the 0xFFFE entry), written to the
    block and rebuilt (host check of every row): device decode == expanded bytes for all 512 experts; on == off bitwise
    at M = 1, 2, 4, 5, 8, 16, 20, 33, 64, all 8 outputs (NaN-safe), rerun."""
    m = _st(ctx)
    li = 0
    h = _upload14(ctx, ctx.rank, li)
    bad, st, n = [], [0] * 5, 0
    try:
        xs = {(M, k): _p14_rows(ctx, ctx.rank, li, M, k) for M in P14_MS for k in range(2 if M < 64 else 1)}
        used = set()
        for x in xs.values():
            used |= set(m.forward(h, x)[1].flatten().tolist())
        gen = torch.Generator()
        gen.manual_seed(14)
        for j, e in enumerate(sorted(used)):
            t = _force(gen, m.p014_get(h, e), j % 3)
            v = m.p014_set(h, e, t)
            st = [a + b for a, b in zip(st, v)]
            if not torch.equal(m.p014_get(h, e), t):
                bad.append(f"expert {e} not written")
        dec, ref = m.p014_unpack(h)
        nb = int((dec != ref).sum())
        if nb:
            bad.append(f"decode: {nb} bytes differ")
        for (M, k), x in xs.items():
            _onoff(m, h, x, bad, f"M{M}.{k}")
            n += 1
    finally:
        m.free(h)
    s, e, it, ei, er = st
    ctx.detail(f"p014 forced: {len(used)} experts · scales {s} · exceptions {e} ({e / s * 100:.2f}%) · items over 8 {ei}/{it} · "
               f"escaped rows {er}")
    return {"ok": not bad, "msg": (f"{len(used)} routed experts forced: exceptions {e / s * 100:.2f}%, items over 8 {ei / it * 100:.1f}%, "
                                   f"escaped rows {er} · decode == expanded (512 experts) · on == off bitwise ({n} calls × 8 outputs)"
                                   if not bad else "differs: " + " ".join(bad[:8]))}


def test_p014_perf(ctx):
    """P014: µs per call off vs on (CUDA graph, 4 layers per graph as test_perf, same inputs → same experts), best of 5
    alternating rounds; GB/s of the useful bytes (router + shared + experts × 1,382,400) and of the bytes read on."""
    m = _st(ctx)
    dec, pre = _steps(ctx)
    lay = LAYERS[:4]
    hs = [_h14(ctx, 0, li) for li in lay]

    def rows(step_list, li):
        return torch.cat([_in(ctx, 0, s, li) for s in step_list])

    cases = {
        1: lambda li: rows(dec[:1], li),
        4: lambda li: rows(dec[:4], li),
        16: lambda li: torch.cat([rows(dec[i:i + 4], li) for i in (0, 31, 46, 60)]),
        20: lambda li: torch.cat([rows(dec[i:i + 4], li) for i in (0, 15, 31, 46, 60)]),
        64: lambda li: _in(ctx, 0, max(pre, key=lambda s: ctx.golden.steps[s]["num_tokens"]), li)[-64:].contiguous(),
    }
    was = m.patch_on(P014)
    lines, short = [], []
    try:
        for M, mk in cases.items():
            vs = [mk(li) for li in lay]
            us = [int(m.forward(h, v)[7].item()) for h, v in zip(hs, vs)]
            best = {}
            for _ in range(5):
                for on in (False, True):
                    m.patch_set(P014, on)
                    t = m.bench(hs, vs, 20, 5) / len(hs)
                    best[on] = min(best.get(on, 1e30), t)
            ex = sum(us) / len(us)
            useful = FIXED + ex * EXPERT
            read = FIXED + ex * (EXPERT - SCALES + SC4)
            f, o = best[False], best[True]
            lines.append(f"M={M}: off {f:.1f} µs ({useful / f / 1e3:.0f} GB/s) → on {o:.1f} µs ({useful / o / 1e3:.0f} GB/s useful, "
                         f"{read / o / 1e3:.0f} GB/s read) · {(o / f - 1) * 100:+.1f}% · experts {ex:.1f}")
            short.append(f"M={M} {f:.1f}→{o:.1f} ({(o / f - 1) * 100:+.1f}%)")
    finally:
        m.patch_set(P014, was)
    ctx.detail("\n".join(["== moe.p014_perf (off → on, µs per layer call)"] + lines))
    return {"ok": True, "msg": "µs " + " · ".join(short)}


# ---- P017 router_pack12 (p017_router_pack12.cuh: k_route reads a 12-bit copy of the bf16 router, same bits) ------------
P017 = 17
ROUTER = 2621440      # bf16 router bytes per layer
ROUTER12 = 1991168    # its 12-bit copy: 32 × 160 × 384 B blocks + 512 B bases + 768 × 8 × 4 B slots


def _router_bits(ctx, name):
    """this rank's router tensor (the bytes moe_upload reads): CPU int16 [512, 2560]"""
    b, shape, dt, _ = ctx.w.slice(name)
    assert dt == "BF16" and list(shape) == [512, 2560], (name, shape, dt)
    return b.view(torch.int16).view(512, 2560)


def _upload17(ctx, rank, li=None):
    """upload with P014 and P017 on (both copies are built only then), whatever the run's default"""
    m = _st(ctx)
    was = (m.patch_on(P014), m.patch_on(P017))
    m.patch_set(P014, True)
    m.patch_set(P017, True)
    try:
        return m.upload(rank, li) if li is not None else m.upload_mtp(rank)
    finally:
        m.patch_set(P014, was[0])
        m.patch_set(P017, was[1])


def _h17(ctx, rank, li):
    m = _st(ctx)
    h = _h(ctx, rank, li)
    if m.p017_built(h) and m.p014_built(h):
        return h
    if ("p17", rank, li) not in _S["h"]:
        _S["h"][("p17", rank, li)] = _upload17(ctx, rank, li)
    return _S["h"][("p17", rank, li)]


def _onoff17(m, h, x, bad, tag):
    """forward with P017 off, on, on again (P014 as the caller set it): all 8 outputs bitwise (NaN-safe)"""
    was = m.patch_on(P017)
    try:
        m.patch_set(P017, False)
        ref = m.forward(h, x)
        m.patch_set(P017, True)
        got, again = m.forward(h, x), m.forward(h, x)
    finally:
        m.patch_set(P017, was)
    for k in range(8):
        if not (_same(ref[k], got[k]) and _same(got[k], again[k])):
            bad.append(f"{tag} out{k}")
            break
    return got


def _p17_stats(v):
    n, e, se, es, mx = v
    return f"exceptions {e} ({e / n * 100:.4f}%) · spans with exceptions {se}/768 · escaped {es} · max/span {mx}"


def test_p017_roundtrip(ctx):
    """P017: every router tensor of this rank (48 layers + MTP) → 12-bit copy (every value checked on the host by the
    builder) → decoded on the device by k_route's load/decode (escaped spans: the bf16 fragments) == the bf16 bits;
    and the copies moe_upload built for layers 0/47 and the MTP layer decode to the checkpoint's bits."""
    m = _st(ctx)
    names = [f"{CK}.{li}.mlp.gate.weight" for li in range(48)] + ["mtp.layers.0.mlp.gate.weight"]
    bad, lines, tot = [], [], [0] * 5
    t0 = time.time()
    for nm in names:
        w = _router_bits(ctx, nm)
        dec, v = m.p017_roundtrip(w)
        nb = int((dec.view(torch.int16).cpu() != w).sum())
        if nb or v[0] != 512 * 2560:
            bad.append(f"{nm}: {nb} values differ")
        tot = [a + b for a, b in zip(tot[:4], v[:4])] + [max(tot[4], v[4])]
        lines.append(f"p017 roundtrip {nm}: {nb} values differ · " + _p17_stats(v))
    for li in (0, 47, None):
        h = _h17(ctx, ctx.rank, li) if li is not None else _upload17(ctx, ctx.rank)
        try:
            nm = f"{CK}.{li}.mlp.gate.weight" if li is not None else "mtp.layers.0.mlp.gate.weight"
            nb = int((m.p017_unpack(h).view(torch.int16).cpu() != _router_bits(ctx, nm)).sum())
            if nb:
                bad.append(f"upload {nm}: {nb} values differ")
            lines.append(f"p017 moe_upload copy {nm}: {nb} values differ · " + _p17_stats(m.p017_info(h)))
        finally:
            if li is None:
                m.free(h)
    for x in lines:
        ctx.detail(x)
    n, e, se, es, mx = tot
    msg = (f"49 routers (rank {ctx.rank}, 48 layers + mtp) + 3 uploaded copies: decoded == bf16 bits" if not bad
           else "differs: " + " · ".join(bad[:4]))
    return {"ok": not bad, "msg": msg + f" · exceptions {e / n * 100:.4f}% · spans with exceptions {se}/{49 * 768} · "
                                        f"escaped spans {es} · max/span {mx} · {time.time() - t0:.0f}s"}


def test_p017_bitwise(ctx):
    """P017 on == off, bitwise, all 8 outputs (+ rerun on), decode path with real routing: M = 1, 2, 4, 5, 8, 16, 20, 33, 64
    from golden MoE inputs (two windows below 64), layers LAYERS × ranks 0/1 weights, with P014 on and with P014 off;
    with both on, rows of an M=64 call == the same rows alone (M=1) and in M=5/20/33 slices."""
    m = _st(ctx)
    bad, n = [], 0
    was = (m.patch_on(P014), m.patch_on(P017))
    try:
        for r in (0, 1):
            for li in LAYERS:
                h = _h17(ctx, r, li)
                for p14 in (True, False):
                    m.patch_set(P014, p14)
                    for M in P14_MS:
                        for k in range(2 if M < 64 else 1):
                            _onoff17(m, h, _p14_rows(ctx, r, li, M, k), bad, f"r{r} l{li} p14={int(p14)} M{M}.{k}")
                            n += 1
                m.patch_set(P014, True)
                m.patch_set(P017, True)
                x = _p14_rows(ctx, r, li, 64, 0)
                full = m.forward(h, x)
                for i in (0, 1, 31, 63):
                    one = m.forward(h, x[i:i + 1].contiguous())
                    if not all(_same(one[j], full[j][i:i + 1]) for j in range(6)):
                        bad.append(f"r{r} l{li} row{i}")
                for a, b in ((0, 5), (7, 27), (31, 64)):
                    part = m.forward(h, x[a:b].contiguous())
                    if not _same(part[0], full[0][a:b]):
                        bad.append(f"r{r} l{li} rows {a}:{b}")
                m.patch_set(P017, was[1])
    finally:
        m.patch_set(P014, was[0])
        m.patch_set(P017, was[1])
    return {"ok": not bad, "msg": (f"on == off bitwise ({n} calls × 8 outputs, rerun; M {','.join(map(str, P14_MS))}; "
                                   f"layers {LAYERS} × ranks 0/1 × P014 on/off) · on: M=1/5/20/33 rows == M=64 rows"
                                   if not bad else "differs: " + " ".join(bad[:8]))}


def _bits16(sign, e, m):
    return (sign << 15) | (e << 7) | m


def _force17(gen, w, mode):
    """w: router bits int16 [512, 2560] → forced exceptions, placed by k_route's spans (tile i = 16 rows, block part p,
    warp → k-steps ks0 … +7/7/6; lane (r, q), value j → row 16i+r(+8), column 16ks+2q(+1)(+8)).
    mode 0: span s gets s % 11 values below its rows' windows (tiny normals, ±0, subnormals) at random (k-step, lane,
    value) — 8 fill the slots, 9 and 10 escape the span (with the checkpoint's own exceptions on top); row 37: every value
    but one tiny (its spans escape); row 70 all zero (base 0); row 100 subnormal-sized (base 0: zeros / subnormals inside
    the window); rows 130 / 131: top exponent odd / even with values at the window's bottom, one below and two below.
    mode 1: mode 0 + ±inf / NaN in rows 200 / 201 (window at the top: every other value of the row an exception)."""
    u = w.to(torch.int32) & 0xFFFF
    tiny = [0x0000, 0x8000, 0x0001, 0x8003, 0x007F, 0x0D8A, 0x9E2B, 0x2000]  # ±0, subnormals, 1e-30, -1e-20, 2^-63
    for sp in range(768):
        c = sp % 11
        if not c:
            continue
        blk, wp = sp // 8, sp % 8
        i, part = blk // 3, blk % 3
        nks = 7 if part < 2 else 6
        ks0 = part * 56 + wp * nks
        pos = torch.randperm(nks * 32 * 8, generator=gen)[:c].tolist()
        for t, x in enumerate(pos):
            s, lane, j = x // 256, (x // 8) % 32, x % 8
            r, q = lane >> 2, lane & 3
            row = 16 * i + r + (8 if j >= 4 else 0)
            col = 16 * (ks0 + s) + 2 * q + (j & 1) + 8 * ((j >> 1) & 1)
            u[row, col] = tiny[(sp + t) % len(tiny)]
    u[37] = (w[37].view(torch.bfloat16).float() * 1e-12).bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
    u[37, 5] = _bits16(0, 125, 0x23)
    u[70] = 0
    u[100] = (torch.randn(2560, generator=gen) * 1e-38).bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
    u[100, ::97] = 0x8000
    for row, top in ((130, 127), (131, 128)):
        b = (top >> 1) - 7
        lo = 2 * b
        u[row] = (torch.randn(2560, generator=gen) * 2.0 ** (top - 127 - 3)).bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
        u[row, 0] = _bits16(0, top, 0x11)
        for k, e in enumerate((lo, lo + 1, lo - 1, lo - 2, lo + 15, lo, lo - 1)):
            u[row, 3 + 41 * k] = _bits16(k & 1, e, (k * 29) & 0x7F)
    if mode == 1:
        u[200, [10, 999, 2000]] = torch.tensor([0x7F80, 0xFF80, 0x7FC0], dtype=torch.int32)
        u[201, [3, 1500]] = torch.tensor([0xFFA5, 0x7F81], dtype=torch.int32)
    return u.to(torch.int16)


def test_p017_forced(ctx):
    """P017 with forced exceptions (_force17) in layer 0's router (rank's upload): mode 0 (finite: 0…10 per span, escaped
    spans, whole-row exceptions, zero / tiny rows, window edges) and mode 1 (+ ±inf / NaN rows): set (fragments + copy
    rebuilt, host-checked) → device decode == bits; on == off bitwise at M = 1 … 64 (golden rows), all 8 outputs
    (NaN-safe), rerun, P014 on."""
    m = _st(ctx)
    h = _upload17(ctx, ctx.rank, 0)
    bad, n, lines = [], 0, []
    was = m.patch_on(P014)
    try:
        m.patch_set(P014, True)
        w0 = _router_bits(ctx, f"{CK}.0.mlp.gate.weight")
        gen = torch.Generator()
        gen.manual_seed(17)
        for mode in (0, 1):
            w = _force17(gen, w0, mode)
            v = m.p017_set(h, w)
            nb = int((m.p017_unpack(h).view(torch.int16).cpu() != w).sum())
            if nb:
                bad.append(f"mode {mode}: decode {nb} values differ")
            if v[3] == 0:
                bad.append(f"mode {mode}: no escaped span")
            for M in P14_MS:
                for k in range(2 if M < 64 else 1):
                    _onoff17(m, h, _p14_rows(ctx, ctx.rank, 0, M, k), bad, f"mode {mode} M{M}.{k}")
                    n += 1
            lines.append(f"mode {mode}: " + _p17_stats(v) + f" · decode {nb} values differ")
    finally:
        m.patch_set(P014, was)
        m.free(h)
    ctx.detail("\n".join(["== moe.p017_forced"] + lines + bad))
    return {"ok": not bad, "msg": (" | ".join(lines) + f" · on == off bitwise ({n} calls × 8 outputs)" if not bad
                                   else "differs: " + " ".join(bad[:8]))}


def test_p017_perf(ctx):
    """P017: µs per layer call off vs on (P014 on; CUDA graph, 4 layers per graph as test_perf, same inputs → same
    experts), best of 5 alternating rounds; GB/s of the bytes read (router bf16 or 12-bit + shared + experts with P014's
    scale records)."""
    m = _st(ctx)
    dec, pre = _steps(ctx)
    lay = LAYERS[:4]
    hs = [_h17(ctx, 0, li) for li in lay]

    def rows(step_list, li):
        return torch.cat([_in(ctx, 0, s, li) for s in step_list])

    cases = {
        1: lambda li: rows(dec[:1], li),
        4: lambda li: rows(dec[:4], li),
        16: lambda li: torch.cat([rows(dec[i:i + 4], li) for i in (0, 31, 46, 60)]),
        20: lambda li: torch.cat([rows(dec[i:i + 4], li) for i in (0, 15, 31, 46, 60)]),
        64: lambda li: _in(ctx, 0, max(pre, key=lambda s: ctx.golden.steps[s]["num_tokens"]), li)[-64:].contiguous(),
    }
    was = (m.patch_on(P014), m.patch_on(P017))
    lines, short = [], []
    try:
        m.patch_set(P014, True)
        for M, mk in cases.items():
            vs = [mk(li) for li in lay]
            us = [int(m.forward(h, v)[7].item()) for h, v in zip(hs, vs)]
            best = {}
            for _ in range(5):
                for on in (False, True):
                    m.patch_set(P017, on)
                    t = m.bench(hs, vs, 20, 5) / len(hs)
                    best[on] = min(best.get(on, 1e30), t)
            ex = sum(us) / len(us)
            read_off = FIXED + ex * (EXPERT - SCALES + SC4)
            read_on = read_off - ROUTER + ROUTER12
            f, o = best[False], best[True]
            lines.append(f"M={M}: off {f:.1f} µs ({read_off / f / 1e3:.0f} GB/s read) → on {o:.1f} µs ({read_on / o / 1e3:.0f} GB/s "
                         f"read) · {o - f:+.1f} µs ({(o / f - 1) * 100:+.1f}%) · experts {ex:.1f}")
            short.append(f"M={M} {f:.1f}→{o:.1f} ({(o / f - 1) * 100:+.1f}%)")
    finally:
        m.patch_set(P014, was[0])
        m.patch_set(P017, was[1])
    ctx.detail("\n".join(["== moe.p017_perf (off → on, µs per layer call, P014 on)"] + lines))
    return {"ok": True, "msg": "µs " + " · ".join(short)}


# ---- P022 route_fast (p022_route_fast.cuh: k_route22 — one barrier stage, faster routing tail, same bits) --------------
P022 = 22
OUTS = ("out", "ids", "weights", "logits", "routed", "shared", "h_s", "experts")
# moe.cu Plan as int32: U, uexp[512], ucnt[512], uoff[512], utok[640], uw[640], tpair[640], NT, tt_u[592], tt_t[592]
PLAN_F = (("U", 1), ("uexp", 512), ("ucnt", 512), ("uoff", 512), ("utok", 640), ("uw", 640), ("tpair", 640), ("NT", 1),
          ("tt_u", 592), ("tt_t", 592))


def _plan(plan, M):
    """the parts of a Plan (int32 cuda) a call with M rows writes → {field: tensor}"""
    f, o = {}, 0
    for name, n in PLAN_F:
        f[name] = plan[o:o + n]
        o += n
    assert o == plan.numel(), (o, plan.numel())
    U, NT = int(f["U"]), int(f["NT"])
    for k in ("uexp", "ucnt", "uoff"):
        f[k] = f[k][:U]
    for k in ("utok", "uw", "tpair"):
        f[k] = f[k][:M * 10]
    for k in ("tt_u", "tt_t"):
        f[k] = f[k][:NT]
    return f


def _call22(m, h, x, on):
    """forward with P022 on/off → the 8 outputs + plan fields, sig, xf of that call"""
    m.patch_set(P022, on)
    o = list(m.forward(h, x))
    plan, sig, xf = m.p022_state(h, x.shape[0])
    return o + [_plan(plan, x.shape[0]), sig, xf]


def _onoff22(m, h, x, bad, tag, stat=None):
    """P022 off, on, on again: the 8 outputs, every plan field, sig and xf bitwise (NaN-safe)"""
    was = m.patch_on(P022)
    try:
        ref = _call22(m, h, x, False)
        got = _call22(m, h, x, True)
        again = m.forward(h, x)
    finally:
        m.patch_set(P022, was)
    for k in range(8):
        if not (_same(ref[k], got[k]) and _same(got[k], again[k])):
            bad.append(f"{tag} {OUTS[k]}")
            return got
    for k, v in ref[8].items():
        if not torch.equal(v, got[8][k]):
            bad.append(f"{tag} plan.{k}")
            return got
    for k, name in ((9, "sig"), (10, "xf")):
        if not _same(ref[k], got[k]):
            bad.append(f"{tag} {name}")
    if stat is not None:
        stat.append(int(got[8]["U"]))
    return got


def _rows22(pool, M, k, seed):
    """test rows for one layer: k = 0 golden window of the layer's pool, 1 random (per-feature std of the pool),
    2 the same golden row M times (every row picks the same experts), 3 random × 8 (peaky routing) with zero rows"""
    n = pool.shape[0]
    if k == 0:
        o = (17 * seed + 7 * M) % (n - M + 1)
        return pool[o:o + M].contiguous()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000 * seed + M)
    if k == 2:
        return pool[(13 * seed + M) % n].expand(M, -1).contiguous()
    x = pool.float().std(0, keepdim=True) * torch.randn(M, 2560, device="cuda", generator=gen)
    if k == 3:
        x = x * 8
        x[::3] = 0
    return x.bfloat16().contiguous()


def test_p022_bitwise(ctx):
    """P022 on == off, bitwise: the 8 outputs (out, ids, weights, logits, routed, shared, h_s, experts) + rerun on, every
    plan field (U, uexp, ucnt, uoff, utok, uw, tpair, NT, tt_u, tt_t), sig, xf; all 48 layers + the MTP layer of this
    rank (default P014/P017), M = 1, 2, 4, 5, 8, 16, 20, 33, 64 × rows: golden window (MTP: layer 47's), random, one
    golden row M times, random × 8 with zero rows."""
    m = _st(ctx)
    r = ctx.rank
    pools = dict(zip(range(48), _pool(ctx, r, list(range(48)))))
    bad, n, us = [], 0, []
    t0 = time.time()
    for li in list(range(48)) + [None]:
        h = m.upload(r, li) if li is not None else m.upload_mtp(r)
        try:
            pool = pools[li if li is not None else 47]
            for M in P14_MS:
                for k in range(4):
                    _onoff22(m, h, _rows22(pool, M, k, li if li is not None else 48), bad, f"l{li} M{M}.{k}", us)
                    n += 1
        finally:
            m.free(h)
    ctx.detail(f"p022 bitwise: {n} calls, experts per call {min(us)}…{max(us)}, {time.time() - t0:.0f}s" + "".join("\n  " + b for b in bad[:20]))
    return {"ok": not bad, "msg": (f"on == off bitwise ({n} calls × 8 outputs + plan + sig + xf, rerun; 48 layers + mtp × "
                                   f"M {','.join(map(str, P14_MS))} × 4 row kinds; experts {min(us)}…{max(us)}) · "
                                   f"{time.time() - t0:.0f}s" if not bad else "differs: " + " ".join(bad[:8]))}


def test_p022_ties(ctx):
    """P022 on == off with forced ties / degenerate routing (layer 0's router replaced through p017_set): (a) experts in
    groups of 4 with identical rows (every logit tied with 3 others), (b) 64 copies of one row at ids 5 mod 8, (c) all
    zero (all logits 0: top-10 = ids 0…9), (d) the router × 64 (peaky: most p underflow to 0, ties at 0), (e) P017's
    forced ±inf / NaN rows; rows: golden, random, one row M times, zero rows; M = 1 … 64; all checks of _onoff22."""
    m = _st(ctx)
    h = _upload17(ctx, ctx.rank, 0)
    bad, n, us, lines = [], 0, [], []
    try:
        w0 = _router_bits(ctx, f"{CK}.0.mlp.gate.weight")
        f0 = w0.view(torch.bfloat16).float()
        gen = torch.Generator()
        gen.manual_seed(22)
        g4 = f0.clone()
        for e in range(0, 512, 4):
            g4[e + 1:e + 4] = g4[e]
        c64 = f0.clone()
        c64[5::8] = f0[123]
        bits = lambda f: f.bfloat16().view(torch.int16).contiguous()  # noqa: E731
        cases = {"groups of 4": bits(g4), "64 copies": bits(c64), "zero": bits(torch.zeros_like(f0)), "x64": bits(f0 * 64),
                 "inf/nan": _force17(gen, w0, 1).contiguous()}
        pool = _pool(ctx, ctx.rank, [0])[0]
        for name, wb in cases.items():
            m.p017_set(h, wb)
            u0 = len(us)
            for M in P14_MS:
                for k in range(4):
                    _onoff22(m, h, _rows22(pool, M, k, 5) if k != 3 else torch.zeros(M, 2560, dtype=torch.bfloat16, device="cuda"),
                             bad, f"{name} M{M}.{k}", us)
                    n += 1
            lines.append(f"{name}: experts per call {min(us[u0:])}…{max(us[u0:])}")
    finally:
        m.free(h)
    ctx.detail("\n".join(["== moe.p022_ties"] + lines + bad[:20]))
    return {"ok": not bad, "msg": (f"5 forced routers × M {','.join(map(str, P14_MS))} × 4 row kinds: on == off bitwise ({n} calls, "
                                   "all outputs + plan + sig + xf)" if not bad else "differs: " + " ".join(bad[:8]))}


def test_p022_invariance(ctx):
    """P022 on: rows of an M=64 call (golden rows) == the same rows alone (M=1, every row) and in M=4/5/20/33 slices, all
    row outputs bitwise; rerun bitwise; layers 0/23/47."""
    m = _st(ctx)
    bad, n = [], 0
    was = m.patch_on(P022)
    try:
        m.patch_set(P022, True)
        for li in (0, 23, 47):
            h = _h(ctx, ctx.rank, li)
            x = _p14_rows(ctx, ctx.rank, li, 64, 0)
            full, again = m.forward(h, x), m.forward(h, x)
            if not all(_same(a, b) for a, b in zip(full, again)):
                bad.append(f"l{li} rerun")
            for i in range(64):
                one = m.forward(h, x[i:i + 1].contiguous())
                n += 1
                if not all(_same(one[j], full[j][i:i + 1]) for j in range(7)):
                    bad.append(f"l{li} row{i}")
            for a, b in ((0, 4), (3, 8), (11, 31), (31, 64), (60, 64)):
                part = m.forward(h, x[a:b].contiguous())
                n += 1
                if not all(_same(part[j], full[j][a:b]) for j in range(7)):
                    bad.append(f"l{li} rows {a}:{b}")
    finally:
        m.patch_set(P022, was)
    return {"ok": not bad, "msg": (f"on: M=1 / slices == M=64 rows bitwise ({n} calls, 7 row outputs, layers 0/23/47), rerun"
                                   if not bad else "differs: " + " ".join(bad[:8]))}


def test_p022_perf(ctx):
    """P022: µs per layer call off vs on (default P014/P017; CUDA graph of layers 0/3/5/23 as test_perf, so weights do not
    stay in L2; the same golden inputs → the same experts), best of 7 alternating rounds × 5 replays. M=5: one sequence's
    5 verify rows; M=20: 4 sequences × 5 consecutive rows."""
    m = _st(ctx)
    r = ctx.rank
    dec, pre = _steps(ctx)
    lay = LAYERS[:4]
    hs = [_h(ctx, r, li) for li in lay]

    def rows(step_list, li):
        return torch.cat([_g(ctx, r).get(s, f"{L}.{li}.mlp#0.in0") for s in step_list]).cuda().contiguous()

    cases = {
        1: lambda li: rows(dec[:1], li),
        4: lambda li: rows(dec[:4], li),
        5: lambda li: rows(dec[:5], li),
        16: lambda li: rows([s for i in (0, 16, 32, 48) for s in dec[i:i + 4]], li),
        20: lambda li: rows([s for i in (0, 16, 32, 48) for s in dec[i:i + 5]], li),
        64: lambda li: rows(dec[:64], li),
    }
    was = m.patch_on(P022)
    lines, short = [], []
    try:
        for M, mk in cases.items():
            vs = [mk(li) for li in lay]
            us = [int(m.forward(h, v)[7].item()) for h, v in zip(hs, vs)]
            best = {}
            for _ in range(7):
                for on in (False, True):
                    m.patch_set(P022, on)
                    t = m.bench(hs, vs, 20, 5) / len(hs)
                    best[on] = min(best.get(on, 1e30), t)
            f, o = best[False], best[True]
            lines.append(f"M={M}: off {f:.1f} µs → on {o:.1f} µs · {o - f:+.1f} µs ({(o / f - 1) * 100:+.1f}%) · experts "
                         f"{sum(us) / len(us):.1f}")
            short.append(f"M={M} {f:.1f}→{o:.1f} ({o - f:+.1f})")
    finally:
        m.patch_set(P022, was)
    ctx.detail("\n".join(["== moe.p022_perf (off → on, µs per layer call)"] + lines))
    return {"ok": True, "msg": "µs " + " · ".join(short)}


def test_zz_free(ctx):
    m = _st(ctx)
    for h in _S["h"].values():
        m.free(h)
    _S["h"].clear()
    return {"ok": True, "msg": "freed"}
