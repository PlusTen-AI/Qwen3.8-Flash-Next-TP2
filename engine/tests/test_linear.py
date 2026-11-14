"""linear family (src/kernels/linear.cu): exact dequant, fp32 accuracy, batch invariance, golden projections, speed.

Golden: decode steps (the reference's W8A16 for M ≲ 8) are the exactness target; prefill rows (activation-quantized
MXFP8 in the reference) are reported separately. RowParallel outputs (out_proj, o_proj) are checked as bf16(p0 + p1)
of both ranks' partials.
"""
import math
import os

import torch
from safetensors import safe_open

P = "model.language_model.layers"
G = "language_model.model.layers"
MXFP8, BF16 = 1, 0
GEMM_TUNE = 4  # patches.def P004
PACK12 = 13  # patches.def P013
_L = None


def _lin(ctx):
    """Q38_TEST_PACK_ALL=1: every bf16 Lin the tests make or upload also gets the P013 12-bit copy, so with the patch on
    (default; Q38_PATCHES=-lin_pack12 for off) the whole suite runs its bf16 decode calls through the packed kernel."""
    global _L
    if _L is None:
        _L = ctx.ext("linear", ["src/kernels/linear/linear.cu", "src/core/weights.cc", "src/model/patches.cc"])
        _L.open(ctx.model_dir, ctx.table)
        _L.pack_all(os.environ.get("Q38_TEST_PACK_ALL") == "1")
    return _L


def _deq(w_u8, s_u8):
    """Host reference: e4m3 [n,k] · 2^(s-127) per 32-block → fp32 (exact)."""
    w = w_u8.view(torch.float8_e4m3fn).float()
    sc = torch.pow(2.0, s_u8.float() - 127).repeat_interleave(32, dim=1)
    return w * sc


def _rand_fp8(n, k, gen):
    w = (torch.randn(n, k, generator=gen) * 64).clamp(-440, 440).to(torch.float8_e4m3fn).view(torch.uint8)
    s = torch.randint(112, 124, (n, k // 32), generator=gen, dtype=torch.uint8)
    return w, s


def _rand_bf16(n, k, gen):
    return (torch.randn(n, k, generator=gen) * 0.02).bfloat16().view(torch.uint8).reshape(n, 2 * k)


# ---- synthetic --------------------------------------------------------------------------------------------------
def test_dequant_exact(ctx):
    """One-hot x picks single weights: y must equal the host dequant bit for bit (all 256 codes × scales)."""
    L = _lin(ctx)
    n, k = 256, 256
    codes = (torch.arange(n).view(-1, 1) * 37 + torch.arange(k).view(1, -1) * 11) % 256
    codes[(codes == 0x7F) | (codes == 0xFF)] = 0  # NaN codes
    w = codes.to(torch.uint8)
    s = ((torch.arange(n).view(-1, 1) * 3 + torch.arange(k // 32).view(1, -1) * 5) % 48 + 96).to(torch.uint8)
    ref = _deq(w, s)
    wb = (torch.randn(n, k) * 3).bfloat16()
    hs = [(L.make(MXFP8, w, s, n, k, 0), ref), (L.make(BF16, wb.view(torch.uint8).reshape(n, 2 * k), torch.empty(0, dtype=torch.uint8), n, k, 0), wb.float())]
    bad = []
    for h, r in hs:
        for off in range(0, k, 64):
            x = torch.zeros(64, k, dtype=torch.bfloat16)
            x[torch.arange(64), torch.arange(64) + off] = 1
            y = L.forward(h, x.cuda(), True).cpu()
            if not torch.equal(y, r[:, off:off + 64].t()):
                bad.append(f"fmt{L.info(h)[0]}@{off}: {(y != r[:, off:off + 64].t()).sum().item()} diff")
        L.free(h)
    return {"ok": not bad, "msg": "e4m3 254 codes (NaN excluded) × scales 2^-31..2^16 · bf16 — bitwise" if not bad
            else " · ".join(bad)}


def _e2m1(v):
    """round-to-nearest-even to e2m1 values (saturating at ±6), as lin_make_nvfp4"""
    a = v.abs()
    # thresholds: ≤0.25→0, <0.75→0.5, ≤1.25→1, <1.75→1.5, ≤2.5→2, <3.5→3, ≤5→4, else 6
    m = torch.where(a <= 5.0, torch.tensor(4.0), torch.tensor(6.0))
    m = torch.where(a < 3.5, torch.tensor(3.0), m)
    m = torch.where(a <= 2.5, torch.tensor(2.0), m)
    m = torch.where(a < 1.75, torch.tensor(1.5), m)
    m = torch.where(a <= 1.25, torch.tensor(1.0), m)
    m = torch.where(a < 0.75, torch.tensor(0.5), m)
    m = torch.where(a <= 0.25, torch.tensor(0.0), m)
    return torch.where(v < 0, -m, m)


def _nvfp4_ref(w):
    """Host reference of lin_make_nvfp4's quantization → dequantized fp64 weights (code · s · g)."""
    w32 = w.float()
    g = (w32.abs().max() / torch.tensor(448.0 * 6.0)).float() if w32.abs().max() > 0 else torch.tensor(1.0)
    n, k = w.shape
    blk = w32.view(n, k // 16, 16)
    amax = blk.abs().amax(-1)
    s = (amax / (torch.tensor(6.0) * g)).to(torch.float8_e4m3fn).float()
    safe = torch.where(s == 0, torch.ones_like(s), s)
    q = _e2m1(blk / (safe * g).unsqueeze(-1))
    q = torch.where((s == 0).unsqueeze(-1), torch.zeros_like(q), q)
    return (q.double() * s.double().unsqueeze(-1) * g.double()).view(n, k)


def test_nvfp4_draft(ctx):
    """NVFP4 draft copy (lin_make_nvfp4): kernel vs the host quantization reference in float64 (fp32-sum accuracy);
    padding rows, zero rows and zero 16-groups, M = 1 … 64; also its error against the bf16 weights (sanity)."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(11)
    n, k = 1320, 2560  # 1320 = 20 CTA tiles + 40 rows: padding
    w = torch.randn(n, k, generator=gen) * torch.exp(torch.randn(n, 1, generator=gen)) * 0.02
    w[7] = 0
    w[100, 32:48] = 0
    w = w.bfloat16()
    ref = _nvfp4_ref(w)
    h = L.make_nvfp4(w, n, k)
    worst, rel_bf16 = 0.0, 0.0
    for M in (1, 5, 17, 64):
        x = (torch.randn(M, k, generator=gen)).bfloat16()
        y = L.forward(h, x.cuda(), True).cpu().double()
        yr = x.double() @ ref.t()
        den = (x.double().abs() @ ref.abs().t()).clamp_min(1e-30)
        worst = max(worst, ((y - yr).abs() / den).max().item())
        yb = x.double() @ w.double().t()
        rel_bf16 = max(rel_bf16, ((y - yb).norm() / yb.norm()).item())
    L.free(h)
    ok = worst < 1e-5 and rel_bf16 < 0.15
    return {"ok": ok, "msg": f"max vs reference {worst:.1e} (relative to Σ|x·w|) · rel vs bf16 {rel_bf16:.3f} · "
            f"M 1…64 · padding·zero groups"}


def test_fp32_accuracy(ctx):
    """Random shapes (padding, splits, both formats) vs float64 reference; error relative to Σ|x·w| per output."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(1)
    worst, lines = 0.0, []
    for fmt, n, k, split in [(MXFP8, 48, 2560, 0), (MXFP8, 200, 3072, 4), (MXFP8, 512, 2560, 1), (BF16, 520, 2560, 0),
                             (BF16, 136, 2560, 5)]:
        if fmt == MXFP8:
            w, s = _rand_fp8(n, k, gen)
            wf = _deq(w, s)
        else:
            w, s = _rand_bf16(n, k, gen), torch.empty(0, dtype=torch.uint8)
            wf = w.view(torch.bfloat16).float()
        h = L.make(fmt, w, s, n, k, split)
        x = torch.randn(64, k, generator=gen).bfloat16()
        y = L.forward(h, x.cuda(), True).cpu().double()
        ref = x.double() @ wf.double().t()
        mag = x.double().abs() @ wf.double().abs().t()
        e = ((y - ref).abs() / mag.clamp_min(1e-30)).max().item()
        yb = L.forward(h, x.cuda(), False).cpu()
        eb = (yb.float() - y.float().bfloat16().float()).abs().max().item()  # bf16 out = round(fp32 out)
        worst = max(worst, e if eb == 0 else 1.0)
        lines.append(f"fmt{fmt} {n}x{k} split{L.info(h)[4]}: rel/Σ|xw| {e:.2e} · bf16 out vs round(f32) {eb:g}")
        L.free(h)
    ctx.detail("\n".join(["== linear.fp32_accuracy"] + lines))
    return {"ok": worst < 1e-5, "msg": f"5 random shapes · max err/Σ|xw| {worst:.1e} · bf16 out = round(fp32)"}


def test_batch_invariance(ctx):
    """A row's result is bitwise the same for every M, row position and run (both formats, split 1 and > 1, f32 out)."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(2)
    cases = []
    w, s = _rand_fp8(2560, 3072, gen)
    cases.append(("fp8 out_proj split4", L.make(MXFP8, w, s, 2560, 3072, 4), False, 3072))
    w, s = _rand_fp8(8192, 2560, gen)
    cases.append(("fp8 qkvz (M>16: 3 CTAs/SM config)", L.make(MXFP8, w, s, 8192, 2560, 0), False, 2560))
    w, s = _rand_fp8(48, 2560, gen)
    cases.append(("fp8 ba", L.make(MXFP8, w, s, 48, 2560, 0), False, 2560))
    wb = _rand_bf16(512, 2560, gen)
    cases.append(("bf16 router split", L.make(BF16, wb, torch.empty(0, dtype=torch.uint8), 512, 2560, 0), False, 2560))
    wb = _rand_bf16(1280, 2560, gen)
    cases.append(("bf16 f32", L.make(BF16, wb, torch.empty(0, dtype=torch.uint8), 1280, 2560, 0), True, 2560))
    bad, n_cmp = [], 0
    for name, h, f32, k in cases:
        x = torch.randn(64, k, generator=gen).bfloat16().cuda()
        y64 = L.forward(h, x, f32)
        if not torch.equal(y64, L.forward(h, x, f32)):
            bad.append(f"{name}: run-to-run")
        for M in (1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 24, 31, 33, 40, 48, 57, 63):
            n_cmp += 1
            if not torch.equal(L.forward(h, x[:M].contiguous(), f32), y64[:M]):
                bad.append(f"{name}: M={M}")
        for i in range(64):  # each row alone, and at the end of a tail slice
            n_cmp += 2
            if not torch.equal(L.forward(h, x[i:i + 1].contiguous(), f32), y64[i:i + 1]):
                bad.append(f"{name}: row {i} M=1")
            if not torch.equal(L.forward(h, x[64 - i - 1:].contiguous(), f32), y64[64 - i - 1:]):
                bad.append(f"{name}: tail {i + 1}")
        perm = torch.randperm(64, generator=gen).cuda()
        n_cmp += 1
        if not torch.equal(L.forward(h, x[perm].contiguous(), f32), y64[perm]):
            bad.append(f"{name}: permuted rows")
        ctx.detail(f"batch_invariance {name}: info {L.info(h)}")
        L.free(h)
    return {"ok": not bad, "msg": f"5 cases × {n_cmp // 5} comparisons bitwise (M 1–64·row position·permutation·repeat)"
            if not bad else " · ".join(bad[:4])}


# ---- golden -----------------------------------------------------------------------------------------------------
def _get(ctx, rank, step, key):
    d = ctx.dump if rank == ctx.rank else os.path.join(os.path.dirname(ctx.dump), f"rank{rank}")
    with safe_open(os.path.join(d, f"step{step:04d}.safetensors"), "pt") as f:
        return f.get_tensor(key)


def _prefill_steps(ctx):
    return [s["step"] for s in ctx.golden.steps if s["num_tokens"] > 1]


def _stats(pairs):
    """pairs [(got, ref)] → max abs, max rel (vs max|ref|), cosine (float64), exact fraction, max bf16 ulp distance."""
    g = torch.cat([a.double().flatten() for a, _ in pairs])
    r = torch.cat([b.double().flatten() for _, b in pairs])
    d = (g - r).abs()
    ulp = torch.ldexp(torch.ones_like(r), torch.frexp(r.float())[1].double() - 8)  # bf16 ulp of |r| (7+1 bits)
    return {"max_abs": d.max().item(), "max_rel": (d.max() / r.abs().max().clamp_min(1e-30)).item(),
            "cos": (g @ r / (g.norm() * r.norm()).clamp_min(1e-300)).item(), "exact": (d == 0).double().mean().item(),
            "ulp": (d / ulp).max().item()}


def _fmt(c):
    return (f"abs {c['max_abs']:.2e} rel {c['max_rel']:.1e} cos {c['cos']:.8f} exact {100 * c['exact']:.2f}% "
            f"max {c['ulp']:.1f}ulp")


def _golden_cols(ctx, tag, layers, names, gkey, row_parallel=False, f32=False):
    """Teacher forcing: golden in0 → our output vs golden out.0, decode steps and prefill rows separately."""
    L = _lin(ctx)
    dec_steps, pre_steps = ctx.golden.decode_steps(), _prefill_steps(ctx)
    dec, pre, lines = [], [], []
    for li in layers:
        hs = [L.upload(r, [n.format(i=li) for n in names]) for r in ((0, 1) if row_parallel else (ctx.rank,))]
        for step in dec_steps + pre_steps:
            k = gkey.format(i=li)
            ref = _get(ctx, ctx.rank, step, k + ".out.0")
            if row_parallel:
                parts = [L.forward(h, _get(ctx, r, step, k + ".in0").cuda(), False).float() for r, h in zip((0, 1), hs)]
                got = (parts[0] + parts[1]).bfloat16().cpu()
            else:
                got = L.forward(hs[0], _get(ctx, ctx.rank, step, k + ".in0").cuda(), f32).cpu()
            ref = ref[-got.shape[0]:]  # prefill: kept rows are the last ones
            (dec if step in dec_steps else pre).append((got, ref))
        for h in hs:
            L.free(h)
        c = _stats(dec[-len(dec_steps):])
        lines.append(f"{tag} layer {li} decode×{len(dec_steps)}: {_fmt(c)}")
    cd, cp = _stats(dec), _stats(pre)
    if row_parallel:  # the dumped output is the all-reduced sum when both ranks hold the same tensor
        k = gkey.format(i=layers[0]) + ".out.0"
        same = torch.equal(_get(ctx, 0, dec_steps[0], k), _get(ctx, 1, dec_steps[0], k))
        lines.append(f"{tag}: golden out rank0 == rank1 (post all-reduce): {same}")
    lines.append(f"{tag} all decode: {_fmt(cd)} · prefill rows ({len(pre)} steps×layers): {_fmt(cp)}")
    ctx.detail("\n".join([f"== linear.{tag}"] + lines))
    ok = cd["cos"] > 0.99999 and cd["max_rel"] < 1e-2
    return {"ok": ok, "msg": f"decode {_fmt(cd)} · prefill rows rel {cp['max_rel']:.1e} cos {cp['cos']:.6f}"}


def test_gdn_qkvz(ctx):
    return _golden_cols(ctx, "gdn_qkvz", (0, 1, 22, 46), [P + ".{i}.linear_attn.in_proj_qkv.weight",
                                                         P + ".{i}.linear_attn.in_proj_z.weight"],
                        G + ".{i}.linear_attn.in_proj_qkvz#0")


def test_gdn_ba(ctx):
    return _golden_cols(ctx, "gdn_ba", (0, 1, 22, 46), [P + ".{i}.linear_attn.in_proj_b.weight",
                                                       P + ".{i}.linear_attn.in_proj_a.weight"],
                        G + ".{i}.linear_attn.in_proj_ba#0")


def test_gdn_out(ctx):
    return _golden_cols(ctx, "gdn_out", (0, 22, 46), [P + ".{i}.linear_attn.out_proj.weight"],
                        G + ".{i}.linear_attn.out_proj#0", row_parallel=True)


def test_qsa_qkv(ctx):
    a = P + ".{i}.self_attn."
    return _golden_cols(ctx, "qsa_qkv", (3, 27, 47), [a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight",
                                                      a + "indexer.index_qk_proj.weight"], G + ".{i}.self_attn.qkv_proj#0")


def test_qsa_o(ctx):
    return _golden_cols(ctx, "qsa_o", (3, 27, 47), [P + ".{i}.self_attn.o_proj.weight"], G + ".{i}.self_attn.o_proj#0",
                        row_parallel=True)


def test_router(ctx):
    return _golden_cols(ctx, "router", (0, 3, 47), [P + ".{i}.mlp.gate.weight"], G + ".{i}.mlp.gate#0")


def test_ple_kv(ctx):
    return _golden_cols(ctx, "ple_kv", (1,), [P + ".{i}.ple.key_proj.weight", P + ".{i}.ple.value_proj.weight"],
                        G + ".{i}.ple.kv_proj#0")


def test_lm_head(ctx):
    """This rank's vocab half, fp32 out; compared to the bf16 golden logits (after rounding) and as floats."""
    L = _lin(ctx)
    h = L.upload(ctx.rank, ["lm_head.weight"])
    n = L.info(h)[1]
    dec, pre, f32 = [], [], []
    dsteps = ctx.golden.decode_steps()
    for step in dsteps + _prefill_steps(ctx):
        x = _get(ctx, ctx.rank, step, "language_model.logits_processor#0.in1")
        ref = _get(ctx, ctx.rank, step, "language_model.logits_processor#0.out")[:, ctx.rank * n:(ctx.rank + 1) * n]
        y = L.forward(h, x.cuda(), True).cpu()
        ref = ref[-y.shape[0]:]
        (dec if step in dsteps else pre).append((y.bfloat16(), ref))
        if step in dsteps:
            f32.append((y, ref))
    L.free(h)
    cd, cf = _stats(dec), _stats(f32)
    cp = _stats(pre) if pre else None
    ctx.detail(f"== linear.lm_head\ndecode bf16(round): {_fmt(cd)}\ndecode fp32 vs bf16 golden: "
               f"{_fmt(cf)}\nprefill rows: {_fmt(cp) if cp else '-'}")
    ok = cd["cos"] > 0.99999 and cd["max_rel"] < 1e-2
    return {"ok": ok, "msg": f"decode×{len(dsteps)} {_fmt(cd)}"
            + (f" · prefill rows cos {cp['cos']:.6f}" if cp else "")}


# ---- speed ------------------------------------------------------------------------------------------------------
SHAPES = [  # name, fmt, n, k, f32 out
    ("gdn_qkvz", MXFP8, 8192, 2560, False), ("gdn_ba", MXFP8, 48, 2560, False), ("gdn_out", MXFP8, 2560, 3072, False),
    ("qsa_qkv", MXFP8, 7296, 2560, False), ("qsa_o", MXFP8, 2560, 3072, False), ("router", BF16, 512, 2560, False),
    ("ple_kv", BF16, 12800, 2560, False), ("lm_head", BF16, 124160, 2560, True),
]
BW_FLOOR = 241.0  # GB/s measured read (tools/bw_probe.py)


def test_perf(ctx):
    """µs and GB/s (useful weight bytes) per shape and M; weights cycle over ≥ 96 MB of copies (beats the 24 MB L2)."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(3)
    lines, low = [], []
    for name, fmt, n, k, f32 in SHAPES:
        nbytes = n * k * (1 + 1 / 32) if fmt == MXFP8 else 2 * n * k
        copies = max(1, math.ceil(96e6 / nbytes))
        if fmt == MXFP8:
            w, s = _rand_fp8(n, k, gen)
        else:
            w, s = _rand_bf16(n, k, gen), torch.empty(0, dtype=torch.uint8)
        hs = [L.make(fmt, w, s, n, k, 0) for _ in range(copies)]
        reps = max(1, math.ceil(200 / copies))
        row = []
        for M in (1, 4, 16, 64):
            x = torch.randn(M, k, generator=gen).bfloat16().cuda()
            us = L.bench(hs, x, f32, reps, 5)
            gbs = nbytes / us / 1e3
            row.append(f"M{M} {us:.1f}µs {gbs:.0f}GB/s")
            if M == 1 and nbytes > 5e6 and gbs < 0.85 * BW_FLOOR:
                low.append(f"{name} {gbs:.0f}")
        lines.append(f"{name:9s} {n}x{k} split{L.info(hs[0])[4]} ×{copies}: " + " · ".join(row))
        for h in hs:
            L.free(h)
        torch.cuda.empty_cache()
    ctx.detail("\n".join(["== linear.perf (best of 5 graph replays; GB/s = weight bytes / time)"] + lines))
    return {"ok": not low, "msg": ("M=1 large shapes ≥85% of 241 GB/s" if not low else "below 85%: " + " ".join(low))
            + f" · table {ctx.log}"}


# ---- prefill (M > 64: tiled GEMM in linear_prefill.cuh) -----------------------------------------------------------
PREFILL_M = (512, 2048, 8192)


def _make(L, fmt, n, k, gen, split=0):
    if fmt == MXFP8:
        w, s = _rand_fp8(n, k, gen)
    else:
        w, s = _rand_bf16(n, k, gen), torch.empty(0, dtype=torch.uint8)
    return L.make(fmt, w, s, n, k, split)


def _decode_rows(L, h, x, f32):
    """Reference: the decode kernel (M ≤ 64) over 64-row slices."""
    return torch.cat([L.forward(h, x[i:i + 64].contiguous(), f32) for i in range(0, x.shape[0], 64)])


def test_prefill_bitwise(ctx):
    """Every production shape (+ padding/odd-N cases): rows of M = 65…8192 calls == the same rows of the decode kernel
    (64-row calls) == M=1 calls, bf16 and fp32 out; random activations with rows of different scale."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(11)
    shapes = [(nm, f, n, k, 0) for nm, f, n, k, _ in SHAPES if nm != "lm_head"] + [
        ("odd fp8", MXFP8, 200, 1024, 0), ("odd bf16", BF16, 136, 640, 0), ("split8 bf16", BF16, 136, 2560, 8)]
    bad, lines, n_cmp = [], [], 0
    for name, fmt, n, k, split in shapes:
        h = _make(L, fmt, n, k, gen, split)
        x = (torch.randn(8192, k, generator=gen) * torch.exp(torch.randn(8192, 1, generator=gen) * 2)).bfloat16().cuda()
        ref = _decode_rows(L, h, x, False)
        for M in (65, 127, 200, 2048, 8192):
            n_cmp += 1
            if not torch.equal(L.forward(h, x[:M].contiguous(), False), ref[:M]):
                bad.append(f"{name} M={M}")
        sub = x[4000:4000 + 333].contiguous()  # rows at other positions of a large call
        n_cmp += 1
        if not torch.equal(L.forward(h, sub, False), ref[4000:4333]):
            bad.append(f"{name} offset rows")
        for i in (0, 4097, 8191):  # single rows (M = 1)
            n_cmp += 1
            if not torch.equal(L.forward(h, x[i:i + 1].contiguous(), False), ref[i:i + 1]):
                bad.append(f"{name} row {i} M=1")
        rf = _decode_rows(L, h, x[:2048], True)
        n_cmp += 1
        if not torch.equal(L.forward(h, x[:2048].contiguous(), True), rf):
            bad.append(f"{name} f32 M=2048")
        lines.append(f"{name} {n}x{k} split{L.info(h)[4]}: ok" if not any(b.startswith(name) for b in bad) else f"{name}: bad")
        L.free(h)
        torch.cuda.empty_cache()
    ctx.detail("\n".join(["== linear.prefill_bitwise"] + lines))
    return {"ok": not bad, "msg": f"{len(shapes)} shapes × {n_cmp // len(shapes)} comparisons: M 65–8192 rows = decode "
            f"path (64 rows) = M=1 bitwise" if not bad else "differs: " + " · ".join(bad[:4])}


def _golden_prefill(ctx, tag, layers, names, gkey, row_parallel=False):
    """Golden prefill rows (step 0: 61 rows, step 32: last 64 of 8,603) placed at the end of an M=2048 call whose other
    rows are other golden rows: our rows bitwise == decode path, and vs golden (the reference's prefill =
    activation-quantized)."""
    L = _lin(ctx)
    pre = [s for s in _prefill_steps(ctx)]
    pairs, bad = [], []
    for li in layers:
        hs = [L.upload(r, [n.format(i=li) for n in names]) for r in ((0, 1) if row_parallel else (ctx.rank,))]
        k = gkey.format(i=li)
        for step in pre:
            ref = _get(ctx, ctx.rank, step, k + ".out.0")
            outs = []
            for r, h in zip((0, 1) if row_parallel else (ctx.rank,), hs):
                xg = _get(ctx, r, step, k + ".in0").cuda()
                filler = xg[torch.arange(2048 - xg.shape[0]) % xg.shape[0]].flip(1).contiguous()
                xb = torch.cat([filler, xg]).contiguous()
                yb = L.forward(h, xb, False)[-xg.shape[0]:]
                yd = L.forward(h, xg.contiguous(), False)
                if not torch.equal(yb, yd):
                    bad.append(f"layer {li} step {step}")
                outs.append(yb)
            got = (outs[0].float() + outs[1].float()).bfloat16().cpu() if row_parallel else outs[0].cpu()
            pairs.append((got, ref[-got.shape[0]:]))
        for h in hs:
            L.free(h)
    c = _stats(pairs)
    ctx.detail(f"== linear.prefill_golden {tag}: {_fmt(c)} · big == decode: {not bad}")
    return c, bad


def test_prefill_golden(ctx):
    """Prefill steps 0 and 32 through the large-M path (rows inside an M=2048 call): bitwise == decode path; error vs
    the reference's prefill (activation-quantized MXFP8, so tolerance: cos > 0.998)."""
    P_ = P
    cases = [("gdn_qkvz", (0, 46), [P_ + ".{i}.linear_attn.in_proj_qkv.weight", P_ + ".{i}.linear_attn.in_proj_z.weight"],
              G + ".{i}.linear_attn.in_proj_qkvz#0", False),
             ("gdn_ba", (0, 46), [P_ + ".{i}.linear_attn.in_proj_b.weight", P_ + ".{i}.linear_attn.in_proj_a.weight"],
              G + ".{i}.linear_attn.in_proj_ba#0", False),
             ("gdn_out", (0, 46), [P_ + ".{i}.linear_attn.out_proj.weight"], G + ".{i}.linear_attn.out_proj#0", True),
             ("qsa_qkv", (3, 47), [P_ + ".{i}.self_attn." + n + ".weight" for n in ("q_proj", "k_proj", "v_proj",
                                                                                 "indexer.index_qk_proj")],
              G + ".{i}.self_attn.qkv_proj#0", False),
             ("qsa_o", (3, 47), [P_ + ".{i}.self_attn.o_proj.weight"], G + ".{i}.self_attn.o_proj#0", True),
             ("router", (0, 47), [P_ + ".{i}.mlp.gate.weight"], G + ".{i}.mlp.gate#0", False),
             ("ple_kv", (1,), [P_ + ".{i}.ple.key_proj.weight", P_ + ".{i}.ple.value_proj.weight"], G + ".{i}.ple.kv_proj#0",
              False)]
    msgs, all_bad, worst = [], [], 1.0
    for tag, layers, names, gkey, rp in cases:
        c, bad = _golden_prefill(ctx, tag, layers, names, gkey, rp)
        all_bad += [f"{tag} {b}" for b in bad]
        worst = min(worst, c["cos"])
        msgs.append(f"{tag} cos {c['cos']:.6f}")
    ok = not all_bad and worst > 0.998
    return {"ok": ok, "msg": ("large-M = decode bitwise · golden " if not all_bad
                              else "differs: " + " ".join(all_bad[:3]) + " · ")
            + " · ".join(msgs)}


def test_prefill_perf(ctx):
    """µs and TFLOPS per shape at M = 512, 2048, 8192 (one weight copy, x resident; best of 5 graph replays, GPU shared)."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(12)
    lines, low = [], []
    for name, fmt, n, k, f32 in SHAPES:
        if name == "lm_head":
            continue
        h = _make(L, fmt, n, k, gen)
        row = []
        for M in PREFILL_M:
            x = torch.randn(M, k, generator=gen).bfloat16().cuda()
            us = L.bench([h], x, False, 3, 5)
            tf = 2 * M * n * k / us / 1e6
            row.append(f"M{M} {us:.0f}µs {tf:.1f}TF")
            if n * k >= 2560 * 2560 and M >= 2048 and tf < 0.6 * 94:
                low.append(f"{name}@{M} {tf:.0f}")
        lines.append(f"{name:9s} {n}x{k} split{L.info(h)[4]}: " + " · ".join(row))
        L.free(h)
        torch.cuda.empty_cache()
    ctx.detail("\n".join(["== linear.prefill_perf (TFLOPS = 2·M·N·K / time; target ≥ 56.4 = 60 % of 94)"] + lines))
    return {"ok": not low, "msg": ("large shapes M≥2048 ≥ 60% of 94 TF" if not low else "below: " + " ".join(low))
            + f" · table {ctx.log}"}


# ---- P004 gemm_tune (p004_gemm_tune.cuh: faster large-M tiles, same bits) -------------------------------------------
def _p004_x(gen, k, rows=8192):
    return (torch.randn(rows, k, generator=gen) * torch.exp(torch.randn(rows, 1, generator=gen) * 2)).bfloat16().cuda()


def _p004_cmp(L, h, x, runs, name, bad):
    """base (patch off) vs patch with tile automatic / 128 / 64 rows, for each (M, fp32 out) in runs → comparisons made."""
    n = 0
    for M, f32 in runs:
        xm = x[:M].contiguous()
        L.patch_set(GEMM_TUNE, False)
        ref = L.forward(h, xm, f32)
        L.patch_set(GEMM_TUNE, True)
        for tile in (-1, 0, 1):
            L.p004_tile(tile)
            n += 1
            if not torch.equal(L.forward(h, xm, f32), ref):
                bad.append(f"{name} M={M}{' f32' if f32 else ''} tile{tile}")
        L.p004_tile(-1)
        del ref
    return n


def test_p004_bitwise(ctx):
    """P004 vs base, bit for bit: every engine shape (SHAPES; lm_head fp32 at M = 65, 2048) and two odd split-1 shapes
    (N padding, short K) at M = 65, 2048, 8192, bf16 and fp32 out; then this rank's checkpoint weights of 4 layers
    (real scale distributions) at M = 2048, 8192. P004 tile automatic and each tile forced. K-split shapes (ba, router)
    keep the base kernels under the patch — compared too."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(13)
    shapes = [(nm, f, n, k, 0) for nm, f, n, k, _ in SHAPES] + [("odd fp8 split1", MXFP8, 200, 1024, 1),
                                                                 ("odd bf16 split1", BF16, 136, 640, 1)]
    real = [("ckpt gdn_qkvz L0", [P + ".0.linear_attn.in_proj_qkv.weight", P + ".0.linear_attn.in_proj_z.weight"]),
            ("ckpt gdn_out L46", [P + ".46.linear_attn.out_proj.weight"]),
            ("ckpt qsa_qkv L3", [P + ".3.self_attn." + n + ".weight" for n in ("q_proj", "k_proj", "v_proj",
                                                                              "indexer.index_qk_proj")]),
            ("ckpt qsa_o L47", [P + ".47.self_attn.o_proj.weight"])]
    was = L.patch_on(GEMM_TUNE)
    bad, lines, n_cmp = [], [], 0
    try:
        for name, fmt, n, k, split in shapes + [(nm, None, 0, 0, 0) for nm, _ in real]:
            if fmt is None:
                h = L.upload(ctx.rank, dict(real)[name])
                runs = [(2048, False), (8192, False)]
            else:
                h = _make(L, fmt, n, k, gen, split)
                runs = [(65, True), (2048, True)] if name == "lm_head" else [(M, f) for M in (65, 2048, 8192) for f in (False, True)]
            _, n, k, _, sp = L.info(h)
            nb = len(bad)
            n_cmp += _p004_cmp(L, h, _p004_x(gen, k), runs, name, bad)
            lines.append(f"{name} {n}x{k} split{sp}: {len(runs)} (M, out) × 3 tiles " + ("ok" if len(bad) == nb else "DIFF"))
            L.free(h)
            torch.cuda.empty_cache()
    finally:
        L.p004_tile(-1)
        L.patch_set(GEMM_TUNE, was)
    ctx.detail("\n".join(["== linear.p004_bitwise"] + lines + bad))
    return {"ok": not bad, "msg": f"{len(shapes)} shapes + checkpoint {len(real)} layers · {n_cmp} comparisons (M "
            f"65·2048·8192 · bf16/fp32 · tile auto/128/64): P004 = base bitwise" if not bad
            else "differs: " + " · ".join(bad[:4])}


def test_p004_perf(ctx):
    """Base vs P004: µs and TFLOPS per shape at M = 2048, 8192 (one weight copy, x resident). The GPU is shared: best of
    10 graph replays, 2 alternating rounds, best kept. Big shapes (≥ 2560²) must be faster and ≥ 75 TF (80 % of 94)."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(14)
    was = L.patch_on(GEMM_TUNE)
    lines, low, summ = [], [], []
    try:
        for name, fmt, n, k, f32 in SHAPES:
            if name == "lm_head":
                continue
            h = _make(L, fmt, n, k, gen)
            big = n * k >= 2560 * 2560 and L.info(h)[4] == 1
            row = []
            for M in (2048, 8192):
                x = torch.randn(M, k, generator=gen).bfloat16().cuda()
                best = {False: math.inf, True: math.inf}
                for _ in range(2):
                    for on in (False, True):
                        L.patch_set(GEMM_TUNE, on)
                        best[on] = min(best[on], L.bench([h], x, False, 3, 10))
                tb, tp = (2 * M * n * k / best[o] / 1e6 for o in (False, True))
                lines.append(f"{name:9s} {n:6d}x{k} M{M:5d}: base {best[False]:7.0f} µs {tb:5.1f} TF · P004 "
                             f"{best[True]:7.0f} µs {tp:5.1f} TF · ×{best[False] / best[True]:.2f}")
                row.append(f"{tb:.0f}→{tp:.0f}")
                if big and (tp < 0.8 * 94 or best[True] > best[False]):
                    low.append(f"{name}@{M} {tp:.0f}")
            if big:
                summ.append(f"{name} {'/'.join(row)}")
            L.free(h)
            torch.cuda.empty_cache()
    finally:
        L.patch_set(GEMM_TUNE, was)
    ctx.detail("\n".join(["== linear.p004_perf (TFLOPS = 2·M·N·K / time; best of 2×10 graph replays, GPU shared)"] + lines))
    return {"ok": not low, "msg": ("TF base→P004 M2048/8192: " + " · ".join(summ) if not low
                                   else "below: " + " ".join(low)) + f" · table {ctx.log}"}


# ---- P013 lin_pack12 (p013_pack12.cuh: decode kernel reads a lossless 12-bit copy of bf16 weights) --------------------
PACK_M = (1, 2, 4, 5, 8, 16, 20, 33, 64)


def _bits(y):
    """exact bits of a float32 / bf16 tensor (NaN-safe comparison)"""
    return y.view(torch.int32 if y.dtype == torch.float32 else torch.int16)


def _lm_bf16(ctx):
    b, shape, dt, _ = ctx.w.slice("lm_head.weight")
    assert dt == "BF16" and len(shape) == 2
    return b.view(shape[0], 2 * shape[1]), shape[0], shape[1]


def _lm_x(ctx, rows=64):
    """lm_head inputs from the golden dump: the first `rows` of the decode rows followed by the prefill rows
    (interleaved by step) — with 77 decode steps, all decode rows."""
    dsteps, psteps = ctx.golden.decode_steps(), _prefill_steps(ctx)
    xs = [_get(ctx, ctx.rank, s, "language_model.logits_processor#0.in1") for s in dsteps]
    pre = [_get(ctx, ctx.rank, s, "language_model.logits_processor#0.in1") for s in psteps]
    xs += [p[i:i + 1] for i in range(max(p.shape[0] for p in pre)) for p in pre if i < p.shape[0]]
    x = torch.cat(xs)[:rows]
    assert x.shape[0] == rows, x.shape
    return x.contiguous()


def _onoff(L, h, x, Ms, f32s=(True, False)):
    """patch off (bf16 copy) vs on (12-bit copy) for each M and output type → list of mismatches"""
    bad = []
    for f32 in f32s:
        for M in Ms:
            xm = x[:M].contiguous()
            L.patch_set(PACK12, False)
            ref = L.forward(h, xm, f32)
            L.patch_set(PACK12, True)
            got = L.forward(h, xm, f32)
            if not torch.equal(_bits(got), _bits(ref)):
                bad.append(f"M={M}{' f32' if f32 else ' bf16'} ({(_bits(got) != _bits(ref)).sum().item()} diff)")
    return bad


def test_pack12_roundtrip(ctx):
    """This rank's real lm_head half → 12-bit copy (host check inside lin_make) → unpacked on the device == bf16 bits."""
    L = _lin(ctx)
    w, n, k = _lm_bf16(ctx)
    h = L.make12(w, n, k, 0)
    same = torch.equal(L.unpack12(h).view(torch.int16).cpu(), w.view(torch.int16))
    nx, mx, nb = L.pack_info(h)
    L.free(h)
    torch.cuda.empty_cache()
    msg = (f"lm_head {n}×{k}: exceptions {nx} ({100 * nx / (n * k):.4f}%) · max per tile {mx} · {nb / 1e6:.1f} MB "
           f"(bf16 {2 * n * k / 1e6:.1f} MB, {100 * nb / (2 * n * k):.1f}%)")
    ctx.detail(f"== linear.pack12_roundtrip\n{msg} · unpack = bf16 bits: {same}")
    return {"ok": same, "msg": msg + (" · unpack = bf16 bitwise" if same else " · unpack differs")}


def test_pack12_lm_head(ctx):
    """fp32 logits (and bf16 out) with the patch on == off, bit for bit, on this rank's lm_head with golden lm_head
    inputs (_lm_x), M = 1 … 64; the packed path's rows also equal its own M=1 calls."""
    L = _lin(ctx)
    w, n, k = _lm_bf16(ctx)
    h = L.make12(w, n, k, 0)
    x = _lm_x(ctx).cuda()
    was = L.patch_on(PACK12)
    try:
        bad = _onoff(L, h, x, PACK_M)
        L.patch_set(PACK12, True)
        y64 = L.forward(h, x, True)
        for i in (0, 7, 31, 63):
            if not torch.equal(_bits(L.forward(h, x[i:i + 1].contiguous(), True)), _bits(y64[i:i + 1])):
                bad.append(f"on: row {i} M=1 != M=64")
    finally:
        L.patch_set(PACK12, was)
    L.free(h)
    torch.cuda.empty_cache()
    ctx.detail("== linear.pack12_lm_head\n" + ("ok" if not bad else "\n".join(bad)))
    return {"ok": not bad, "msg": f"lm_head golden inputs · M {','.join(map(str, PACK_M))} · fp32/bf16 out: on = off "
            f"bitwise" if not bad else "differs: " + " · ".join(bad[:4])}


def _forced(gen, n, k):
    """bf16 [n, k] with every exception kind: values below the window, ±0, subnormals, ±inf, NaNs, a whole row and a
    whole tile of exceptions (> 32 per tile and per k32 block: window reloads), window edges (even/odd top exponent),
    an all-zero row, a row of tiny values (window base 0: zeros/subnormals in the window)."""
    w = (torch.randn(n, k, generator=gen) * torch.exp(torch.randn(n, 1, generator=gen)) * 0.02).bfloat16()
    u = w.view(torch.int16)

    def put(r, cols, bits):
        u[r, cols] = torch.tensor(bits, dtype=torch.int32).to(torch.int16)

    w[3, ::7] = 1e-30                                   # far below the window
    w[4, 100:164] = torch.randn(64, generator=gen).bfloat16() * 1e-6
    w[5, ::5] = 0.0
    w[5, 1::5] = -0.0                                   # ±0 (below the window when b > 0)
    put(6, [0, 33, 64, 2000 % k], [0x0001, 0x8003, 0x007F, 0x8040])  # subnormals
    put(9, [10, 11, 12, 13, 40], [0x7F80, 0xFF80, 0x7FC0, 0xFFA5, 0x7F81])  # ±inf, NaNs: window at the top → row 9 all exceptions
    w[20 % n] = 0.0                                      # all zero: base 0, no exception
    w[21 % n] = (torch.randn(k, generator=gen) * 1e-37).bfloat16()  # tiny row: base 0, subnormals/zeros inside the window
    w[21 % n, ::9] = 0.0
    put(21 % n, [5, 6], [0x0005, 0x8001])
    if n >= 48:
        w[32:48, 7 * 32 + 3] = 1e30                      # tile 2: every other value an exception (~16·k entries)
    for r, top in ((50 % n, 127), (51 % n, 128)):        # window edges: top exponent odd / even
        b = (top >> 1) - 7
        e_lo = 2 * b
        w[r] = (torch.randn(k, generator=gen) * 1e-3).bfloat16()  # the rest below 2^top
        cols = torch.arange(0, k, 4)
        exps = [top, e_lo, e_lo - 1, min(e_lo + 15, top), max(e_lo - 2, 1)]
        bits = [(((i * 37) & 1) << 15) | (exps[i % len(exps)] << 7) | ((i * 29) & 0x7F) for i in range(len(cols))]
        u[r, cols] = torch.tensor(bits, dtype=torch.int32).to(torch.int16)
        u[r, 1] = top << 7                               # the row's largest exponent
    return w


def test_pack12_synthetic(ctx):
    """Forced exceptions (_forced) on N-padding and K-split shapes (split > 1: lists start inside a tile's entries):
    device unpack == bf16 bits; patch on == off bit for bit (NaN-safe) at M = 1 … 64, fp32 and bf16 out."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(21)
    cases = [("padding 1320x2560", 1320, 2560, 0), ("split5 136x2560", 136, 2560, 5), ("split5 136x640", 136, 640, 5),
             ("router 512x2560", 512, 2560, 0), ("split2 64x1024", 64, 1024, 2), ("1 tile 16x2560 (chunk 2)", 16, 2560, 0),
             ("split2 136x640 (chunk 2)", 136, 640, 2), ("100 CTAs 6400x1024 (M>16: 32 KB)", 6400, 1024, 0),
             ("100 CTAs 6400x960 (chunk 2/1)", 6400, 960, 0)]
    bad, lines = [], []
    was = L.patch_on(PACK12)
    try:
        for name, n, k, split in cases:
            w = _forced(gen, n, k)
            h = L.make12(w.view(torch.uint8).reshape(n, 2 * k), n, k, split)
            nx, mx, _ = L.pack_info(h)
            if not torch.equal(L.unpack12(h).view(torch.int16).cpu(), w.view(torch.int16)):
                bad.append(f"{name}: unpack")
            x = (torch.randn(64, k, generator=gen) * torch.exp(torch.randn(64, 1, generator=gen) * 2)).bfloat16().cuda()
            b = _onoff(L, h, x, PACK_M)
            bad += [f"{name}: {e}" for e in b]
            lines.append(f"{name} split{L.info(h)[4]}: exceptions {nx} (max {mx}/tile) · " + ("ok" if not b else "DIFF"))
            L.free(h)
    finally:
        L.patch_set(PACK12, was)
    ctx.detail("\n".join(["== linear.pack12_synthetic"] + lines + bad))
    return {"ok": not bad, "msg": f"{len(cases)} shapes (forced exceptions: below the window·±0·subnormal·±inf·NaN·"
            f"whole tile·window edge) · unpack = bf16 · M 1–64 on = off bitwise" if not bad
            else "differs: " + " · ".join(bad[:4])}


def test_pack12_perf(ctx):
    """µs per call, patch off (bf16 copy) vs on (12-bit copy), M = 1, 4, 16, 20, 64: this rank's real lm_head (fp32 out;
    one 635 MB copy beats the 24 MB L2) and bf16 engine/generic shapes (random weights, copies cycled over ≥ 96 MB).
    GB/s = useful bf16 bytes / time. Best of 2 alternating rounds × 5 graph replays."""
    L = _lin(ctx)
    gen = torch.Generator().manual_seed(22)
    w, n, k = _lm_bf16(ctx)
    shapes = [("lm_head", n, k, True, w), ("router", 512, 2560, False, None), ("ple_kv", 12800, 2560, False, None),
              ("bf16 8192x2560", 8192, 2560, False, None)]
    lines, summ, slow = [], [], []
    was = L.patch_on(PACK12)
    try:
        for name, n_, k_, f32, wt in shapes:
            nbytes = 2 * n_ * k_
            copies = max(1, math.ceil(96e6 / nbytes))
            wt = _rand_bf16(n_, k_, gen) if wt is None else wt
            hs = [L.make12(wt, n_, k_, 0) for _ in range(copies)]
            pb = L.pack_info(hs[0])[2]
            reps = 20 if name == "lm_head" else max(1, math.ceil(200 / copies))
            row = []
            for M in (1, 4, 16, 20, 64):
                x = torch.randn(M, k_, generator=gen).bfloat16().cuda()
                best = {False: math.inf, True: math.inf}
                for _ in range(2):
                    for on in (False, True):
                        L.patch_set(PACK12, on)
                        best[on] = min(best[on], L.bench(hs, x, f32, reps, 5))
                b0, b1 = best[False], best[True]
                lines.append(f"{name:15s} {n_:6d}x{k_} split{L.info(hs[0])[4]} M{M:2d}: off {b0:8.1f} µs "
                             f"{nbytes / b0 / 1e3:4.0f} GB/s · on {b1:8.1f} µs {nbytes / b1 / 1e3:4.0f} GB/s "
                             f"(read {pb / b1 / 1e3:4.0f}) · ×{b0 / b1:.3f}")
                row.append(f"{b0:.0f}→{b1:.0f}")
                if name == "lm_head" and b1 >= b0:
                    slow.append(f"lm_head M{M}")
            summ.append(f"{name} " + "/".join(row))
            for h in hs:
                L.free(h)
            torch.cuda.empty_cache()
    finally:
        L.patch_set(PACK12, was)
    ctx.detail("\n".join(["== linear.pack12_perf (µs; GB/s = bf16 bytes / time; read = 12-bit copy bytes / time)"] + lines))
    return {"ok": not slow, "msg": "µs off→on M1/4/16/20/64: " + summ[0] + f" · table {ctx.log}"}
