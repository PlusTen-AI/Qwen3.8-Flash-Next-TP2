"""hc family (src/kernels/hc.cu): HC GatedResidual modules, final mixer, embedding, PLE — golden dump (teacher forcing),
a torch fp32 reference with the reference deployment's rounding points, batch invariance (bitwise) and timing.

Norm weights: the checkpoint stores hc_norm and the PLE norms in fp32 (not bf16-exact); the reference deployment loads
them into bf16 params (rounded). The engine keeps fp32 (lossless). HC tests report the golden gap for both: `fp32`
(engine) and `bf` (norm rounded to bf16 like the reference, test-only variant) — the difference between the two is that
systematic gap.
"""
import json
import os

import torch
from safetensors import safe_open

from golden import cmp

BF = torch.bfloat16
G = "language_model.model."
WP = "model.language_model."
EOS = 248044
HC_BYTES = (324 + 320) * 10240 * 2  # one decoder HC module: down+inject [324,10240] + up [10240,320] bf16
_m = None
_f = {}
_wc = {}


def _mod(ctx):
    global _m
    if _m is None:
        torch.backends.cuda.matmul.allow_tf32 = False
        _m = ctx.ext("hc", ["src/kernels/hc/hc.cu", "src/core/weights.cc", "src/model/patches.cc"])
        _m.open(ctx.model_dir, ctx.table, ctx.rank)
    return _m


def _file(ctx, k):
    if k not in _f:
        _f.clear()
        f = safe_open(os.path.join(ctx.golden.dir, f"step{k:04d}.safetensors"), "pt", device="cpu")
        _f[k] = (f, set(f.keys()))
    return _f[k]


def _has(ctx, k, key):
    return G + key in _file(ctx, k)[1]


def _g(ctx, k, key):
    return _file(ctx, k)[0].get_tensor(G + key).cuda()


class Acc:
    """Worst-case error per output name over many comparisons (max abs, max rel vs max|ref|, min cosine, exact %)."""

    def __init__(self):
        self.d = {}

    def add(self, name, got, ref):
        r = cmp(name, got, ref)
        r["eq"] = (got.float() == ref.float().to(got.device)).float().mean().item()
        w = self.d.setdefault(name, dict(r, n=0))
        w["n"] += r["n"]
        w["max_abs"], w["max_rel"] = max(w["max_abs"], r["max_abs"]), max(w["max_rel"], r["max_rel"])
        w["cos"], w["eq"] = min(w["cos"], r["cos"]), min(w["eq"], r["eq"])
        return r

    def get(self, name):
        return self.d[name]

    def text(self, names=None):
        return " · ".join(f"{k} rel {v['max_rel']:.1e} cos {v['cos']:.6f} eq {100 * v['eq']:.1f}%"
                          for k, v in self.d.items() if names is None or k in names)


# ---- request layout ---------------------------------------------------------------------------------------------
def _requests(ctx):
    """[(name, [(step, positions of the dumped rows)])] — prefill keeps the last rows_kept rows."""
    names = iter(["short", "long", "short2"])
    out = []
    for s in ctx.golden.steps:
        (r,) = s["reqs"].values()
        if r["new"]:
            out.append((next(names), []))
            pos = list(range(s["num_tokens"] - s["rows_kept"], s["num_tokens"]))
        else:
            pos = [r["computed"]]
        out[-1][1].append((s["step"], pos))
    return out


def _tokens(ctx, name):
    t = ctx.golden.tokens[name]
    return t["prompt_ids"] + t["output_ids"]


def _prev(toks, pos):
    return [[toks[p - 2] if p >= 2 else EOS, toks[p - 1] if p >= 1 else EOS] for p in pos]


# ---- torch reference (fp32 math, the reference deployment's bf16 materialisation points) ---------------------------
def _wt(ctx, name, shape):
    if name not in _wc:
        b, shp, dt, tp = ctx.w.slice(name)
        _wc[name] = b.view(torch.bfloat16).view(*shape).cuda().float()
    return _wc[name]


def _hc_ref(ctx, prefix, inject, norm, X, bo, inj):
    M = X.shape[0]
    wd = _wt(ctx, prefix + ".input_mix_weight_down.weight", (320, 10240))
    if inject:
        wd = torch.cat([wd, _wt(ctx, prefix + ".block_inject_weight.weight", (4, 10240))])
    wu = _wt(ctx, prefix + ".input_mix_weight_up.weight", (10240, 320))
    x = X.float().view(M, 4, 2560)
    if bo is not None:
        w = 2 * torch.sigmoid(inj.float() / 4) if inj is not None else torch.ones(M, 4, device=X.device)
        x = (x + bo.float()[:, None, :] * w[:, :, None]).to(BF).float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    xn = (y + y * norm.view(4, 2560)).to(BF).float()
    z = (xn.view(M, -1) @ wd.T).to(BF).float()
    lora = (z[:, :320] / 4 * torch.sigmoid(z[:, :320] / 4)).to(BF)
    g = (lora.float() @ wu.T).to(BF).float().view(M, 4, 2560)
    u = ((torch.sigmoid(g) * xn).sum(1) / 4).to(BF)
    return x.view(M, -1).to(BF), u, lora, z[:, 320:324].to(BF) if inject else None


def _mix(m, key, X, bo, inj, inject=True):
    if _PAD:  # test_prefill_golden: the same rows at the end of a large call
        return _mix_padded(m, key, X, bo, inj, inject)
    return _mix_raw(m, key, X, bo, inj, inject)


def _mix_raw(m, key, X, bo, inj, inject=True):
    M = X.shape[0]
    u = torch.empty(M, 2560, dtype=BF, device="cuda")
    io = torch.empty(M, 4, dtype=BF, device="cuda") if inject else None
    lora = m.combine_mix(key, X, bo, inj, u, io)
    return u, io, lora


def _attn(i):
    return "self_attn" if i % 4 == 3 else "linear_attn"


# ---- tests ------------------------------------------------------------------------------------------------------
def test_embed(ctx):
    """Vocab-half partial: rows of this rank bit-equal to the (all-reduced) golden, other rows 0."""
    m, bad = _mod(ctx), []
    for k in (0, 1, 32, 33, 48, 79):
        ids = _g(ctx, k, "embed_tokens#0.in0").int()
        ref = _g(ctx, k, "embed_tokens#0.out")
        out = torch.empty_like(ref)
        m.embed_lookup(ids, out)
        own = (ids.long() // 124160) == ctx.rank
        if own.any() and not torch.equal(out[own], ref[own]):
            bad.append(f"step {k} own")
        if (~own).any() and out[~own].abs().max().item() != 0:
            bad.append(f"step {k} other≠0")
    return {"ok": not bad, "msg": "own rows bit-exact, other rows 0" if not bad else " ".join(bad)}


def _layer_chain(ctx, m, k, i, rnd, acc, tag, ref_acc=None):
    """Layer i at step k: X, prev → (attn side) u; golden attention output → (MLP side) v, X, inj_m."""
    ap, mp = f"{WP}layers.{i}.attn_hyper_connection", f"{WP}layers.{i}.mlp_hyper_connection"
    ka, km = m.hc_load(ap, True, rnd), m.hc_load(mp, True, rnd)
    L = f"layers.{i}"
    X = _g(ctx, k, f"{L}#0.kw.hidden_states").contiguous()
    bo = _g(ctx, k, f"{L}#0.kw.prev_block_output") if _has(ctx, k, f"{L}#0.kw.prev_block_output") else None
    inj = _g(ctx, k, f"{L}#0.kw.prev_injection") if _has(ctx, k, f"{L}#0.kw.prev_injection") else None
    prefill = _has(ctx, k, f"{L}.attn_hyper_connection.input_mix_weight_down_block_inject#0.out")
    if i == 1:  # PLE sees the materialised state; then the attention-side mix has no pending combine
        Xc = X.clone()
        m.combine(Xc, bo, inj)
        acc.add(f"{tag}X1c", Xc, _g(ctx, k, "layers.1.ple#0.in0"))
        X, bo, inj = _g(ctx, k, "layers.1.ple#0.out").contiguous(), None, None
    X0 = X.clone()
    u, inj_a, lora = _mix(m, ka, X, bo, inj)
    a = _attn(i)
    acc.add(f"{tag}u", u, _g(ctx, k, f"{L}.{a}#0.kw.hidden_states"))
    acc.add(f"{tag}lora", lora, _g(ctx, k, f"{L}.attn_hyper_connection.input_mix_weight_up#0.in0"))
    if i not in (0, 1):
        acc.add(f"{tag}Xa", X, _g(ctx, k, f"layers.{i - 1}#0.resid"))
    if prefill:
        acc.add(f"{tag}inj", inj_a, _g(ctx, k, f"{L}.attn_hyper_connection.input_mix_weight_down_block_inject#0.out")[:, 320:324])
    if ref_acc is not None:
        xr, ur, lr, ir = _hc_ref(ctx, ap, True, m.hc_norm(ka), X0, bo, inj)
        ref_acc.add("X", X, xr), ref_acc.add("u", u, ur), ref_acc.add("lora", lora, lr), ref_acc.add("inj", inj_a, ir)
    ao = _g(ctx, k, f"{L}.{a}#0.out")
    v, inj_m, lora2 = _mix(m, km, X, ao, inj_a)
    acc.add(f"{tag}v", v, _g(ctx, k, f"{L}.mlp#0.in0"))
    acc.add(f"{tag}X", X, _g(ctx, k, f"{L}#0.out.0"))
    acc.add(f"{tag}inj", inj_m, _g(ctx, k, f"{L}#0.out.2"))
    acc.add(f"{tag}lora", lora2, _g(ctx, k, f"{L}.mlp_hyper_connection.input_mix_weight_up#0.in0"))


def test_layers(ctx):
    """Decoder HC modules (both sides, chained) for layers 0,1,2,3,23,47 at prefill and decode steps."""
    m, acc, ref = _mod(ctx), Acc(), Acc()
    for k in (0, 1, 31, 32, 33, 48, 79):
        for i in (0, 1, 2, 3, 23, 47):
            _layer_chain(ctx, m, k, i, False, acc, "", ref)
            _layer_chain(ctx, m, k, i, True, acc, "bf.")
    ctx.detail("== hc.layers vs golden: " + acc.text())
    ctx.detail("== hc.layers vs torch ref (same inputs, fp32 norm): " + ref.text())
    u, ub, r = acc.get("u"), acc.get("bf.u"), ref.get("u")
    # kernel = the reference's math: with its bf16-rounded norm ≥ 97 % bit-equal; engine fp32 norm: small systematic
    # gap
    ok = ub["eq"] > 0.97 and ub["cos"] > 0.99999 and acc.get("bf.v")["eq"] > 0.97 and u["cos"] > 0.99999 and \
        acc.get("v")["cos"] > 0.99999 and acc.get("Xa")["eq"] > 0.999 and acc.get("X1c")["eq"] > 0.999 and \
        r["eq"] > 0.9 and r["max_rel"] < 1e-2
    return {"ok": ok, "msg": f"bf-norm u eq {100 * ub['eq']:.1f}% cos {ub['cos']:.6f} · fp32-norm u rel "
                             f"{u['max_rel']:.1e} cos {u['cos']:.6f} eq {100 * u['eq']:.1f}% · v cos "
                             f"{acc.get('v')['cos']:.6f} · resid eq {100 * acc.get('Xa')['eq']:.2f}% · "
                             f"torch-ref u eq {100 * r['eq']:.1f}%"}


def test_final(ctx):
    """Final mixer at every step: (layer 47 out.0, out.1, out.2) → y; lora."""
    m, acc, ref = _mod(ctx), Acc(), Acc()
    pre = f"{WP}hyper_connection_mixer"
    for rnd, tag in ((False, ""), (True, "bf.")):
        key = m.hc_load(pre, False, rnd)
        for s in ctx.golden.steps:
            k, ph = s["step"], ("pre." if s["num_tokens"] > 1 else "dec.")
            X = _g(ctx, k, "layers.47#0.out.0").contiguous()
            bo, inj = _g(ctx, k, "layers.47#0.out.1"), _g(ctx, k, "layers.47#0.out.2")
            X0 = X.clone()
            y, _, lora = _mix(m, key, X, bo, inj, inject=False)
            acc.add(tag + ph + "y", y, _file(ctx, k)[0].get_tensor("language_model.model#0.out").cuda())
            acc.add(tag + ph + "lora", lora, _g(ctx, k, "hyper_connection_mixer.input_mix_weight_up#0.in0"))
            acc.add(tag + "X", X, _g(ctx, k, "layers.47#0.resid"))
            if not rnd:
                xr, yr, lr, _ = _hc_ref(ctx, pre, False, m.hc_norm(key), X0, bo, inj)
                ref.add("y", y, yr), ref.add("lora", lora, lr), ref.add("X", X, xr)
    ctx.detail("== hc.final vs golden: " + acc.text())
    ctx.detail("== hc.final vs torch ref: " + ref.text())
    # decode golden ran cuBLAS bf16 split-K with reduced-precision (bf16) reduction for this down GEMM (≈ 23 % of z
    # off by ≤ 1 ulp vs fp32); prefill golden reduced in fp32 → the bit-level check uses prefill.
    y, yb, yp, r = acc.get("dec.y"), acc.get("bf.dec.y"), acc.get("bf.pre.y"), ref.get("y")
    ok = yp["eq"] > 0.97 and y["cos"] > 0.99999 and acc.get("pre.y")["cos"] > 0.99999 and r["eq"] > 0.97 and \
        r["max_rel"] < 1e-2 and acc.get("X")["eq"] > 0.999
    return {"ok": ok, "msg": f"{len(ctx.golden.steps)} steps · bf-norm y eq prefill {100 * yp['eq']:.1f}% decode "
                             f"{100 * yb['eq']:.1f}% · fp32-norm y rel {y['max_rel']:.1e} cos {y['cos']:.6f} · X eq "
                             f"{100 * acc.get('X')['eq']:.2f}% · torch-ref y eq {100 * r['eq']:.1f}%"}


def test_ple_ids(ctx):
    """n-gram ids (int64 [M,16]) at every step must equal the golden ids exactly (EOS rules, kept prefill rows)."""
    m = _mod(ctx)
    m.ple_load()
    bad, n = [], 0
    for name, steps in _requests(ctx):
        toks = _tokens(ctx, name)
        for k, pos in steps:
            ids = _g(ctx, k, "layers.1.ple#0.in1").int()
            if ids.tolist() != [toks[p] for p in pos]:
                bad.append(f"step {k} ids≠tokens")
            prev = torch.tensor(_prev(toks, pos), dtype=torch.int32, device="cuda")
            if len(pos) == 1 and prev.tolist() != _g(ctx, k, "layers.1.ple#0.in3").tolist():
                bad.append(f"step {k} ctx")
            E = torch.empty(len(pos), 2560, dtype=BF, device="cuda")
            got = m.ple_lookup(ids, prev, E)
            ref = _g(ctx, k, "layers.1.ple.ple_embedding.ngram_embedding#0.in0")
            n += ref.numel()
            if not torch.equal(got, ref):
                bad.append(f"step {k} ({(got != ref).sum().item()} differ)")
    return {"ok": not bad, "msg": f"{n} ids exact over {len(ctx.golden.steps)} steps" if not bad else " ".join(bad[:4])}


def test_ple_lookup(ctx):
    """NVFP4 table half: owned heads bit-equal to the (all-reduced) golden rows, other heads 0."""
    m = _mod(ctx)
    m.ple_load()
    bad, own_n, n = [], 0, 0
    reqs = dict((k, (name, pos)) for name, st in _requests(ctx) for k, pos in st)
    half = 320001536 // 2
    for k in (0, 1, 31, 32, 33, 47, 48, 79):
        name, pos = reqs[k]
        toks = _tokens(ctx, name)
        ids = _g(ctx, k, "layers.1.ple#0.in1").int()
        prev = torch.tensor(_prev(toks, pos), dtype=torch.int32, device="cuda")
        E = torch.empty(len(pos), 2560, dtype=BF, device="cuda")
        gid = m.ple_lookup(ids, prev, E)
        ref = _g(ctx, k, "layers.1.ple.ple_embedding.ngram_embedding#0.out").float()
        got = E.float().view(-1, 16, 160)
        own = (gid // half) == ctx.rank
        own_n, n = own_n + own.sum().item(), n + own.numel()
        if not torch.equal(got[own], ref[own]):
            bad.append(f"step {k} own ≠ ({(got[own] != ref[own]).sum().item()})")
        if got[~own].abs().max().item() if (~own).any() else 0:
            bad.append(f"step {k} other≠0")
        if not torch.equal(_g(ctx, k, "layers.1.ple.ple_embedding#0.out").float().view(-1, 16, 160), ref):
            bad.append(f"step {k} ple_embedding.out ≠ ngram out")
    return {"ok": not bad, "msg": f"owned heads bit-exact ({own_n}/{n} heads on rank {ctx.rank}), others 0"
            if not bad else " ".join(bad[:4])}


def test_ple_apply(ctx):
    """PLE per request in step order (conv state carried): golden E → kv, X. Long prefill: kept rows ≥ 9 only.
    `bf.` = PLE norms rounded to bf16 as the reference deployment loads them (test variant) — isolates the fp32-norm
    gap."""
    m = _mod(ctx)
    m.ple_load()
    acc = Acc()
    for bfn, pre in ((False, ""), (True, "bf.")):
        for name, steps in _requests(ctx):
            state = torch.zeros(1, 9, 10240, dtype=BF, device="cuda")
            for k, pos in steps:
                X = _g(ctx, k, "layers.1.ple#0.in0").contiguous()
                E = _g(ctx, k, "layers.1.ple.ple_embedding#0.out").contiguous()
                M = X.shape[0]
                kv = m.ple_apply(X, E, state, [0], [0, M], bfn)
                tag = pre + ("pre." if M > 1 else "dec.")
                acc.add(tag + "kv", kv, _g(ctx, k, "layers.1.ple.kv_proj#0.out.0"))
                ref = _g(ctx, k, "layers.1.ple#0.out")
                r0 = 9 if pos[0] > 0 and M > 1 else 0  # truncated prefill: rows < 9 need rows that were not kept
                acc.add(tag + "X", X[r0:], ref[r0:])
    ctx.detail("== hc.ple_apply vs golden: " + acc.text())
    d, p, bd, bp = acc.get("dec.X"), acc.get("pre.X"), acc.get("bf.dec.X"), acc.get("bf.pre.X")
    ok = d["cos"] > 0.99999 and p["cos"] > 0.99999 and bp["eq"] > 0.99 and acc.get("dec.kv")["cos"] > 0.99999
    return {"ok": ok, "msg": f"X decode eq {100 * d['eq']:.2f}% rel {d['max_rel']:.1e} (bf-norm {100 * bd['eq']:.2f}%) · "
                             f"prefill eq {100 * p['eq']:.2f}% (bf-norm {100 * bp['eq']:.2f}%) · kv eq "
                             f"{100 * acc.get('dec.kv')['eq']:.2f}%/{100 * acc.get('pre.kv')['eq']:.2f}%"}


def test_invariance(ctx):
    """Bitwise: row results independent of M / other rows / chunking (M > 64) / run; PLE with 2 sequences."""
    m = _mod(ctx)
    key = m.hc_load(f"{WP}layers.2.attn_hyper_connection", True, False)
    X0 = _g(ctx, 0, "layers.2#0.kw.hidden_states")
    bo0, inj0 = _g(ctx, 0, "layers.2#0.kw.prev_block_output"), _g(ctx, 0, "layers.2#0.kw.prev_injection")
    X1, bo1, inj1 = (_g(ctx, 48, f"layers.2#0.kw.{n}") for n in ("hidden_states", "prev_block_output", "prev_injection"))
    Xa, boa, inja = torch.cat([X0, X1]), torch.cat([bo0, bo1]), torch.cat([inj0, inj1])  # 122 rows → 2 chunks

    def run(rows):
        X = Xa[rows].clone()
        u, io, _ = _mix(m, key, X, boa[rows].contiguous(), inja[rows].contiguous())
        return torch.cat([X.view(len(rows), -1), u, io], 1)

    full = run(list(range(122)))
    bad = []
    if not torch.equal(full, run(list(range(122)))):
        bad.append("run-to-run")
    for rows in ([0], [60], [121], list(range(10, 15)), list(range(3, 20)), list(range(64)), list(range(50, 114)),
                 [5, 70, 33, 100]):
        if not torch.equal(run(rows), full[rows]):
            bad.append(f"rows {rows[0]}..{rows[-1]} (M={len(rows)})")
    # PLE: two sequences in one call (M=122) vs each alone
    m.ple_load()
    E0 = _g(ctx, 0, "layers.1.ple.ple_embedding#0.out")
    E1 = _g(ctx, 48, "layers.1.ple.ple_embedding#0.out")
    P0, P1 = _g(ctx, 0, "layers.1.ple#0.in0"), _g(ctx, 48, "layers.1.ple#0.in0")
    st = torch.zeros(2, 9, 10240, dtype=BF, device="cuda")
    Xb = torch.cat([P1, P0]).contiguous()
    m.ple_apply(Xb, torch.cat([E1, E0]).contiguous(), st, [1, 0], [0, 61, 122], False)
    sa = torch.zeros(1, 9, 10240, dtype=BF, device="cuda")
    xa = P0.clone()
    m.ple_apply(xa, E0.contiguous(), sa, [0], [0, 61], False)
    if not (torch.equal(xa, Xb[61:]) and torch.equal(sa[0], st[0])):
        bad.append("ple 2-seq")
    return {"ok": not bad, "msg": "HC rows bitwise equal for M=1..122 subsets, chunked, run-to-run · PLE 2-seq"
            if not bad else "differ: " + " ".join(bad[:4])}


def test_perf(ctx):
    """hc_combine_mix µs at M=1,4,16,64 (cycling 8 modules, 105 MB > L2; best of 5, GPU shared) and PLE."""
    m = _mod(ctx)
    keys = [m.hc_load(f"{WP}layers.{i}.{s}_hyper_connection", True, False) for i in range(4) for s in ("attn", "mlp")]
    pk = m.pack_blob(keys[0]).numel()  # P012 on (and packed at upload): the decode kernels read the 12-bit copy
    was = m.p012(True)
    m.p012(was)
    on = pk > 0 and was
    nb =pk + 4 * 10240 * 2 if on else HC_BYTES
    parts = ["P012" if on else "bf16"]
    for M in (1, 4, 16, 64):
        us = m.bench_mix(keys, M, 64, 5)
        parts.append(f"M{M} {us:.1f}µs {nb / us / 1e3:.0f}GB/s")
        ctx.detail(f"hc perf M={M} ({parts[0]}, {nb / 1e6:.2f} MB read): {us:.2f} us/module, {nb / us / 1e3:.1f} GB/s "
                   f"({nb / us / 1e3 / 2.41:.1f}% of 241)")
    m.ple_load()
    for M in (1, 64):
        us = m.bench_ple(M, 5)
        ctx.detail(f"ple perf M={M}: {us:.1f} us (kv 65.5 MB → {65.5e6 / us / 1e3:.0f} GB/s)")
        parts.append(f"PLE M{M} {us:.0f}µs")
    us1 = m.bench_mix(keys, 1, 64, 5)
    return {"ok": nb / us1 / 1e3 > 0.85 * 241, "msg": " · ".join(parts)}


# ---- prefill (M > 64: hc_prefill.cuh) ------------------------------------------------------------------------------
def _tile(t, M, noise, gen):
    """M rows cycling through t, each scaled elementwise by (1 + noise·N(0,1)) → distinct but realistic rows."""
    r = t[torch.arange(M, device="cuda") % t.shape[0]].float()
    return (r * (1 + noise * torch.randn(r.shape, generator=gen, device="cuda"))).to(BF).contiguous()


def _chunked_mix(m, key, X, bo, inj, inject):
    """Reference: the decode kernels (64-row calls)."""
    outs = []
    for c in range(0, X.shape[0], 64):
        x = X[c:c + 64].clone()
        u, io, lora = _mix(m, key, x, bo[c:c + 64].contiguous(), inj[c:c + 64].contiguous(), inject)
        outs.append((x, u, io, lora))
    return [torch.cat([o[i] for o in outs]) if outs[0][i] is not None else None for i in range(4)]


def test_prefill_bitwise(ctx):
    """HC modules (inject / final mixer) and PLE: rows of M = 65…8192 calls == the decode kernels (64-row calls),
    X, u, inj, lora; PLE with a carried state, two sequences, 8,192 rows (X and conv state)."""
    m, bad = _mod(ctx), []
    gen = torch.Generator(device="cuda").manual_seed(5)
    src = [(k, "layers.2#0.kw.") for k in (0, 32, 48)]
    X0 = torch.cat([_g(ctx, k, p + "hidden_states") for k, p in src])
    bo0 = torch.cat([_g(ctx, k, p + "prev_block_output") for k, p in src])
    inj0 = torch.cat([_g(ctx, k, p + "prev_injection") for k, p in src])
    MM = 8192
    X, bo, inj = _tile(X0, MM, 0.02, gen), _tile(bo0, MM, 0.02, gen), _tile(inj0, MM, 0.1, gen)
    n_cmp = 0
    for name, prefix, inject in (("attn2", f"{WP}layers.2.attn_hyper_connection", True),
                                 ("mlp47", f"{WP}layers.47.mlp_hyper_connection", True),
                                 ("final", f"{WP}hyper_connection_mixer", False)):
        key = m.hc_load(prefix, inject, False)
        ref = _chunked_mix(m, key, X, bo, inj, inject)
        for lo, hi in ((0, 65), (0, 200), (0, 2048), (0, MM), (3000, 3000 + 777)):
            x = X[lo:hi].clone()
            u, io, lora = _mix(m, key, x, bo[lo:hi].contiguous(), inj[lo:hi].contiguous(), inject)
            got = (x, u, io, lora)
            n_cmp += 1
            for nm, a, b in zip(("X", "u", "inj", "lora"), got, ref):
                if a is not None and not torch.equal(a, b[lo:hi]):
                    bad.append(f"{name} rows {lo}..{hi} {nm} ({(a != b[lo:hi]).sum().item()})")
    # PLE: slot 0 continues a 61-row prefill, slot 1 starts fresh; one call of 8192 rows vs 64-row steps
    m.ple_load()
    E0, P0 = _g(ctx, 0, "layers.1.ple.ple_embedding#0.out"), _g(ctx, 0, "layers.1.ple#0.in0")
    E, Xp = _tile(E0, MM, 0.05, gen), _tile(P0, MM, 0.02, gen)
    seqs = [0, 3000, MM]
    st0 = torch.zeros(2, 9, 10240, dtype=BF, device="cuda")
    m.ple_apply(P0.clone(), E0.contiguous(), st0, [0], [0, 61], False)  # warm slot 0
    st_ref, Xr = st0.clone(), Xp.clone()
    for b in range(2):
        for c in range(seqs[b], seqs[b + 1], 64):
            e = min(c + 64, seqs[b + 1])
            xc = Xr[c:e].clone()
            m.ple_apply(xc, E[c:e].contiguous(), st_ref, [b], [0, e - c], False)
            Xr[c:e] = xc
    st, Xb = st0.clone(), Xp.clone()
    m.ple_apply(Xb, E, st, [0, 1], seqs, False)
    n_cmp += 1
    if not torch.equal(Xb, Xr):
        bad.append(f"ple X ({(Xb != Xr).sum().item()})")
    if not torch.equal(st, st_ref):
        bad.append("ple state")
    return {"ok": not bad, "msg": f"HC 3 modules × 5 ranges (M 65–8192, offsets) · PLE 8192 rows B=2 state carried — "
            f"bitwise == decode path" if not bad else "differs: " + " · ".join(bad[:4])}


_PAD = 0


def _mix_padded(m, key, X, bo, inj, inject=True):
    """The same rows at the end of an M=_PAD call (other rows: the same rows reversed/rolled) → large-M path."""
    M = X.shape[0]
    idx = torch.arange(_PAD - M, device="cuda") % M
    Xf = torch.cat([X[idx].flip(1), X]).contiguous()
    bof = torch.cat([bo[idx], bo]).contiguous() if bo is not None else None
    injf = torch.cat([inj[idx], inj]).contiguous() if inj is not None else None
    u, io, lora = _mix_raw(m, key, Xf, bof, injf, inject)
    X.copy_(Xf[-M:])
    return u[-M:], io[-M:] if io is not None else None, lora[-M:]


def test_prefill_golden(ctx):
    """Golden prefill steps 0 (61 rows) and 32 (last 64 rows of 8,603) through the large-M path (inside M=2048 calls):
    bitwise == decode path on the same rows, and the golden comparison of test_layers / test_final / test_ple_apply."""
    global _PAD
    m, acc_big, acc_dec, bad = _mod(ctx), Acc(), Acc(), []
    steps = [s["step"] for s in ctx.golden.steps if s["num_tokens"] > 1][:2]
    for big in (False, True):
        _PAD = 2048 if big else 0
        try:
            for k in steps:
                for i in (0, 2, 3, 23, 47):
                    _layer_chain(ctx, m, k, i, True, acc_big if big else acc_dec, "bf.")
        finally:
            _PAD = 0
    for name, v in acc_big.d.items():
        d = acc_dec.d[name]
        if (v["max_abs"], v["cos"], v["eq"]) != (d["max_abs"], d["cos"], d["eq"]):
            bad.append(name)
    # final mixer and PLE through the large-M path
    key = m.hc_load(f"{WP}hyper_connection_mixer", False, True)
    accf = Acc()
    for k in steps:
        X = _g(ctx, k, "layers.47#0.out.0").contiguous()
        bo, inj = _g(ctx, k, "layers.47#0.out.1"), _g(ctx, k, "layers.47#0.out.2")
        Xd = X.clone()
        yd, _, _ = _mix_raw(m, key, Xd, bo, inj, inject=False)
        _PAD = 2048
        y, _, _ = _mix_padded(m, key, X, bo, inj, inject=False)
        _PAD = 0
        if not (torch.equal(y, yd) and torch.equal(X, Xd)):
            bad.append(f"final step {k}")
        accf.add("y", y, _file(ctx, k)[0].get_tensor("language_model.model#0.out").cuda())
    m.ple_load()
    accp = Acc()
    for k in steps:
        X = _g(ctx, k, "layers.1.ple#0.in0").contiguous()
        E = _g(ctx, k, "layers.1.ple.ple_embedding#0.out").contiguous()
        M = X.shape[0]
        st = torch.zeros(2, 9, 10240, dtype=BF, device="cuda")
        Xd = X.clone()
        m.ple_apply(Xd, E, st[1:].clone(), [0], [0, M], True)
        fill = 2048 - M  # sequence 0 = filler rows (slot 1), sequence 1 = these rows (slot 0, fresh)
        Xb = torch.cat([X[torch.arange(fill, device="cuda") % M].flip(1), X]).contiguous()
        Eb = torch.cat([E[torch.arange(fill, device="cuda") % M], E]).contiguous()
        m.ple_apply(Xb, Eb, st, [1, 0], [0, fill, 2048], True)
        if not torch.equal(Xb[fill:], Xd):
            bad.append(f"ple step {k}")
        r0 = 9 if k != steps[0] else 0
        accp.add("X", Xb[fill + r0:], _g(ctx, k, "layers.1.ple#0.out")[r0:])
    u, v, y, p = acc_big.get("bf.u"), acc_big.get("bf.v"), accf.get("y"), accp.get("X")
    ctx.detail("== hc.prefill_golden (large-M path, bf-norm): " + acc_big.text() + " · final " + accf.text() +
               " · ple " + accp.text())
    ok = not bad and u["eq"] > 0.97 and v["eq"] > 0.97 and y["cos"] > 0.99999 and p["eq"] > 0.99
    return {"ok": ok, "msg": (f"large-M = decode bitwise · " if not bad else "differs: " + " ".join(bad[:3]) + " · ") +
            f"golden prefill {len(steps)} steps (bf-norm): u eq {100 * u['eq']:.1f}% · v eq {100 * v['eq']:.1f}% · "
            f"final y cos {y['cos']:.6f} · PLE X eq {100 * p['eq']:.2f}%"}


def test_prefill_perf(ctx):
    """hc_combine_mix µs per module (cycling 8 modules) and PLE (lookup + apply) at M = 512, 2048, 8192; best of 3."""
    m = _mod(ctx)
    keys = [m.hc_load(f"{WP}layers.{i}.{s}_hyper_connection", True, False) for i in range(4) for s in ("attn", "mlp")]
    m.ple_load()
    lines, parts = [], []
    for M in (512, 2048, 8192):
        us = m.bench_mix(keys, M, 16, 3)
        tf = 2 * M * (324 + 320) * 10240 / us / 1e6
        pus = m.bench_ple(M, 3)
        lines.append(f"M={M}: HC {us:.0f} µs/module ({us / M:.3f} µs/row, {tf:.1f} TF on the GEMMs, ×97 = "
                     f"{97 * us / 1e3:.0f} ms) · PLE {pus:.0f} µs ({pus / M:.2f} µs/row)")
        parts.append(f"M{M} HC {us:.0f}µs {tf:.0f}TF · PLE {pus / 1e3:.1f}ms")
    ctx.detail("\n".join(["== hc.prefill_perf"] + lines))
    return {"ok": True, "msg": " | ".join(parts)}


# ---- P003 hc_fused_prefill (p003_hc_fused_prefill.cuh) ------------------------------------------------------------
def test_p003(ctx):
    """P003 on vs off in one process: X (in place), u, inj, lora bitwise for M = 384…8192 (odd sizes, offset rows) ×
    attn / mlp inject modules, the final mixer, bo=None and inj=None (realistic rows: golden tiles + noise); then µs per
    module (bench_mix, cycling 8 modules) base vs patch at M = 2048 and 8192, best of 3 alternating runs."""
    m, bad = _mod(ctx), []
    was = m.p003(False)
    try:
        gen = torch.Generator(device="cuda").manual_seed(7)
        src = [(k, "layers.2#0.kw.") for k in (0, 32, 48)]
        X0 = torch.cat([_g(ctx, k, p + "hidden_states") for k, p in src])
        bo0 = torch.cat([_g(ctx, k, p + "prev_block_output") for k, p in src])
        inj0 = torch.cat([_g(ctx, k, p + "prev_injection") for k, p in src])
        MM = 8192
        X, bo, inj = _tile(X0, MM, 0.02, gen), _tile(bo0, MM, 0.02, gen), _tile(inj0, MM, 0.1, gen)
        attn = m.hc_load(f"{WP}layers.2.attn_hyper_connection", True, False)
        mlp = m.hc_load(f"{WP}layers.47.mlp_hyper_connection", True, False)
        fin = m.hc_load(f"{WP}hyper_connection_mixer", False, False)
        cases = (("attn2", attn, True, True, True), ("mlp47", mlp, True, True, True), ("final", fin, False, True, True),
                 ("bo=None", attn, True, False, False), ("inj=None", mlp, True, True, False))
        n_cmp = 0
        for lo, hi in ((0, 384), (0, 385), (3000, 3777), (0, 2048), (5000, 8001), (0, MM)):
            for name, key, inject, with_bo, with_inj in cases:
                b = bo[lo:hi].contiguous() if with_bo else None
                i = inj[lo:hi].contiguous() if with_inj else None
                out = []
                for on in (False, True):
                    m.p003(on)
                    x = X[lo:hi].clone()
                    u, io, lora = _mix_raw(m, key, x, b, i, inject)
                    out.append((x, u, io, lora))
                n_cmp += 1
                for nm, a, p in zip(("X", "u", "inj", "lora"), *out):
                    if a is not None and not torch.equal(a, p):
                        bad.append(f"{name} M={hi - lo} {nm} ({(a != p).sum().item()})")
        keys = [m.hc_load(f"{WP}layers.{i}.{s}_hyper_connection", True, False) for i in range(4) for s in ("attn", "mlp")]
        parts, lines = [], []
        for M in (2048, 8192):
            t = {False: [], True: []}
            for _ in range(3):
                for on in (False, True):
                    m.p003(on)
                    t[on].append(m.bench_mix(keys, M, 16, 3))
            b, p = min(t[False]), min(t[True])
            parts.append(f"M{M} {b:.0f}→{p:.0f}µs ({b / p:.2f}×)")
            lines.append(f"M={M}: base {b:.0f} µs · P003 {p:.0f} µs/module ({b / p:.2f}×; runs base "
                         f"{' '.join(f'{v:.0f}' for v in t[False])} | P003 {' '.join(f'{v:.0f}' for v in t[True])})")
        ctx.detail("\n".join(["== hc.p003"] + lines))
    finally:
        m.p003(was)
    return {"ok": not bad, "msg": (f"on = off bitwise ({n_cmp} cases: M 384–8192 × attn/mlp/final/bo=None/inj=None)"
                                   if not bad else "differs: " + " · ".join(bad[:4])) + " · " + " · ".join(parts)}


# ---- P012 hc_pack12 (p012_hc_pack12.cuh) ----------------------------------------------------------------------------
# packed block offsets (p12::kDnB, kUpB, kOScD, kOScU, kOEoD, kOEoU, kOEnt)
_P12 = dict(dnB=4915200, upB=4915200, scD=9830400, scU=9831040, eoD=9851520, eoU=9852656, ent=9855232)
_COLOFS = [0, 1, 0, 1, 8, 9, 8, 9]  # A fragment element → column offset (pack_tiles); rows g, g+8 (elements 2,3,6,7)


def _hc_modules():
    """Every HC module of the checkpoint: 48 layers × (attn, mlp), the final mixer, the MTP head's three."""
    out = [(f"{WP}layers.{i}.{s}_hyper_connection", True) for i in range(48) for s in ("attn", "mlp")]
    out += [(f"{WP}hyper_connection_mixer", False), ("mtp.layers.0.attn_hyper_connection", True),
            ("mtp.layers.0.mlp_hyper_connection", True), ("mtp.hyper_connection_mixer", False)]
    return out


def _u32(b, lo, hi):
    return b[lo:hi].view(torch.int32).long() & 0xFFFFFFFF


def _p12_minis(words):
    """words [U, G, 3, 32 lanes, 4 k-steps] → mini values [U, G, 32, 4, 8 elements] (gap fields → M3)."""
    W0, W1, W2 = words[:, :, 0], words[:, :, 1], words[:, :, 2]
    K = 0x87FF87FF
    Ms = [W0 & K, W1 & K, W2 & K,
          ((W0 >> 11) & 0x000F000F) | ((W1 >> 7) & 0x00F000F0) | ((W2 >> 3) & 0x07000700) | ((W2 << 1) & 0x80008000)]
    return torch.stack([(Ms[e >> 1] >> (16 * (e & 1))) & 0xFFFF for e in range(8)], -1)


def _p12_value(mini, b):
    """mini s·c·m + row base b → bf16 bits (integer math: c = 0 → ±0, else exponent c + b − 1)."""
    c = (mini >> 7) & 15
    return torch.where(c == 0, mini & 0x8000, (mini & 0x807F) | ((c + b - 1) << 7))


def _p12_bases(sc, tiles):
    """scale words [tiles·8] (s_g | s_g+8 << 16, s = (b + 126) << 7) → bases [tiles, 16] (row r = g + 8h)."""
    sc = sc.view(tiles, 8)
    return torch.stack([(sc & 0xFFFF) >> 7, (sc >> 23) & 0x1FF], 1).view(tiles, 16) - 126


def _p12_unpack(blob):
    """Independent decode of a packed block (integer math, no bf16 multiply): (down [320, 10240], up [10240, 320]) bf16
    bits as int64, and the number of exception entries."""
    P, dev = _P12, "cuda"
    ent = _u32(blob, P["ent"], blob.numel())
    lane, el = torch.arange(32, device=dev), torch.arange(8, device=dev)
    rin = (lane >> 2)[:, None] + 8 * ((el >> 1) & 1)[None, :]  # [32, 8] row in the tile
    cin = (2 * (lane & 3))[:, None] + torch.tensor(_COLOFS, device=dev)[None, :]  # [32, 8] column in the k-step
    # down [kc 40][tile 20][G 4][3][32][4]; exception lists per (kc, warp), blk = 4·(4G + q) + slot, tile = warp + 7·slot
    md = _p12_minis(_u32(blob, 0, P["dnB"]).view(800, 4, 3, 32, 4)).view(40, 20, 4, 32, 4, 8)
    bd = _p12_bases(_u32(blob, P["scD"], P["scD"] + 640), 20)
    vd = _p12_value(md, bd[:, rin].view(1, 20, 1, 32, 1, 8))
    eod = _u32(blob, P["eoD"], P["eoD"] + 281 * 4)
    n_dn = int(eod[-1])
    if n_dn:
        e = ent[:n_dn]
        unit = torch.repeat_interleave(torch.arange(280, device=dev), eod[1:] - eod[:-1])
        blk = (e >> 8) & 0xFF
        j = blk >> 2
        vd[unit // 7, unit % 7 + 7 * (blk & 3), j >> 2, (e >> 3) & 31, j & 3, e & 7] = e >> 16
    ks = torch.arange(640, device=dev).view(40, 4, 4)  # [kc, G, q] → k-step
    col = (ks.view(40, 1, 4, 1, 4, 1) * 16 + cin.view(1, 1, 1, 32, 1, 8)).expand(40, 20, 4, 32, 4, 8)
    row = (torch.arange(20, device=dev).view(20, 1, 1) * 16 + rin).view(1, 20, 1, 32, 1, 8).expand_as(col)
    dn = torch.full((320 * 10240,), -1, dtype=torch.long, device=dev)
    dn[(row * 10240 + col).reshape(-1)] = vd.reshape(-1)
    # up [tile 640][G 5][3][32][4]; lists per tile, blk = k-step; tile row r ↔ Wup row (r >> 2)·2560 + 4·tile + (r & 3)
    mu = _p12_minis(_u32(blob, P["dnB"], P["dnB"] + P["upB"]).view(640, 5, 3, 32, 4))
    bu = _p12_bases(_u32(blob, P["scU"], P["scU"] + 640 * 32), 640)
    vu = _p12_value(mu, bu[:, rin].view(640, 1, 32, 1, 8))
    eou = _u32(blob, P["eoU"], P["eoU"] + 641 * 4)
    n_all = int(eou[-1])
    if n_all > n_dn:
        e = ent[n_dn:n_all]
        unit = torch.repeat_interleave(torch.arange(640, device=dev), eou[1:] - eou[:-1])
        blk = (e >> 8) & 0xFF
        vu[unit, blk >> 2, (e >> 3) & 31, blk & 3, e & 7] = e >> 16
    ksu = torch.arange(20, device=dev).view(5, 4)
    colu = (ksu.view(1, 5, 1, 4, 1) * 16 + cin.view(1, 1, 32, 1, 8)).expand(640, 5, 32, 4, 8)
    r = rin.view(1, 1, 32, 1, 8).expand(640, 5, 32, 4, 8)
    rowu = (r >> 2) * 2560 + 4 * torch.arange(640, device=dev).view(640, 1, 1, 1, 1) + (r & 3)
    up = torch.full((10240 * 320,), -1, dtype=torch.long, device=dev)
    up[(rowu * 320 + colu).reshape(-1)] = vu.reshape(-1)
    return dn.view(320, 10240), up.view(10240, 320), n_all


def test_p012_pack(ctx):
    """P012 packing on every HC module of this rank (48×2 + final mixer + MTP 3), uploaded with the patch on: the upload
    decodes every fragment on the device as the kernels do (fails on any difference); here an independent integer decode
    of the packed block (layout, row bases, exception lists) == the checkpoint's bf16 rows, bitwise."""
    m, bad = _mod(ctx), []
    n_vals = n_exc = n_bytes = 0
    mods = _hc_modules()
    for prefix, inject in mods:
        key = m.hc_load_packed(prefix, inject)
        blob = m.pack_blob(key)
        if blob.numel() == 0:
            bad.append(f"{prefix} not packed")
            continue
        dn, up, ne = _p12_unpack(blob)
        for nm, got, shape in (("input_mix_weight_down", dn, (320, 10240)), ("input_mix_weight_up", up, (10240, 320))):
            b, shp, dt, tp = ctx.w.slice(f"{prefix}.{nm}.weight")
            ref = b.view(torch.int16).view(*shape).cuda().long() & 0xFFFF
            if not torch.equal(got, ref):
                bad.append(f"{prefix}.{nm} ({(got != ref).sum().item()} differ)")
        n_vals += 2 * 320 * 10240
        n_exc += ne
        n_bytes += blob.numel()
    bf = 2 * 2 * 320 * 10240
    ctx.detail(f"== hc.p012_pack: {len(mods)} modules · {m.pack_stats()} · packed blocks {n_bytes / 1e6:.1f} MB "
               f"({n_bytes / len(mods) / 1e6:.3f} MB/module, bf16 down+up {bf / 1e6:.3f})")
    return {"ok": not bad, "msg": (f"{len(mods)} modules bitwise (on-device decode at upload + independent integer "
                                   f"decode) · exceptions {100 * n_exc / max(n_vals, 1):.4f}% · "
                                   f"{n_bytes / len(mods) / 1e6:.3f} MB/module (bf16 {bf / 1e6:.3f})") if not bad
            else "differs: " + " · ".join(bad[:4])}


def test_p012(ctx):
    """P012 on vs off in one process: X (in place), u, inj, lora bitwise for M = 1 … 383 (decode path, two row offsets)
    × attn / mlp / final mixer / MTP modules, bo=None, inj=None (realistic rows: golden tiles + noise); rows alone ==
    rows of an M=64 call with the patch on; then µs per module on vs off (bench_mix, 8 modules, best of 3 alternating)."""
    m, bad = _mod(ctx), []
    was = m.p012(True)
    try:
        gen = torch.Generator(device="cuda").manual_seed(12)
        src = [(k, "layers.2#0.kw.") for k in (0, 32, 48)]
        X0 = torch.cat([_g(ctx, k, p + "hidden_states") for k, p in src])
        bo0 = torch.cat([_g(ctx, k, p + "prev_block_output") for k, p in src])
        inj0 = torch.cat([_g(ctx, k, p + "prev_injection") for k, p in src])
        MM = 1024
        X, bo, inj = _tile(X0, MM, 0.02, gen), _tile(bo0, MM, 0.02, gen), _tile(inj0, MM, 0.1, gen)
        mods = {n: (m.hc_load_packed(p, i), i) for n, p, i in (
            ("attn2", f"{WP}layers.2.attn_hyper_connection", True), ("mlp47", f"{WP}layers.47.mlp_hyper_connection", True),
            ("attn0", f"{WP}layers.0.attn_hyper_connection", True), ("final", f"{WP}hyper_connection_mixer", False),
            ("mtp.attn", "mtp.layers.0.attn_hyper_connection", True), ("mtp.mlp", "mtp.layers.0.mlp_hyper_connection", True),
            ("mtp.mixer", "mtp.hyper_connection_mixer", False))}
        cases = [(n, n, True, True) for n in mods] + [("bo=None", "attn2", False, False), ("inj=None", "mlp47", True, False)]
        n_cmp = 0
        for M in (1, 2, 3, 4, 5, 8, 9, 16, 20, 33, 47, 64, 65, 100, 130, 200, 383):
            for lo in ((0, 517) if M <= 64 else (0, 301)):
                for name, mk, with_bo, with_inj in cases:
                    key, inject = mods[mk]
                    b = bo[lo:lo + M].contiguous() if with_bo else None
                    i = inj[lo:lo + M].contiguous() if with_inj else None
                    out = []
                    for on in (False, True):
                        m.p012(on)
                        x = X[lo:lo + M].clone()
                        u, io, lora = _mix_raw(m, key, x, b, i, inject)
                        out.append((x, u, io, lora))
                    n_cmp += 1
                    for nm, a, p in zip(("X", "u", "inj", "lora"), *out):
                        if a is not None and not torch.equal(a, p):
                            bad.append(f"{name} M={M}@{lo} {nm} ({(a != p).sum().item()})")
        m.p012(True)
        key = mods["attn2"][0]
        x = X[:64].clone()
        u, io, lora = _mix_raw(m, key, x, bo[:64].contiguous(), inj[:64].contiguous())
        for r in (0, 13, 63):
            x1 = X[r:r + 1].clone()
            u1, io1, l1 = _mix_raw(m, key, x1, bo[r:r + 1].contiguous(), inj[r:r + 1].contiguous())
            if not (torch.equal(u1, u[r:r + 1]) and torch.equal(io1, io[r:r + 1]) and torch.equal(x1, x[r:r + 1])):
                bad.append(f"row {r} alone ≠ in M=64")
        keys = [m.hc_load_packed(f"{WP}layers.{i}.{s}_hyper_connection", True) for i in range(4) for s in ("attn", "mlp")]
        pk_bytes = sum(m.pack_blob(k).numel() for k in keys) / len(keys) + 4 * 10240 * 2  # + the bf16 inject rows
        parts, lines = [], []
        for M in (1, 4, 16, 20, 64):
            t = {False: [], True: []}
            for _ in range(3):
                for on in (False, True):
                    m.p012(on)
                    t[on].append(m.bench_mix(keys, M, 64, 5))
            b, p = min(t[False]), min(t[True])
            parts.append(f"M{M} {b:.1f}→{p:.1f}µs")
            lines.append(f"M={M}: off {b:.2f} µs ({HC_BYTES / b / 1e3:.0f} GB/s of {HC_BYTES / 1e6:.2f} MB) · on {p:.2f} µs "
                         f"({pk_bytes / p / 1e3:.0f} GB/s of {pk_bytes / 1e6:.2f} MB; bf16-equivalent "
                         f"{HC_BYTES / p / 1e3:.0f} GB/s) · {b / p:.3f}× · runs off {' '.join(f'{v:.1f}' for v in t[False])}"
                         f" | on {' '.join(f'{v:.1f}' for v in t[True])}")
        ctx.detail("\n".join(["== hc.p012"] + lines))
    finally:
        m.p012(was)
    return {"ok": not bad, "msg": (f"on = off bitwise ({n_cmp} cases: M 1–383 × 7 modules/bo=None/inj=None, "
                                   f"2 offsets) · " if not bad else "differs: " + " · ".join(bad[:4]) + " · ")
            + " · ".join(parts)}
