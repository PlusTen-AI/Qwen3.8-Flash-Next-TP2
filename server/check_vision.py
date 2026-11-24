"""check_vision.py — vision front end (server/vision.py) and tower (server/vision_tower.py) against the golden dump.

  docker run --rm --network none … -v <dump>:/gold:ro -v <models>:/models:ro \
      qwen38-plus:<tag> python3 /w/src/server/check_vision.py [--device cuda] [--items all] [--details FILE]

For every golden vision case (tokens.json "vision"):
  [pixels]  our uint8 patches == the golden visual#0.in0 rows of that item (the rank that ran it), grid ↔ token count
  [expand]  golden prompt collapsed to template form (each media span → one placeholder) → vision.expand: ids, mRoPE
            positions (golden language_model.model kw.positions), rows (golden is_multimodal) and delta (positions of
            the first decode step) all exact
  [bad input] broken / wrong-kind bytes → ValueError
Tower, per item that a rank ran (default rgba, tiny, chart, clip; --items all adds scene and long):
  exact     input norm, pos-embed sum, rotary tables, rotary, residual adds (fail the run); LayerNorm and GELU on
            golden inputs (reported: torch's CPU kernels differ from CUDA in the last bit)
  per op    golden input → our op vs golden output (patch embed, qkv, attention, proj, fc1, fc2, merger)
  chain     our pixels through all 27 blocks: per block our fc1 output vs golden (act_fn#k.in0), block-26 in / out,
            merger output vs the golden per-item embeddings (embed_multimodal; cached items via embed_input_ids)
  diagnostics (never part of the tower): the same ops / chain with the reference's lossy linears emulated (MXFP8
            activations, bias added after rounding) and, on the CPU, attention with bf16 P; a noise run = our tower
            with only the attention changed, compared with our own output (how much the tower amplifies a last-bit
            difference)
  --tp (two nodes, tools/tp2.sh … --device cuda --tp): the tower split over both ranks as the server runs it (rank 1
            only follows); the chain and final checks as above, plus each item against the whole tower on one GPU
            (same pixels) and both timings. The per-op and emulation diagnostics are skipped (they need whole layers).
  --ref fp32 (diagnostic): the same tower in fp32 (RefFp32) as the high-precision reference; relL2 / cosine of our
            bf16 tower and of the golden against it (fc1 output of blocks 0/6/13/20/26, block-26 output, final
            embeddings with per-row cosine), and where the worst final rows go wrong (row norm, residual outliers,
            per-block cosine)
Exact checks fail the run (exit 1); tower numbers are reported (max |Δ|, relative L2, cosine, share of equal values).
One line per check on screen, details (per block, per op) to --details.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "tools")]
import qlog  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

import vision as V  # noqa: E402

BF16 = torch.bfloat16
ap = argparse.ArgumentParser()
ap.add_argument("--gold", default="/gold")
ap.add_argument("--model", default="/models/Qwen3.8-Flash-Next-FP4-Local")
ap.add_argument("--device", default="cpu")
ap.add_argument("--items", default="rgba.png,tiny.png,chart.png,clip.mp4", help="tower items (file names) or 'all'")
ap.add_argument("--no-tower", action="store_true")
ap.add_argument("--no-emulate", action="store_true", help="skip the reference-emulation diagnostic chain")
ap.add_argument("--ref", default=None, choices=["fp32"],
                help="also compare ours and the golden with an fp32 reference tower")
ap.add_argument("--no-ops", action="store_true", help="skip the per-op checks on golden inputs")
ap.add_argument("--details", default=None)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--tp", action="store_true", help="the tower over both ranks (run on both with tools/tp2.sh)")
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--tp-ports", default="29651,29613", help="TP job socket (worker) and NCCL store (head) ports")
args = ap.parse_args()
TP = None
if args.tp:
    from vision_tower import TP as _TP, Tower as _Tower
    sp, stp = (int(x) for x in args.tp_ports.split(","))
    TP = _TP.connect(args.rank, os.environ["Q38_HEAD_IP"], os.environ["Q38_PEER_IP"], sp, stp)
    args.no_ops = args.no_emulate = True
    args.ref = None
    if args.rank == 1:  # rank 1 runs its half of what rank 0 sends, until rank 0 closes the socket
        t0 = time.time()
        _Tower(args.model, "cuda", tp=TP).follow()
        qlog.ok("tp", f"rank 1 followed for {time.time() - t0:.1f} s")
        sys.exit(0)
if args.threads:
    torch.set_num_threads(args.threads)
if args.details:
    qlog.detail_file(args.details)
G = args.gold
bad = 0


def fail(tag, msg):
    global bad
    bad += 1
    qlog.error(tag, msg)
    qlog.detail(f"FAIL [{tag}] {msg}")


def good(tag, msg):
    qlog.ok(tag, msg)
    qlog.detail(f"ok [{tag}] {msg}")


def info(tag, msg):
    qlog.info(tag, msg)
    qlog.detail(f"[{tag}] {msg}")


class Step:
    """One golden step file (lazy, mmap)."""

    def __init__(self, rank, step):
        self.path = f"{G}/rank{rank}/step{step:04d}.safetensors"
        self.f = safe_open(self.path, "pt")
        self.keys = set(self.f.keys())

    def __contains__(self, k):
        return k in self.keys

    def get(self, k):
        return self.f.get_tensor(k) if k in self.keys else None


def cmp(ours, gold):
    """ours vs golden (same element count): equal share, max |Δ|, relative L2, cosine."""
    a, b = ours.reshape(-1).double().cpu(), gold.reshape(-1).double().cpu()
    d = a - b
    eq = (a == b).float().mean().item()
    nb = b.norm().item()
    rel = d.norm().item() / nb if nb else float("nan")
    cos = (a @ b).item() / (a.norm().item() * nb) if nb else float("nan")
    return {"eq": eq, "max": d.abs().max().item(), "rel": rel, "cos": cos}


def fmt(m):
    return f"eq {m['eq'] * 100:.2f}% max|Δ| {m['max']:.3g} relL2 {m['rel']:.2e} cos {m['cos']:.6f}"


def exact(ours, gold):
    a, b = ours.reshape(-1).cpu(), gold.reshape(-1).cpu()
    if a.shape != b.shape:
        return False, f"shape {tuple(ours.shape)} vs {tuple(gold.shape)}"
    ne = int((a.view(torch.int16) != b.view(torch.int16)).sum()) if a.dtype == BF16 else int((a != b).sum())
    return ne == 0, f"{ne} of {a.numel()} differ" if ne else "bit-exact"


# ---- golden metadata ------------------------------------------------------------------------------------------------
tok_json = json.load(open(f"{G}/tokens.json"))["vision"]
cases = tok_json["cases"]
sha = json.load(open(f"{G}/media/sha256.json"))
reqs = [json.loads(x) for x in open(f"{G}/rank0/requests.jsonl")]
manifest = {d["step"]: d for d in (json.loads(x) for x in open(f"{G}/rank0/manifest.jsonl"))}
from tokenizers import Tokenizer  # noqa: E402

tokenizer = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))

# case → request (prompt ids equal; cases in run order, so repeated prompts map in order)
case_req, nxt = {}, 0
for name, c in cases.items():
    for i in range(nxt, len(reqs)):
        if reqs[i]["prompt_ids"] == c["prompt_ids"]:
            case_req[name], nxt = reqs[i], i + 1
            break
    else:
        fail("map", f"{name}: no golden request with its prompt ids")

media_cache = {}  # file → Media
t_load = {}


def load(fn):
    if fn not in media_cache:
        data = open(f"{G}/media/{fn}", "rb").read()
        import hashlib
        if hashlib.sha256(data).hexdigest() != sha[fn]:
            fail("media", f"{fn}: sha256 differs from sha256.json")
        t0 = time.time()
        media_cache[fn] = V.load_video(data) if fn.endswith(".mp4") else V.load_image(data)
        t_load[fn] = time.time() - t0
    return media_cache[fn]


# ---- (a) pixels and (b) expansion per case --------------------------------------------------------------------------
where = {}  # file → (rank, step, row offset) where the golden tower ran it
for name, c in cases.items():
    r = case_req.get(name)
    if r is None:
        continue
    step, files = r["step"], c["files"]
    media = [load(fn) for fn in files]
    if [m.kind for m in media] != [x["modality"] for x in r["media"]]:
        fail(f"{name} media", "kinds / order differ from the golden request")
        continue
    steps = [Step(0, step), Step(1, step)]
    # (a) pixels: find each item's rows in the visual input of the rank that ran it
    for fn, m, rm in zip(files, media, r["media"]):
        span = c["prompt_ids"][rm["offset"]:rm["offset"] + rm["length"]]
        pad = V.IMAGE_PAD if m.kind == "image" else V.VIDEO_PAD
        npad = sum(t == pad for t in span)
        ngrp = sum(t == V.VISION_START for t in span) if m.kind == "video" else 1
        if npad != m.tokens or ngrp != m.grid[0]:
            fail(f"{name} grid", f"{fn}: grid {m.grid} → {m.tokens} tokens / {m.grid[0]} groups, golden {npad} / {ngrp}")
        found = None
        for rk, st in enumerate(steps):
            g = st.get("visual#0.in0")
            if g is None:
                continue
            cu = st.get("visual.blocks.0.attn#0.kw.cu_seqlens").tolist()
            for o in cu[:-1]:
                if o + m.pixels.shape[0] <= g.shape[0] and torch.equal(g[o:o + m.pixels.shape[0]], m.pixels):
                    found = (rk, o)
                    break
            if found:
                break
        tag = f"{name} pixels"
        desc = f"{fn} {m.kind} grid {m.grid} {m.pixels.shape[0]}×{m.pixels.shape[1]} ({t_load[fn] * 1000:.0f} ms)"
        if m.kind == "video":
            desc += f", frames {m.frames} @ {m.fps:g} fps, ts {' '.join(s[1:-9] for s in m.timestamps)}"
        if found:
            good(tag, f"{desc}: exact (rank{found[0]} rows {found[1]}…)")
            where.setdefault(fn, (found[0], step, found[1]))
        elif any("visual#0.in0" in st for st in steps):
            fail(tag, f"{desc}: not equal to any golden item rows")
        else:
            info(tag, f"{desc}: tower not run in this step (encoder cache) — checked in its first case")
    # (b) expansion: collapse each media span to the template placeholder, expand, compare
    ids = c["prompt_ids"]
    if ids != r["prompt_ids"]:
        fail(f"{name} expand", "tokens.json prompt_ids differ from requests.jsonl")
    collapsed, pos = [], 0
    for m, rm in zip(media, r["media"]):
        collapsed += ids[pos:rm["offset"]] + [V.IMAGE_PAD if m.kind == "image" else V.VIDEO_PAD]
        pos = rm["offset"] + rm["length"]
    collapsed += ids[pos:]
    out, pos3, delta, rows = V.expand(collapsed, media, tokenizer)
    gp = steps[0].get("language_model.model#0.kw.positions")
    gm = steps[0].get("Qwen4ExpForConditionalGeneration.fn.embed_input_ids#0.in.is_multimodal")
    probs = []
    if out != ids:
        k = next((i for i, (x, y) in enumerate(zip(out, ids)) if x != y), min(len(out), len(ids)))
        probs.append(f"ids differ at {k} (ours {len(out)}, golden {len(ids)})")
    gpos = gp[:, :len(ids)].t().reshape(-1).tolist() if gp is not None else None
    if gpos is None or pos3 != gpos:
        probs.append("positions differ" if gpos is not None else "no golden positions")
    grows = torch.nonzero(gm[:len(ids)]).view(-1).tolist() if gm is not None else None
    if grows is None or rows != grows:
        probs.append("rows differ" if grows is not None else "no golden is_multimodal")
    # delta: the first decode step after the prefill puts token p (= computed) at p + delta on all three rows
    dstep = manifest.get(step + 1, {}).get("reqs", {}).get(r["req"])
    dnote = "no decode step"
    if dstep and not dstep.get("new"):
        dp = Step(0, step + 1).get("language_model.model#0.kw.positions")
        want = dstep["computed"] + delta
        if dp is None or dp.shape[0] != 3 or any(int(dp[i, 0]) != want for i in range(3)):
            probs.append(f"decode positions {None if dp is None else dp[:, 0].tolist()} ≠ {want}")
        dnote = f"decode step {step + 1} at {want}"
    desc = f"{len(collapsed)} → {len(out)} ids, {len(rows)} media rows, delta {delta}"
    if probs:
        fail(f"{name} expand", f"{desc}: {'; '.join(probs)}")
    else:
        good(f"{name} expand", f"{desc}: ids, positions [3, {len(ids)}], rows, delta ({dnote}) exact")
    qlog.detail(f"  {name}: collapsed head {collapsed[:40]}")
    # golden sanity: embed_input_ids rows at is_multimodal = concatenated embed_multimodal items
    em = [steps[0].get(f"Qwen4ExpForConditionalGeneration.fn.embed_multimodal#0.ret.{i}") for i in range(len(media))]
    ret = steps[0].get("Qwen4ExpForConditionalGeneration.fn.embed_input_ids#0.ret")
    if all(e is not None for e in em) and ret is not None and gm is not None:
        ok_rows = torch.equal(ret[gm[:len(ids)]], torch.cat(em))
        (good if ok_rows else fail)(f"{name} rows", "golden embeddings at our rows = the golden per-item tower output"
                                    if ok_rows else "golden embeddings at our rows ≠ golden tower output")

# bad input → ValueError with a short message (never another exception or a crash)
png = open(f"{G}/media/chart.png", "rb").read()
mp4 = open(f"{G}/media/clip.mp4", "rb").read()
res = []
for kind, data, what in (("image", b"", "empty"), ("image", b"not an image", "text"), ("image", png[:400], "cut png"),
                         ("image", mp4, "mp4 as image"), ("video", b"", "empty"), ("video", b"not a video", "text"),
                         ("video", png, "png as video"), ("video", mp4[:3000], "cut mp4")):
    try:
        (V.load_image if kind == "image" else V.load_video)(data)
        res.append(f"{kind}/{what}: accepted")
    except ValueError as e:
        res.append(f"{kind}/{what}: ValueError")
        qlog.detail(f"[bad input] {kind}/{what}: {e}")
    except Exception as e:  # noqa: BLE001
        res.append(f"{kind}/{what}: {type(e).__name__}!")
odd = [r for r in res if not r.endswith("ValueError")]
(info if not any(r.endswith("!") for r in odd) else fail)("bad input", " · ".join(res))

if args.no_tower:
    sys.exit(1 if bad else 0)

# ---- (c) tower --------------------------------------------------------------------------------------------------------
from vision_tower import Tower, mm_f32  # noqa: E402

t0 = time.time()
tower = Tower(args.model, device=args.device, tp=TP)
info("tower", f"loaded on {args.device}{' (half of every layer, rank 0 of 2)' if TP else ''} in {time.time() - t0:.1f} s, "
     f"weights {tower.nbytes() / 2**30:.2f} GiB, threads {torch.get_num_threads()}")
single = Tower(args.model, device=args.device) if TP else None  # the whole tower on this GPU, for the comparison
dev = tower.device
items = list(where) if args.items == "all" else [x for x in args.items.split(",") if x in where]


def to(x):
    return x.to(dev)


def op_line(tag, ours, gold, extra="", quiet=False):
    m = cmp(ours, gold)
    if not quiet:
        qlog.info(tag, fmt(m) + extra)
    qlog.detail(f"[{tag}] {fmt(m)}{extra}")
    return m


# ---- reference emulation (diagnostic only: shows where and why our tower differs; never used by the tower) ----------
def mxfp8_act(x):
    """The reference's MXFP8 activation quantisation (b12x): per 32 along K a UE8M0 scale 2^ceil(log2(amax / 448)),
    values e4m3 (RNE, saturating)."""
    sh = x.shape
    xf = x.float().reshape(-1, sh[-1] // 32, 32)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
    return q.reshape(sh).to(BF16)


class RefLinear:
    """The reference's linear: [MXFP8 activations], fp32 GEMM (· global scale) rounded to bf16, then + bias in bf16."""

    def __init__(self, lin, act_q):
        self.lin, self.act_q = lin, act_q

    def __call__(self, x):
        L = self.lin
        acc = mm_f32(mxfp8_act(x) if self.act_q else x, L.w)
        if L.g is not None:
            acc = acc * L.g
        y = acc.to(BF16)
        return y + L.b if L.b is not None else y


def attention_bf16p(q, k, v, cu):
    """Flash-attention-like: fp32 scores and softmax statistics, P rounded to bf16 for P·V (one global max per row,
    not FA's per-block running max — an approximation of the reference kernel)."""
    scale = tower.head_dim ** -0.5
    out = torch.empty_like(q)
    for a, b in zip(cu[:-1], cu[1:]):
        qs, ks, vs = (z[a:b].transpose(0, 1).float() for z in (q, k, v))
        s = (qs @ ks.transpose(1, 2)) * scale
        p = torch.exp(s - s.amax(-1, keepdim=True))
        o = (p.to(BF16).float() @ vs) / p.sum(-1, keepdim=True)
        out[a:b] = o.transpose(0, 1).to(BF16)
    return out


class ref_emulation:
    """Context: swap the tower's quantised linears for RefLinear (qkv / proj with MXFP8 activations) and, with
    attn=True, its attention for attention_bf16p; linears=False keeps our linears (noise-floor run)."""

    def __init__(self, linears=True, attn=True):
        self.linears, self.attn = linears, attn

    def __enter__(self):
        self.saved = [dict(p) for p in tower.blocks], (tower.mfc1, tower.mfc2)
        if self.linears:
            for p in tower.blocks:
                for k in ("qkv", "proj", "fc1", "fc2"):
                    p[k] = RefLinear(p[k], k in ("qkv", "proj"))
            tower.mfc1, tower.mfc2 = RefLinear(tower.mfc1, False), RefLinear(tower.mfc2, False)
        if self.attn and dev.type == "cpu":
            tower.attention = attention_bf16p  # instance attribute shadows the method
        return self

    def __exit__(self, *a):
        blocks, (tower.mfc1, tower.mfc2) = self.saved
        for p, q in zip(tower.blocks, blocks):
            p.update(q)
        tower.__dict__.pop("attention", None)
        return False


def ndiff(ours, gold):
    """Values that differ: count, the largest |Δ|, total."""
    a, b = ours.reshape(-1).cpu().float(), gold.reshape(-1).cpu().float()
    d = (a - b).abs()
    n = int((d > 0).sum())
    return n, float(d.max()) if n else 0.0, a.numel()


def per_op(st, rank, step):
    """Golden input → our op → vs golden output, for blocks 0 and 26 and the merger (whole rank batch)."""
    g = st.get
    L = g("visual#0.in0").shape[0]
    tag = f"op r{rank}s{step}"
    # exact ops
    xn = tower.input_norm(to(g("visual#0.in0")))
    e, msg = exact(xn, g("visual.input_norm#0.out"))
    (good if e else fail)(f"{tag} input norm", msg)
    pe_g = to(g("visual.patch_embed.proj#0.out").reshape(L, -1))
    op_line(f"{tag} patch embed", tower.patch(to(g("visual.patch_embed.proj#0.in0").reshape(L, -1))), pe_g,
            " (bf16 GEMM, bias in fp32)")
    cu = g("visual.blocks.0.attn#0.kw.cu_seqlens").tolist()
    # the grid per sequence is not in the dump: rebuild it from the items located on this rank
    grids = [media_cache[fn].grid for fn, (rk, s, o) in sorted(where.items(), key=lambda kv: kv[1][2])
             if rk == rank and s == step]
    pos = torch.cat([tower.pos_embed(gr) for gr in grids])
    e, msg = exact(pe_g + pos, g("visual.blocks.0.norm1#0.in0").reshape(L, -1))
    (good if e else fail)(f"{tag} pos embed", f"golden patch embed + our pos = golden block-0 input: {msg}")
    cs = [tower.rotary(gr) for gr in grids]
    for k, i in (("cos", 0), ("sin", 1)):
        e, msg = exact(torch.cat([c[i] for c in cs]), g(f"visual.blocks.0.attn#0.kw.rotary_pos_emb_{k}"))
        (good if e else fail)(f"{tag} rotary {k}", msg)
    ln = [0, 0, 0]

    def lin(name, ours, x, gold, what):
        op_line(f"{tag} {name}", ours(x), gold, f" ({what}) — ours")
        op_line(f"{tag} {name}", RefLinear(ours, "MXFP8" in what)(x), gold, " — reference emulated (diagnostic)")

    for b in (0, 26):
        p = tower.blocks[b]
        pre = f"visual.blocks.{b}"
        for n, wb in (("norm1", p["n1"]), ("norm2", p["n2"])):
            d = ndiff(tower.layer_norm(to(g(f"{pre}.{n}#0.in0").reshape(L, -1)), wb), g(f"{pre}.{n}#0.out"))
            ln = [ln[0] + d[0], max(ln[1], d[1]), ln[2] + d[2]]
        lin(f"b{b} qkv", p["qkv"], to(g(f"{pre}.attn.qkv#0.in0").reshape(L, -1)),
            g(f"{pre}.attn.qkv#0.out.0").reshape(L, -1), "MXFP8 W, bf16 act")
        rin = g(f"{pre}.attn.apply_rotary_emb#0.in0")  # [2, L, 16, 72]
        rc, rs = to(g(f"{pre}.attn.apply_rotary_emb#0.in1")), to(g(f"{pre}.attn.apply_rotary_emb#0.in2"))
        rot = torch.stack([tower.apply_rotary(to(rin[j]), rc, rs) for j in range(2)])
        e, msg = exact(rot, g(f"{pre}.attn.apply_rotary_emb#0.out"))
        (good if e else fail)(f"{tag} b{b} rotary", f"on golden q/k: {msg}")
        q, k, v = (to(g(f"{pre}.attn.attn#0.kw.{n}")[0]) for n in ("query", "key", "value"))
        op_line(f"{tag} b{b} attention", tower.attention(q, k, v, cu), g(f"{pre}.attn.attn#0.out")[0],
                f" ({'flash_attn' if dev.type == 'cuda' else 'fp32 SDPA'} vs the reference's flash attention) — ours")
        if dev.type == "cpu":
            op_line(f"{tag} b{b} attention", attention_bf16p(q, k, v, cu), g(f"{pre}.attn.attn#0.out")[0],
                    " — bf16-P emulated (diagnostic)")
        lin(f"b{b} proj", p["proj"], to(g(f"{pre}.attn.proj#0.in0").reshape(L, -1)),
            g(f"{pre}.attn.proj#0.out.0").reshape(L, -1), "MXFP8 W, bf16 act")
        # residuals: mid = block input + attn out; block output (the next input) = mid + mlp out
        e, msg = exact(to(g(f"{pre}.norm1#0.in0")) + to(g(f"{pre}.attn#0.out")), g(f"{pre}.norm2#0.in0"))
        (good if e else fail)(f"{tag} b{b} residual", f"x + attn = golden mid: {msg}")
        if b == 26:
            e, msg = exact(to(g(f"{pre}.norm2#0.in0")) + to(g(f"{pre}.mlp#0.out")), g("visual.merger.norm#0.in0"))
            (good if e else fail)(f"{tag} b{b} residual2", f"mid + mlp = golden block output: {msg}")
        lin(f"b{b} fc1", p["fc1"], to(g(f"{pre}.mlp.linear_fc1#0.in0").reshape(L, -1)),
            g(f"{pre}.mlp.linear_fc1#0.out").reshape(L, -1), "NVFP4 W4A16")
        lin(f"b{b} fc2", p["fc2"], to(g(f"{pre}.mlp.linear_fc2#0.in0").reshape(L, -1)),
            g(f"{pre}.mlp.linear_fc2#0.out").reshape(L, -1), "NVFP4 W4A16")
    d = ndiff(tower.layer_norm(to(g("visual.merger.norm#0.in0").reshape(L, -1)), tower.mnorm),
              g("visual.merger.norm#0.out"))
    (good if ln[0] + d[0] == 0 else qlog.warn)(
        f"{tag} layernorm", f"blocks 0/26 norm1/norm2 on golden input: {ln[0]} of {ln[2]} differ (max |Δ| "
                            f"{ln[1]:.2g}); merger norm {d[0]} of {d[2]} (max |Δ| {d[1]:.2g})")
    qlog.detail(f"[{tag} layernorm] blocks {ln} merger {d}")
    nd, tot, neg, dmax = 0, 0, 0, 0.0
    for k in range(tower.depth):
        a_in = to(g(f"visual.blocks.0.mlp.act_fn#{k}.in0"))
        a_out = g(f"visual.blocks.0.mlp.act_fn#{k}.out")
        a = F.gelu(a_in, approximate="tanh")
        n, u, t = ndiff(a, a_out)
        diff = (a.cpu() != a_out).reshape(-1)
        neg += int((diff & (a_in.cpu().reshape(-1) < 0)).sum())
        nd, dmax, tot = nd + n, max(dmax, u), tot + t
        qlog.detail(f"[{tag} gelu b{k}] {n} of {t} differ, max |Δ| {u:.2g}")
    (good if nd == 0 else qlog.warn)(f"{tag} gelu tanh", f"27 blocks on golden input: {nd} of {tot} differ "
                                     f"({neg} at x < 0, where 1 + tanh cancels), max |Δ| {dmax:.2g}")
    lin("merger fc1", tower.mfc1, to(g("visual.merger.linear_fc1#0.in0")), g("visual.merger.linear_fc1#0.out.0"),
        "NVFP4 W4A16")
    e, msg = exact(F.gelu(to(g("visual.merger.act_fn#0.in0")), approximate="none"), g("visual.merger.act_fn#0.out"))
    (good if e else qlog.warn)(f"{tag} merger gelu", f"erf GELU on golden input: {msg}")
    lin("merger fc2", tower.mfc2, to(g("visual.merger.linear_fc2#0.in0")), g("visual.merger.linear_fc2#0.out.0"),
        "NVFP4 W4A16")


def chain(fn, rank, step, off, emulate=None, ref=None):
    """Our pixels through the whole tower; compare with the golden at every dumped point of this item.
    emulate: None (our tower) or a ref_emulation context (diagnostic); ref: our own output to compare with too."""
    m = media_cache[fn]
    st = Step(rank, step)
    L = m.pixels.shape[0]
    rows = slice(off, off + L)
    got = {}

    def hook(name, val):
        if name in ("hidden0", "patch_embed", "input_norm") or name.endswith(".fc1") or \
                name in ("blocks.25.out", "blocks.26.out") or name in (f"blocks.{k}.out" for k in REF_BLOCKS):
            got[name] = val.detach().to("cpu")

    t0 = time.time()
    if emulate is not None:
        with emulate:
            out = tower.embed([m], hook=hook)
    else:
        out = tower.embed([m], hook=hook)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    kind = "chain" if emulate is None else "emul" if emulate.linears else "noise"
    tag = f"{kind} {fn}"
    if not emulate:
        e, msg = exact(got["input_norm"], st.get("visual.input_norm#0.out")[rows])
        (good if e else fail)(f"{tag} input norm", msg)
        op_line(f"{tag} patch embed", got["patch_embed"],
                st.get("visual.patch_embed.proj#0.out").reshape(-1, 1152)[rows])
        op_line(f"{tag} block-0 input", got["hidden0"], st.get("visual.blocks.0.norm1#0.in0").reshape(-1, 1152)[rows])
    per = []
    for k in range(tower.depth):
        per.append(op_line(f"{tag} b{k:02d} fc1", got[f"blocks.{k}.fc1"],
                           st.get(f"visual.blocks.0.mlp.act_fn#{k}.in0").reshape(-1, tower.inter)[rows], quiet=True))
    pick = (0, 1, 2, 8, 13, 20, 25, 26)
    info(f"{tag} fc1 relL2", " ".join(f"b{k}:{per[k]['rel']:.1e}" for k in pick) + " (all blocks in details)")
    op_line(f"{tag} b26 input", got["blocks.25.out"], st.get("visual.blocks.26.norm1#0.in0").reshape(-1, 1152)[rows])
    op_line(f"{tag} b26 output", got["blocks.26.out"], st.get("visual.merger.norm#0.in0").reshape(-1, 1152)[rows])
    mrows = slice(off // 4, (off + L) // 4)
    what = {"chain": "", "emul": " (diagnostic: reference linears emulated" +
            (", attention with bf16 P)" if dev.type == "cpu" else ", flash_attn)"),
            "noise": " (diagnostic: our tower, only attention with bf16 P)"}[kind]
    m_out = op_line(f"{tag} merger out", out, st.get("visual#0.out")[mrows], what)
    if ref is not None:
        op_line(f"{tag} vs ours", out, ref, " (same tower, attention kernel changed: the tower's own sensitivity)")
    if emulate is None:
        info(f"{tag} time", f"{L} patches → {out.shape[0]} rows on {dev.type}: {dt:.2f} s")
    return out, m_out, dt, got


# ---- high-precision reference (--ref fp32; diagnostic): is our bf16 tower or the golden closer to the exact value? ---
REF_BLOCKS = (0, 6, 13, 20, 26)


class RefFp32:
    """The same tower with fp32 activations everywhere: weights dequantised exactly (the Tower's bf16 weights are exact)
    and used in fp32, fp32 GEMMs (CPU; TF32 off on CUDA) with the NVFP4 global scale and bias in fp32, fp32 LayerNorm /
    GELU / rotary / attention (fp32 SDPA) / residuals. Inputs: the same uint8 pixels; input normalisation, position
    embedding and rotary tables kept in fp32 (no bf16 rounding): x·w + b, the bilinear interpolation (float64 weights
    and sum) and cos / sin of the fp32 angle t·inv_freq, each rounded once to fp32."""

    def __init__(self):
        pc = json.load(open(os.path.join(args.model, "preprocessor_config.json")))
        mean = torch.tensor(pc.get("image_mean", [0.5] * 3), dtype=torch.float32)
        std = torch.tensor(pc.get("image_std", [0.5] * 3), dtype=torch.float32)
        w = std.reciprocal() * pc.get("rescale_factor", 1 / 255)
        self.nw = w.double().repeat_interleave(V.PATCH_DIM // 3).to(dev)  # c-major rows: c = column // 512
        self.nb = (-mean / std).double().repeat_interleave(V.PATCH_DIM // 3).to(dev)

    @staticmethod
    def lin(L, x):
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            y = x @ L.w.float().t()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        if L.g is not None:
            y = y * L.g.float()
        return y + L.b.float() if L.b is not None else y

    @staticmethod
    def ln(x, wb):
        return F.layer_norm(x, (x.shape[-1],), wb[0].float(), wb[1].float(), 1e-6)

    @staticmethod
    def pos(h, w):
        from vision_tower import _merge_order_rc
        n = tower.grid_side
        row, col = _merge_order_rc(h, w)
        hf = row.double() * ((n - 1) / (h - 1) if h > 1 else 0.0)
        wf = col.double() * ((n - 1) / (w - 1) if w > 1 else 0.0)
        hfl, wfl = torch.floor(hf), torch.floor(wf)
        dh, dw = hf - hfl, wf - wfl
        hc = torch.clamp(hfl.long() + 1, max=n - 1)
        wc = torch.clamp(wfl.long() + 1, max=n - 1)
        hfl, wfl = hfl.long(), wfl.long()
        e = tower.pos.cpu().double()
        val = ((1 - dh) * (1 - dw))[:, None] * e[hfl * n + wfl] + ((1 - dh) * dw)[:, None] * e[hfl * n + wc] + \
              (dh * (1 - dw))[:, None] * e[hc * n + wfl] + (dh * dw)[:, None] * e[hc * n + wc]
        return val.float().to(dev)

    @staticmethod
    def rope(h, w):
        from vision_tower import _merge_order_rc
        rd = tower.head_dim // 2
        inv = (1.0 / (10000.0 ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd).double())).float()
        row, col = _merge_order_rc(h, w)
        ang = torch.stack([row, col], 1).float()[:, :, None] * inv  # fp32 angle t·inv_freq, [hw, 2, 18]
        return ang.double().cos().float().flatten(1).to(dev), ang.double().sin().float().flatten(1).to(dev)

    def embed(self, m, hook):
        t, h, w = m.grid
        x = (m.pixels.to(dev).double() * self.nw + self.nb).float()
        x = self.lin(tower.patch, x) + self.pos(h, w).repeat(t, 1)
        cos, sin = (z.repeat(t, 1) for z in self.rope(h, w))
        L = x.shape[0]
        cu = [i * h * w for i in range(t + 1)]
        hd, nh = tower.head_dim, tower.heads
        for i, p in enumerate(tower.blocks):
            qkv = self.lin(p["qkv"], self.ln(x, p["n1"])).view(L, 3, nh, hd)
            half = hd // 2
            c, s = cos[:, None, :], sin[:, None, :]

            def rot(z):
                return torch.cat([z[..., :half] * c - z[..., half:] * s, z[..., :half] * s + z[..., half:] * c], -1)

            q, k, v = rot(qkv[:, 0]), rot(qkv[:, 1]), qkv[:, 2]
            o = torch.empty_like(q)
            for a, b in zip(cu[:-1], cu[1:]):
                o[a:b] = F.scaled_dot_product_attention(*(z[a:b].transpose(0, 1)[None] for z in (q, k, v)),
                                                        scale=hd ** -0.5)[0].transpose(0, 1)
            x = x + self.lin(p["proj"], o.reshape(L, -1))
            f1 = self.lin(p["fc1"], self.ln(x, p["n2"]))
            x = x + self.lin(p["fc2"], F.gelu(f1, approximate="tanh"))
            hook(f"blocks.{i}.fc1", f1)
            hook(f"blocks.{i}.out", x)
        hm = self.ln(x, tower.mnorm).view(-1, tower.hidden * V.MERGE * V.MERGE)
        return self.lin(tower.mfc2, F.gelu(self.lin(tower.mfc1, hm), approximate="none"))


def rowcos(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-300)


def ref_compare(fn, rank, step, off, ours_out, ours_got):
    """(a) our bf16 tower vs the fp32 reference, (b) the golden (reference deployment) vs the fp32 reference;
    fc1 output at REF_BLOCKS (the golden keeps every block's fc1), block-26 output, final embeddings (+ per-row
    cosine); low-cosine rows vs norm."""
    m = media_cache[fn]
    st = Step(rank, step)
    L = m.pixels.shape[0]
    rows, mrows = slice(off, off + L), slice(off // 4, (off + L) // 4)
    ref = {}
    keep = {f"blocks.{k}.{w}" for k in REF_BLOCKS for w in ("fc1", "out")}
    pc = {"ours": [], "golden": []}  # per block: per-patch cosine of the fc1 output vs the fp32 reference

    def hook(name, val):
        if name in keep:
            ref[name] = val.detach().cpu()
        if name.endswith(".fc1"):
            k = int(name.split(".")[1])
            r = val.detach().cpu()
            pc["ours"].append(rowcos(ours_got[name], r))
            pc["golden"].append(rowcos(st.get(f"visual.blocks.0.mlp.act_fn#{k}.in0").reshape(-1, tower.inter)[rows], r))

    t0 = time.time()
    with torch.inference_mode():
        rout = REF.embed(m, hook).cpu()
    dt = time.time() - t0
    gold = {f"fc1 b{k}": st.get(f"visual.blocks.0.mlp.act_fn#{k}.in0").reshape(-1, tower.inter)[rows]
            for k in REF_BLOCKS}
    gold["out b26"] = st.get("visual.merger.norm#0.in0").reshape(-1, tower.hidden)[rows]
    gold["final"] = st.get("visual#0.out")[mrows]
    ours = {f"fc1 b{k}": ours_got[f"blocks.{k}.fc1"] for k in REF_BLOCKS}
    ours["out b26"], ours["final"] = ours_got["blocks.26.out"], ours_out.cpu()
    refs = {f"fc1 b{k}": ref[f"blocks.{k}.fc1"] for k in REF_BLOCKS}
    refs["out b26"], refs["final"] = ref["blocks.26.out"], rout
    res = {}
    for who, src in (("ours", ours), ("golden", gold)):
        ms = {k: cmp(src[k], refs[k]) for k in refs}
        rc = rowcos(src["final"], rout)
        res[who] = (ms, rc)
        line = " ".join(f"{k.replace(' ', '')}:{ms[k]['rel']:.1e}" for k in refs)
        info(f"ref {fn} {who}~fp32", f"relL2 {line} · final cos {ms['final']['cos']:.6f}, row cos mean "
                                     f"{rc.mean():.6f} min {rc.min():.4f}")
        for k in refs:
            qlog.detail(f"[ref {fn} {who}~fp32 {k}] {fmt(ms[k])}")
    for k in REF_BLOCKS:  # our block outputs too (the golden has no per-block output before block 25)
        qlog.detail(f"[ref {fn} ours~fp32 out b{k}] {fmt(cmp(ours_got[f'blocks.{k}.out'], ref[f'blocks.{k}.out']))}")
    # low-cosine rows (ours vs golden): their norm against the item's median row norm
    og = rowcos(ours["final"], gold["final"])
    nrm = rout.double().norm(dim=-1)
    med = nrm.median()
    worst = torch.argsort(og)[:5].tolist()
    lo = nrm <= torch.quantile(nrm, 0.1)
    oc, gc = res["ours"][1], res["golden"][1]
    info(f"ref {fn} low rows", f"ours~golden row cos mean {og.mean():.4f} min {og.min():.4f}; 5 worst rows: "
         + "norm/median " + " ".join(f"{float(nrm[i] / med):.2f}" for i in worst)
         + ", ours~fp32 cos " + " ".join(f"{float(oc[i]):.3f}" for i in worst)
         + ", golden~fp32 cos " + " ".join(f"{float(gc[i]):.3f}" for i in worst)
         + f"; ours~golden mean cos in the lowest-norm 10% {og[lo].mean():.4f} vs rest {og[~lo].mean():.4f}"
         + f" (fp32 ref {dt:.1f} s)")
    # where the worst rows go wrong: per block the lowest fc1 cosine (vs fp32) among the row's 4 patches
    for who in ("golden", "ours"):
        cs = torch.stack(pc[who]).view(len(pc[who]), -1, 4).amin(-1)  # [blocks, rows]
        first = [next((k for k in range(cs.shape[0]) if cs[k, i] < 0.99), None) for i in worst]
        info(f"ref {fn} worst rows {who}", "fc1 cos vs fp32 at b0/6/13/20/26: " + " | ".join(
            "/".join(f"{float(cs[k, i]):.3f}" for k in REF_BLOCKS) for i in worst)
             + f"; first block < 0.99: {first}; patches < 0.99 per block (b0/6/13/20/26): "
             + "/".join(str(int((torch.stack(pc[who])[k] < 0.99).sum())) for k in REF_BLOCKS) + f" of {L}")
    # outlier tokens: per merged row the largest peak/RMS of its 4 patches in the fp32 residual stream
    for k in (6, 26):
        x = ref[f"blocks.{k}.out"].double()
        pk = (x.abs().amax(-1) / x.pow(2).mean(-1).sqrt()).view(-1, 4).amax(-1)
        top = pk >= torch.quantile(pk, 0.95)
        info(f"ref {fn} outliers b{k}", f"residual peak/RMS: median {pk.median():.1f}, 5 worst rows "
             + " ".join(f"{float(pk[i]):.1f}" for i in worst)
             + f"; golden~fp32 row cos in the top-5% peak rows {gc[top].mean():.4f} (min {gc[top].min():.4f}) vs rest "
             + f"{gc[~top].mean():.4f} (min {gc[~top].min():.4f}); ours~fp32 {oc[top].mean():.4f} vs {oc[~top].mean():.4f}")
    return res


done_ops = set()
outs = {}
REF = RefFp32() if args.ref == "fp32" else None
for fn in items:
    rank, step, off = where[fn]
    if (rank, step) not in done_ops and not args.no_ops:
        done_ops.add((rank, step))
        per_op(Step(rank, step), rank, step)
    outs[fn] = chain(fn, rank, step, off)
    if single is not None:  # the same pixels through the whole tower on one GPU (warm-up pass first, then timed)
        m1 = media_cache[fn]
        with torch.inference_mode():
            single.embed([m1])
            torch.cuda.synchronize()
            t1 = time.time()
            one = single.embed([m1])
            torch.cuda.synchronize()
            t_one = time.time() - t1
            t1 = time.time()
            two = tower.embed([m1])
            torch.cuda.synchronize()
            t_two = time.time() - t1
        eq = float((two == one).float().mean())
        op_line(f"tp {fn} vs one GPU", two, one, f" · equal {eq:.4%} · {m1.pixels.shape[0]} patches: one GPU "
                f"{t_one:.2f} s, two ranks {t_two:.2f} s")
    if REF is not None:
        ref_compare(fn, rank, step, off, outs[fn][0], outs[fn][3])
    if not args.no_emulate:
        chain(fn, rank, step, off, emulate=ref_emulation())
        if dev.type == "cpu":
            chain(fn, rank, step, off, emulate=ref_emulation(linears=False), ref=outs[fn][0])

# final per-item embeddings vs the golden embed_multimodal (all-gathered) and cached cases via embed_input_ids rows
for name, c in cases.items():
    r = case_req.get(name)
    if r is None:
        continue
    st = Step(0, r["step"])
    gm = st.get("Qwen4ExpForConditionalGeneration.fn.embed_input_ids#0.in.is_multimodal")
    ret = st.get("Qwen4ExpForConditionalGeneration.fn.embed_input_ids#0.ret")
    o = 0
    for i, fn in enumerate(c["files"]):
        m = media_cache[fn]
        if fn not in outs:
            o += m.tokens
            continue
        gold = st.get(f"Qwen4ExpForConditionalGeneration.fn.embed_multimodal#0.ret.{i}")
        src = "embed_multimodal"
        if gold is None:
            gold = ret[gm[:ret.shape[0]]][o:o + m.tokens]
            src = "embed_input_ids (encoder cache)"
        op_line(f"{name} final {fn}", outs[fn][0], gold, f" vs golden {src}")
        o += m.tokens

if TP is not None:
    TP.close()  # rank 1 leaves
qlog.info("summary", f"{bad} exact check(s) failed" if bad else "all exact checks passed")
sys.exit(1 if bad else 0)
