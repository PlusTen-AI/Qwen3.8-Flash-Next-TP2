"""MTP vs DFlash2 drafts under the server's sampling (T 1.0 · top_k 20 · top_p 0.95), measured offline (run through
tools/tp2.sh). Each prompt: a trajectory sampled from the target (plain decoding), then a replay along it. At every
position P (the target has run through P; x_{P+1} is the next emitted token, the round's anchor), each drafter makes
its drafts exactly as in serving — sampled from its own q — and the target verifies them (Engine::verify on
[x_{P+1}, d_1, …], then commit: 0 rows, or 1 row = x_{P+1} for the last verify, which advances the replay):
  MTP      k sampled drafts from the head chain; q = the engine's truncated draft distribution (P006 NVFP4 lm_head copy)
  DFlash2  block b: b − 1 drafts from engine/tests/dflash2_ref.py (sampled lattice walk, q over 16 candidates)
For drafts d with q_i(d_i) and the verified, prefix-conditioned target p̃_i(d_i) (truncated as the server does), the
chance that the first i drafts are all accepted is Π_{j≤i} min(1, p̃_j(d_j) / q_j(d_j)); averaged over positions this
is an unbiased estimate of the expected tokens per round 1 + Σ_i Π_{j≤i} a_j (no accept-draw noise).
Target features for DFlash2 (its training capture): the completed (materialised) multi-stream state after zero-based
layers 5, 15, 25, 35, 45, averaged over the 4 streams, from the engine taps. Both ranks compute identical drafts.
  tools/tp2.sh dflash python3 engine/tests/dflash_study.py [--decode 300 --blocks 4,8]"""
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "tools"), os.path.join(ROOT, "server")]
import qlog  # noqa: E402

import ext  # noqa: E402
from chat import Chat  # noqa: E402
from dflash2_ref import ContextKV, DFlash2Ref  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--decode", type=int, default=300)
ap.add_argument("--depth", type=int, default=4, help="MTP drafts per round")
ap.add_argument("--blocks", default="4,8", help="DFlash2 block sizes (drafts = block − 1)")
ap.add_argument("--only", default="", help="prompt indices, comma list ('' = all)")
ap.add_argument("--layer-shift", type=int, default=0, help="capture the state after layer i + shift (feature check)")
ap.add_argument("--no-emulate", action="store_true", help="drafter in fp32 without bf16 rounding points")
a = ap.parse_args()
MD = "/models/Qwen3.8-Flash-Next-FP4-Local"
DD = "/models/Qwen3.8-Flash-Next-DFlash2"
OUT = "/work/results/tests/dflash"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
chat = Chat(MD)
PROMPTS = [
    ("think en", True,
     "Write a reflective essay of about 800 words on what it means to grow up in a small coastal town."),
    ("think ko", True, "작은 바닷가 마을에서 자란다는 것이 어떤 의미인지에 대해 800단어 정도의 수필을 써 줘."),
    ("prose en", False,
     "Write a short story of about 800 words about an old clockmaker who repairs a stranger's watch."),
    ("prose ko", False, "낡은 시계방 주인이 낯선 손님의 시계를 고쳐 주는 이야기를 800단어 정도의 단편 소설로 써 줘."),
    ("code", True, "Write a Python function that parses an Apache access log line into a dict, with pytest tests."),
]
sel = [int(x) for x in a.only.split(",")] if a.only else list(range(len(PROMPTS)))
toks_of = {i: chat.encode(chat.render([{"role": "user", "content": PROMPTS[i][2]}],
                                      reasoning_effort=None if PROMPTS[i][1] else "none")) for i in sel}
blocks = [int(x) for x in a.blocks.split(",")]
K = a.depth
KB = max(blocks + [K + 1])
L_max = max(len(t) for t in toks_of.values())
CH = 64  # prefill rows per step: taps need M ≤ logit_rows
eng.init(MD, "/work/results/tensor_table.tsv", a.rank, os.environ["Q38_HEAD_IP"], 1,
         (L_max + a.decode + 2 * KB + 64 + 127) // 64 * 64, CH, mtp=True)
eng.set_taps(True)
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)
T, TOPK, TOPP = 1.0, 20, 0.95
LAYERS = [i + a.layer_shift for i in (5, 15, 25, 35, 45)]

# target embedding / lm_head (full tensors) for the drafter
tab = {l.split("\t")[0]: l.split("\t")[1] for l in open("/work/results/tensor_table.tsv") if l.count("\t") > 2}
def full(name):
    with safe_open(os.path.join(MD, tab[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)
EMB = full("model.language_model.embed_tokens.weight")  # [V, 2560] bf16
LMH = full("lm_head.weight")
dw = {}
with safe_open(os.path.join(DD, "model.safetensors"), "pt", device="cuda") as f:
    for k in f.keys():
        dw[k] = f.get_tensor(k)
dcfg = json.load(open(os.path.join(DD, "config.json")))
LMHF = LMH.float()
drafter = DFlash2Ref(dw, dcfg, embed=lambda ids: EMB[ids], lm_head=lambda h: h.float() @ LMHF.t(),
                     emulate_bf16=not a.no_emulate)


def trunc_probs(lg, temp, topk, topp):
    v, i = torch.topk(lg.float() / temp, topk)
    p = torch.softmax(v, -1)
    c = torch.cumsum(p, -1)
    p = torch.where((c - p) < topp, p, torch.zeros_like(p))
    return i, p / p.sum()


def overlap(pd, qd):
    pi, pp = pd
    qi, qp = qd
    m = (pi.unsqueeze(1) == qi.unsqueeze(0))
    return torch.minimum(pp.unsqueeze(1).expand(m.shape), qp.unsqueeze(0).expand(m.shape))[m].sum().item()


def prob_of(pd, t):
    pi, pp = pd
    hit = (pi == t).nonzero()
    return pp[hit[0, 0]].item() if hit.numel() else 0.0


def feat_rows(rows):
    """[rows, 12800] target features of the last step's first `rows` rows (stream mean, bf16 as the capture)"""
    fs = [eng.tap(f"X.{i}", rows, 10240).unflatten(-1, (4, 2560)).mean(dim=-2) for i in LAYERS]
    return torch.cat(fs, -1)


def prefill(toks):
    """rows 0 … L−2 in 64-row steps with the head (next token known) and taps; then the last token alone"""
    L = len(toks)
    eng.reset(0)
    feats = []
    for c0 in range(0, L - 1, CH):
        n = min(CH, L - 1 - c0)
        eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [0], [0, n], False)
        feats.append(feat_rows(n))
        eng.mtp_step(toks[c0 + 1:c0 + 1 + n], list(range(c0, c0 + n)), [0], [0, n], list(range(n)), True, False)
    lg = eng.step([toks[L - 1]], [L - 1], [0], [0, 1])[0]
    feats.append(feat_rows(1))
    return lg, torch.cat(feats, 0)


res = {}
gen = torch.Generator(device="cuda")
dgen = torch.Generator(device="cuda")


def add_ctx(ctx, f_rows, pos0):
    """context K/V of new feature rows (positions pos0 …), appended (SGLang caches it across rounds)"""
    new = drafter.prepare_context(f_rows, torch.arange(pos0, pos0 + f_rows.shape[0], device="cuda"))
    if ctx is None:
        return new
    return ContextKV(pos=torch.cat([ctx.pos, new.pos]), k=[torch.cat([a_, b_]) for a_, b_ in zip(ctx.k, new.k)],
                     v=[torch.cat([a_, b_]) for a_, b_ in zip(ctx.v, new.v)])


def chain(lgs, d, qd):
    """[min(1, p̃_i(d_i) / q_i(d_i))] for verify rows lgs (row i predicts d_i)"""
    out = []
    for i, t in enumerate(d):
        pp = prob_of(trunc_probs(lgs[i], T, TOPK, TOPP), t)
        out.append(min(1.0, pp / qd[i]) if qd[i] > 0 else 0.0)
    return out


for pi_ in sel:
    kind = PROMPTS[pi_][0]
    toks = toks_of[pi_]
    L = len(toks)
    # 1. trajectory
    gen.manual_seed(1000 + pi_)
    lg, _ = prefill(toks)
    x = []
    for s in range(a.decode + 1):
        idx, pr = trunc_probs(lg, T, TOPK, TOPP)
        t = int(idx[torch.multinomial(pr, 1, generator=gen)].item())
        x.append(t)
        lg = eng.step([t], [L + s], [0], [0, 1])[0]
    # 2. replay: x[s] is the token at position L + s; P = L − 1 + s
    dgen.manual_seed(2000 + pi_)
    _, feats = prefill(toks)
    ctx = add_ctx(None, feats, 0)
    al = {"mtp": [], **{f"df{b}": [] for b in blocks}}
    for s in range(a.decode):
        P = L - 1 + s
        # MTP: sampled chain from the target's hidden at P (row 0 of the last target step)
        d, qd = [], []
        lgh = eng.mtp_step([x[s]], [P], [0], [0, 1], [0], True, False)[0]
        for i in range(K):
            idx, pr = trunc_probs(lgh, T, TOPK, TOPP)
            j = int(torch.multinomial(pr, 1, generator=dgen).item())
            d.append(int(idx[j].item()))
            qd.append(float(pr[j].item()))
            if i + 1 < K:
                lgh = eng.mtp_step([d[-1]], [P + 1 + i], [0], [0, 1], [0], False, False)[0]
        vl = eng.verify([x[s]] + d, list(range(P + 1, P + 2 + K)), [0], [0, K + 1])
        al["mtp"].append(chain(vl, d, qd))
        eng.commit([0])
        # DFlash2 blocks; the last verify keeps row 0 (x[s] at P + 1) and gives the features of P + 1
        for bi, b in enumerate(blocks):
            u = torch.rand(b - 1, generator=dgen, device="cuda")
            r = drafter.draft(None, None, x[s], P + 1, b, temperature=T, top_k=TOPK, uniforms=u, ctx_kv=ctx)
            d = [int(t_) for t_ in r.tokens.tolist()]
            qd = [float(r.q_rows[i, r.path_index[i]].item()) for i in range(b - 1)]
            vl = eng.verify([x[s]] + d, list(range(P + 1, P + 1 + b)), [0], [0, b])
            al[f"df{b}"].append(chain(vl, d, qd))
            last = bi + 1 == len(blocks)
            if last:
                ctx = add_ctx(ctx, feat_rows(1), P + 1)
            eng.commit([1 if last else 0])
    rr = res.setdefault(kind, {k_: [] for k_ in al})
    for k_, v_ in al.items():
        rr[k_] += v_
    say(qlog.info, f"dflash {kind}", f"#{pi_} input {L:,} · positions {a.decode}")


def tpr(al):
    tot = 0.0
    for v in al:
        p = 1.0
        for x_ in v:
            p *= x_
            tot += p
    return 1 + tot / len(al)


def mean_a(al):
    return "/".join(f"{sum(v[i] for v in al) / len(al):.2f}" for i in range(len(al[0])))


for kind, r in res.items():
    parts = [f"MTP k{K} {tpr(r['mtp']):.2f} (a {mean_a(r['mtp'])})"]
    parts += [f"DFlash2 b{b} {tpr(r[f'df{b}']):.2f} (a {mean_a(r[f'df{b}'])})" for b in blocks]
    say(qlog.ok, f"dflash {kind}",
        f"{len(r['mtp']):,} positions · feature layers {LAYERS} · expected tokens per round " + " | ".join(parts))
json.dump({k: {m: v for m, v in r.items()} for k, r in res.items()}, open(f"{OUT}/dflash{a.rank}.json", "w"))
sys.stdout.flush()
os._exit(0)
