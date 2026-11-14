"""Block verification study (run through tools/tp2.sh): expected tokens per round of block verification (Sun et al. 2024,
arXiv 2403.10444) against today's token-by-token speculative sampling, on the same sampled draft chains.

Along a sampled trajectory (server defaults T 1.0 · top_k 20 · top_p 0.95), at every --stride-th position: --chains draft
chains of --depth tokens sampled from the head (q̃, the drafts' real distribution), each verified by the target (verify +
commit(0): no state change), then for each chain and each depth k ≤ --depth, with P_i / Q_i the truncated target / draft
rows of draft i and x_i the drafts:
  token by token   E = 1 + Σ_{i≤k} Π_{j≤i} min(1, P_j(x_j) / Q_j(x_j))                (the accept test of kernels/sample)
  block            p_0 = 1, p_i = min(1, p_{i−1} · P_i(x_i) / Q_i(x_i));
                   h_i = R_i / (R_i + 1 − p_i), R_i = Σ_v max(p_i · P_{i+1}(v) − Q_{i+1}(v), 0) (i < k), h_k = p_k;
                   τ = the largest i with η_i ≤ h_i (0 if none) → P(τ = i) = h_i · Π_{j>i} (1 − h_j); E = 1 + E[τ]
Both are exact expectations given the chain (the uniforms integrated out); averaged over chains and positions (per-position
means: both methods are compared at the same positions). Screen: per prompt and depth, tokens per round of both and the
gain. Output (rank 0): /work/results/tests/bstudy/block_study.json
  tools/tp2.sh bstudy python3 engine/tests/block_study.py [--decode 600 --stride 1 --chains 6 --depth 8]"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "tools"), os.path.join(ROOT, "server")]
import qlog  # noqa: E402

import ext  # noqa: E402
from chat import Chat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--decode", type=int, default=600)
ap.add_argument("--skip", type=int, default=300, help="trajectory tokens before the first studied position")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--chains", type=int, default=6)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--chunk", type=int, default=2048)
a = ap.parse_args()
MD = "/models/Qwen3.8-Flash-Next-FP4-Local"
OUT = "/work/results/tests/bstudy"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
chat = Chat(MD)
bench_doc = (open(f"{MD}/README.md").read() + open(f"{MD}/chat_template.jinja").read()) * 2
PROMPTS = [
    ("think en bench", None,
     "[0f3a9c]\n" + bench_doc[:36000] + "\n\nBased on the document above, explain in detail how its "
     "parts fit together and what a reader should check first."),
    ("prose ko", "none", "낡은 시계방 주인이 낯선 손님의 시계를 고쳐 주는 이야기를 800단어 정도의 단편 소설로 써 줘."),
]
toks_of = [chat.encode(chat.render([{"role": "user", "content": p}], reasoning_effort=e)) for _, e, p in PROMPTS]
K = a.depth
eng.init(MD, "/work/results/tensor_table.tsv", a.rank, os.environ["Q38_HEAD_IP"], 1,
         (max(len(t) for t in toks_of) + a.skip + a.decode + 2 * K + 64 + 127) // 64 * 64, a.chunk, mtp=True)
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)
T, TOPK, TOPP = 1.0, 20, 0.95
V = 248320


def trunc(lg):
    """truncated, renormalised distribution as a dense fp64 vector over the top-k ids (ids, probs)"""
    v, i = torch.topk(lg.float() / T, TOPK)
    p = torch.softmax(v.double(), -1)
    keep = (torch.cumsum(p, -1) - p) < TOPP
    p = torch.where(keep, p, torch.zeros_like(p))
    return i, p / p.sum()


def prob(d, t):
    hit = (d[0] == t).nonzero()
    return d[1][hit[0, 0]].item() if hit.numel() else 0.0


def resid_mass(pi, P, Q):
    """Σ_v max(pi·P(v) − Q(v), 0) over the union of both supports"""
    ids = torch.cat([P[0], Q[0]]).unique()
    pv = torch.zeros(len(ids), dtype=torch.float64, device=ids.device)
    qv = torch.zeros_like(pv)
    pos = {int(t): j for j, t in enumerate(ids.tolist())}
    pv[[pos[int(t)] for t in P[0].tolist()]] = P[1]
    qv[[pos[int(t)] for t in Q[0].tolist()]] = Q[1]
    return torch.clamp(pi * pv - qv, min=0).sum().item()


def prefill(toks):
    L = len(toks)
    eng.reset(0)
    for c0 in range(0, L - 1, a.chunk):
        n = min(a.chunk, L - 1 - c0)
        eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [0], [0, n], False)
        eng.mtp_step(toks[c0 + 1:c0 + 1 + n], list(range(c0, c0 + n)), [0], [0, n], list(range(n)), True, False)
    return eng.step([toks[L - 1]], [L - 1], [0], [0, 1])[0]


res = []
gen = torch.Generator(device="cuda")
for pi_, (kind, eff, _) in enumerate(PROMPTS):
    toks = toks_of[pi_]
    L = len(toks)
    gen.manual_seed(3000 + pi_)
    lg = prefill(toks)
    # trajectory: the target state walks along it; the head's KV over it is written by the from-target pass at each step
    x, per = [], []  # per[s] = ([tbt E_k], [block E_k]) for k = 1..K, averaged over chains
    N = a.skip + a.decode
    for s in range(N):
        P = L - 1 + s  # the target has run through position P; its hidden there is X_ row 0
        d0 = trunc(lg)
        t = int(d0[0][torch.multinomial(d0[1].float(), 1, generator=gen)].item())
        eng.mtp_step([t], [P], [0], [0, 1], [0], True, False)  # the head's KV at P from the target (as in decoding)
        if s >= a.skip and (s - a.skip) % a.stride == 0:
            eng.hold_hidden(1)  # the target's hidden at P (verify below overwrites the step buffer)
            acc_t, acc_b = [0.0] * K, [0.0] * K
            for _ in range(a.chains):
                # draft chain: pass 1 from the target's hidden at P with token t (position P + 1), then the head's own
                ql = eng.mtp_step_held([t], [P], [0], [0, 1], [0])[0]
                xs, Qs = [], []
                for i in range(K):
                    q = trunc(ql)
                    d = int(q[0][torch.multinomial(q[1].float(), 1, generator=gen)].item())
                    xs.append(d)
                    Qs.append(q)
                    if i + 1 < K:
                        ql = eng.mtp_step([d], [P + 1 + i], [0], [0, 1], [0], False, False)[0]
                # one more head pass for Q_{K+1} is not needed (h_K = p_K); the target over t, x_1 … x_K
                vl = eng.verify([t] + xs, list(range(P + 1, P + K + 2)), [0], [0, K + 1])
                eng.commit([0])
                Ps = [trunc(vl[i]) for i in range(K + 1)]  # Ps[i] = target row predicting draft i+1 (Ps[K]: bonus)
                r = [min(1.0, prob(Ps[i], xs[i]) / max(prob(Qs[i], xs[i]), 1e-300)) for i in range(K)]
                pb, pbs = 1.0, []
                for i in range(K):
                    pb = min(1.0, pb * prob(Ps[i], xs[i]) / max(prob(Qs[i], xs[i]), 1e-300))
                    pbs.append(pb)
                Rs = [resid_mass(pbs[i], Ps[i + 1], Qs[i + 1]) for i in range(K - 1)]
                for k in range(1, K + 1):
                    e, pr = 1.0, 1.0
                    for i in range(k):
                        pr *= r[i]
                        e += pr
                    acc_t[k - 1] += e
                    h = [Rs[i] / (Rs[i] + 1 - pbs[i]) if Rs[i] + 1 - pbs[i] > 0 else 1.0 for i in range(k - 1)] + [pbs[k - 1]]
                    et, tail = 0.0, 1.0
                    for i in range(k - 1, -1, -1):  # P(τ = i+1) = h_i · Π_{j>i} (1 − h_j)
                        et += (i + 1) * h[i] * tail
                        tail *= 1 - h[i]
                    acc_b[k - 1] += 1 + et
            per.append(([v / a.chains for v in acc_t], [v / a.chains for v in acc_b]))
        x.append(t)
        if t in chat.eos:
            break
        lg = eng.step([t], [P + 1], [0], [0, 1])[0]
    n = len(per)
    if n == 0:
        say(qlog.warn, f"bstudy {kind}", "no positions (the answer ended early)")
        continue
    mt = [sum(p[0][k] for p in per) / n for k in range(K)]
    mb = [sum(p[1][k] for p in per) / n for k in range(K)]
    res.append({"kind": kind, "positions": n, "tbt": mt, "block": mb, "per": per})
    say(qlog.ok, f"bstudy {kind}",
        f"{n} positions × {a.chains} chains · tokens/round per depth (token by token → block): " + " · ".join(
            f"k{k + 1} {mt[k]:.2f}→{mb[k]:.2f} ({100 * (mb[k] / mt[k] - 1):+.1f}%)" for k in range(1, K)))
if a.rank == 0:
    json.dump({"depth": K, "chains": a.chains, "runs": res}, open(f"{OUT}/block_study.json", "w"))
sys.stdout.flush()
os._exit(0)
