"""Draft distribution study (run through tools/tp2.sh): which draft q gives the most accepted tokens, measured without
sampling noise. Speculative sampling is exact for any q (the accept test uses the q a draft came from), so the draft's
temperature and truncation are free choices; only acceptance changes.

1. Trajectory: plain decoding of each prompt (chat template, thinking on/off) sampled from the target with the
   server defaults (T 1.0 · top_k 20 · top_p 0.95), --decode tokens.
2. Replay along it: at each position P the target step, then the head chain with teacher forcing (pass 1 from the
   target's hidden at P with token x_{P+1}, passes 2…k from the head's own hidden with x_{P+i}) — the contexts an
   accepted chain sees (accepted tokens are distributed as p). For every variant (draft temperature ratio r = T_d/T,
   draft top-k') the acceptance probability α_{P,i} = Σ_v min(p̃_{P+i}(v), q̃_{P,i}(v)) (p̃, q̃: truncated, renormalised).
   Expected tokens per round = 1 + Σ_i Π_{j≤i} α_{P,j}, averaged over P.
Screen (rank 0): per prompt kind and variant, α per depth and tokens/round.
  tools/tp2.sh dtemp python3 engine/tests/draft_temp.py [--decode 400 --depth 4]"""
import argparse
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
ap.add_argument("--decode", type=int, default=400)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--ratios", default="1.0,0.9,0.8,0.7,0.6,0.5")
ap.add_argument("--qtopk", default="20,10,5")
a = ap.parse_args()
MD = "/models/Qwen3.8-Flash-Next-FP4-Local"
OUT = "/work/results/tests/dtemp"
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
    ("prose en", False,
     "Write an 800-word opinion piece on why public libraries still matter in the age of the internet."),
    ("prose ko", False, "인터넷 시대에도 공공 도서관이 여전히 중요한 이유에 대해 800단어 정도의 칼럼을 써 줘."),
    ("code", True, "Write a Python function that parses an Apache access log line into a dict, with pytest tests."),
]
toks_of = {i: chat.encode(chat.render([{"role": "user", "content": p}], reasoning_effort=None if th else "none"))
           for i, (_, th, p) in enumerate(PROMPTS)}
L_max = max(len(t) for t in toks_of.values())
K = a.depth
eng.init(MD, "/work/results/tensor_table.tsv", a.rank, os.environ["Q38_HEAD_IP"], 1,
         (L_max + a.decode + K + 64 + 127) // 64 * 64, a.chunk, mtp=True)
say = (lambda f, t, m: f(t, m)) if a.rank == 0 else (lambda f, t, m: None)
T, TOPK, TOPP = 1.0, 20, 0.95


def trunc_probs(lg, temp, topk, topp):
    """truncated, renormalised distribution as (indices [topk], probs [topk]) — top-k then top-p (≥ 1 token)"""
    v, i = torch.topk(lg.float() / temp, topk)
    p = torch.softmax(v, -1)  # = softmax over the top-k (renormalised)
    c = torch.cumsum(p, -1)
    keep = (c - p) < topp
    p = torch.where(keep, p, torch.zeros_like(p))
    return i, p / p.sum()


def overlap(pd, qd):
    pi, pp = pd
    qi, qp = qd
    m = (pi.unsqueeze(1) == qi.unsqueeze(0)).float()  # [kp, kq]
    return torch.minimum(pp.unsqueeze(1).expand_as(m), qp.unsqueeze(0).expand_as(m))[m.bool()].sum().item()


def prefill(toks):
    """all rows but the last with the head (next token known), then the last token alone (its hidden in row 0)"""
    L = len(toks)
    eng.reset(0)
    for c0 in range(0, L - 1, a.chunk):
        n = min(a.chunk, L - 1 - c0)
        eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [0], [0, n], False)
        eng.mtp_step(toks[c0 + 1:c0 + 1 + n], list(range(c0, c0 + n)), [0], [0, n], list(range(n)), True, False)
    return eng.step([toks[L - 1]], [L - 1], [0], [0, 1])[0]


ratios = [float(x) for x in a.ratios.split(",")]
qtopk = [int(x) for x in a.qtopk.split(",")]
variants = [(r, 20) for r in ratios] + [(1.0, k) for k in qtopk if k != 20]
acc = {}  # kind → variant → list of per-position α vectors
gen = torch.Generator(device="cuda")
for pi_, (kind, think, _) in enumerate(PROMPTS):
    toks = toks_of[pi_]
    L = len(toks)
    # 1. trajectory (plain sampled decode)
    gen.manual_seed(1000 + pi_)
    lg = prefill(toks)
    x = []
    for s in range(a.decode + K):
        idx, pr = trunc_probs(lg, T, TOPK, TOPP)
        t = int(idx[torch.multinomial(pr, 1, generator=gen)].item())
        x.append(t)
        lg = eng.step([t], [L + s], [0], [0, 1])[0]
    # 2. replay with teacher-forced head chains; p̃ rows kept for k steps
    lg = prefill(toks)
    P_rows = [trunc_probs(lg, T, TOPK, TOPP)]  # P_rows[s] = p̃ at position L−1+s (predicts x[s])
    Q = []  # Q[s] = raw head logits [K] made at position L−1+s (draft i predicts x[s+i])
    for s in range(a.decode):
        P = L - 1 + s
        ql = [eng.mtp_step([x[s]], [P], [0], [0, 1], [0], True, False)[0].clone()]
        for i in range(1, K):
            ql.append(eng.mtp_step([x[s + i]], [P + i], [0], [0, 1], [0], False, False)[0].clone())
        Q.append(ql)
        lg = eng.step([x[s]], [P + 1], [0], [0, 1])[0]
        P_rows.append(trunc_probs(lg, T, TOPK, TOPP))
    # the target rows the last chains are compared with (up to x[decode + K − 1])
    for s in range(a.decode, a.decode + K - 1):
        lg = eng.step([x[s]], [L + s], [0], [0, 1])[0]
        P_rows.append(trunc_probs(lg, T, TOPK, TOPP))
    for var in variants:
        r, k2 = var
        al = []
        for s in range(a.decode):
            al.append([overlap(P_rows[s + 1 + i], trunc_probs(Q[s][i], T * r, k2, TOPP)) for i in range(K)])
        acc.setdefault(kind, {}).setdefault(var, []).extend(al)
    say(qlog.info, f"dtemp {kind}", f"#{pi_} input {L:,} · positions {a.decode}")


def summary(al):
    n = len(al)
    mean_a = [sum(v[i] for v in al) / n for i in range(K)]
    tpr = 1 + sum(sum(_prod(v[:i + 1]) for i in range(K)) for v in al) / n
    return mean_a, tpr


def _prod(v):
    out = 1.0
    for x in v:
        out *= x
    return out


for kind, d in acc.items():
    base = summary(d[(1.0, 20)])[1]
    parts = []
    for var in variants:
        ma, tpr = summary(d[var])
        parts.append(f"r {var[0]:.1f}·k{var[1]} {tpr:.2f} ({100 * (tpr / base - 1):+.1f}%) α " +
                     "/".join(f"{x:.2f}" for x in ma))
    say(qlog.ok, f"dtemp {kind}",
        f"{len(next(iter(d.values()))):,} positions · expected tokens per round " + " | ".join(parts))
sys.stdout.flush()
os._exit(0)
