"""MTP head check (run through tools/tp2.sh): draft quality of the MTP head against the target's own greedy decode.

Prefill a golden prompt (chunks), with the MTP head run over every chunk (its KV cache needs every position: row p gets
the target's hidden at p and the token at p + 1). Then plain greedy decode N tokens; at each position the head drafts
a chain of --depth tokens (first pass from the target's hidden, then from its own) and the chain is compared with the
tokens the target actually produces. A broken head gives ~0 %; the reference deployment serves this head at depth
up to 12.
One line per prompt (rank 0): acceptance of draft i given drafts 1..i−1 accepted, mean accepted length at depth k.
  tools/tp2.sh mtp python3 engine/tests/mtp_check.py [--decode 96 --depth 6]"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
import qlog  # noqa: E402

import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--only", default="short,long")
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--decode", type=int, default=96)
ap.add_argument("--depth", type=int, default=6)
a = ap.parse_args()
OUT = "/work/results/tests/mtp"
os.makedirs(OUT, exist_ok=True)
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
gold = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))
L_max = max(len(gold[k]["prompt_ids"]) for k in a.only.split(","))
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], 1, (L_max + a.decode + a.depth + 127) // 64 * 64, a.chunk, mtp=True)
log = open(f"{OUT}/mtp{a.rank}.log", "w")


def am(lg):
    return int(lg.argmax())


for name in a.only.split(","):
    toks = gold[name]["prompt_ids"]
    L = len(toks)
    eng.reset(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for c0 in range(0, L, a.chunk):
        n = min(a.chunk, L - c0)
        print(f"{name}: step rows {c0}..{c0 + n}", file=log, flush=True)
        lg = eng.step(toks[c0:c0 + n], list(range(c0, c0 + n)), [0], [0, n], False)
        m = n if c0 + n < L else n - 1  # the last prompt row waits for the sampled token
        if m > 0:
            print(f"{name}: mtp rows {c0}..{c0 + m}", file=log, flush=True)
            eng.mtp_step(toks[c0 + 1:c0 + 1 + m], list(range(c0, c0 + m)), [0], [0, m], list(range(m)), True, False)
    last_row = (L - 1) - (L - 1) // a.chunk * a.chunk
    torch.cuda.synchronize()
    t_pf = time.perf_counter() - t0
    # greedy decode: seq[i] = token at position L + i
    seq = [am(lg[-1])]
    chains = []  # chains[i] = drafts for positions L+i+1 … made after the target produced seq[i]
    src, from_t = [last_row], True
    t_mtp = t_dec = 0.0
    for i in range(a.decode):
        p = L + i - 1  # position of the hidden the head starts from (the row that produced seq[i])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ch, tok = [], seq[i]
        for d in range(a.depth):
            dl = eng.mtp_step([tok], [p + d], [0], [0, 1], src if d == 0 else [0], from_t if d == 0 else False, False)
            tok = am(dl[-1])
            ch.append(tok)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        chains.append(ch)
        lg = eng.step([seq[i]], [L + i], [0], [0, 1])
        seq.append(am(lg[0]))
        torch.cuda.synchronize()
        t_dec += time.perf_counter() - t2
        t_mtp += t2 - t1
        src, from_t = [0], True
    # acceptance: chain i drafts positions L+i+1 … ; truth seq[i+1 …]
    acc = [[0, 0] for _ in range(a.depth)]  # [accepted, tried] for draft d given d−1 accepted
    lens = []
    for i, ch in enumerate(chains):
        k = 0
        for d in range(a.depth):
            if i + 1 + d >= len(seq):
                break
            acc[d][1] += 1
            if ch[d] != seq[i + 1 + d]:
                break
            acc[d][0] += 1
            k += 1
        if i + a.depth < len(seq):
            lens.append(k)
    rates = [f"{100 * x / max(n, 1):.0f}%" for x, n in acc]
    mean = sum(lens) / max(len(lens), 1)
    gold_out = gold[name]["output_ids"]
    same = next((j for j, (x, y) in enumerate(zip(seq, gold_out)) if x != y), min(len(seq), len(gold_out)))
    print(f"{name}: chains {chains[:4]} seq {seq[:12]} acc {acc} lens {lens}", file=log, flush=True)
    if a.rank == 0:
        qlog.ok(f"mtp {name}",
                f"input {L:,} · decode {a.decode} · draft acceptance {' · '.join(rates)} (depth 1…{a.depth}) · "
                f"mean accepted {mean:.2f} tokens/step · MTP {1e3 * t_mtp / a.decode / a.depth:.1f} ms/draft · "
                f"decode {1e3 * t_dec / a.decode:.1f} ms/step · first {same} tokens equal the reference output")
sys.stdout.flush()
os._exit(0)
