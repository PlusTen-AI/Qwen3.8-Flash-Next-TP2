"""End-to-end check of the decode step against the golden dump (run through tools/tp2.sh on C and D).

For each golden request (short, long) the engine is teacher-forced: the prompt is fed in chunks of --chunk rows
(multi-token steps of one sequence), then each golden output token one step at a time. Compared on rank 0:
  - logits at every golden decode step and at the last prompt row: top-1 agreement, max |Δlogit|, total variation of
    the softmax distributions, and the same numbers for the reference vs itself (short vs short2) as the noise floor;
  - per-layer materialised state (taps "X.<i>") vs golden `layers.<i>#0.resid` at the decode steps (cosine, max rel).
One line per request on screen; per-step and per-layer details in /work/results/tests/e2e/e2e.log.
"""
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
from golden import Golden  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--dump", default="/work/results/raw/dump/20261111-1632")
ap.add_argument("--chunks", default="64,2048", help="prompt chunkings; logits must be bitwise equal across them")
ap.add_argument("--only", default="short,long")
a = ap.parse_args()
OUT = "/work/results/tests/e2e"
os.makedirs(OUT, exist_ok=True)
qlog.detail_file(f"{OUT}/e2e{a.rank}.log")
srcs = ext.engine_sources()
eng = ext.load("engine", srcs, log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
CH = [int(x) for x in a.chunks.split(",")]
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank,
         os.environ["Q38_HEAD_IP"], 1, 16384, max(CH))
tok = json.load(open(f"{a.dump}/tokens.json"))
g = Golden(f"{a.dump}/rank0", f"{a.dump}/tokens.json") if a.rank == 0 else None
LG = "language_model.logits_processor#0.out"


def golden_steps(name):
    """golden step ids of request `name` in order (prefill first)."""
    order = {"short": 0, "long": 1, "short2": 2}[name]
    seen = []
    for s in g.steps:
        rid = next(iter(s["reqs"]))
        if not seen or seen[-1][0] != rid:
            seen.append([rid, []])
        seen[-1][1].append(s["step"])
    return seen[order][1]


def dist(x, y):
    p, q = torch.softmax(x.float(), -1), torch.softmax(y.float(), -1)
    return 0.5 * (p - q).abs().sum().item()


first = {}
for name, chunk in [(n, c) for n in a.only.split(",") for c in CH]:
    r = tok[name]
    ids, outs = r["prompt_ids"], r["output_ids"]
    eng.reset(0)
    torch.cuda.synchronize()
    t0 = time.time()
    # prompt in chunks (teacher forcing is exact here: the prompt is fixed); logits of the last row only
    p = 0
    eng.set_taps(False)
    while p < len(ids):
        n = min(chunk, len(ids) - p)
        lg = eng.step(ids[p:p + n], list(range(p, p + n)), [0], [0, n], False)
        p += n
    last_prompt = lg[-1]
    torch.cuda.synchronize()
    t_prompt = time.time() - t0
    res = {"top1": 0, "n": 0, "dl": 0.0, "tv": 0.0, "cos_min": 1.0}
    steps = golden_steps(name) if g else []
    # decode: feed golden outputs one by one (output j is produced by the logits after token j-1)
    logits = [last_prompt]
    eng.set_taps(a.rank == 0)
    for j, t in enumerate(outs[:-1]):
        lg = eng.step([t], [len(ids) + j], [0], [0, 1])
        logits.append(lg[0])
        if g:
            k = steps[j + 1]
            for i in range(48):
                x = eng.tap(f"X.{i}", 1, 10240)
                ref = g.get(k, f"language_model.model.layers.{i}#0.resid").to(x.device)
                c = torch.nn.functional.cosine_similarity(x.float().flatten(), ref.float().flatten(), 0).item()
                res["cos_min"] = min(res["cos_min"], c)
                if j < 3:
                    qlog.detail(f"{name} step{j + 1} layer{i} cos {c:.6f} maxrel {((x.float() - ref.float()).abs().max() / ref.float().abs().max()).item():.3e}")
    if name in first:  # same request, other chunking: must be bitwise equal
        ref_l = first[name]
        same = all(torch.equal(x, y) for x, y in zip(logits, ref_l))
        nd = sum(int((x != y).sum()) for x, y in zip(logits, ref_l))
        if a.rank == 0:
            (qlog.ok if same else qlog.warn)(
                f"e2e {name}", f"chunk {chunk} vs {CH[0]} · logits {'bitwise equal' if same else f'{nd} values differ'}"
                               f" · input {len(ids):,} ({len(ids) / t_prompt:,.0f} tok/s)")
        continue
    first[name] = [x.clone() for x in logits]
    if g:
        for j, lgj in enumerate(logits):
            k = steps[j] if j < len(steps) else None
            ref = g.get(k, LG)[-1].to(lgj.device).float()
            top = int(lgj.argmax())
            res["top1"] += int(top == outs[j])
            res["n"] += 1
            res["dl"] = max(res["dl"], (lgj - ref).abs().max().item())
            res["tv"] = max(res["tv"], dist(lgj, ref))
            pg = torch.softmax(ref, -1)
            qlog.detail(f"{name} out{j} golden {outs[j]} ours {top} max|dlogit| {(lgj - ref).abs().max().item():.3f} tv {dist(lgj, ref):.4f}"
                        f" · golden p(golden) {pg[outs[j]].item():.3f} p(ours) {pg[top].item():.3f}")
        (qlog.ok if res["top1"] == res["n"] else qlog.warn)(
            f"e2e {name}", f"chunk {chunk} · input {len(ids):,} ({len(ids) / t_prompt:,.0f} tok/s) · "
                           f"top-1 {res['top1']}/{res['n']} · max|Δlogit| {res['dl']:.3f} · "
                           f"TV max {res['tv']:.4f} · layer resid cos min {res['cos_min']:.5f}")
if g:  # reference noise floor: short vs short2 golden logits on the steps before they diverge
    a1, a2 = golden_steps("short"), golden_steps("short2")
    o1, o2 = tok["short"]["output_ids"], tok["short2"]["output_ids"]
    d = next((i for i, (x, y) in enumerate(zip(o1, o2)) if x != y), len(o1))
    tv = max(dist(g.get(a1[j], LG)[-1], g.get(a2[j], LG)[-1]) for j in range(min(d, len(a1))))
    dl = max((g.get(a1[j], LG)[-1].float() - g.get(a2[j], LG)[-1].float()).abs().max().item() for j in range(min(d, len(a1))))
    cmin = 1.0
    for j in range(1, min(d, len(a1))):
        for i in range(48):
            k = f"language_model.model.layers.{i}#0.resid"
            x, y = g.get(a1[j], k).float().flatten(), g.get(a2[j], k).float().flatten()
            cmin = min(cmin, torch.nn.functional.cosine_similarity(x, y, 0).item())
    qlog.info("e2e reference self-diff", f"short vs short2 · same input for {d} steps · max|Δlogit| {dl:.3f} · "
                                          f"TV max {tv:.4f} · layer resid cos min {cmin:.5f}")
sys.stdout.flush()
os._exit(0)
