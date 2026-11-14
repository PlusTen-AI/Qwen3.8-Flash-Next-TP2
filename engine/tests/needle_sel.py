"""Needle diagnostic (run through tools/tp2.sh): are a record's tokens among the QSA selection of the row that predicts
the answer? Input JSON (built on the Mac from the server's /tokenize): {"prefix": ids shared by all cases, "cases": [{
"rid", "tail": ids after the prefix, "first": the first answer token id of the asked record}], "records": [{"rid",
"pos": token positions of its number, "first"}]}. The engine reads the prefix once (16K-row chunks), keeps a
snapshot, then for each case restores it and runs the tail with taps: per QSA layer, whether the asked record's number
is among the selected positions of the last row and how many other records' numbers are, and the greedy first answer
token (the asked record's first digit, another record's, or something else — first digits are shared, so a match is a
hint only).
  tools/tp2.sh nsel python3 engine/tests/needle_sel.py --ids /work/scratch/needle_sel/ids.json --yarn 4 --ramp 262144"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
import qlog  # noqa: E402

import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--yarn", type=float, default=4.0)
ap.add_argument("--ramp", type=int, default=262144)
ap.add_argument("--chunk", type=int, default=16384)
a = ap.parse_args()
OUT = "/work/results/tests/needle_sel"
os.makedirs(OUT, exist_ok=True)
D = json.load(open(a.ids))
P = D["prefix"]
L = len(P) + max(len(c["tail"]) for c in D["cases"])
eng = ext.load("engine", ext.engine_sources(), log=f"{OUT}/build{a.rank}.log", ldflags=ext.ENGINE_LDFLAGS)
eng.init("/models/Qwen3.8-Flash-Next-FP4-Local", "/work/results/tensor_table.tsv", a.rank, os.environ["Q38_HEAD_IP"], 1,
         (L + 64 + 3) // 4 * 4, a.chunk, mtp=False, yarn_factor=a.yarn, yarn_ramp=a.ramp)
for c0 in range(0, len(P), a.chunk):
    c1 = min(c0 + a.chunk, len(P))
    eng.step(P[c0:c1], list(range(c0, c1)), [0], [0, c1 - c0], False)
eng.snapshot(0)
recs = {r["rid"]: r for r in D["records"]}
firsts = {r["first"]: r["rid"] for r in D["records"]}
QSA = [i for i in range(48) if i % 4 == 3]
lines, out = [], []
for c in D["cases"]:
    assert eng.restore(0) == len(P)
    T = c["tail"]
    eng.set_taps(True)
    lg = eng.step(T, list(range(len(P), len(P) + len(T))), [0], [0, len(T)], True)
    eng.set_taps(False)
    tok = int(lg[-1].argmax())
    who = "its first digit" if tok == c["first"] else ("another record's first digit" if tok in firsts else
                                                         f"token {tok}")
    per = []
    for i in QSA:
        sel = eng.tap(f"sel.{i}", len(T), 2052 * 2).view(torch.int32)[-1].cpu()
        got = set(sel[:int(sel[2051])].tolist())
        mine = all(p in got for p in recs[c["rid"]]["pos"])
        others = sum(all(p in got for p in r["pos"]) for rid, r in recs.items() if rid != c["rid"])
        per.append({"layer": i, "asked": mine, "others": others})
    out.append({"rid": c["rid"], "depth": c.get("depth"), "answer": who, "layers": per})
    lines.append(f"record {c['rid']} ({c.get('depth', '?')}%): answer {who} · asked record selected in "
                 f"{sum(p['asked'] for p in per)}/12 layers [{''.join('✓' if p['asked'] else '·' for p in per)}] · "
                 f"other records selected per layer {[p['others'] for p in per]}")
if a.rank == 0:
    json.dump(out, open(f"{OUT}/needle_sel.json", "w"), indent=1)
    for ln in lines:
        qlog.ok("nsel", ln)
sys.stdout.flush()
os._exit(0)
