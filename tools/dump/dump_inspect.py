"""Golden dump overview + the reference's run-to-run noise floor per module (short vs short2: same inputs up to the
first divergent token). Usage (dev image): dump_inspect.py <dump dir> → screen summary, details to
<dump dir>/inspect.txt, noise to noise.json"""
import json
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qlog  # noqa: E402

D = sys.argv[1]
qlog.detail_file(os.path.join(D, "inspect.txt"))
tok = json.load(open(os.path.join(D, "tokens.json")))
man = [json.loads(x) for x in open(os.path.join(D, "rank0", "manifest.jsonl"))]
reqs = []
for s in man:
    rid = next(iter(s["reqs"]))
    if not reqs or reqs[-1][0] != rid:
        reqs.append([rid, []])
    reqs[-1][1].append(s["step"])
names = ["short", "long", "short2"]
qlog.info("dump", " · ".join(f"{n} steps {v[1][0]}–{v[1][-1]}" for n, v in zip(names, reqs)))
a, b = tok["short"]["output_ids"], tok["short2"]["output_ids"]
div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))

# key structure from one decode step: family pattern -> example key, shape, dtype
t = load_file(os.path.join(D, "rank0", f"step{reqs[0][1][1]:04d}.safetensors"))
fam = {}
for k, v in t.items():
    p = re.sub(r"layers\.\d+\.", "layers.N.", k)
    fam.setdefault(p, (k, tuple(v.shape), str(v.dtype)))
qlog.detail(f"# key families in decode step {reqs[0][1][1]} (rank0): {len(fam)}")
for p in sorted(fam):
    qlog.detail(f"{p}\t{fam[p][1]}\t{fam[p][2]}")
qlog.info("dump keys", f"{len(t)} decode-step keys · {len(fam)} patterns · list {D}/inspect.txt")

# noise floor: steps with identical inputs in short and short2 (prefill + decodes before the divergence)
noise = defaultdict(float)
pairs = list(zip(reqs[0][1][:div + 1], reqs[2][1][:div + 1]))
for r in ("rank0", "rank1"):
    for s1, s2 in pairs:
        x, y = load_file(os.path.join(D, r, f"step{s1:04d}.safetensors")), load_file(os.path.join(D, r, f"step{s2:04d}.safetensors"))
        for k in x.keys() & y.keys():
            if x[k].dtype in (torch.float32, torch.bfloat16, torch.float16) and x[k].shape == y[k].shape:
                ref = x[k].float()
                d = (x[k].float() - y[k].float()).abs().max().item() / max(ref.abs().max().item(), 1e-12)
                p = re.sub(r"layers\.\d+\.", "layers.N.", k)
                noise[p] = max(noise[p], d)
nz = {k: v for k, v in noise.items() if v > 0}
json.dump(dict(noise), open(os.path.join(D, "noise.json"), "w"), indent=0)
first = []
for k in fam:  # earliest module (in call order of the step) whose output differs between runs
    pass
order = man[reqs[0][1][1]]["order"]
seen = set()
for name in order:
    p = re.sub(r"layers\.\d+\.", "layers.N.", name)
    for kk, v in nz.items():
        if kk.startswith(p + "#") and ".out" in kk and p not in seen:
            first.append(p)
            seen.add(p)
qlog.info("dump repeat", f"{len(pairs)} same-input step pairs · differing key patterns {len(nz)}/{len(noise)} · "
                         f"max rel diff {max(nz.values(), default=0):.2e} · "
                         f"first differing module {first[0] if first else 'none'}")
for k in sorted(nz, key=lambda k: -nz[k])[:40]:
    qlog.detail(f"noise\t{k}\t{nz[k]:.3e}")
