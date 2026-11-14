"""bf16 lossless packing study (CPU, numpy; dev container on a node): how many bits per weight a fixed-width packed format
would need for the bf16 tensors every decode step reads (HC down/up, lm_head, router).
Format studied (decode ④): per block of G consecutive values along K, base = the block's largest exponent − (2^e − 2);
a value is stored as sign 1 · exponent code e bits · mantissa 7: code 0 = ±0, codes 1 … 2^e − 1 = exponents base … max;
a nonzero value whose exponent is below the window (or a subnormal) is an exception (kept in a side list, 32 bits each:
position + value). Reported per tensor class: exception rate and effective bits/weight for e = 3, 4 and G = 32, 128,
row; plus the exponent entropy (the floor of an entropy coder such as DFloat11: 8 + H bits).
  python3 tools/bf16_pack_study.py <model dir> <tensor_table.tsv>"""
import math
import sys
from collections import Counter

import numpy as np

md, table = sys.argv[1], sys.argv[2]
rows = [l.rstrip("\n").split("\t") for l in open(table)][1:]
pick = {}
for r in rows:
    name, f, off, nb, dt, shape, layer, block, part = r[:9]
    if dt != "BF16":
        continue
    if name.startswith("mtp."):
        continue
    if block in ("attn_hc", "mlp_hc") and part.startswith("input_mix_weight_"):
        pick.setdefault("hc " + part.split("_")[-1].split(".")[0], []).append((name, f, int(off), int(nb), shape))
    elif block in ("lm_head", "router"):
        pick.setdefault(block, []).append((name, f, int(off), int(nb), shape))


def load(f, off, nb):
    with open(f"{md}/{f}", "rb") as fh:
        fh.seek(off)
        return np.frombuffer(fh.read(nb), dtype=np.uint16)


def study(vals, K):
    e = ((vals >> 7) & 0xFF).astype(np.int32)
    nz = (vals & 0x7FFF) != 0
    out = {}
    hist = Counter(e[nz].tolist())
    n = int(nz.sum())
    H = -sum(c / n * math.log2(c / n) for c in hist.values())
    out["H"] = H
    out["zero"] = 1 - n / vals.size
    for G in (32, 128, K):
        m = vals.size // G
        eb = e[: m * G].reshape(m, G)
        nzb = nz[: m * G].reshape(m, G)
        mx = np.where(nzb, eb, -1).max(1, keepdims=True)
        for eb_bits in (3, 4):
            w = (1 << eb_bits) - 2  # window of exponents base … max = 2^e − 1 codes
            exc = nzb & ((eb < mx - w) | (eb == 0))
            rate = exc.sum() / vals.size
            bits = 1 + eb_bits + 7 + 8 / G + 32 * rate  # + base byte per block
            out[(G if G != K else "row", eb_bits)] = (rate, bits)
    return out


for cls, ts in pick.items():
    tot = {}
    nbytes = 0
    for name, f, off, nb, shape in ts[:40]:
        K = int(shape.split("x")[-1])
        v = load(f, off, nb)
        if v.size > 64_000_000:
            v = v[:64_000_000 // K * K]  # lm_head: the first rows suffice
        r = study(v, K)
        for k, x in r.items():
            tot.setdefault(k, []).append((x, v.size))
        nbytes += nb
    print(f"== {cls}: {len(ts)} tensors (studied {min(len(ts), 40)}) · e.g. {ts[0][0]} {ts[0][4]}")
    def avg(k):
        xs = tot[k]
        n = sum(s for _, s in xs)
        if isinstance(xs[0][0], tuple):
            return tuple(sum(x[i] * s for x, s in xs) / n for i in range(2))
        return sum(x * s for x, s in xs) / n
    print(f"   exponent entropy {avg('H'):.2f} bits → entropy-coded ≥ {8 + avg('H'):.2f} bits/weight · zeros {100 * avg('zero'):.3f}%")
    for G in (32, 128, "row"):
        print("   G=" + str(G) + ": " + " · ".join(f"e{eb}: exc {100 * avg((G, eb))[0]:.3f}% → {avg((G, eb))[1]:.2f} bits"
                                      for eb in (3, 4)))
