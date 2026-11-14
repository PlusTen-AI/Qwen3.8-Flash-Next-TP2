#!/usr/bin/env python3
"""csf_ref.py — reference numpy decoder for the "csf" NVFP4 block-scale encoding
(quantized_layers[...].weight_scale_encoding == "csf", routed experts only).

Output is the standard modelopt NVFP4 `weight_scale`: float8_e4m3fn bytes, shape [N, C], C = K/16, row-major
(same layout as the MTP experts' directly stored `weight_scale`).

Stored tensors per projection (`<proj>.weight_scale.nvfp4_csf_fixed`, `<proj>.weight_scale.nvfp4_csf_exceptions`):
  fixed  u8  [N/16, 16 + 8*C]   row group g = rows 16g .. 16g+15
         fixed[g, i]                    base byte of row r = 16g+i                      (i < 16)
         fixed[g, 16 + i*C/2 + j/2]     4-bit delta d of S[r, j]; even j = low nibble, odd j = high nibble
         S[r, j] = base[r] + d          plain uint8 add on E4M3 bit patterns (d = 0..15, no carry issues: S <= 0x7E)
  exc    u32 [E]                        word = (byte << 24) | (r*C + j), ascending; overrides: S.flat[pos] = byte
                                        (the delta nibble at an exception position is stored as 0)
Geometry: N = 16*fixed.shape[0], C = (fixed.shape[1] - 16) // 8.

CLI decodes sampled (or all) routed-expert scale tensors, checks the layout against the MTP experts' stored
`weight_scale` and the decoded bytes' range, and times the decode. Needs numpy only; tensors are read straight from the
safetensors files.
"""
import argparse
import json
import os
import random
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlog  # noqa: E402

FIXED, EXC = ".nvfp4_csf_fixed", ".nvfp4_csf_exceptions"
PROJS = ("gate_proj", "up_proj", "down_proj")


def decode_fixed(fixed, exc):
    """fixed u8 [G, 16+8C], exc u32 [E] -> u8 [16G, C] (E4M3 bit patterns, row-major)."""
    g, w = fixed.shape
    rows, cols = 16 * g, (w - 16) // 8
    assert fixed.dtype == np.uint8 and (w - 16) % 8 == 0, (fixed.dtype, fixed.shape)
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    out = np.empty((rows, cols), np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = packed >> 4
    out += fixed[:, :16].reshape(rows, 1)
    if exc.size:
        exc = exc.astype(np.uint32, copy=False)
        out.reshape(-1)[exc & 0xFFFFFF] = (exc >> 24).astype(np.uint8)
    return out


# ---- minimal safetensors reader (numpy only) ----
_NP = {"U8": np.uint8, "U32": np.uint32, "F32": np.float32, "F8_E4M3": np.uint8, "I32": np.int32, "I64": np.int64}


class Ckpt:
    def __init__(self, root):
        self.root = root
        self.wmap = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.hdr, self.fd = {}, {}

    def _file(self, f):
        if f not in self.hdr:
            fd = os.open(os.path.join(self.root, f), os.O_RDONLY)
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            self.hdr[f], self.fd[f] = (json.loads(os.pread(fd, n, 8)), 8 + n), fd
        return self.hdr[f], self.fd[f]

    def get(self, name):
        (h, base), fd = self._file(self.wmap[name])
        t = h[name]
        a, b = t["data_offsets"]
        return np.frombuffer(os.pread(fd, b - a, base + a), _NP[t["dtype"]]).reshape(t["shape"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", help="checkpoint dir (has model.safetensors.index.json)")
    ap.add_argument("--layers", default="0,1,2,3,11,12,23,24,35,36,46,47", help="sample layers")
    ap.add_argument("--experts", type=int, default=32, help="random experts per sampled layer")
    ap.add_argument("--full-layer", type=int, default=47, help="also check all 512 experts of this layer (-1 = off)")
    ap.add_argument("--all", action="store_true", help="check every routed-expert projection (48x512x3)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ck = Ckpt(a.ckpt)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    tc = cfg.get("text_config", cfg)
    nl, ne = tc["num_hidden_layers"], tc["num_experts"]

    # layout reference: MTP experts store the standard weight_scale directly
    for p in PROJS:
        s = ck.get(f"mtp.layers.0.mlp.experts.0.{p}.weight_scale").shape
        w = ck.get(f"mtp.layers.0.mlp.experts.0.{p}.weight").shape
        d = decode_fixed(ck.get(f"model.language_model.layers.0.mlp.experts.0.{p}.weight_scale{FIXED}"),
                         ck.get(f"model.language_model.layers.0.mlp.experts.0.{p}.weight_scale{EXC}")).shape
        qlog.check("layout", d == s and s == (w[0], w[1] * 2 // 16),
                   f"{p}: decoded {d} == MTP weight_scale {s} == [N, K/16] of weight {w}",
                   f"{p}: decoded {d} vs MTP weight_scale {s} vs weight {w}")

    rnd = random.Random(a.seed)
    if a.all:
        jobs = [("all", L, e, p) for L in range(nl) for e in range(ne) for p in PROJS]
    else:
        jobs = [("sample", int(L), e, p) for L in a.layers.split(",")
                for e in sorted(rnd.sample(range(ne), min(a.experts, ne))) for p in PROJS]
        if a.full_layer >= 0:
            jobs += [(f"layer{a.full_layer}", a.full_layer, e, p) for e in range(ne) for p in PROJS]

    stats = {}  # set -> [n, exc, scales, max_byte, t_io, t_dec]
    for tag, L, e, p in jobs:
        st = stats.setdefault(tag, [0, 0, 0, 0, 0.0, 0.0])
        b = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}.weight_scale"
        t0 = time.perf_counter()
        fx, ex = ck.get(b + FIXED), ck.get(b + EXC)
        t1 = time.perf_counter()
        out = decode_fixed(fx, ex)
        t2 = time.perf_counter()
        st[0] += 1
        st[1] += ex.size
        st[2] += out.size
        st[3] = max(st[3], int(out.max()))
        st[4] += t1 - t0
        st[5] += t2 - t1

    for tag, (n, nx, ns, mx, tio, tdec) in stats.items():
        line = (f"{tag}: {n} tensors · exceptions {nx / ns * 100:.2f}% of scales · max byte 0x{mx:02x} · "
                f"read {tio / n * 1e3:.3f} ms · decode {tdec / n * 1e3:.3f} ms /tensor")
        qlog.check("range", mx <= 0x7E, f"{tag}: all decoded bytes finite positive E4M3 (max 0x{mx:02x})",
                   f"{tag}: decoded byte 0x{mx:02x} > 0x7E")
        qlog.info("csf", line)
    tot = sum(s[0] for s in stats.values())
    per = sum(s[5] for s in stats.values()) / tot
    per_io = sum(s[4] for s in stats.values()) / tot
    full = nl * ne * len(PROJS)
    qlog.info("bench", f"full decode {full} tensors: decode {per * full:.1f} s + read {per_io * full:.1f} s "
                       f"(1 thread, from {tot}-tensor run)")


if __name__ == "__main__":
    main()
