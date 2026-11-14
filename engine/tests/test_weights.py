"""Production loader vs an independent Python slice (safetensors + tools/tp_plan rules + tools/csf_ref decode)."""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from csf_ref import decode_fixed  # noqa: E402

L = "model.language_model.layers"
CASES = [  # one per rule
    f"{L}.0.linear_attn.in_proj_qkv.weight",            # seg
    f"{L}.0.linear_attn.conv1d.weight",                 # seg (3-D)
    f"{L}.0.linear_attn.in_proj_z.weight_scale",        # col
    f"{L}.0.linear_attn.out_proj.weight",               # row
    f"{L}.3.self_attn.q_proj.weight",                   # col
    f"{L}.3.self_attn.indexer.index_qk_proj.weight",    # rep
    f"{L}.5.mlp.experts.17.gate_proj.weight",           # col (packed fp4)
    f"{L}.5.mlp.experts.17.down_proj.weight",           # row (packed fp4)
    f"{L}.5.mlp.experts.17.up_proj.weight_scale.nvfp4_csf_fixed",    # csf:col
    f"{L}.5.mlp.experts.17.down_proj.weight_scale.nvfp4_csf_fixed",  # csf:row
    "lm_head.weight",                                   # col
]


def _ref(model, wmap, name, tp, rank):
    def full(n):
        with safe_open(os.path.join(model, wmap[n]), "pt") as f:
            t = f.get_tensor(n)
        return t.view(torch.uint8).numpy() if t.dtype != torch.uint8 else t.numpy()

    if tp.startswith("csf:"):
        pre = name.rsplit(".", 1)[0]
        dec = decode_fixed(full(pre + ".nvfp4_csf_fixed"), full(pre + ".nvfp4_csf_exceptions").view(np.uint32))
        n, c = dec.shape
        return dec[rank * n // 2:(rank + 1) * n // 2] if tp == "csf:col" else dec[:, rank * c // 2:(rank + 1) * c // 2]
    with safe_open(os.path.join(model, wmap[name]), "pt") as f:
        t = f.get_tensor(name)
    b = t.view(torch.uint8).reshape(*t.shape[:-1], -1).numpy() if t.dtype != torch.uint8 else t.numpy()
    if tp == "rep":
        return b
    if tp == "col":
        return b[rank * b.shape[0] // 2:(rank + 1) * b.shape[0] // 2]
    if tp == "row":
        return b[:, rank * b.shape[1] // 2:(rank + 1) * b.shape[1] // 2]
    if tp.startswith("seg:"):
        parts, base = [], 0
        for s in map(int, tp[4:].split(",")):
            parts.append(b[base + rank * s // 2: base + (rank + 1) * s // 2])
            base += s
        return np.concatenate(parts)
    raise ValueError(tp)


def test_slices(ctx):
    wmap = json.load(open(os.path.join(ctx.model_dir, "model.safetensors.index.json")))["weight_map"]
    bad = []
    for n in CASES:
        got, shape, dtype, tp = ctx.w.slice(n)
        ref = _ref(ctx.model_dir, wmap, n, tp, ctx.rank)
        same = got.numel() == ref.size and np.array_equal(got.numpy(), ref.reshape(-1))
        if not same:
            bad.append(f"{n.split('.')[-2]}.{n.split('.')[-1]}({tp})")
        ctx.detail(f"slice {n} tp={tp} shape={shape} bytes={got.numel()} {'OK' if same else 'DIFF'}")
    return {"ok": not bad, "msg": f"{len(CASES)} rules byte-identical" if not bad else f"differs: {' '.join(bad)}"}
