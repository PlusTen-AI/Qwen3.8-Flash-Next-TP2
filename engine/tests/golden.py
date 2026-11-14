"""Golden dump access (tools/dump): steps, requests, module inputs/outputs.

Key format in a step file: `<module>#<call>.<in{i}|kw.{name}|out[.{i}]|resid>` and for wrapped functions
`<module>.fn.<method>#<call>.<in.{arg}|post.{arg}|ret[.{i}]>`. Prefill steps keep the last `rows_kept` token rows.
"""
import json
import os

import torch
from safetensors.torch import load_file


class Golden:
    def __init__(self, rank_dir, tokens_json=None):
        self.dir = rank_dir
        self.steps = [json.loads(x) for x in open(os.path.join(rank_dir, "manifest.jsonl"))]
        self.modules = json.load(open(os.path.join(rank_dir, "modules.json")))
        self.tokens = json.load(open(tokens_json)) if tokens_json and os.path.exists(tokens_json) else None
        self._cache = {}

    def step(self, k):
        if k not in self._cache:
            self._cache = {k: load_file(os.path.join(self.dir, f"step{k:04d}.safetensors"))}
        return self._cache[k]

    def get(self, k, key):
        t = self.step(k)
        if key not in t:
            raise KeyError(f"step {k}: no {key}")
        return t[key]

    def keys(self, k, prefix=""):
        return [x for x in self.step(k) if x.startswith(prefix)]

    def requests(self):
        """req id -> ordered list of (step, num_tokens, computed) — prefill first, then decode steps."""
        out = {}
        for s in self.steps:
            for rid, r in s["reqs"].items():
                out.setdefault(rid, []).append((s["step"], r.get("scheduled", s["num_tokens"]), r.get("computed", -1)))
        return out

    def decode_steps(self):
        return [s["step"] for s in self.steps if s["num_tokens"] == 1]


def cmp(name, got, ref, atol=None, rtol=None):
    """Error summary between engine output and golden: max abs, max rel (vs ref scale), cosine. Returns dict."""
    g, r = got.float().flatten(), ref.float().flatten().to(got.device)
    d = (g - r).abs()
    scale = r.abs().max().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(g, r, dim=0).item() if g.numel() > 1 else 1.0
    return {"name": name, "n": g.numel(), "max_abs": d.max().item(), "max_rel": (d.max() / scale).item(), "cos": cos,
            "mismatch": int((d > (atol or 0) + (rtol or 0) * r.abs()).sum().item()) if atol is not None else None}
