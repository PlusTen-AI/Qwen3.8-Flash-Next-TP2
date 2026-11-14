"""qdump — vLLM general plugin that records golden tensors from the reference image (eager mode only).

Per engine step and rank it saves every hooked module's inputs/outputs (plus selected internal kernel
entry points) to  $QDUMP_DIR/rank{r}/step{k:04d}.safetensors  and one manifest line per step.
Prefill steps keep only the last $QDUMP_ROWS token rows; tensors over $QDUMP_TENSOR_MB are skipped
(listed in the manifest). Decoder layers also get `.resid` = the materialised multi-stream state.

Env: QDUMP_DIR (enables), QDUMP_ROWS=16, QDUMP_MAX_STEPS=96, QDUMP_TENSOR_MB=64, QDUMP_SKIP=visual
     (comma list of module-name parts; "none" hooks the vision tower too), QDUMP_SKIP_RE=<regex>: module names not
     hooked (e.g. the insides of the middle vision blocks — their outputs are still kept),
     QDUMP_FULLROWS=<regex>: keys kept with all rows (cap QDUMP_FULL_MB=1024) — inputs needed to rebuild the
     recurrent/KV state after a long prefill (GDN mixed_qkv/a/b, QSA fused projection, merged input embeddings),
     QDUMP_MM_DECODE_STEPS=N: requests with images/videos keep only their first N decode steps (default: all)
Per new request (rank 0 and 1): requests.jsonl = request id, step, full prompt token ids, media items (modality,
hash, placeholder offset and length).
"""
import functools
import importlib
import json
import os
import time

_ENV = os.environ.get


def register():
    if not _ENV("QDUMP_DIR"):
        return
    from vllm.v1.worker import gpu_worker

    W = gpu_worker.Worker
    if getattr(W, "_qdump", False):
        return
    W._qdump = True
    o_load, o_exec, o_sample = W.load_model, W.execute_model, W.sample_tokens

    def load_model(self, *a, **k):
        r = o_load(self, *a, **k)
        _REC.install(self.model_runner.model)
        return r

    def execute_model(self, so, *a, **k):
        _REC.end()  # a step that never reached sample_tokens
        if so.total_num_scheduled_tokens > 0:
            _REC.begin(so)
        out = o_exec(self, so, *a, **k)
        if out is not None:
            _REC.end()
        return out

    def sample_tokens(self, *a, **k):
        try:
            return o_sample(self, *a, **k)
        finally:
            _REC.end()

    W.load_model, W.execute_model, W.sample_tokens = load_model, execute_model, sample_tokens


# Internal entry points that are not module forwards: "module:Class.method" -> args to record again after the call
# (in-place outputs), by argument name. Module forwards (e.g. QSAIndexer) are already covered by the hooks.
FN_TARGETS = {
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn:QwenGatedDeltaNetAttention._forward_core": ["core_attn_out"],
    "vllm.models.qwen4_exp.nvidia.qsa:Qwen4ExpQSAAttention._run_qsa": ["output"],
    "vllm.models.qwen4_exp.nvidia.qsa:Qwen4ExpQSAFlashAttentionImpl.forward_qsa": ["output"],
    # vision: tower outputs per media item, then token embeddings with the media rows replaced
    "vllm.models.qwen4_exp.nvidia.model:Qwen4ExpForConditionalGeneration.embed_multimodal": [],
    "vllm.models.qwen4_exp.nvidia.model:Qwen4ExpForConditionalGeneration.embed_input_ids": [],
}


class _Recorder:
    def __init__(self):
        self.on = False
        self.step = 0
        self.names = {}
        self.ready = False

    # ---- setup -------------------------------------------------------------------------------
    def install(self, model):
        import torch

        self.torch = torch
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.dir = os.path.join(_ENV("QDUMP_DIR"), f"rank{self.rank}")
        os.makedirs(self.dir, exist_ok=True)
        self.rows = int(_ENV("QDUMP_ROWS", "16"))
        self.max_steps = int(_ENV("QDUMP_MAX_STEPS", "96"))
        self.cap = int(float(_ENV("QDUMP_TENSOR_MB", "64")) * 2**20)
        self.full_cap = int(float(_ENV("QDUMP_FULL_MB", "1024")) * 2**20)
        import re
        fr = _ENV("QDUMP_FULLROWS", "")
        self.full = re.compile(fr) if fr else None
        self.mm_decode = int(_ENV("QDUMP_MM_DECODE_STEPS", "-1"))
        self.mm_reqs = {}  # request id → decode steps seen (requests with media only)
        skip = tuple(s for s in _ENV("QDUMP_SKIP", "visual").split(",") if s)
        skip_re = re.compile(_ENV("QDUMP_SKIP_RE")) if _ENV("QDUMP_SKIP_RE") else None
        mods = {}
        for name, m in model.named_modules():
            if not name or any(p in name.split(".") for p in skip) or (skip_re and skip_re.search(name)):
                continue
            self.names[id(m)] = name
            m.register_forward_pre_hook(functools.partial(self._pre, name), with_kwargs=True)
            m.register_forward_hook(functools.partial(self._post, name), with_kwargs=True)
            mods[name] = {
                "class": type(m).__name__,
                "params": {p: [list(t.shape), str(t.dtype)] for p, t in m.named_parameters(recurse=False)},
                "buffers": {p: [list(t.shape), str(t.dtype)] for p, t in m.named_buffers(recurse=False)},
            }
        wrapped = []
        for target, post in FN_TARGETS.items():
            mod, _, attr = target.partition(":")
            cls_name, meth = attr.split(".")
            try:
                cls = getattr(importlib.import_module(mod), cls_name)
                orig = getattr(cls, meth)
            except Exception as e:  # noqa: BLE001
                wrapped.append(f"{attr}: missing ({type(e).__name__})")
                continue
            setattr(cls, meth, self._wrap(orig, meth, post))
            wrapped.append(attr)
        with open(os.path.join(self.dir, "modules.json"), "w") as f:
            json.dump({"modules": mods, "fn_targets": wrapped}, f)
        self.ready = True
        print(f"[qdump] rank{self.rank} · {len(mods)} modules · {len(wrapped)} functions · {self.dir}", flush=True)

    # ---- step window -------------------------------------------------------------------------
    def _media_step(self, so):
        """False = a step that only advances media requests already past their first QDUMP_MM_DECODE_STEPS decodes."""
        new = getattr(so, "scheduled_new_reqs", []) or []
        for r in new:
            if getattr(r, "mm_features", None):
                self.mm_reqs[r.req_id] = 0
        if self.mm_decode < 0 or new:
            return True
        ids = list((so.num_scheduled_tokens or {}).keys())
        if not ids or any(i not in self.mm_reqs for i in ids):
            return True
        keep = any(self.mm_reqs[i] < self.mm_decode for i in ids)
        for i in ids:
            self.mm_reqs[i] += 1
        return keep

    def _log_requests(self, so):
        with open(os.path.join(self.dir, "requests.jsonl"), "a") as f:
            for r in getattr(so, "scheduled_new_reqs", []) or []:
                media = [{"modality": m.modality, "hash": m.identifier, "offset": int(m.mm_position.offset),
                          "length": int(m.mm_position.length)} for m in (getattr(r, "mm_features", None) or [])]
                f.write(json.dumps({"req": r.req_id, "step": self.step,
                                    "prompt_ids": list(getattr(r, "prompt_token_ids", None) or []),
                                    "media": media}) + "\n")

    def begin(self, so):
        if not self.ready or self.step >= self.max_steps:
            return
        if not self._media_step(so):
            return
        self.on = True
        self.t0 = time.time()
        self.n = int(so.total_num_scheduled_tokens)
        self.buf, self.skipped, self.dtypes, self.calls, self.order = {}, [], {}, {}, []
        self._log_requests(so)
        reqs = {}
        for r in getattr(so, "scheduled_new_reqs", []) or []:
            reqs[r.req_id] = {"new": True, "computed": int(getattr(r, "num_computed_tokens", 0)),
                              "prompt_len": len(getattr(r, "prompt_token_ids", None) or []),
                              "media": len(getattr(r, "mm_features", None) or [])}
        cached = getattr(so, "scheduled_cached_reqs", None)
        if cached is not None:
            for i, rid in enumerate(getattr(cached, "req_ids", []) or []):
                comp = getattr(cached, "num_computed_tokens", None)
                reqs[rid] = {"new": False, "computed": int(comp[i]) if comp is not None else -1}
        for rid, n in (so.num_scheduled_tokens or {}).items():
            reqs.setdefault(rid, {})["scheduled"] = int(n)
        self.reqs = reqs

    def end(self):
        if not self.on:
            return
        self.on = False
        from safetensors.torch import save_file

        path = os.path.join(self.dir, f"step{self.step:04d}.safetensors")
        save_file(self.buf, path)
        nbytes = sum(t.numel() * t.element_size() for t in self.buf.values())
        line = {"step": self.step, "t": round(self.t0, 3), "sec": round(time.time() - self.t0, 2),
                "num_tokens": self.n, "rows_kept": min(self.n, self.rows), "reqs": self.reqs,
                "tensors": len(self.buf), "bytes": nbytes, "order": self.order,
                "dtypes": self.dtypes, "skipped": self.skipped}
        with open(os.path.join(self.dir, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")
        self.buf = {}
        self.step += 1

    # ---- capture -----------------------------------------------------------------------------
    def _put(self, key, t):
        torch = self.torch
        full = self.full is not None and self.full.search(key) is not None
        if not full and t.dim() >= 1 and t.shape[0] == self.n and self.n > self.rows:
            t = t[-self.rows:]
        nb = t.numel() * t.element_size()
        if nb > (self.full_cap if full else self.cap):
            self.skipped.append([key, list(t.shape), str(t.dtype), "size"])
            return
        c = t.detach().to("cpu", copy=True).contiguous()
        if c.dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.float64, torch.int64, torch.int32,
                           torch.int16, torch.int8, torch.uint8, torch.bool, torch.float8_e4m3fn, torch.float8_e5m2):
            self.dtypes[key] = [str(c.dtype), list(c.shape)]
            c = c.view(torch.uint8) if c.dim() else c.reshape(1).view(torch.uint8)
        self.buf[key] = c

    def _walk(self, key, x, depth=0):
        torch = self.torch
        if isinstance(x, torch.Tensor):
            self._put(key, x)
        elif isinstance(x, (tuple, list)) and depth < 2:
            for i, y in enumerate(x):
                self._walk(f"{key}.{i}", y, depth + 1)

    def _key(self, name):
        k = self.calls.get(name, 0)
        return f"{name}#{k}"

    def _pre(self, name, mod, args, kwargs):
        if not self.on:
            return None
        key = self._key(name)
        self.order.append(name)
        for i, a in enumerate(args):
            self._walk(f"{key}.in{i}", a)
        for k, v in kwargs.items():
            self._walk(f"{key}.kw.{k}", v)
        return None

    def _post(self, name, mod, args, kwargs, out):
        if not self.on:
            return None
        key = self._key(name)
        self.calls[name] = self.calls.get(name, 0) + 1
        self._walk(f"{key}.out", out)
        if type(mod).__name__ == "Qwen4ExpDecoderLayer" and isinstance(out, tuple) and len(out) == 3:
            try:  # materialised multi-stream state after this layer (combine is pure: hc_combine returns a new tensor)
                with self.torch.no_grad():
                    self._put(f"{key}.resid", mod.mlp_hyper_connection.combine(*out))
            except Exception as e:  # noqa: BLE001
                self.skipped.append([f"{key}.resid", [], "", f"combine: {type(e).__name__}"])
        return None

    def _wrap(self, orig, meth, post):
        import inspect

        sig = inspect.signature(orig)
        rec = self

        @functools.wraps(orig)
        def w(this, *args, **kwargs):
            if not rec.on:
                return orig(this, *args, **kwargs)
            owner = rec.names.get(id(this))
            if owner is None:  # attention impl objects are not modules: name them by their layer argument
                lay = kwargs.get("layer", args[0] if args else None)
                owner = rec.names.get(id(lay), type(this).__name__)
            base = f"{owner}.fn.{meth}"
            k = rec.calls.get(base, 0)
            rec.calls[base] = k + 1
            key = f"{base}#{k}"
            rec.order.append(base)
            try:
                bound = sig.bind(this, *args, **kwargs)
                named = list(bound.arguments.items())[1:]
            except TypeError:
                named = [(str(i), a) for i, a in enumerate(args)] + list(kwargs.items())
            for n, a in named:
                rec._walk(f"{key}.in.{n}", a)
            r = orig(this, *args, **kwargs)
            for n, a in named:
                if n in post:
                    rec._walk(f"{key}.post.{n}", a)
            rec._walk(f"{key}.ret", r)
            return r

        return w


_REC = _Recorder()
