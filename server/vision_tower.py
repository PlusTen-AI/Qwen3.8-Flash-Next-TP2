"""vision_tower.py — the Qwen3.8-Flash-Next vision tower (Qwen3-VL ViT, 27 blocks, deepstack off) in PyTorch.

Tower(model_dir, device).embed(media) → bf16 [Σ L/4, 2560]: the embeddings of the image_pad / video_pad tokens, rows in
item order (vision.expand gives their prompt positions).

Precision (lossless = the checkpoint's own precision):
  - weights dequantised exactly to bf16: MXFP8 (attn.qkv / attn.proj) e4m3 · 2^(s−127); NVFP4 W4A16 (mlp.fc1/fc2,
    merger.fc1/fc2) e2m1 · e4m3 — both exact in bf16; the NVFP4 global scale (weight_scale_2) multiplies the fp32
    accumulator, never folded into bf16 weights;
  - activations bf16 between ops (the reference quantises the MXFP8 layers' activations to MXFP8 — we do not);
  - matmuls: bf16 inputs, fp32 accumulation, bias (and global scale) in fp32, one rounding to bf16 (the reference
    rounds the GEMM output to bf16 and adds the bias in bf16 — two roundings);
  - LayerNorm, softmax, GELU, rotary: fp32 inside, one rounding to bf16.
Ops that are exact or elementwise in the reference are reproduced bit for bit (checked by server/check_vision.py):
  input normalisation (uint8 → bf16, fp32 FMA x·2/255 − 1), the bilinear position-embedding interpolation (vLLM's
  Triton kernel), the 2-D rotary tables and the rotary itself, residual adds; LayerNorm and GELU are torch's own
  kernels on bf16 input as in the reference (fp32 inside) — identical code on CUDA, ulp-level differences on the CPU.
On CUDA attention runs in flash_attn varlen when it is installed (else torch SDPA per sequence); on the CPU in fp32
SDPA per sequence.

Two nodes (TP, the server's default): every layer is split over both ranks — attention by heads (qkv rows and proj
columns of 8 of the 16 heads), the MLPs by halves of their inner width (fc1 rows, fc2 columns) — and the two fp32
partial products of proj / fc2 / merger.fc2 are summed across the ranks (all-reduce in fp32) before the global scale, the
bias and the one bf16 rounding: the same fp32 accumulation split in two (the GEMM's own order is not fixed either).
The collectives are torch.distributed's own NCCL communicator (not the engine's). Its kernels wait on the GPU for the
other rank, as the engine's all-reduce kernels do: the two must never run at the same time, or each rank can end up
holding the GPU in one waiting for the other in the other (a server soak hung so, 2026-11-14). The server therefore runs
the tower in the engine's call order — job_begin / job_blocks / job_end, made by rank 0's scheduler thread between
engine calls and replayed by rank 1 from the command link, the device idle before and after each — in slices of a few
blocks, so replies keep flowing while a picture is encoded. (embed() / follow(): rank 0 leads over a job socket, rank 1
follows on its own thread — for the checks only, with no engine beside them.) Tower.profile() runs the largest picture
and video once at start so the peak working memory is held by PyTorch before the KV pool takes the rest.
"""
import json
import os
import pickle
import socket
import struct
import threading
import time

import torch
import torch.nn.functional as F

from vision import MERGE, PATCH_DIM

BF16 = torch.bfloat16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _ue8m0(s):
    """uint8 scale bytes → fp32 2^(s−127) (s = 0 → 2^−127), built from the bits as the engine does."""
    b = s.to(torch.int32) << 23
    b = torch.where(s == 0, torch.full_like(b, 0x00400000), b)
    return b.view(torch.float32)


def _exact_bf16(w32, what):
    w = w32.to(BF16)
    if not torch.equal(w.float(), w32):
        raise ValueError(f"{what}: dequantised weights are not exact in bf16")
    return w


def dequant_mxfp8(q, s):
    """e4m3 [N, K] + UE8M0 [N, K/32] → bf16 [N, K] (exact)."""
    n, k = q.shape
    w = q.float().view(n, k // 32, 32) * _ue8m0(s).view(n, k // 32, 1)
    return w.view(n, k)


def dequant_nvfp4(packed, scale):
    """packed e2m1 [N, K/2] (low nibble = even k) + e4m3 [N, K/16] → fp32 [N, K] = e2m1 · e4m3 (exact in bf16)."""
    n, k2 = packed.shape
    p = packed.to(torch.int64)
    codes = torch.stack([p & 15, p >> 4], dim=-1).view(n, k2 * 2)
    w = E2M1[codes].view(n, k2 * 2 // 16, 16) * scale.float().view(n, k2 * 2 // 16, 1)
    return w.view(n, k2 * 2)


def _round_bf16(d):
    """float64 → bf16 with one correct rounding: round to odd into fp32 (truncate, set the last bit if inexact), then
    RNE into bf16 (fp32 keeps ≥ 2 extra bits, so the result is the correctly rounded bf16)."""
    f = d.to(torch.float32)
    back = f.double()
    bits = f.view(torch.int32)
    bits = torch.where(back.abs() > d.abs(), bits - 1, bits)  # rounded away from zero → one ulp back
    bits = torch.where(back != d, bits | 1, bits)
    return bits.view(torch.float32).to(BF16)


def trim():
    """give freed host memory back to the system (glibc keeps it in its arenas otherwise; on GB10 the host's memory is
    the GPU's: what the process keeps, the KV pool cannot have)"""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _merge_order_rc(h, w):
    """(row, col) of each patch of an h × w grid in merge order (2×2 windows, row-major)."""
    idx = torch.arange(h * w)
    blk, loc = idx // (MERGE * MERGE), idx % (MERGE * MERGE)
    row = (blk // (w // MERGE)) * MERGE + loc // MERGE
    col = (blk % (w // MERGE)) * MERGE + loc % MERGE
    return row, col


class Linear:
    """y = round_bf16(x·Wᵀ (fp32 acc) · g + b). Rows in pieces of ROWS (rows are independent): the fp32 product of a
    large item stays bounded (≤ 32,768 × 4,608 × 4 B ≈ 0.6 GiB). reduce (row-parallel over two ranks): the fp32 partial
    product is summed across the ranks before g and b."""
    ROWS = 32768

    def __init__(self, w, b=None, g=None, reduce=None):
        self.w, self.b, self.g = w, b, g  # bf16 [N, K], bf16 [N] | None, fp32 scalar tensor | None
        self.reduce = reduce

    def _rows(self, x):
        y = mm_f32(x, self.w)
        if self.reduce is not None:
            self.reduce(y)
        if self.g is not None:  # in place: the same fp32 operations without another product-sized temporary
            y.mul_(self.g)
        if self.b is not None:
            y.add_(self.b.float())
        return y.to(BF16)

    def __call__(self, x):
        if x.shape[0] <= self.ROWS:
            return self._rows(x)
        out = torch.empty(*x.shape[:-1], self.w.shape[0], dtype=BF16, device=x.device)
        for r in range(0, x.shape[0], self.ROWS):
            out[r:r + self.ROWS] = self._rows(x[r:r + self.ROWS])
        return out


def mm_f32(x, w):
    """bf16 x [M, K] · bf16 w [N, K]ᵀ → fp32 [M, N], fp32 accumulation (bf16 products are exact in fp32)."""
    if x.is_cuda:
        try:
            return torch.mm(x, w.t(), out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            pass
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # fp32 GEMM, not TF32
        try:
            return x.float() @ w.float().t()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    return x.float() @ w.float().t()


class TP:
    """The two ranks of the tower: rank (0 leads), the job socket (rank 0 → rank 1) and torch.distributed (NCCL) for the
    collectives. connect() opens both; rank 1's socket listens on its own CX7 address."""
    def __init__(self, rank, sock):
        import torch.distributed as dist
        self.rank, self.size, self.sock, self.dist = rank, 2, sock, dist
        self.rf = sock.makefile("rb") if rank == 1 and sock is not None else None

    @classmethod
    def connect(cls, rank, head, peer, sock_port, store_port, timeout=600):
        """rank 0 hosts the NCCL store (head:store_port), rank 1 joins it. sock_port (checks only: embed / follow): rank 0
        connects to rank 1 (peer:sock_port), rank 1 accepts on its address (peer); None: no job socket (the server)."""
        import datetime
        import torch.distributed as dist
        s = None
        if sock_port is None:
            pass
        elif rank == 0:
            t0 = time.time()
            while True:
                try:
                    s = socket.create_connection((peer, sock_port), timeout=5)
                    break
                except OSError:
                    if time.time() - t0 > timeout:
                        raise RuntimeError(f"vision: rank 1 at {peer}:{sock_port} not reachable ({timeout} s)")
                    time.sleep(0.5)
        else:
            ls = socket.socket()
            ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ls.bind((peer, sock_port))
            ls.listen(1)
            s, _ = ls.accept()
            ls.close()
        if s is not None:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(None)
        store = dist.TCPStore(head, store_port, 2, rank == 0, timeout=datetime.timedelta(seconds=timeout))
        opt = dist.ProcessGroupNCCL.Options()  # 4 channels: 0.5 GiB of NCCL buffers instead of 2.0 GiB per rank, the
        opt.config.min_ctas, opt.config.max_ctas = 1, 4  # tower's all-reduce no slower (151 MB fp32: 7.7 vs 8.1 ms)
        dist.init_process_group("nccl", store=store, rank=rank, world_size=2, pg_options=opt,
                                device_id=torch.device("cuda", torch.cuda.current_device()))
        return cls(rank, s)

    def send(self, obj):
        b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack("<I", len(b)) + b)

    def recv(self):
        h = self.rf.read(4)
        if len(h) < 4:
            return None
        return pickle.loads(self.rf.read(struct.unpack("<I", h)[0]))

    def all_reduce(self, t):
        self.dist.all_reduce(t)

    def gather(self, t, dim):
        """both ranks' pieces of a split tensor, concatenated along dim (hooks only: the checks see whole tensors)"""
        parts = [torch.empty_like(t) for _ in range(self.size)]
        self.dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim)

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass


class Tower:
    def __init__(self, model_dir, device="cuda", dtype=BF16, tp=None):
        if dtype != BF16:
            raise ValueError("the tower runs in bf16")
        self.device = torch.device(device)
        self.tp = tp  # TP: this rank's half of every layer; None: the whole tower here
        self.lock = threading.Lock()  # rank 0: one item through the tower at a time (both ranks in the same order)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        vc = cfg["vision_config"]
        self.hidden, self.heads = vc["hidden_size"], vc["num_heads"]
        self.head_dim = self.hidden // self.heads
        self.depth, self.inter, self.out_hidden = vc["depth"], vc["intermediate_size"], vc["out_hidden_size"]
        n = tp.size if tp else 1
        if self.heads % n or self.inter % n or (self.hidden * MERGE * MERGE) % n:
            raise ValueError("the vision tower does not split evenly over the ranks")
        self.heads_local = self.heads // n
        self.grid_side = int(vc["num_position_embeddings"] ** 0.5)
        if vc.get("deepstack_visual_indexes"):
            raise ValueError("deepstack visual layers are not supported")
        if vc.get("hidden_act") != "gelu_pytorch_tanh":
            raise ValueError(f"unexpected vision hidden_act {vc.get('hidden_act')}")
        qcfg = (cfg.get("quantization_config") or {}).get("quantized_layers", {})
        self._load(model_dir, qcfg)
        self._input_norm_lut(model_dir)
        self._rope_tables(8192)
        trim()

    # ---- weights ---------------------------------------------------------------------------------------------------
    def _load(self, model_dir, qcfg):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        pre = "model.visual."
        names = {}
        for k, f in idx.items():
            if k.startswith(pre):
                names.setdefault(f, []).append(k)
        raw = {}
        for f, ks in names.items():
            with safe_open(os.path.join(model_dir, f), framework="pt", device="cpu") as h:
                for k in ks:
                    raw[k[len(pre):]] = h.get_tensor(k)
        dev = self.device

        def t(name):
            return raw.pop(name).to(dev)

        tp = self.tp
        r, n = (tp.rank, tp.size) if tp else (0, 1)

        def linear(name, split=None):
            """split (TP): "qkv" (rows of this rank's heads in q, k and v), "rows" (output half), "cols" (input half,
            the fp32 partial product summed across the ranks)"""
            w = raw.pop(name + ".weight")
            algo = qcfg.get(pre + name, {}).get("quant_algo")
            g = None
            if name + ".weight_scale_2" in raw:  # NVFP4 (W4A16)
                if algo not in (None, "W4A16_NVFP4", "NVFP4"):
                    raise ValueError(f"{name}: config says {algo}, checkpoint has NVFP4 tensors")
                w32 = dequant_nvfp4(w, raw.pop(name + ".weight_scale"))
                g = raw.pop(name + ".weight_scale_2").float().reshape(()).to(dev)
                raw.pop(name + ".input_scale", None)  # weight-only: an activation scale is not used
            elif w.dtype == torch.float8_e4m3fn:  # MXFP8
                if algo not in (None, "MXFP8"):
                    raise ValueError(f"{name}: config says {algo}, checkpoint has MXFP8 tensors")
                w32 = dequant_mxfp8(w, raw.pop(name + ".weight_scale"))
            elif w.dtype == BF16:
                w32 = w.float()
            else:
                raise ValueError(f"{name}: unsupported weight dtype {w.dtype}")
            b = raw.pop(name + ".bias") if name + ".bias" in raw else None
            red = None
            if split and n > 1:
                if split == "qkv":  # [3·heads·head_dim, K]: this rank's heads of q, k and v
                    hd = self.heads_local * self.head_dim
                    idx = torch.cat([torch.arange(p * self.hidden + r * hd, p * self.hidden + (r + 1) * hd)
                                     for p in range(3)])
                    w32, b = w32[idx], (b[idx] if b is not None else None)
                elif split == "rows":
                    k = w32.shape[0] // n
                    w32, b = w32[r * k:(r + 1) * k], (b[r * k:(r + 1) * k] if b is not None else None)
                else:  # "cols"
                    k = w32.shape[1] // n
                    w32, red = w32[:, r * k:(r + 1) * k], tp.all_reduce
            wb = _exact_bf16(w32.contiguous(), name).to(dev)
            return Linear(wb, b.to(dev) if b is not None else None, g, red)

        pw = raw.pop("patch_embed.proj.weight")
        self.patch = Linear(pw.reshape(pw.shape[0], -1).contiguous().to(dev), t("patch_embed.proj.bias"))
        if self.patch.w.shape[1] != PATCH_DIM:
            raise ValueError("unexpected patch embedding shape")
        self.pos = t("pos_embed.weight")  # bf16 [2304, 1152]
        self.blocks = []
        for i in range(self.depth):
            p = f"blocks.{i}."
            self.blocks.append({
                "n1": (t(p + "norm1.weight"), t(p + "norm1.bias")),
                "n2": (t(p + "norm2.weight"), t(p + "norm2.bias")),
                "qkv": linear(p + "attn.qkv", "qkv"), "proj": linear(p + "attn.proj", "cols"),
                "fc1": linear(p + "mlp.linear_fc1", "rows"), "fc2": linear(p + "mlp.linear_fc2", "cols"),
            })
        self.mnorm = (t("merger.norm.weight"), t("merger.norm.bias"))
        self.mfc1, self.mfc2 = linear("merger.linear_fc1", "rows"), linear("merger.linear_fc2", "cols")
        if raw:
            raise ValueError(f"unused visual tensors: {sorted(raw)[:5]}")

    def _input_norm_lut(self, model_dir):
        """uint8 → bf16 table of the reference's device normalisation y = fp32 FMA(x, w, b) → bf16, w = rescale/std,
        b = −mean/std computed in fp32 (vLLM FusedMMInputNorm buffers; its Triton kernel contracts x·w + b)."""
        pc = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
        mean = torch.tensor(pc.get("image_mean", [0.5] * 3), dtype=torch.float32)
        std = torch.tensor(pc.get("image_std", [0.5] * 3), dtype=torch.float32)
        rescale = pc.get("rescale_factor", 1 / 255)
        w = std.reciprocal() * rescale  # torch: scalar / tensor = reciprocal · scalar, in fp32
        b = -mean / std
        x = torch.arange(256, dtype=torch.float64)[:, None]
        exact = x * w.double()[None] + b.double()[None]  # exact in float64 (≤ 32-bit product + a power of 2 range)
        lut = exact.to(torch.float32).to(BF16)  # one fp32 rounding (= the FMA), then RNE to bf16
        self.norm_lut = lut.t().contiguous().to(self.device)  # [3, 256]

    def _rope_tables(self, max_pos):
        """vLLM RotaryEmbedding(head 72, partial 0.5 → rotary_dim 36, theta 10000).cos_sin_cache → bf16 [max_pos, 18]
        each. The reference builds it on the GPU; its fp32 inv_freq equals the correctly rounded 1 / 10000^(2i/36)
        (the CPU's pow is an ulp off for some i), so inv_freq and cos / sin go through float64 here; the angle t·inv is
        an fp32 product as there. Same table on any device."""
        rd = self.head_dim // 2
        e = torch.arange(0, rd, 2, dtype=torch.float32) / rd
        inv = (1.0 / (10000.0 ** e.double())).float()
        freqs = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv[None]
        self.rope_cos = freqs.double().cos().float().to(BF16).to(self.device)
        self.rope_sin = freqs.double().sin().float().to(BF16).to(self.device)

    def nbytes(self):
        ts = [self.pos, self.patch.w, self.patch.b, *self.mnorm, self.mfc1.w, self.mfc1.b, self.mfc2.w, self.mfc2.b]
        for p in self.blocks:
            ts += [*p["n1"], *p["n2"]] + [x for k in ("qkv", "proj", "fc1", "fc2") for x in (p[k].w, p[k].b)]
        return sum(t.numel() * t.element_size() for t in ts if t is not None)

    # ---- per-item metadata -----------------------------------------------------------------------------------------
    def input_norm(self, px):
        """uint8 [L, 1536] (c-major) → bf16 [L, 1536] (a table lookup per channel; int32 indices, never the whole
        input as int64)."""
        L = px.shape[0]
        x = px.to(self.device).view(L, 3, -1)
        out = torch.empty(L, 3, x.shape[2], dtype=BF16, device=self.device)
        for c in range(3):
            out[:, c] = torch.index_select(self.norm_lut[c], 0, x[:, c].reshape(-1).int()).view(L, -1)
        return out.view(L, -1)

    def pos_embed(self, grid):
        """vLLM _bilinear_pos_embed_kernel for one (t, h, w): bf16 [t·h·w, hidden] in merge order, bit-exact.

        As the compiled Triton kernel evaluates it (fp contraction on): h_scale = fp32(47 / (h−1)); hf = floor(fp32(
        row·h_scale)); dh = fma(row, h_scale, −hf) (one rounding of the exact product minus hf); w11 = dh·dw,
        w10 = dh − w11, w01 = dw − w11, w00 = (1 − dh) − w01 in fp32, each → bf16; then in bf16
        val = fma(w11, e11, fma(w10, e10, fma(w01, e01, w00·e00))), every fma rounded once to bf16."""
        t, h, w = grid
        key = (h, w)
        cache = self.__dict__.setdefault("_pos_cache", {})
        val = cache.get(key)
        if val is None:
            val = self._pos_embed_hw(h, w).to(self.device)
            cache[key] = val  # recent sizes, up to ~256 MiB (one 4096² picture's table is 151 MiB)
            while len(cache) > 1 and sum(v.numel() * v.element_size() for v in cache.values()) > 256 * 2**20:
                cache.pop(next(iter(cache)))
        return val.repeat(t, 1) if t > 1 else val

    def _pos_embed_hw(self, h, w, rows=8192):
        """in pieces of `rows` patches (the float64 temporaries of a 4096² picture would be ~3 GiB of host memory)"""
        out = [self._pos_embed_rows(h, w, r, min(r + rows, h * w)) for r in range(0, h * w, rows)]
        trim()
        return torch.cat(out)

    def _pos_embed_rows(self, h, w, r0, r1):
        n = self.grid_side
        hs = torch.tensor(float(n - 1) / float(h - 1) if h > 1 else 0.0, dtype=torch.float32)
        ws = torch.tensor(float(n - 1) / float(w - 1) if w > 1 else 0.0, dtype=torch.float32)
        row, col = _merge_order_rc(h, w)
        row, col = row[r0:r1], col[r0:r1]
        hfl = torch.floor(row.float() * hs)
        wfl = torch.floor(col.float() * ws)
        dh = (row.double() * hs.double() - hfl.double()).float()  # exact product − floor, one fp32 rounding
        dw = (col.double() * ws.double() - wfl.double()).float()
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = (1.0 - dh) - w01
        hc = torch.clamp(hfl.long() + 1, max=n - 1)
        wc = torch.clamp(wfl.long() + 1, max=n - 1)
        hfl, wfl = hfl.long(), wfl.long()
        e = self.pos.cpu()
        ids = [hfl * n + wfl, hfl * n + wc, hc * n + wfl, hc * n + wc]
        wt = [z.to(BF16) for z in (w00, w01, w10, w11)]
        val = wt[0][:, None] * e[ids[0]]  # bf16 product (exact in fp32, rounded once)
        for i in (1, 2, 3):
            val = _round_bf16(wt[i].double()[:, None] * e[ids[i]].double() + val.double())
        return val

    def rotary(self, grid):
        """cos, sin bf16 [t·h·w, 36] = [f(h·θ), f(w·θ)] per patch in merge order."""
        t, h, w = grid
        row, col = _merge_order_rc(h, w)
        if max(h, w) > self.rope_cos.shape[0]:
            self._rope_tables(max(h, w))
        pid = torch.stack([row, col], dim=1).to(self.device)
        cos = self.rope_cos[pid].flatten(1)
        sin = self.rope_sin[pid].flatten(1)
        if t > 1:
            cos, sin = cos.repeat(t, 1), sin.repeat(t, 1)
        return cos, sin

    # ---- ops ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def layer_norm(x, wb):
        """torch LayerNorm on bf16 input and bf16 weights (fp32 statistics and math inside, one rounding), eps 1e-6."""
        w, b = wb
        return F.layer_norm(x, (x.shape[-1],), w, b, 1e-6)

    @staticmethod
    def apply_rotary(x, cos, sin):
        """neox over the whole head: x [L, H, 72] bf16, cos/sin [L, 36] → bf16 (fp32 math, one rounding)."""
        half = x.shape[-1] // 2
        xf = x.float()
        x0, x1 = xf[..., :half], xf[..., half:]
        c, s = cos.float()[:, None, :], sin.float()[:, None, :]
        return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).to(BF16)

    def attention(self, q, k, v, cu):
        """Non-causal attention per sequence (cu = boundaries): q/k/v bf16 [L, H, D] → bf16 [L, H, D]."""
        scale = self.head_dim ** -0.5
        if q.is_cuda:
            try:
                from flash_attn import flash_attn_varlen_func
                cut = torch.tensor(cu, dtype=torch.int32, device=q.device)
                ml = max(b - a for a, b in zip(cu[:-1], cu[1:]))
                return flash_attn_varlen_func(q, k, v, cut, cut, ml, ml, dropout_p=0.0, softmax_scale=scale,
                                              causal=False)
            except ImportError:
                pass
        out = torch.empty_like(q)
        for a, b in zip(cu[:-1], cu[1:]):
            qs, ks, vs = (z[a:b].transpose(0, 1)[None] for z in (q, k, v))
            if not q.is_cuda:
                qs, ks, vs = qs.float(), ks.float(), vs.float()
            o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
            out[a:b] = o[0].transpose(0, 1).to(BF16)
        return out

    def _whole(self, t, dim):
        """a split tensor as a whole one (hooks; both ranks take part)"""
        return self.tp.gather(t, dim) if self.tp else t

    def block(self, i, x, cu, cos, sin, hook=None):
        p = self.blocks[i]
        L = x.shape[0]
        if hook is None:  # the same operations, each intermediate freed once used (a 4096² picture's peak: 2.4 → ~1.6 GiB)
            h = self.layer_norm(x, p["n1"])
            qkv = p["qkv"](h).view(L, 3, self.heads_local, self.head_dim)
            del h
            q = self.apply_rotary(qkv[:, 0], cos, sin)
            k = self.apply_rotary(qkv[:, 1], cos, sin)
            v = qkv[:, 2].contiguous()
            del qkv
            o = self.attention(q, k, v, cu)
            del q, k, v
            a = p["proj"](o.reshape(L, self.heads_local * self.head_dim))
            del o
            x = x + a
            del a
            h2 = self.layer_norm(x, p["n2"])
            f1 = p["fc1"](h2)
            del h2
            act = F.gelu(f1, approximate="tanh")  # bf16 in/out, fp32 inside
            del f1
            m = p["fc2"](act)
            del act
            return x + m
        h = self.layer_norm(x, p["n1"])
        qkv = p["qkv"](h).view(L, 3, self.heads_local, self.head_dim)
        q = self.apply_rotary(qkv[:, 0], cos, sin)
        k = self.apply_rotary(qkv[:, 1], cos, sin)
        o = self.attention(q, k, qkv[:, 2].contiguous(), cu)
        a = p["proj"](o.reshape(L, self.heads_local * self.head_dim))
        x = x + a
        h2 = self.layer_norm(x, p["n2"])
        f1 = p["fc1"](h2)
        act = F.gelu(f1, approximate="tanh")  # bf16 in/out, fp32 inside
        m = p["fc2"](act)
        out = x + m
        W = self._whole
        for name, val in (("norm1", h), ("qkv", W(qkv, 2)), ("q", W(q, 1)), ("k", W(k, 1)), ("attn", W(o, 1)),
                          ("proj", a), ("mid", x), ("norm2", h2), ("fc1", W(f1, -1)), ("act", W(act, -1)),
                          ("fc2", m), ("out", out)):
            hook(f"blocks.{i}.{name}", val)
        return out

    def merger(self, x, hook=None):
        h = self.layer_norm(x, self.mnorm).view(-1, self.hidden * MERGE * MERGE)
        f1 = self.mfc1(h)
        act = F.gelu(f1, approximate="none")
        out = self.mfc2(act)
        if hook is not None:
            W = self._whole
            for name, val in (("norm", h), ("fc1", W(f1, -1)), ("act", W(act, -1)), ("out", out)):
                hook(f"merger.{name}", val)
        return out

    # ---- whole tower -------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def embed(self, media, hook=None):
        """media: list of vision.Media → bf16 [Σ L/4, 2560] on the tower's device, rows in item order. With TP (rank 0):
        the job goes to rank 1, which runs its half alongside (hook: rank 1 joins the gathers)."""
        if not media:
            return torch.empty(0, self.out_hidden, dtype=BF16, device=self.device)
        grids = [tuple(m.grid) for m in media]
        px = torch.cat([m.pixels for m in media])
        if not self.tp:
            return self._run(grids, px, hook)
        with self.lock:
            px = px.to(self.device)
            self.tp.send((grids, tuple(px.shape), hook is not None))
            self.tp.dist.broadcast(px, 0)
            return self._run(grids, px, hook)

    @torch.inference_mode()
    def follow(self, on_error=None):
        """rank 1 (its own thread and CUDA stream): run rank 0's jobs until the socket closes. An error here would leave
        rank 0 waiting in a collective: on_error(msg) ends the process."""
        stream = torch.cuda.Stream()
        try:
            with torch.cuda.stream(stream):
                while True:
                    job = self.tp.recv()
                    if job is None:
                        return
                    grids, shape, hooked = job
                    px = torch.empty(shape, dtype=torch.uint8, device=self.device)
                    self.tp.dist.broadcast(px, 0)
                    self._run(grids, px, (lambda name, val: None) if hooked else None)
                    stream.synchronize()
        except Exception as e:  # noqa: BLE001
            if on_error is None:
                raise
            on_error(f"vision tower (rank 1): {type(e).__name__}: {e}")

    def profile(self, video_patches=None, hold_bytes=0):
        """Run the largest picture (16,777,216 pixels: 256 × 256 patches, one attention sequence) and the largest video
        (the clip budget in two temporal groups) once, so PyTorch holds the peak working memory before the KV pool
        takes the rest → (GiB held, seconds). hold_bytes: memory in use beside the tower meanwhile (rank 0: the
        embeddings cache and a request's embeddings), allocated through the runs so the peak includes it, then freed into
        PyTorch's reserve. Both ranks call it at start (TP: rank 1 through its own call, not a job)."""
        import vision
        t0 = time.time()
        side = int(vision.IMAGE_MAX_PIXELS ** 0.5) // vision.PATCH
        vp = video_patches or vision.VIDEO_MAX_PIXELS // (vision.PATCH * vision.PATCH * vision.TEMPORAL)
        vh = side // 2
        shapes = [(1, side, side), (2, vh, vp // (2 * vh) // MERGE * MERGE)]
        with torch.inference_mode():
            held = torch.zeros(int(hold_bytes), dtype=torch.uint8, device=self.device) if hold_bytes else None
            for g in shapes:
                L = g[0] * g[1] * g[2]
                self._run([g], torch.zeros(L, PATCH_DIM, dtype=torch.uint8, device=self.device), None)
                torch.cuda.synchronize()
            del held
        trim()
        return torch.cuda.memory_reserved() / 2**30, time.time() - t0

    # ---- the server's ordered path (see the module doc) ----------------------------------------------------------------
    @torch.inference_mode()
    def job_begin(self, grids, shape, px=None):
        """an item's input: rank 0 passes its pixels (uint8 [L, 1536], CPU), rank 1 gets them by broadcast"""
        torch.cuda.synchronize()  # the engine's kernels are done: no collective of it overlaps the tower's
        if px is None:
            px = torch.empty(tuple(shape), dtype=torch.uint8, device=self.device)
        else:
            px = px.to(self.device)
        if self.tp:
            self.tp.dist.broadcast(px, 0)
        self._job = list(self._prep([tuple(g) for g in grids], px))
        torch.cuda.synchronize()

    @torch.inference_mode()
    def job_blocks(self, k0, k1):
        torch.cuda.synchronize()
        x, cu, cos, sin = self._job
        for i in range(k0, k1):
            x = self.block(i, x, cu, cos, sin)
        self._job[0] = x
        torch.cuda.synchronize()

    @torch.inference_mode()
    def job_end(self):
        """→ the item's embeddings (both ranks have them; rank 1 drops them)"""
        torch.cuda.synchronize()
        out = self.merger(self._job[0])
        self._job = None
        torch.cuda.synchronize()
        return out

    def _prep(self, grids, px, hook=None):
        """input norm, patch and position embeddings, rotary tables, sequence bounds → (x, cu, cos, sin)"""
        xn = self.input_norm(px)
        pe = self.patch(xn)
        pos = torch.cat([self.pos_embed(g) for g in grids])
        x = pe + pos
        cs = [self.rotary(g) for g in grids]
        cos, sin = torch.cat([c for c, _ in cs]), torch.cat([s for _, s in cs])
        cu = [0]
        for t, h, w in grids:
            for _ in range(t):
                cu.append(cu[-1] + h * w)
        if hook is not None:
            for name, val in (("input_norm", xn), ("patch_embed", pe), ("pos", pos), ("hidden0", x), ("cos", cos),
                              ("sin", sin)):
                hook(name, val)
        return x, cu, cos, sin

    def _run(self, grids, px, hook=None):
        x, cu, cos, sin = self._prep(grids, px, hook)
        for i in range(self.depth):
            x = self.block(i, x, cu, cos, sin, hook)
        return self.merger(x, hook)
