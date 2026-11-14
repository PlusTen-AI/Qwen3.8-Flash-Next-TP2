"""Pure-PyTorch reference of the DFlash2 draft side (one round, one sequence) — a study tool (engine/tests/
dflash_study.py), written from reading SGLang @3c560089 (Apache-2.0) and z-lab/dflash; nothing imported from them.
Drafter: incoai/Qwen3.8-Flash-Next-DFlash2 @a22a1744 (<models>/Qwen3.8-Flash-Next-DFlash2).

Mirrors SGLang @3c560089 (models/dflash.py DFlash2DraftModel + speculative/dflash_worker_v2.py) and the
z-lab HF reference (dflash/model.py DFlash2DraftModel). See dflash2_spec.md for file:line references.
No SGLang / flash-attn / triton imports. Readable over fast. fp32 math, with optional bf16 rounding at the
points where SGLang materializes bf16 tensors (emulate_bf16=True) so greedy near-ties behave closer.

One round, anchor at absolute position P:
  context  = target features of committed positions < P (only P-2047 <= pos < P are visible)
  block    = [anchor, MASK x (B-1)] at positions P..P+B-1, embedded with the TARGET embedding
  5 layers = pre-norm, two-tap dynamic grouped conv around attention and MLP, bidirectional attention
             over [context K/V ; block K/V] with a 2047 left/right window
  head     = draft norm -> TARGET lm_head on rows 1..B-1 -> top-16 candidates per slot
  selector = K x K lattice between adjacent slots (codebooks + hidden projection), one walk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# Checkpoint names are used as-is (dflash2_tensors.txt). SGLang fuses some of them at load (M:845-852):
#   layers.{i}.self_attn.{q,k,v}_proj.weight -> layers.{i}.self_attn.qkv_proj.weight  (rows q|k|v)
#   layers.{i}.mlp.{gate,up}_proj.weight     -> layers.{i}.mlp.gate_up_proj.weight   (rows gate|up)
# Codebooks are bare Parameters in SGLang (M:1124-1129); z-lab maps them to nn.Embedding ".weight" (Z:630-637).
_LAYER_TENSORS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "attention_conv.base_kernel",
    "attention_conv.kernel_projection.weight",
    "mlp_conv.base_kernel",
    "mlp_conv.kernel_projection.weight",
)
_GLOBAL_TENSORS = (
    "fc.weight",
    "hidden_norm.weight",
    "norm.weight",
    "candidate_selector.hidden_projection.weight",
    "candidate_selector.predecessor_codebook",
    "candidate_selector.successor_codebook",
)
_CODEBOOKS = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")


@dataclass
class ContextKV:
    """Per-layer context K/V built from target features (spec (a)). k, v: [T, n_kv, head_dim] fp32."""

    pos: Tensor  # [T] long, absolute positions
    k: List[Tensor]
    v: List[Tensor]


@dataclass
class DraftResult:
    """Output of DFlash2Ref.draft. S = block_size - 1 draft slots.

    tokens[e]          drafted token for absolute position anchor_pos + 1 + e.
    block_tokens       [B] = [anchor, tokens...]: the verify block fed to the target at anchor_pos..+B-1.
    candidate_ids[e]   the 16 candidate ids of slot e (top-16 of the draft logits, sorted by unary desc).
    q_rows[e]          THE DRAFT DISTRIBUTION q FOR THE ACCEPT TEST at slot e: a distribution over
                       candidate_ids[e] only (zero on every other vocab id), conditioned on the candidate
                       actually chosen at slot e-1 (slot 0: on the anchor). Sampling: softmax(score_row / T),
                       T = max(temperature, 1e-5), NO top-k / top-p applied. Greedy: one-hot at the argmax.
                       Dense form for rejection sampling: q_dense(V) (scatter into zeros), exactly what
                       SGLang hands the verify kernel (W:1516-1547).
    path_index[e]      index into candidate_ids[e] of tokens[e].
    unary, scores      [S, K] and [S, K(pred), K(cand)] lattice terms (debugging).
    hidden             [B, H] final-normed draft hidden (row 0 = anchor row, unused downstream).
    """

    tokens: Tensor
    block_tokens: Tensor
    candidate_ids: Tensor
    q_rows: Tensor
    path_index: Tensor
    unary: Tensor
    scores: Tensor
    hidden: Tensor
    greedy: bool

    def q_dense(self, vocab_size: int) -> Tensor:
        q = torch.zeros(self.q_rows.shape[0], vocab_size, dtype=torch.float32, device=self.q_rows.device)
        return q.scatter_(-1, self.candidate_ids, self.q_rows.float())


def is_greedy(temperature: float, top_k: Optional[int]) -> bool:
    # SGLang: temperature < 1e-6 -> (temperature=1, top_k=1) (sampling_params.py:222-225);
    # greedy rows = top_k <= 1 (dspark_draft.py:111-119). top_k -1/0/None = whole vocab.
    return temperature < 1e-6 or top_k == 1


class DFlash2Ref:
    def __init__(
        self,
        weights: Dict[str, Tensor],
        cfg: dict,
        embed: Callable[[Tensor], Tensor],
        lm_head: Callable[[Tensor], Tensor],
        *,
        device: Optional[torch.device] = None,
        emulate_bf16: bool = True,
        noise_embed_scale: float = 1.0,
        mask_embedding: Optional[Tensor] = None,
    ) -> None:
        """
        weights: checkpoint tensors by the names in dflash2_tensors.txt (no "model." prefix).
        cfg:     the drafter config.json as a dict (dflash_config nested).
        embed:   target input embedding, LongTensor [N] -> [N, 2560] (any float dtype).
        lm_head: target output head, Tensor [N, 2560] -> logits [N, vocab]. Must return the real vocab only
                 (248320; crop any padded tail). Receives bf16 input when emulate_bf16 (SGLang casts the
                 hidden to the head weight dtype, M:95).
        noise_embed_scale: 1.0 unless the target defines get_dflash_noise_embedding_scale (W:522-526).
            UNSURE: the drafter repo's SGLang patch might define it for the Qwen3.8 target; this checkout has none.
        mask_embedding: [2560] vector from the drafter folder's mask_embedding.pt, if it exists (W:1282-1342).
            UNSURE: whether the drafter folder ships one; if it does, it replaces the target row 248077.
        """
        self.device = torch.device(device) if device is not None else next(iter(weights.values())).device
        self.embed = embed
        self.lm_head = lm_head
        self.emulate_bf16 = emulate_bf16
        self.noise_embed_scale = float(noise_embed_scale)
        self.mask_embedding = None if mask_embedding is None else mask_embedding.to(self.device).float()

        text = cfg.get("text_config", cfg)
        dcfg = cfg.get("dflash_config", {}) or {}

        def dget(key, default=None):  # dflash_config first, then top level (U:621-846)
            return dcfg.get(key, text.get(key, cfg.get(key, default)))

        self.H = int(text["hidden_size"])
        self.n_layers = int(text["num_hidden_layers"])
        self.n_heads = int(text["num_attention_heads"])
        self.n_kv = int(text.get("num_key_value_heads", self.n_heads))
        self.head_dim = int(text.get("head_dim", self.H // self.n_heads))
        self.eps = float(text.get("rms_norm_eps", 1e-6))
        if text.get("attention_bias", False):
            raise NotImplementedError("attention_bias=True not handled")
        if text.get("hidden_act", "silu") != "silu":
            raise NotImplementedError("only silu (M:471-475)")

        # Draft extras this checkpoint does not use; refuse rather than silently ignore.
        if dget("attention_value_scale") is not None or dget("attention_sink_bias", False):
            raise NotImplementedError("attention_value_scale / attention_sink_bias not handled")
        if dget("projector_type") is not None:
            raise NotImplementedError("Domino projector not handled")

        self.cfg_block_size = int(dget("block_size", 16))
        self.taps = int(dget("conv_kernel_size", 0))
        self.group_size = int(dget("conv_group_size", 0))
        if not self.taps or not self.group_size:
            raise ValueError("DFlash2 needs conv_kernel_size and conv_group_size")
        self.n_groups = self.H // self.group_size
        self.mask_token_id = int(dget("mask_token_id"))
        self.top_k_sel = int(dget("selector_top_k"))
        self.output_multiplier = float(dget("output_multiplier", 1.0))
        softcap = float(dget("final_logit_softcapping") or 0.0)
        self.softcap = softcap if softcap > 0 else None

        # Attention type and window per layer (M:155-189, U:433-450).
        layer_types = text.get("layer_types")
        is_causal = text.get("is_causal")
        sliding_window = text.get("sliding_window")
        self.layer_window: List[int] = []  # window_left, -1 = none
        self.layer_causal: List[bool] = []
        for i in range(self.n_layers):
            lt = None if layer_types is None else layer_types[i]
            if lt in (None, "full_attention"):
                self.layer_window.append(-1)
                self.layer_causal.append(False if is_causal is None else bool(is_causal))
            elif lt == "sliding_attention":
                self.layer_window.append(int(sliding_window) - 1)  # HF window includes self (U:449-450)
                self.layer_causal.append(True if is_causal is None else bool(is_causal))
            else:
                raise ValueError(f"layer_types[{i}]={lt!r}")

        # RoPE (M:249-276; rotary_embedding/base.py:233-266; factory.py:156-157).
        rope_params = text.get("rope_parameters") or text.get("rope_scaling")
        theta = text.get("rope_theta", 10000)
        if rope_params is not None:
            theta = rope_params.get("rope_theta", theta)
            rtype = rope_params.get("rope_type", rope_params.get("type", "default"))
            if rtype != "default":
                raise NotImplementedError(f"rope_type={rtype!r} not handled")
        self.rope_theta = float(theta)
        rotary_dim = self.head_dim
        prf = float(text.get("partial_rotary_factor", 1.0))
        if rope_params is not None and "rope_dim" in rope_params:
            rotary_dim, prf = int(rope_params["rope_dim"]), 1.0
        if prf < 1.0:
            # UNSURE: the task says default rope; if config.json carries a partial_rotary_factor copied from
            # the target, SGLang honours it (rotates only the first dims) while the comment at M:253-254 says
            # HF Qwen3 ignores it. Check config.json.
            rotary_dim = int(rotary_dim * prf)
        self.rotary_dim = rotary_dim
        self.inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=self.device) / rotary_dim)
        )

        # Weights: exact checkpoint names; fp32 copies except the two [vocab, 256] codebooks (row-gathered).
        need = list(_GLOBAL_TENSORS) + [
            f"layers.{i}.{n}" for i in range(self.n_layers) for n in _LAYER_TENSORS
        ]
        missing = [n for n in need if n not in weights]
        if missing:
            raise KeyError(f"missing draft tensors: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        self.unused = sorted(set(weights) - set(need))
        self.w: Dict[str, Tensor] = {}
        for n in need:
            t = weights[n].to(self.device)
            self.w[n] = t if n in _CODEBOOKS else t.float()
        assert self.w["fc.weight"].shape[0] == self.H
        assert self.w["layers.0.attention_conv.kernel_projection.weight"].shape[0] == 2 * self.taps * self.n_groups
        assert self.w["layers.0.attention_conv.base_kernel"].shape == (2, self.taps, self.H)

    # ------------------------------------------------------------------ primitives
    def r(self, x: Tensor) -> Tensor:
        """Round to bf16 where SGLang stores a bf16 tensor (no-op when emulate_bf16=False)."""
        return x.to(torch.bfloat16).float() if self.emulate_bf16 else x

    def lin(self, x: Tensor, name: str) -> Tensor:
        return self.r(x @ self.w[name].T)

    def rms(self, x: Tensor, name: str) -> Tensor:
        # x * rsqrt(mean(x^2) + eps) * w  (layernorm.py:821-864; Qwen3 style, not 1+w)
        x = x.float()
        return self.r(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w[name])

    def rope(self, x: Tensor, pos: Tensor) -> Tensor:
        """Neox RoPE on the first rotary_dim dims. x: [N, heads, head_dim], pos: [N]."""
        freqs = pos.float()[:, None] * self.inv_freq[None, :]  # [N, rot/2]
        cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
        half = self.rotary_dim // 2
        x1, x2 = x[..., :half], x[..., half : self.rotary_dim]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return self.r(torch.cat([rot, x[..., self.rotary_dim :]], dim=-1))

    def head_norm(self, x: Tensor, name: str) -> Tensor:
        """Per-head RMSNorm, x: [N, heads, head_dim], weight [head_dim] (M:245-247, 388; U:1256-1265)."""
        return self.rms(x, name)

    # ------------------------------------------------------------------ (a) context
    def prepare_context(self, ctx_feats: Tensor, ctx_pos: Tensor) -> ContextKV:
        """Target features -> per-layer context K/V (W:1951-1970, M:783-803).

        ctx_feats: [T, 12800] = concat of target layers [5,15,25,35,45] (each the 4-stream mean), one row
                   per committed position (the target's hidden at that INPUT position).
        ctx_pos:   [T] absolute positions, all < anchor_pos. Rows older than anchor_pos-2047 are masked anyway.
        No input_layernorm and no conv on context rows; the same ctx_hidden feeds every layer.
        UNSURE: the concat order (layer 5 first) and the 4-stream mean come from the drafter repo's SGLang
        patch, not visible in the reference code; z-lab concatenates in target_layer_ids order (Z:37-43).
        """
        ctx_pos = ctx_pos.to(self.device).long()
        x = ctx_feats.to(self.device).float()
        ctx_hidden = self.rms(self.lin(x, "fc.weight"), "hidden_norm.weight")  # M:800-803
        ks, vs = [], []
        T = ctx_hidden.shape[0]
        for i in range(self.n_layers):
            p = f"layers.{i}.self_attn."
            k = self.lin(ctx_hidden, p + "k_proj.weight").view(T, self.n_kv, self.head_dim)
            v = self.lin(ctx_hidden, p + "v_proj.weight").view(T, self.n_kv, self.head_dim)
            k = self.rope(self.head_norm(k, p + "k_norm.weight"), ctx_pos)
            ks.append(k)
            vs.append(v)
        return ContextKV(pos=ctx_pos, k=ks, v=vs)

    # ------------------------------------------------------------------ (c) conv
    def conv(self, x: Tensor, delta: Tensor, base: Tensor, block_size: int) -> Tensor:
        """Grouped dynamic depthwise causal conv over block rows (M:486-499).

        x: [N, H]; delta: [N, taps, G] (from the CURRENT row's projection); base: [taps, H].
        out[i,c] = sum_t (base[t,c] + delta[i,t,c//16]) * x[i-t,c],  x[i-t] := 0 if (i % block_size) < t.
        """
        N = x.shape[0]
        xb = x.view(N, self.n_groups, self.group_size)
        coef = base.view(1, self.taps, self.n_groups, self.group_size) + delta.unsqueeze(-1)
        out = coef[:, 0] * xb
        pos_in_block = torch.arange(N, device=x.device) % block_size
        for t in range(1, self.taps):
            shifted = F.pad(xb[:-t], (0, 0, 0, 0, t, 0))  # shifted[i] = xb[i-t]
            shifted = torch.where((pos_in_block >= t).view(-1, 1, 1), shifted, torch.zeros_like(shifted))
            out = out + coef[:, t] * shifted
        return self.r(out.reshape(N, self.H))

    def conv_prepare(self, x: Tensor, prefix: str, block_size: int) -> Tuple[Tensor, Tensor]:
        # M:547-554: one projection of the sublayer INPUT gives both sides' dynamic kernels.
        d = self.lin(x, prefix + "kernel_projection.weight").view(x.shape[0], 2, self.taps, self.n_groups)
        base = self.w[prefix + "base_kernel"]  # [2(side), taps, H]
        return self.conv(x, d[:, 0], base[0], block_size), d[:, 1]

    def conv_finish(self, y: Tensor, d_out: Tensor, prefix: str, block_size: int) -> Tensor:
        return self.conv(y, d_out, self.w[prefix + "base_kernel"][1], block_size)  # M:556-557

    # ------------------------------------------------------------------ (c) attention
    def attention(self, i: int, h: Tensor, block_pos: Tensor, ctx: ContextKV) -> Tensor:
        """Block queries over [context K/V ; block K/V] (M:362-400, W:2622-2670, FA:1557-1571)."""
        p = f"layers.{i}.self_attn."
        B = h.shape[0]
        q = self.lin(h, p + "q_proj.weight").view(B, self.n_heads, self.head_dim)
        k = self.lin(h, p + "k_proj.weight").view(B, self.n_kv, self.head_dim)
        v = self.lin(h, p + "v_proj.weight").view(B, self.n_kv, self.head_dim)
        q = self.rope(self.head_norm(q, p + "q_norm.weight"), block_pos)
        k = self.rope(self.head_norm(k, p + "k_norm.weight"), block_pos)

        K = torch.cat([ctx.k[i], k], 0)  # [T+B, n_kv, d]
        V = torch.cat([ctx.v[i], v], 0)
        kpos = torch.cat([ctx.pos, block_pos], 0)
        qpos = block_pos
        # Visibility on absolute positions == bottom-right aligned FA/flashinfer window (spec (a)).
        # UNSURE: alignment is kernel-internal (FA documents bottom-right; flashinfer assumed the same; z-lab
        # Z:157-171 does exactly this). Only matters once anchor_pos > 2047.
        allow = torch.ones(B, K.shape[0], dtype=torch.bool, device=h.device)
        if self.layer_causal[i]:
            allow &= kpos[None, :] <= qpos[:, None]
        wl = self.layer_window[i]
        if wl >= 0:
            allow &= (qpos[:, None] - kpos[None, :]) <= wl
            if not self.layer_causal[i]:
                allow &= (kpos[None, :] - qpos[:, None]) <= wl  # FA window=(wl, wl) when non-causal

        rep = self.n_heads // self.n_kv  # GQA: q head j uses kv head j // rep
        Kx = K.repeat_interleave(rep, dim=1)  # [T+B, n_heads, d]
        Vx = V.repeat_interleave(rep, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, Kx) * (self.head_dim**-0.5)  # M:278
        s = s.masked_fill(~allow[None], float("-inf"))
        a = torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), Vx)
        a = self.r(a.reshape(B, self.n_heads * self.head_dim))
        return self.lin(a, p + "o_proj.weight")

    def mlp(self, i: int, h: Tensor) -> Tensor:
        p = f"layers.{i}.mlp."
        g = self.lin(h, p + "gate_proj.weight")
        u = self.lin(h, p + "up_proj.weight")
        return self.lin(self.r(F.silu(g) * u), p + "down_proj.weight")  # M:478-482

    # ------------------------------------------------------------------ (b)+(c) block forward
    def embed_block(self, block_ids: Tensor) -> Tensor:
        e = self.embed(block_ids).to(self.device).float()
        if self.mask_embedding is not None:  # copied into the bf16 target table in SGLang (W:1332)
            me = self.r(self.mask_embedding)[None]
            e = torch.where((block_ids == self.mask_token_id)[:, None], me, e)
        if self.noise_embed_scale != 1.0:
            e = self.r(e * self.noise_embed_scale)  # W:2563-2564
        return e

    def forward_block(self, block_ids: Tensor, block_pos: Tensor, ctx: ContextKV) -> Tensor:
        """Returns the final-normed hidden [B, H] (M:806-842 + DFlashDecoderLayer M:593-633)."""
        B = block_ids.shape[0]
        # Residual stream starts at the target embeddings. SGLang's fused add+norm normalises the unrounded
        # fp32 sum `xs` and stores the residual `x` in bf16 (layernorm.py:831-839); the MLP output is added
        # inside the NEXT layer's input norm, and after the last layer inside `norm` (M:607-611, 625, 837).
        x = self.embed_block(block_ids)
        xs = x
        for i in range(self.n_layers):
            p = f"layers.{i}."
            h = self.rms(xs, p + "input_layernorm.weight")
            h, d_attn = self.conv_prepare(h, p + "attention_conv.", B)
            a = self.attention(i, h, block_pos, ctx)
            a = self.conv_finish(a, d_attn, p + "attention_conv.", B)
            xs = x + a
            x = self.r(xs)
            h = self.rms(xs, p + "post_attention_layernorm.weight")
            h, d_mlp = self.conv_prepare(h, p + "mlp_conv.", B)
            m = self.mlp(i, h)
            m = self.conv_finish(m, d_mlp, p + "mlp_conv.", B)
            xs = x + m
            x = self.r(xs)
        return self.rms(xs, "norm.weight")

    # ------------------------------------------------------------------ (d) candidates
    def candidates(self, hidden_pred: Tensor) -> Tuple[Tensor, Tensor]:
        """Top-16 of the target lm_head logits (M:89-152, 1242-1260). -> ids [S,K] long, unary [S,K] fp32."""
        hin = hidden_pred.to(torch.bfloat16) if self.emulate_bf16 else hidden_pred
        logits = self.lm_head(hin)
        if self.emulate_bf16:
            logits = logits.to(torch.bfloat16)  # bf16 matmul output in SGLang
        vals, ids = torch.topk(logits, self.top_k_sel, dim=-1, sorted=True)
        unary = vals.float() * self.output_multiplier
        if self.softcap is not None:
            unary = torch.tanh(unary / self.softcap) * self.softcap
        return ids.long(), unary

    # ------------------------------------------------------------------ (e) selector
    def lattice(self, hidden_pred: Tensor, cand: Tensor, unary: Tensor, anchor_token: int) -> Tensor:
        """score[e,p,c] = unary[e,c] + <A[pred[e,p]] * Wh h[e], Bk[cand[e,c]]> (M:1066-1087, 1132-1158)."""
        S, K = cand.shape
        hp = self.lin(hidden_pred, "candidate_selector.hidden_projection.weight")  # [S, r]
        A = self.w["candidate_selector.predecessor_codebook"]
        Bk = self.w["candidate_selector.successor_codebook"]
        anchor = torch.full((1, K), int(anchor_token), dtype=torch.long, device=cand.device)
        pred_ids = torch.cat([anchor, cand[:-1]], 0)  # slot 0: anchor; slot e: cand[e-1]
        preds = A[pred_ids].float()  # [S, K(p), r]
        keys = Bk[cand].float()  # [S, K(c), r]
        pair = self.r(torch.einsum("spr,scr->spc", self.r(preds * hp[:, None, :]), keys))
        return unary[:, None, :] + pair

    def walk(
        self, cand: Tensor, scores: Tensor, greedy: bool, temperature: float, uniforms: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """One path through the lattice (K:249-292; M:1160-1219). -> tokens [S], q_rows [S,K], idx [S]."""
        S, K = cand.shape
        T = max(float(temperature), 1e-5)  # W:282-286
        prev = 0  # slot 0: all K predecessor rows are the anchor; read row 0
        toks, qs, idxs = [], [], []
        for e in range(S):
            row = scores[e, prev].float()
            if greedy:
                idx = int(torch.argmax(row))  # first max
                q = F.one_hot(torch.tensor(idx, device=row.device), K).float()
            else:
                q = torch.softmax(row / T, dim=-1)
                idx = int((uniforms[e] >= torch.cumsum(q, 0)).sum().clamp_max(K - 1))
            toks.append(cand[e, idx])
            qs.append(q)
            idxs.append(idx)
            prev = idx
        return torch.stack(toks), torch.stack(qs), torch.tensor(idxs, device=cand.device)

    # ------------------------------------------------------------------ one round
    @torch.no_grad()
    def draft(
        self,
        ctx_feats: Optional[Tensor],
        ctx_pos: Tensor,
        anchor_token: int,
        anchor_pos: int,
        block_size: Optional[int] = None,
        temperature: float = 0.0,
        top_k: Optional[int] = -1,
        top_p: float = 1.0,
        uniforms: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        ctx_kv: Optional[ContextKV] = None,
    ) -> DraftResult:
        """Draft block_size-1 tokens after the anchor.

        ctx_feats/ctx_pos: target features of committed positions < anchor_pos (rows at positions
            >= anchor_pos - 2047 are the only ones visible; older rows are masked). Pass ctx_kv instead to
            reuse prepare_context output (SGLang caches it across rounds, spec (g)).
        anchor_token/anchor_pos: the bonus token from the last verify (or the prefill sample) and its position
            (= number of tokens the target has already processed).
        block_size: served 4 (3 drafts); checkpoint trained with 8. Not a prefix relation (spec (g)).
        temperature/top_k: decide greedy vs sampled walk (is_greedy). top_k > 1 and top_p are NOT applied to
            the draft distribution in SGLang (only temperature); they matter for the target p (verify_chain).
        uniforms: [block_size-1] U(0,1) per slot for the sampled walk (else drawn from `generator`).
        Returns DraftResult; see its docstring for what q is.
        """
        del top_p  # accepted for API symmetry; not used by the draft side (spec (e))
        B = int(block_size or self.cfg_block_size)
        if B < 2:
            raise ValueError("block_size must be >= 2")
        if ctx_kv is None:
            ctx_kv = self.prepare_context(ctx_feats, ctx_pos)
        if ctx_kv.pos.numel() and int(ctx_kv.pos.max()) >= anchor_pos:
            raise ValueError("context positions must be < anchor_pos")

        block_ids = torch.full((B,), self.mask_token_id, dtype=torch.long, device=self.device)
        block_ids[0] = int(anchor_token)  # K:177-178
        block_pos = anchor_pos + torch.arange(B, device=self.device)  # K:172

        hidden = self.forward_block(block_ids, block_pos, ctx_kv)
        hpred = hidden[1:]  # rows 1..B-1; anchor row dropped (W:296, 1477)
        cand, unary = self.candidates(hpred)
        scores = self.lattice(hpred, cand, unary, anchor_token)

        greedy = is_greedy(temperature, top_k)
        if not greedy and uniforms is None:
            gdev = generator.device if generator is not None else "cpu"
            uniforms = torch.rand(B - 1, generator=generator, device=gdev)
        if uniforms is not None:
            uniforms = uniforms.to(self.device).float()
        tokens, q_rows, idx = self.walk(cand, scores, greedy, temperature, uniforms)
        return DraftResult(
            tokens=tokens,
            block_tokens=torch.cat([block_ids[:1], tokens]),
            candidate_ids=cand,
            q_rows=q_rows,
            path_index=idx,
            unary=unary,
            scores=scores,
            hidden=hidden,
            greedy=greedy,
        )


# ---------------------------------------------------------------------- (f) verify (target side helpers)
def target_probs(logits: Tensor, temperature: float, top_k: Optional[int] = -1, top_p: float = 1.0) -> Tensor:
    """Target p per verify row as SGLang builds it (DA:106-120, U:1150-1217): /T, then top-k, then top-p."""
    V = logits.shape[-1]
    if is_greedy(temperature, top_k):
        return F.one_hot(logits.argmax(-1), V).float()
    scaled = logits.float() / float(temperature)
    need_k = top_k is not None and 0 < top_k < V
    need_p = top_p < 1.0

    def top_p_renorm(pr: Tensor) -> Tensor:  # sampler.py:1006-1015 (torch fallback of flashinfer)
        srt, order = pr.sort(-1, descending=True)
        srt = srt.masked_fill(srt.cumsum(-1) - srt > top_p, 0.0)
        srt = srt / srt.sum(-1, keepdim=True)
        return torch.zeros_like(srt).scatter_(-1, order, srt)

    if need_k:
        vals, ids = torch.topk(scaled, top_k, dim=-1)
        pk = torch.softmax(vals, -1)
        if need_p:
            pk = top_p_renorm(pk)
        return torch.zeros_like(scaled).scatter_(-1, ids, pk)
    p = torch.softmax(scaled, -1)
    return top_p_renorm(p) if need_p else p


@torch.no_grad()
def verify_chain(
    draft: DraftResult,
    target_logits: Tensor,
    temperature: float,
    top_k: Optional[int] = -1,
    top_p: float = 1.0,
    uniforms: Optional[Tensor] = None,
    final_uniform: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int]:
    """SGLang chain accept (RS:128-239 sampling, K:6-60 greedy). Returns (accept_len, bonus_token).

    target_logits: [B, V] target logits for the verify block draft.block_tokens at anchor_pos..+B-1
                   (row e predicts position anchor_pos+e+1). Apply penalties/bias before calling.
    """
    S = draft.tokens.shape[0]
    toks = draft.tokens.tolist()
    if draft.greedy != is_greedy(temperature, top_k):
        raise ValueError("draft and verify must use the same sampling params (SGLang never mixes them)")
    if is_greedy(temperature, top_k):
        tgt = target_logits.argmax(-1).tolist()
        n = 0
        while n < S and toks[n] == tgt[n]:
            n += 1
        return n, int(tgt[n])
    V = target_logits.shape[-1]
    p = target_probs(target_logits, temperature, top_k, top_p)  # [B, V]
    q = draft.q_dense(V).to(p.device)  # [S, V]
    if uniforms is None:
        uniforms = torch.rand(S, generator=generator)
    if final_uniform is None:
        final_uniform = float(torch.rand(1, generator=generator))
    n = 0
    while n < S:
        x = toks[n]
        qx, px = float(q[n, x]), float(p[n, x])
        if 0.0 < qx <= 1.0 and float(uniforms[n]) * qx < px:
            n += 1
        else:
            break
    dist = p[n] if n == S else torch.clamp(p[n] - q[n], min=0.0)
    total = float(dist.sum())
    cdf = torch.cumsum(dist, 0)
    hit = torch.nonzero(cdf > final_uniform * total)
    bonus = int(hit[0]) if hit.numel() else V - 1  # RS:200-205 fallback
    return n, bonus
