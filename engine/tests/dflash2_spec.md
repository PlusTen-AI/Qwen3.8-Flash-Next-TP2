# DFlash2 draft side — one speculative round, single sequence

Sources (read-only). Abbreviations used for file:line references:
- `M`  = sglang `python/sglang/srt/models/dflash.py`
- `W`  = sglang `python/sglang/srt/speculative/dflash_worker_v2.py`
- `U`  = sglang `python/sglang/srt/speculative/dflash_utils.py`
- `K`  = sglang `python/sglang/kernels/ops/speculative/dflash.py`
- `DA` = sglang `python/sglang/kernels/ops/speculative/dspark/dspark_accept.py`
- `RS` = sglang `python/sglang/kernels/ops/speculative/reject_sampling.py`
- `FA` = sglang `python/sglang/srt/layers/attention/flashattention_backend.py`, `FI` = `.../flashinfer_backend.py`
- `Z`  = z-lab `dflash/model.py` (HF reference; it DOES contain the DFlash2 path: conv + selector, Z:478-547, Z:630-659)

Notation: `P` = anchor position (= number of tokens already processed by the target), `B` = block size
(served 4, trained 8), `H`=2560, `G` = H/16 = 160 conv groups, `K`=16 candidates, `r`=256 selector rank.

Both sources agree on every point below unless stated.

## (a) Target features → context K/V

- Which positions: every committed position `< P`. At prefill, all prompt positions `0..L-1` are
  materialized (W:2372-2396); the token sampled by prefill becomes the first anchor at `P=L` (W:2401-2404).
  After each verify, the first `commit_lens = accept_len+1` rows of the verify block (anchor + accepted
  drafts, positions `P..P+accept_len`) are materialized (W:2953-2970, prefix-valid write W:1924-1971).
  The new bonus token (position `P+accept_len+1`) has no features yet; it is the next anchor.
  Z does the same: `target_hidden` = rows `[:produced]` of the verify output (Z:307-308), cache cropped to `start` (Z:250, 304).
- Feature of position `i` = target hidden at input position `i` (the row that predicts token `i+1`),
  12,800 = concat over target_layer_ids `[5,15,25,35,45]` in that order (Z:37-43 cats in `layer_ids` order;
  M:705-717 sizes fc as `len(target_layer_ids)*H`). The 4-stream averaging is the drafter-repo patch (not in these files).
- Projection: `ctx_hidden = RMSNorm_hidden_norm(fc(feat))` (M:788-803; Z:584). Computed once per position.
- Per layer `l`: `k = k_norm(k_proj_l(ctx_hidden))` per head, RoPE at the position's absolute index;
  `v = v_proj_l(ctx_hidden)` (W:1951-1970, W:2001-2024; `kv_proj_only` M:407-433, `apply_k_norm` M:435-438,
  `apply_k_rope` M:440-444). The SAME `ctx_hidden` feeds all 5 layers.
  NO `input_layernorm` (`prepare_context_hidden_for_kv` is identity for this class, M:783-786; only Laguna
  overrides it M:1032-1035) and NO conv on context rows. Z agrees (`k_proj(target_hidden)` Z:384-390).
- Context rows are only K/V, never queries. Query rows = the block (B rows).
- Sliding window: all layers `sliding_attention`, `sliding_window` 2048 → SGLang `window_left = 2047`
  (U:433-450). `is_causal: false` → `AttentionType.ENCODER_ONLY` (M:155-161, M:164-189).
  FA: `causal=False`, `window=(2047, 2047)` (FA:1557-1571); FI: `causal=False`, `window_left=2047` (FI:1372-1390).
  Those lines only set `causal=False` + window 2047; the query alignment is kernel-internal: FA documents
  bottom-right alignment (query row j ↔ key index `kv_len - q_len + j`), flashinfer is assumed to match
  (UNSURE 7). Under that, block row at absolute position `t` sees keys at positions `max(0, t-2047) .. P+B-1`.
  Z states it explicitly: `q-k < 2048` and `k-q < 2048` with bottom-right query positions (Z:157-171).
  No custom mask is attached to the draft forward (`custom_mask=None`, draft_worker_common.py:134-145).
  ⇒ the drafter needs context positions `max(0, P-2047) .. P-1` (at most 2047 rows).
- RoPE: neox style, rotary_dim = head_dim = 128 unless config has `partial_rotary_factor<1` or
  `rope_scaling.rope_dim` (M:249-276); theta from config (10000), `inv_freq = theta^(-arange(0,128,2)/128)`
  (rotary_embedding/base.py:233-266). Context rows: their own absolute positions; block rows: `P..P+B-1`
  (W:2519-2523, K:172-186). Z: same position ids (Z:246).

## (b) Block input tokens and embedding

- `block_ids = [anchor, MASK, …, MASK]` (B entries), `anchor` = bonus token from the previous verify
  (or the prefill sample), `MASK = mask_token_id = 248077` (W:2517-2518, K:177-181; Z:197-199, 235).
- Embedding: the TARGET's input embedding table (W:217-225, 2562; Z:135-138, 238-242), times
  `noise_embed_scale` = `target_model.get_dflash_noise_embedding_scale()` if the target class defines it,
  else 1.0 (W:522-526, 2563-2564). No model in this SGLang checkout defines it → 1.0.
  Z: `dflash_config.input_embedding_scale` default 1.0 (Z:241) — absent from our config → 1.0.
- If the draft folder has `mask_embedding.pt`, its vector overwrites row `mask_token_id` of the target
  embedding table at load (W:1282-1342).
- The residual stream starts from these embeddings; target features never enter the residual.

## (c) One draft layer (pre-norm; M:593-633, Z:433-475)

```
h  = RMSNorm_in(x)                          # input_layernorm, eps 1e-6, x*rsqrt(mean x²+eps)*w
h, dA = attention_conv.prepare(h)           # M:547-554
a  = Attention(h)                           # q,k,v all from the CONV'D h (M:368-389)
a  = attention_conv.finish(a, dA)           # M:556-557
x  = x + a
h  = RMSNorm_post(x)
h, dM = mlp_conv.prepare(h)
m  = down(silu(gate(h)) * up(h))            # M:447-482 (SiluAndMul: first half gate, second up)
m  = mlp_conv.finish(m, dM)
x  = x + m
```
(SGLang fuses the residual add into the next norm, M:607-611, 625, 837 — same math.) After 5 layers:
`out = RMSNorm_norm(x)` (M:833-837; Z:597). Residual is stored in bf16 in SGLang (layernorm.py:831-839).

Attention (M:362-400; Z:369-420): `q = q_proj(h)` → [B,32,128], `k,v = k_proj(h), v_proj(h)` → [B,8,128];
per-head RMSNorm with `q_norm`/`k_norm` weight [128] (eps 1e-6), then RoPE at block positions
(fused bf16 kernel U:1227-1303 rounds the normed value to bf16 before rotation). K/V of the block are
appended after the context K/V; GQA 32/8 (q head `h` uses kv head `h//4`); scale `128^-0.5` (M:278);
bidirectional inside the block, window as in (a). No sinks, no v-scale (`attention_value_scale` absent,
U:746-758). Output `[B, 4096]` → `o_proj` → `[B, 2560]`.

Two-tap dynamic grouped convolution (`DFlashGroupedConv`, M:486-557; Z:478-512):
- params per conv: `base_kernel [2(side), taps=2, H]`, `kernel_projection: H → 2*taps*G = 640` (no bias).
- `prepare(x)`: `δ = kernel_projection(x).reshape(N, 2, taps, G)` (feature index = side*320 + tap*160 + g);
  returns `conv(x, δ[:,0], base_kernel[0])` and keeps `δ[:,1]` for `finish(y) = conv(y, δ[:,1], base_kernel[1])`.
  Both δ sides come from the SAME pre-conv input `x` (the norm output).
- `conv(x, δ, base)[i, c] = (base[0,c] + δ[i,0,g(c)]) * x[i,c] + [i%B ≥ 1] * (base[1,c] + δ[i,1,g(c)]) * x[i-1,c]`,
  `g(c) = c // 16`. base is per channel, δ per group (broadcast over its 16 channels) and taken from the
  CURRENT row `i`. Tap 1 mixes the previous block row; causal; resets at each block start
  (`position & (block_size-1)` / `% block_size`, M:490-497). Single block ⇒ row 0 has no tap-1 term.
  Context rows never enter the conv. `block_size` here = served block size (`set_block_size`, M:764-775, W:507).
- init values (`base[:,0]=1`, M:524-525) are irrelevant: loaded from the checkpoint.

## (d) Final norm → logits

- Only rows `1..B-1` (the mask rows) are used; row 0 (anchor) is discarded (W:296, 1477; Z:249 `[:, 1-B:]`).
- `logits = out.to(lm_head dtype) @ W_lm[:vocab].T` with the TARGET lm_head (no target final norm; the
  draft's own `norm` was applied) (M:89-102, 1251-1260; W:2476). Org vocab only (padded tail excluded).
- `unary, cand = top16(logits)` sorted descending (flashinfer `top_k(sorted=True, deterministic=True)` or
  `torch.topk`, M:64-68, 105-152); `unary = vals.float() * output_multiplier` then optional softcap
  (M:1242-1249). Our config has neither → `output_multiplier=1`, no softcap (U:684-688).
  Z: `topk(..., sorted=False)` on the transformed logits (Z:525, 599-605) — same set, different order.

## (e) Candidate selector (M:1066-1219, W:232-308, K:249-320; Z:515-547)

- `hp[e] = hidden_projection(out[e])` → [r] for each draft slot `e = 0..B-2` (M:1146).
- Predecessor ids: slot 0 → the anchor token (repeated over the K rows); slot e≥1 → `cand[e-1, p]` (M:1080-1083).
- Lattice: `score[e, p, c] = unary[e, c] + Σ_r A[pred[e,p], r] * hp[e, r] * Bk[cand[e,c], r]`
  with `A = predecessor_codebook [V, r]`, `Bk = successor_codebook [V, r]` (M:1076-1087, 1140-1142).
  Slot 0's K rows are identical (all anchor); the walk reads row 0.
- Walk (one path, no tree): slot 0 uses row `score[0, 0, :]`; slot e uses row `score[e, idx[e-1], :]`
  (K:268-292; torch fallback M:1180-1209 is the same).
  - greedy (request `top_k ≤ 1`; SGLang turns temperature < 1e-6 into `temperature=1, top_k=1`,
    sampling_params.py:222-225; mask dspark_draft.py:111-119): `idx = argmax` (first max), q = one-hot (K:273-280, M:1214-1218).
  - sampling: `prob = softmax(row / T)`, `T = max(request temperature, 1e-5)` (W:282-286, 1485-1489);
    `idx = #{k : u ≥ cumsum(prob)[k]}` clamped to K-1, one uniform per slot (K:281-289, M:1182-1197).
    The request's top-k / top-p are NOT applied to the draft distribution; only temperature.
- Drafted tokens: `tok[e] = cand[e, idx[e]]`, e = 0..B-2 (K:291, M:1209); block sent to verify =
  `[anchor, tok[0..B-2]]` (W:2792-2794). Z: same walk, `torch.multinomial` instead of inverse CDF (Z:524-547).

## (f) Accept rule

- Greedy batch (all rows greedy): selector sample is not kept; accept = longest prefix with
  `tok[e] == argmax target_logits[e]`, bonus = target argmax at the first mismatch / last row (W:2211-2253, K:6-60).
- Sampling: q is NOT one-hot. `q_e` = the K-vector `prob` of slot e at its realized predecessor row,
  supported only on `cand[e]`, scattered into a dense [B-1, V] buffer (W:1516-1547, M:1210-1213).
  Target `p` = `softmax(logits/T)` (DA:106-111), or with request top-k then top-p renorm (U:1150-1217;
  `build_speculative_verify_target_probs`), plus any logit bias / penalties (W:2888-2893).
  Chain rule (RS:128-160, thresholds 1.0, DA:144-159): for e = 0.. : accept `tok[e]` iff `u_e * q_e(tok) < p_e(tok)`
  (and `0 < q ≤ 1`); first reject at e → sample from `normalize(relu(p_e - q_e))` over the full vocab
  (RS:164-239); all accepted → sample from `p_{B-1}`. Z identical (Z:94-124, 277-285).
- Greedy rows inside a sampling batch: q one-hot (M:1214-1218), p one-hot (top_k=1) → exact match test.
- Optional `--speculative-use-block-verification` (RS:52-127) changes the rule; default off.

## (g) State between rounds; block size 4 vs 8

- Kept: draft KV cache of context K/V per layer (from target features, computed once at prefill / after
  verify, W:2953-2970), the bonus token, `seq_len` (W:2975-2978, dflash_info_v2.py:36-46).
  Block K/V are written to slots `P..P+B-1` during the draft forward, then the committed prefix of those
  slots is overwritten by context K/V; the rest are beyond `seq_len` and are rewritten next round.
  Nothing is recomputed except the B block rows. Z: same via cache crop (Z:250, 304).
- Optional `--speculative-draft-window-size` (compact draft cache) is off by default (spec.py:164-167);
  the native 2047 window still applies inside the attention kernel.
- Block size: worker uses `--speculative-num-draft-tokens` (W:491-507) and pushes it into the convs
  (M:764-775). With B=4: 3 mask rows, 3 selector slots, conv resets every 4 rows. Because block rows attend
  to each other bidirectionally, B=4 hidden states are NOT a prefix of B=8 ones (the extra mask rows change
  attention) — drafts for slots 0..2 can differ between B=4 and B=8. The "QSA" limit is not in this code.

## Weight names (checkpoint → SGLang param)
Checkpoint names (dflash2_tensors.txt) have no `model.` prefix. SGLang fuses `q_proj/k_proj/v_proj` →
`self_attn.qkv_proj` and `gate_proj/up_proj` → `mlp.gate_up_proj` (M:845-852); everything else loads
by its own name, incl. `candidate_selector.{predecessor,successor}_codebook` as bare Parameters
(M:1124-1130; Z maps them to `.weight` of nn.Embedding, Z:630-637). No embed_tokens / lm_head.

## UNSURE (cannot be settled from these files)
1. `mask_embedding.pt` in the drafter folder? If present, it replaces target embedding row 248077 (W:1282-1342).
2. Target-side `noise_embed_scale`: the drafter repo's SGLang patch could add `get_dflash_noise_embedding_scale`
   to the Qwen3.8 target class (W:522-526). This checkout has none → 1.0.
3. Feature layout: concat order 5,15,25,35,45 and the 4-stream mean come from the drafter patch, not visible here.
4. RoPE: confirm drafter config.json has no `partial_rotary_factor` / `rope_scaling.rope_dim`; if present SGLang
   rotates only part of the head (M:249-276, factory.py:156-157).
5. Top-16 order and ties: SGLang sorted, Z unsorted. Only affects which sample a given uniform maps to
   and argmax tie-breaks; distributions are identical.
6. Numerics: SGLang keeps bf16 activations and a bf16 residual; matmuls bf16 with fp32 accumulate; the
   lattice einsum runs in bf16. Greedy argmax can flip on near-ties vs an fp32 reference.
7. Window alignment in the attention kernels (FA/flashinfer internals, not in these files): assumed
   bottom-right, i.e. the 2047 window is measured from each block row's absolute position. Matters only
   when P > 2047 (i.e. the long-context regime). Z's mask (Z:157-171) agrees with this reading.
