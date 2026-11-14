// qsa.h — QSA full-attention core for one layer on this rank (12 q heads + gate, 1 kv head, head_dim 256): spec §2.5.
// Projection is Lin (linear.h): qkv [M, 7296] = [q0,g0,…,q11,g11 (6144) | k 256 | v 256 | iq 4×128 | ik 128].
// qsa_core: q/k Gemma RMSNorm + NeoX RoPE on dims 0..63 (cos/sin table as vLLM's cos_sin_cache, bf16: qsa_rope_set —
// default YaRN factor 1.5625, mscale folded), K/V → cache at the row's position, indexer q (norm+rope), raw ik ring, compressed key every 4th position,
// selection = top-512 blocks by Σ_h ReLU(q_h·k̂) (+ open tail), sparse attention (scale 1/16, online softmax fp32),
// gate: out = o·sigmoid(gate) → attn [M, 3072] (input of o_proj).
#pragma once
#include <string>

#include "core/common.h"
#include "kernels/linear/linear.h"
#include "core/weights.h"

namespace q38 {

// Position cap. The RoPE table and the selection workspace follow the configured window (QsaRope::rows ≤ the cap):
// a window ≤ 262,144 allocates the 256K-era workspace and runs only the base selection tier (the 256K-era selection
// code plus a bounds trap; same bits); the table has `rows` rows (the window, not 262,144).
constexpr int kQsaMaxPos = 1 << 20;
constexpr int kQsaOrigPos = 262144;  // config.json max_position_embeddings = rope original_max_position_embeddings

// RoPE of the QSA layers (rotary dims 0..63 of q, k, indexer q and compressed keys; θ = 1e7): vLLM's cos_sin_cache
// computed op by op in fp32 on the GPU, then bf16 — row p = cos(p·inv_freq) ‖ sin(p·inv_freq) (32 + 32).
//   factor ≤ 1: plain RoPE (vLLM RotaryEmbedding, rope_type "default": inv_freq = 1/θ^(2i/64), no mscale).
//   factor > 1: YaRN (vLLM YaRNScalingRotaryEmbedding: beta_fast 32, beta_slow 1, correction range floor/ceil-truncated
//     and clamped, mscale = 0.1·ln(factor) + 1 folded into cos/sin), correction range from ramp_len: 262,144 (HF
//     transformers, original_max_position_embeddings) → dims 14..22; vLLM's MRotaryEmbedding uses max(original, served
//     window), e.g. 1,048,576 → 16..24. ramp_len ≤ 0 → original.
// The defaults reproduce the 256K-era table's values bit for bit (rows < 262,144).
struct QsaRope {
  double factor = 1.5625;
  int original = kQsaOrigPos;
  int ramp_len = kQsaOrigPos;
  int rows = kQsaOrigPos;  // table rows = the served window (≤ kQsaMaxPos); every position must be < rows
};
// Builds the table shared by all QSA layers and the MTP head and sizes the selection workspace for `rows`; returns the
// table. Call before the first qsa_upload (the engine: rows = ctx); without a call qsa_upload builds the defaults.
// Called again (tests) it rebuilds both: QsaW from earlier uploads must take the returned table.
const bf16* qsa_rope_set(const QsaRope& r);
const QsaRope& qsa_rope();  // the current parameters
// Test hook: the table for r (the same kernel) into dst [r.rows, 64] bf16 (device memory); synchronous.
void qsa_rope_fill(const QsaRope& r, bf16* dst);

struct QsaW {
  Lin qkv, o;
  const bf16 *q_norm = nullptr, *k_norm = nullptr, *iq_norm = nullptr, *ik_norm = nullptr;
  const bf16* rope = nullptr;  // [rows, 64] cos(32) ‖ sin(32), bf16 (vLLM cos_sin_cache, qsa_rope_set); shared by all layers
};
// Core fields only (norm weights, RoPE tables); the Lin projections are filled by the caller (model.cc, linear.h).
QsaW qsa_upload(const Weights& W, int layer);
// Same by tensor prefix (".../self_attn."), e.g. the MTP layer "mtp.layers.0.self_attn." (same shapes, bf16 norms).
QsaW qsa_upload(const Weights& W, const std::string& attn_prefix);

// Per sequence, per layer (rank-local). Capacity in tokens `cap` (multiple of 4).
struct QsaCache {
  bf16* kv = nullptr;        // [cap, 512]  K(256) ‖ V(256) of this rank's kv head
  bf16* ik_raw = nullptr;    // [cap, 128]  raw indexer keys (pre-norm, pre-rope)
  bf16* ik_comp = nullptr;   // [cap/4, 128] compressed indexer keys (normed, roped at group start)
  int cap = 0;
};

// pos [M] (device): absolute position of each row (its cache index); rows of one sequence consecutive (seq_start [B+1]).
// rp (optional, device, [M, 6] int32): the rows' RoPE positions when they differ from pos (images/videos: mRoPE) —
// (t, h, w) of the row, then (t, h, w) of position pos − 3 (the rotation of the compressed key a row completes when
// pos % 4 == 3); null = pos for every component (text, bit for bit the plain table).
// sel_out (optional, may be null): [M, 2052] int32 selected token indices as in vLLM's indexer output (for tests).
// Selected tokens are listed in ascending order (blocks, then the open tail); top-k ties go to the lower block index.
// M ≤ 64 runs the decode kernels (qsa.cu), larger M the prefill kernels (qsa_prefill.cu: indexer GEMM, per-row
// attention with the LSE merge on chip). Both compute every row with the same operations in the same order, so a
// row's selection, output and cache writes are bitwise the same for any M, B, path or chunking of its sequence.
// max_pos (optional host hint, −1 = unknown): the largest row position of the call; if given it must be ≥ every row's
// position and < the table rows. It picks between kernel paths that give the same bits (P021: k_attn + k_merge for
// small work, the cluster kernel otherwise) and sizes the selection: calls that see ≤ 65,536 blocks (positions <
// 262,144) run the base tier (256K-era selection code: static smem bitsets, 16-bit P009 candidate blocks); larger the wide
// tier (bitsets in dynamic smem sized for the call's blocks, 32-bit candidate blocks; same operations ⇒ same bits).
// Unknown, or while the stream is being captured into a CUDA graph (positions are not part of a graph), the selection
// is sized for the whole window. A selection CTA that sees more blocks than it was sized for traps.
void qsa_core(const QsaW& w, const bf16* qkv, const int* pos, bf16* attn, QsaCache* const* caches, const int* seq_start,
              int B, int M, int* sel_out, cudaStream_t s, int max_pos = -1, const int* rp = nullptr);
// Only qsa_core's cache writes for these rows (k_prepare: K‖V, raw indexer keys, completed compressed keys) — no
// attention. A later qsa_core over some of the rows writes the same cache values again (P023 mtp_last_rows).
void qsa_prepare(const QsaW& w, const bf16* qkv, const int* pos, QsaCache* const* caches, const int* seq_start, int B,
                 int M, cudaStream_t s, const int* rp = nullptr);

// qsa_core keeps a grow-only internal workspace: 15 KB per row (Q, indexer q, selection) + the decode path's scores and
// attention partials for up to 64 rows (≈ 70 MB) + for M > 64 the prefill path's fp32 scores of 1,024 rows (268 MB);
// M = 8,192 ≈ 460 MB. With P009 qsa_fast_index on: scores of ≤ 511 rows (134 MB, calls below 512 rows) + candidates of
// ≤ 8,192 rows with fallback slices (176 MB) instead of the 268 MB: M = 8,192 ≈ 500 MB. Those sizes hold for windows ≤
// 262,144; above, score rows (decode 64, prefill 511 / 1,024, P009 fallback slices 96) take window/4 floats instead of
// 65,536 and P009 candidate blocks 4 B instead of 2 (1M window, M = 8,192: +0.58 GB with P009 on, +0.86 GB off). Reserving it
// up front for the largest M keeps cudaMalloc/attribute calls out of the step (do it before CUDA-graph capture). One
// workspace per process: qsa_core calls must not run concurrently on different streams.
void qsa_reserve(int max_rows);
// Test hook: device pointers to the last call's normed+roped main Q [M,12,256] and indexer q [M,4,128] (bf16).
void qsa_last_q(const bf16** q, const bf16** qi);
// Test hook (P021 qsa_decode_attn): the decode path's attention stage alone (after the selection) on given inputs —
// q [M,12,256] normed+roped Q, qkv [M,7296] (gates), sel [M,2052]; variant 0 = k_attn + k_merge (base), 1 = k_pattn,
// 2 = P021 as qsa_core runs it for this max_pos hint (base kernels for small work, see p021_qsa_decode_attn.cuh), 3 = the
// P021 cluster kernel for any call. M ≤ 64. No sync.
void qsa_decode_attn_stage(int variant, const bf16* q, const bf16* qkv, const int* sel, QsaCache* const* caches,
                           const int* seq_start, int B, int M, bf16* attn, cudaStream_t s, int max_pos = -1);
// Test hook (P009 qsa_fast_index): device pointers to the candidate counts [rows of the last round, splits] and the
// fallback list ([0] = rows that fell back in the last call), number of key splits.
void qsa_p009_last(const int** cnt, const int** fb, int* splits);

}  // namespace q38
