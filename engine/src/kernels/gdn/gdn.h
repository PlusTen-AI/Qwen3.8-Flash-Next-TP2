// gdn.h — Gated DeltaNet core for one layer on this rank (8 k-heads, 24 v-heads, d 128): arch spec §2.4.
// Projections are Lin (linear.h): qkvz [M, 8192] = [q 1024 | k 1024 | v 3072 | z 3072], ba [M, 48] = [b 24 | a 24].
// gdn_core: causal conv1d update (+silu) on q|k|v, l2norm q,k, q·128^-½, g/β, delta-rule recurrence with the fp32
// state, gated RMSNorm (plain w, sigmoid(z)) → core [M, 3072] (input of out_proj).
// State per sequence (rank-local): S [24, 128(V), 128(K)] fp32 (config mamba_ssm_dtype; lossless), conv [3, 5120] bf16.
// Rows of one sequence are consecutive (seq_start [B+1], device) and advance its state in order (T tokens, all kept;
// speculative rows: gdn_core_spec + gdn_commit below).
#pragma once
#include "core/common.h"
#include "kernels/linear/linear.h"
#include "core/weights.h"

namespace q38 {

struct GdnW {
  Lin qkvz, ba, out;
  const bf16* conv_w = nullptr;  // [5120, 4] (q|k|v channels of this rank)
  const float* A = nullptr;      // [24] = -exp(A_log) as fp32
  const float* dt_bias = nullptr;  // [24]
  const bf16* norm_w = nullptr;  // [128]
};
// Core fields only (conv_w, A, dt_bias, norm_w); the Lin projections are filled by the caller (model.cc, linear.h).
GdnW gdn_upload(const Weights& W, int layer);

struct GdnState {
  float* S = nullptr;   // [24,128,128]
  bf16* conv = nullptr; // [3,5120]
};

void gdn_core(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, GdnState* const* states,
              const int* seq_start, int B, int M, cudaStream_t s);
// Same, plus o [M, 3072] bf16 (delta-rule output before the gated norm; golden `post.core_attn_out`) if o != nullptr.
void gdn_core_tap(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, bf16* o, GdnState* const* states,
                  const int* seq_start, int B, int M, cudaStream_t s);
// P005 gdn_fast_prefill (patches.def): with the patch on, calls with M ≥ B · rows (default 32) run the warp-specialised
// kernel (p005_gdn_fast_prefill.cuh, bit-identical); smaller ones (decode) keep the base kernel. Tests lower it to 1.
// Returns the previous value.
int gdn_set_fast_min_rows(int rows);

// ---- speculative decoding (MTP verify): record the rows, keep the first n afterwards ------------------------------
// Once the per-token δ = β(v − (S·e^g)·k) is known, the delta-rule update S ← fma(δ, k, S·e^g) is elementwise, so the
// state after any prefix of the rows can be rebuilt from (e^g, δ, k) without the reductions — bit for bit, as the
// replay repeats the kernel's two roundings (S·e^g, then the fma) in the same order. gdn_core_spec runs a verify step
// (T ≤ kGdnRecRows rows per sequence), writes core exactly as gdn_core would and records each row, but leaves the
// state untouched; gdn_commit then advances the state by the first n rows. Spec + commit(n) == gdn_core on those n
// rows: S, conv state and core rows 0 … n−1 bitwise equal (tests/test_gdn.py spec_*).
constexpr int kGdnRecRows = 16;  // rows per sequence in one spec call (last real token + up to 15 drafts)
struct GdnRec {  // one sequence, one layer; rows of the last gdn_core_spec call (row t = t-th row of the sequence)
  float* eg = nullptr;  // [16, 24]       e^g per row and v-head
  float* dl = nullptr;  // [16, 24, 128]  δ per row, v-head and V row (S row)
  float* k = nullptr;   // [16, 8, 128]   l2-normalised k per row and k-head (as used in the update)
  bf16* x = nullptr;    // [16, 5120]     pre-conv q|k|v channels of the row (= qkvz[row, 0:5120]) for the conv state
};
// One record = one 16-B aligned block of gdn_rec_bytes() = 427,520 B (≈ 15.4 MB per sequence slot for 36 layers);
// gdn_rec_at carves the four arrays out of it (host only, no allocation).
size_t gdn_rec_bytes();
GdnRec gdn_rec_at(void* base);

// As gdn_core (base kernel only, never P005), but S and the conv state are only read; recs [B] (device array, like
// states) receive the rows. Every sequence must have ≤ kGdnRecRows rows (M > B · 16 fails on the host, a longer
// sequence traps in the kernel with a message).
void gdn_core_spec(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, GdnState* const* states,
                   GdnRec* const* recs, const int* seq_start, int B, int M, cudaStream_t s);
// Commit after a verify step, all layers in one launch: pair p = l·B + b (l < L layers, b < B sequences) advances
// *states[p] by the first n[b] rows of *recs[p] (S replay, conv state = last 3 rows of old conv ‖ x rows 0 … n−1).
// states, recs [L·B] and n [B] are device arrays; 0 ≤ n[b] ≤ rows of sequence b in the spec call (n = 0: unchanged).
// Bound by S read + write (36 layers: 108 MiB per sequence).
void gdn_commit(GdnState* const* states, GdnRec* const* recs, const int* n, int L, int B, cudaStream_t s);

// ---- P020 gdn_fused_commit (patches.def): the commit of a verify step folded into the next verify step -------------
// A verify step leaves its kept rows PENDING (record + n per sequence) instead of launching gdn_commit; the next verify
// of that sequence applies them on the way. gdn_core_spec_pending = per sequence b: gdn_commit of rows
// 0 … npend[b]−1 of *pend[b] (the same replay and conv shift, bit for bit), the committed S and conv state written
// back, then gdn_core_spec of the new rows on that state (core, record into *recs[b]; they are the next pending rows).
// S is read once and written once per round instead of read by the verify and read + written by the commit.
// npend [B] is a device array like seq_start; npend[b] ≤ 0: nothing pending (pend[b] is not read, S and conv state are
// not written — exactly gdn_core_spec). npend[b] ≤ kGdnRecRows (more traps). A sequence with pending rows must have
// ≥ 1 row in the call (one with none returns before anything, its pending rows stay unapplied; Engine::verify always
// has 1 … 16).
// Records: pend[b] must hold the rows of sequence b's previous verify step. Double-buffered (two records per slot,
// alternating: the call writes the one not pending) or in place (pend == recs): the kernel reads every pending value
// before its first cluster barrier and stores records only after it, and every record value is shared within one
// cluster only, so a record may be rewritten while it is consumed (tests/test_gdn.py p020_* run both).
// Chain contract: any sequence of rounds (pending verify, keep n) + gdn_flush leaves S, conv state, every core row and
// every record bitwise as gdn_core_spec + gdn_commit after each round (tests/test_gdn.py p020_*).
void gdn_core_spec_pending(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, GdnState* const* states,
                           GdnRec* const* recs, GdnRec* const* pend, const int* npend, const int* seq_start, int B,
                           int M, cudaStream_t s);
// Apply pending rows without a new verify step — before any other use of the state (prefill / plain steps, snapshot,
// reading it), all layers in one launch: gdn_commit itself (pair p = l·B + b, n [B] device). A state whose pending rows
// are dropped (restore, reset) needs no flush, only its pending n set to 0.
inline void gdn_flush(GdnState* const* states, GdnRec* const* pend, const int* n, int L, int B, cudaStream_t s) {
  gdn_commit(states, pend, n, L, B, s);
}

}  // namespace q38
