// hc.h — hyper-connections (4 residual streams, X [M, 4·2560] bf16), token embedding and PLE (layer 1).
// Math: arch spec §2.2 / §2.3 / §2.1 (scratchpad arch_decode_spec.md). HC and PLE weights are replicated (plan P0).
#pragma once
#include <string>

#include "core/common.h"
#include "core/weights.h"

namespace q38 {

// Row limit of every call (workspaces are allocated once for it). Small M runs the decode kernels (64-row passes), large
// M the prefill kernels (hc_prefill.cuh: HC from 384 rows, PLE above 64) — a row's values are bitwise the same either
// way (and never depend on M or on the other rows).
constexpr int kHcMaxRows = 8192;

// ---- hyper-connection GatedResidual ------------------------------------------------------------------------------
struct HcW {
  const float* norm = nullptr;   // [10240] hc_norm weight as fp32 (checkpoint fp32 for decoder, bf16 for MTP)
  const void* down = nullptr;    // [320 (+4 inject) , 10240] bf16, kernel layout
  const void* up = nullptr;      // [10240, 320] bf16, kernel layout
  bool inject = true;            // false for the final mixer
  const void* pk = nullptr;      // P012 hc_pack12: lossless 12-bit copy of down (lora rows) and up for the decode kernels
};
HcW hc_upload(const Weights& W, const std::string& prefix, bool inject);  // prefix ".../attn_hyper_connection"

// If bo != nullptr: X ← combine(X, bo [M,2560], inj_prev [M,4] or nullptr = unit weight). Then
// u [M,2560] = gate_mix(norm(X)); inj_out [M,4] = injection logits of this module (if inject).
void hc_combine_mix(const HcW& w, bf16* X, const bf16* bo, const bf16* inj_prev, bf16* u, bf16* inj_out, int M,
                    cudaStream_t s);
// X ← combine(X, bo, inj) only (layer 1, before PLE)
void hc_combine(bf16* X, const bf16* bo, const bf16* inj, int M, cudaStream_t s);
// x0 [M,2560] → X [M,10240] (4 identical streams)
void hc_expand(const bf16* x0, bf16* X, int M, cudaStream_t s);

// ---- embedding (vocab-split, plan P0: rows [124160·r, +124160) on rank r; others → 0; caller all-reduces) ----------
void embed_lookup(const bf16* table_half, int rank, const int* ids, bf16* out, int M, cudaStream_t s);
// Media rows of a partial embedding (Engine::set_media): out[rows[k]] = src[k] (bf16 [2560], 16-byte aligned), or zeros
// where src[k] is null (rank 1: the all-reduce then gives rank 0's row exactly).
void embed_media(bf16* out, const int* rows, const bf16* const* src, int K, cudaStream_t s);

// ---- PLE / n-gram embedding (layer 1) ----------------------------------------------------------------------------
struct PleW;  // opaque, defined in hc.cu
PleW* ple_upload(const Weights& W, int rank);
// n-gram ids from ids [M] and prev [M,2] (tokens at p-2, p-1 per row; EOS rules of §2.3) → partial lookup E [M,2560]
// (rows of the other rank's table half are 0; caller all-reduces E).
void ple_lookup(const PleW* w, const int* ids, const int* prev, bf16* E, int M, cudaStream_t s);
// X ← X + PLE branch (gate, dilated conv). conv_state: per row's sequence [9, 10240] bf16 (last 9 conv inputs),
// rows of one sequence are consecutive (seq_start [B+1]) and advance its state in order.
void ple_apply(const PleW* w, bf16* X, const bf16* E, bf16* const* conv_state, const int* seq_start, int B, int M,
               cudaStream_t s);

// ---- test hooks (device buffers of the last call, valid until the next call) --------------------------------------
const bf16* hc_last_lora();              // [M, 320] lora of the last hc_combine_mix
// P012: the packed block of a module (device bytes, size) and the packing totals of this process
size_t hc_pack_bytes(const HcW& w);
std::string hc_pack_stats();
const bf16* ple_last_kv(const PleW* w);  // [M, 12800] key ‖ value projection of the last ple_apply (M ≤ 1024)
// [M, 10240] conv inputs of the last ple_apply with M ≤ 64 (row m = row m of the call): what the conv state keeps
// (last 9 rows), used to rebuild the state after a speculative verify step (Engine::commit)
const bf16* ple_last_cin(const PleW* w);
// n-gram ids [M, 16] int64 (same hash as ple_lookup)
void ple_ngram_ids(const PleW* w, const int* ids, const int* prev, long long* out, int M, cudaStream_t s);
// shallow copy sharing the table and projections (own workspaces);
// round_norms: PLE norms rounded to bf16 (as the reference loads them)
PleW* ple_debug_clone(const PleW* w, bool round_norms);

}  // namespace q38
