// linear.h — y[M,N] = x[M,K] · Wᵀ. M ≤ 64 (decode): skinny GEMM, weights streamed once per call. 64 < M ≤ 8192
// (prefill): tiled tensor-core GEMM over the same packed weights (linear_prefill.cuh) whose every output is bitwise the
// decode kernel's (same MMA operands, chain and split order) — the switch at M = 64 changes no value. Patch P004
// gemm_tune (p004_gemm_tune.cuh) replaces the prefill tiles of split-1 shapes with faster ones, same bits.
// Formats: MXFP8 (W8A16: bf16 activations, weights dequantized exactly e4m3·2^(s−127), fp32 accumulate) and bf16.
// Patch P013 lin_pack12 (p013_pack12.cuh): a bf16 Lin may also hold a lossless 12-bit copy (pack12 at upload) that the
// decode kernel reads instead (M ≤ 64, same MMA operands → same bits, 25 % fewer bytes); M > 64 reads the bf16 copy.
// Several checkpoint tensors can be fused along N at upload (vLLM's local output order):
//   GDN qkvz = in_proj_qkv(seg slice) ‖ in_proj_z   ·  GDN ba = in_proj_b ‖ in_proj_a
//   QSA qkv  = q_proj ‖ k_proj ‖ v_proj ‖ indexer.index_qk_proj   ·  shared gate_up = gate_proj ‖ up_proj
// Contract (all kernels in this engine): deterministic and batch-invariant — a row's result never depends on M or on
// the other rows (fixed reduction order, no float atomics).
#pragma once
#include <string>
#include <vector>

#include "core/common.h"
#include "core/weights.h"

namespace q38 {

constexpr int kMaxRows = 8192;   // M limit of lin_forward
constexpr int kDecodeRows = 64;  // M ≤ this: skinny decode kernel; above: tiled prefill GEMM (same per-row arithmetic)

struct Lin {
  int fmt = WMat::BF16;  // WMat::BF16 | WMat::MXFP8 | WMat::NVFP4 (lin_make_nvfp4, decode only)
  int n = 0, k = 0;
  void* w = nullptr;      // device, kernel layout (owned)
  uint8_t* s = nullptr;   // device scales, kernel layout (MXFP8)
  int npad = 0;           // n rounded up to the 64-row CTA tile (zero rows)
  int split = 1;          // K splits (fixed per shape at upload, summed in fixed order; never depends on M)
  float gs = 1.f;         // NVFP4: global scale · 2^8
  // P013 lin_pack12 (bf16 only): 12-bit copy for the decode kernel, layout in p013_pack12.cuh; null when not built
  uint8_t* pk = nullptr;       // device: 768 B per (16-row tile, k32 block)
  uint8_t* pk_base = nullptr;  // device: per row exponent window base, npad bytes
  uint32_t* pk_xo = nullptr;   // device: per tile first exception [npad/16 + 1]
  uint32_t* pk_x = nullptr;    // device: exceptions (kb, lane, slot, exact high byte), sorted per tile
  int pk_nx = 0;               // exceptions in pk_x
};

// Upload this rank's slices of `names` (weights; for MXFP8 the matching `<name>_scale` = `...weight_scale` is used),
// concatenated along N in the given order. pack12 (bf16 only): also build the P013 12-bit copy (checked bit-exact).
Lin lin_upload(const Weights& W, const std::vector<std::string>& names, bool pack12 = false);
// Same from host row-major bytes: w [n,k] (e4m3 or bf16), s [n,k/32] ue8m0 (MXFP8). split 0 = chosen from the shape.
Lin lin_make(int fmt, int n, int k, const uint8_t* w, const uint8_t* s, int split = 0, bool pack12 = false);
// Tests: the P013 copy unpacked on the device to bf16 [n, k] row-major (same unpacking as the decode kernel).
void lin_pack12_unpack(const Lin& L, bf16* out, cudaStream_t s);
// NVFP4 copy of a bf16 [n, k] matrix (quantized here; k % 256 == 0), for draft-only weights: decode kernel only (M ≤ 64).
Lin lin_make_nvfp4(int n, int k, const bf16* w);
void lin_free(Lin& L);

// x [M, k] bf16 row-major, 16-byte aligned; M ≤ kMaxRows. No workspace or global state (split-K partials are reduced
// inside a thread-block cluster), so calls may run concurrently on any streams.
void lin_forward(const Lin& L, const bf16* x, bf16* y, int M, cudaStream_t s);    // y bf16 [M, n]
void lin_forward_f32(const Lin& L, const bf16* x, float* y, int M, cudaStream_t s);  // y fp32 [M, n] (logits)

// Tests only: force the P004 gemm_tune tile (-1 = automatic, 0 = 128×128, 1 = 128×64); applies while the patch is on.
void lin_p004_tile(int t);

}  // namespace q38
