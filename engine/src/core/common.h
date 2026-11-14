// common.h — shared types and number formats for the Qwen3.8-Flash-Next engine (GB10, sm_121a, TP=2).
//
// Weight formats as stored in the checkpoint (row-major [N, K], N = output rows, K = input):
//   MXFP8  : E4M3 bytes [N, K] + UE8M0 scale bytes [N, K/32];       w = e4m3(q) · 2^(s-127)
//   NVFP4  : packed e2m1 [N, K/2] (low nibble = even k) + E4M3 scale [N, K/16] + fp32 global g;
//            w = e2m1(code) · e4m3(scale) · g          (routed experts: CSF-encoded scales, decoded at load)
//   bf16   : HC, router, lm_head, embed, PLE projections, norms, conv weights
// Activations between kernels are bf16; math inside kernels is fp32 (see tools/byte_floor.py / golden dump).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace q38 {

using bf16 = __nv_bfloat16;

// ---- model constants (config.json text_config; checked against the config at load) ------------------------------
constexpr int kHidden = 2560, kLayers = 48, kStreams = 4, kHcRank = 320, kHcDown = 324;  // 320 lora + 4 inject
constexpr int kVocab = 248320, kVocabHalf = 124160;
constexpr int kExperts = 512, kTopK = 10, kExpInter = 640, kSharedInter = 640;
// GDN (per rank: 8 k-heads, 24 v-heads)
constexpr int kGdnKHeads = 16, kGdnVHeads = 48, kGdnDk = 128, kGdnDv = 128, kGdnConv = 4;
// QSA (per rank: 12 q heads, 1 kv head)
constexpr int kQHeads = 24, kKvHeads = 2, kHeadDim = 256, kRotDim = 64;
constexpr int kIdxHeads = 4, kIdxDim = 128, kIdxRatio = 4, kIdxBudgetBlocks = 512;  // 2048 tokens
constexpr float kEps = 1e-6f;
inline bool is_qsa_layer(int i) { return i % 4 == 3; }

// ---- number formats (device) -------------------------------------------------------------------------------------
__host__ __device__ inline float e2m1_to_float(uint32_t c) {  // 4-bit code: s eem
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[c & 7];
  return (c & 8) ? -v : v;
}
__host__ __device__ inline float e4m3_to_float(uint8_t b) {
#ifdef __CUDA_ARCH__
  __nv_fp8_e4m3 x;
  x.__x = b;
  return float(x);
#else
  int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = (e == 0) ? (m / 8.f) * (1.f / 64.f) : (1.f + m / 8.f) * float(1 << e) / 128.f;  // bias 7
  if (e == 15 && m == 7) v = __builtin_nanf("");
  return s ? -v : v;
#endif
}
__host__ __device__ inline float ue8m0_to_float(uint8_t s) {  // 2^(s-127)
#ifdef __CUDA_ARCH__
  return __int_as_float(s == 0 ? 0x00400000 : (int(s) << 23));
#else
  union { uint32_t u; float f; } c;
  c.u = s == 0 ? 0x00400000u : (uint32_t(s) << 23);
  return c.f;
#endif
}

// ---- small helpers -----------------------------------------------------------------------------------------------
__host__ __device__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// A weight matrix resident on one rank, in the checkpoint's row-major layout unless a kernel says otherwise.
struct WMat {
  enum Fmt : int { BF16 = 0, MXFP8 = 1, NVFP4 = 2 } fmt = BF16;
  int n = 0, k = 0;              // logical rows (outputs) × cols (inputs) on this rank
  const void* w = nullptr;       // bf16 [n,k] | e4m3 [n,k] | packed e2m1 [n,k/2]
  const uint8_t* s = nullptr;    // MXFP8: ue8m0 [n,k/32] | NVFP4: e4m3 [n,k/16]
  float g = 1.f;                 // NVFP4 global scale (weight_scale_2)
};

}  // namespace q38
