// qsa_dev.cuh — QSA internals shared by qsa.cu (prepare, decode path, top-k selection) and qsa_prefill.cu (prefill
// path). Both paths compute every per-row value with the same operations in the same order (see qsa.cu header), so a
// row's selection, output and cache writes are bitwise independent of which path ran it.
#pragma once
#include <algorithm>
#include <cmath>

#include "kernels/qsa/qsa.h"
#include "core/log.h"

namespace q38::qsa_dev {

constexpr int kQH = kQHeads / 2;                                      // 12 q heads per rank
constexpr int kQkv = 7296, kOffK = 6144, kOffV = 6400, kOffIq = 6656, kOffIk = 7168;
constexpr int kBudget = kIdxBudgetBlocks;                             // 512 blocks
constexpr int kSelCols = kBudget * kIdxRatio + kIdxRatio - 1;         // 2051 token columns
constexpr int kSelW = kSelCols + 1;                                   // + count column
constexpr int kTileN = 32;                                            // tokens per attention tile
constexpr int kMaxTiles = (kSelCols + kTileN - 1) / kTileN;           // 65
// Selection tiers: calls that see ≤ kMaxBlk blocks (positions < 262,144) run the base tier: the 256K-era code (stride
// kMaxBlk, static smem bitsets, 16-bit P009 candidate blocks); larger calls the wide tier (runtime stride and bitset
// words, dynamic smem, 32-bit candidate blocks) up to kMaxBlkW blocks. Same operations in both ⇒ same bits.
constexpr int kMaxBlk = kQsaOrigPos / kIdxRatio;                      // 65,536: base tier score row stride
constexpr int kMaxBlkW = kQsaMaxPos / kIdxRatio;                      // 262,144: wide tier limit
constexpr unsigned kFull = 0xffffffffu;
constexpr int kBins = 2048, kBinBase = 123 << 7;  // score histogram: bf16-granular bins over [2^-4, 2^12)
constexpr int kMG = 4, kMT = (kMaxTiles + kMG - 1) / kMG;  // LSE merge: tile groups, tiles per group (17)
constexpr int kAT = 128;   // attention threads
constexpr int kRS = 264;   // smem row stride (bf16) of Q/K/V: 528 B → conflict-free ldmatrix
constexpr int kPS = 40;    // smem row stride (bf16) of P: 80 B
constexpr int kKS = 136;   // smem row stride (bf16) of 128-dim indexer rows: 272 B → conflict-free ldmatrix
constexpr int kDecRows = 64;  // calls with M ≤ kDecRows take the decode path, larger ones the prefill path

// ---- small helpers -----------------------------------------------------------------------------------------------
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ int score_bin(float s) {
  return min(max(int(__float_as_uint(s) >> 16) - kBinBase, 0), kBins - 1);
}
__device__ __forceinline__ void ld8(const bf16* p, float* f) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x, f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ void ld4(const bf16* p, float* f) {
  uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
}
__device__ __forceinline__ void st8(bf16* p, const float* f) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
__device__ __forceinline__ void st4(bf16* p, const float* f) {
  uint2 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
  h[0] = __floats2bfloat162_rn(f[0], f[1]), h[1] = __floats2bfloat162_rn(f[2], f[3]);
  *reinterpret_cast<uint2*>(p) = u;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
  return v;
}
__device__ __forceinline__ int seq_of(const int* seq_start, int B, int r) {
  int b = 0;
  while (b + 1 < B && seq_start[b + 1] <= r) b++;
  return b;
}
// Programmatic dependent launch: kernels are launched early (PDL attribute) and wait here for their predecessor.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---- indexer score (both paths) ------------------------------------------------------------------------------------
// One m16n8k16 chain per (key, q-vector): A = 16 compressed keys (rows) × 128 dims, B = 128 dims × 8 query vectors
// (q-vector n = 4·row + head of a row pair). Fragments come straight from global memory with 16-B loads: lane (g =
// lane>>2, c = lane&3) loads dims [32p+8c, 32p+8c+8) (p = 0..3) of key rows g, g+8 and of q-vector g; k-step 2p takes
// dims 32p+8c+{0..3} in the k-slots {2c,2c+1,2c+8,2c+9}, k-step 2p+1 dims 32p+8c+{4..7}. c = 0, then k-steps 0..7 in
// order — the same chain in the decode kernel (k_score) and the prefill GEMM (k_pscore) ⇒ bitwise equal fp32 dots.
__device__ __forceinline__ void idx_load(uint4* f, const bf16* v, bool ok, int c) {  // 4 chunks of one 128-dim row
#pragma unroll
  for (int p = 0; p < 4; p++)
    f[p] = ok ? __ldg(reinterpret_cast<const uint4*>(v + 32 * p + 8 * c)) : make_uint4(0, 0, 0, 0);
}
__device__ __forceinline__ void idx_dot(float* acc, const uint4* ka, const uint4* kb, const uint4* q) {
  acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
#pragma unroll
  for (int p = 0; p < 4; p++) {
    const uint32_t a0[4] = {ka[p].x, kb[p].x, ka[p].y, kb[p].y}, a1[4] = {ka[p].z, kb[p].z, ka[p].w, kb[p].w};
    mma_bf16(acc, a0, q[p].x, q[p].y);
    mma_bf16(acc, a1, q[p].z, q[p].w);
  }
}
// Score = Σ_h ReLU: lanes hold heads {2(lane&1), +1} of row (lane>>1)&1 for keys g (acc0,1) and g+8 (acc2,3);
// (h0+h1)+(h2+h3) via one xor-1 shuffle (commutative ⇒ both lanes hold the same value). ReLU as x > 0 ? x : +0.
__device__ __forceinline__ float relu0(float x) { return x > 0.f ? x : 0.f; }
__device__ __forceinline__ float2 idx_relu_sum(const float* acc) {
  float a = __fadd_rn(relu0(acc[0]), relu0(acc[1])), b = __fadd_rn(relu0(acc[2]), relu0(acc[3]));
  a = __fadd_rn(a, __shfl_xor_sync(kFull, a, 1));
  b = __fadd_rn(b, __shfl_xor_sync(kFull, b, 1));
  return make_float2(a, b);
}

// ---- launch ------------------------------------------------------------------------------------------------------
template <class... P, class... A>
void launch_smem(void (*k)(P...), dim3 g, dim3 b, size_t smem, cudaStream_t s, A... args) {
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t c{g, b, smem, s, at, 1};
  CUDA_CHECK(cudaLaunchKernelEx(&c, k, P(args)...));
}
template <class... P, class... A>
void launch(void (*k)(P...), dim3 g, dim3 b, cudaStream_t s, A... args) {
  launch_smem(k, g, b, 0, s, args...);
}

// ---- prefill path (qsa_prefill.cu) ---------------------------------------------------------------------------------
struct PScoreArgs {
  const bf16* qi;
  const int *pos, *seq_start;
  QsaCache* const* caches;
  int B, ra, rb;  // rows [ra, rb) of this call
  float* score;   // [rb − ra, stride] (row r at r − ra); only rows that see > 512 blocks, columns < visible
  int stride;     // score row stride: kMaxBlk (base tier) or the workspace stride (wide tier)
  bool wide;      // wide tier (the call sees > kMaxBlk blocks)
};
// Indexer scores of rows [ra, rb) (tensor-core GEMM per 16-row tile × key splits).
void pscore(const PScoreArgs& a, cudaStream_t s);

struct PAttnArgs {
  const bf16 *q, *qkv;
  const int *sel, *seq_start;
  QsaCache* const* caches;
  int B;
  bf16* attn;
};
// Sparse attention + LSE merge + gate for M rows (one CTA per row, 97 KB dynamic smem: call pattn_init() once first).
void pattn_init();
void pattn(const PAttnArgs& a, int M, cudaStream_t s);

}  // namespace q38::qsa_dev
