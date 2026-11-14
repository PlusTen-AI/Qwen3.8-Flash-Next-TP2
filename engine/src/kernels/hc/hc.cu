// hc.cu — hyper-connections, token embedding and PLE (see hc.h; arch spec §2.1–2.3, §2.7).
//
// HC module = 4 kernels chained by programmatic dependent launch (each prefetches its weights before the wait):
//   K1 combine   grid (4 streams, M)      X ← bf16(X + bo·w_s) in place, rrms[m,s] = rsqrt(mean(X_s²) + eps)
//   K2 down      40 CTAs × 7 warps        partial[kc] = xn[:, 256-col chunk kc] · Wdᵀ   (324 rows: 320 lora + 4 inject)
//   K2b reduce   one thread per output    z = Σ_kc partial (kc ascending) → lora = bf16(silu(bf16(z)/4)), inj = bf16(z)
//   K3 up+mix    640 warps (160 CTAs)     G = bf16(lora · Wupᵀ) for a 16-row tile (4 hidden × 4 streams),
//                                         u = bf16(¼ Σ_s σ(G_s)·xn_s)
// xn = bf16(y + y·g), y = X'·rrms (grouped Gemma RMSNorm) is rebuilt identically in K2 and K3 (never stored).
// GEMMs: mma.sync m16n8k16 bf16 → fp32, weight rows on the MMA M side (pre-packed fragments, one 16 B load per lane),
// activation rows on the N side. A value's arithmetic is fixed by the shapes (k-steps ascending, 40 fixed K-chunks
// summed in order) and MMA columns are independent → deterministic and batch-invariant (never depends on M).
// PLE: n-gram hash → NVFP4 row lookup of this rank's table half (exact dequant) → caller all-reduces E →
// kv GEMV (same MMA scheme) → gate (bf16 materialisation points of the reference) → dilated conv + residual,
// state = last 9 inputs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

#include "kernels/hc/hc.h"
#include "core/log.h"
#include "model/patches.h"

namespace q38 {

namespace {

constexpr int H = kHidden, D = kStreams * kHidden;  // 2560, 10240
constexpr int kChunkM = 64;                          // rows per pass: ≤ 8 n8 tiles
constexpr int kDnKs = D / 16, kDnChunks = 40, kDnStep = kDnKs / kDnChunks, kDnWarps = 7;  // 640 k-steps, 16 per CTA
constexpr int kUpKs = kHcRank / 16, kUpTiles = D / 16, kUpWarps = 4;  // 20 k-steps, 640 tiles, 4 per CTA
constexpr int kXsW = 132, kLsW = 164;  // smem row pitch in 32-bit words (≡ 4 mod 32 → conflict-free B fragments)
constexpr int kPleHeads = 16, kPleDim = 160, kPleRowB = 80, kPleSclB = 10, kKvN = D + H;  // kv rows 12800

constexpr int kPleChunk = 1024;                      // PLE rows per pass above kChunkM (workspace rows)
// HC: from this M on the prefill kernels are faster than 64-row passes of the decode kernels (GB10: 365 vs 470 µs at
// M = 384, 330 vs 272 µs at 256 — the large-M down kernel walks all of K in one CTA). Both give the same bits.
constexpr int kHcBigRows = 384;

struct Ws {
  float* rrms = nullptr;  // [kHcMaxRows, 4]
  float* part = nullptr;  // [40, 324, 64] (decode path)
  bf16* lora = nullptr;   // [kHcMaxRows, 320]
} g_ws;  // shared by all modules: calls are stream-ordered (one stream)

// ---- device helpers ----------------------------------------------------------------------------------------------
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }  // round to bf16
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
}
__device__ __forceinline__ void load8f(const float* p, float (&f)[8]) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

// Sum over the block (NW warps), fixed order: xor tree in each warp, then warps 0..NW-1. Every thread gets the sum.
template <int NW>
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = 0.f;
#pragma unroll
  for (int i = 0; i < NW; i++) r += red[i];
  __syncthreads();
  return r;
}

// xn = bf16(y + y·g), y = x·rrms — identical in K2 and K3 (explicit roundings, no contraction).
__device__ __forceinline__ float hc_norm1(float x, float r, float g) {
  const float y = __fmul_rn(x, r);
  return bfr(__fmaf_rn(y, g, y));
}

// ---- K1: combine + per-stream rrms ---------------------------------------------------------------------------
// X'_s = bf16(X_s + bo · w_s), w_s = 2σ(inj_s/4) (inj == nullptr → 1). bo == nullptr: no combine (stats only).
__global__ void __launch_bounds__(320) k_combine(bf16* X, const bf16* bo, const bf16* inj, float* rrms) {
  __shared__ float red[10];
  pdl_launch();
  pdl_wait();  // P015: launched early behind the all-reduce that produced bo (no-op without PDL)
  const int s = blockIdx.x, m = blockIdx.y, t = threadIdx.x;
  uint4* xp = reinterpret_cast<uint4*>(X + size_t(m) * D + s * H) + t;
  float x[8];
  unpack8(*xp, x);
  if (bo) {
    const float w = inj ? 2.f * sigm(b2f(inj[m * kStreams + s]) * 0.25f) : 1.f;
    float b[8];
    unpack8(reinterpret_cast<const uint4*>(bo + size_t(m) * H)[t], b);
#pragma unroll
    for (int e = 0; e < 8; e++) x[e] = bfr(__fmaf_rn(b[e], w, x[e]));
    *xp = pack8(x);
  }
  if (!rrms) return;
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) ss = __fmaf_rn(x[e], x[e], ss);
  ss = block_sum<10>(ss, red);
  if (t == 0) rrms[m * kStreams + s] = rsqrtf(ss / float(H) + kEps);
}

// ---- K2: down projection, split-K partials -------------------------------------------------------------------
// Warp w owns row tiles w, w+7, w+14 (tile 20 = the 4 inject rows, compact fragments: lanes 0..15, rows g < 4).
template <bool INJ>
__device__ __forceinline__ void dn_load(uint4 (&a)[3], const uint4* Wd, const uint2* Wi, int w, int lane, int ks) {
#pragma unroll
  for (int i = 0; i < 3; i++) {
    const int rt = w + kDnWarps * i;
    if (rt < 20) {
      a[i] = __ldcs(Wd + (size_t(rt) * kDnKs + ks) * 32 + lane);
    } else if (INJ && lane < 16) {
      const uint2 v = __ldcs(Wi + ks * 16 + lane);
      a[i] = make_uint4(v.x, 0u, v.y, 0u);
    } else {
      a[i] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

template <int NT, bool INJ>
__global__ void __launch_bounds__(kDnWarps * 32) k_down(const uint4* __restrict__ Wd, const uint2* __restrict__ Wi,
                                                         const bf16* X, const float* rrms, const float* g, float* part,
                                                         int M) {
  extern __shared__ __align__(16) uint32_t xs[];  // xn chunk [NT·8][256 (+8)] bf16
  constexpr int N = INJ ? kHcDown : kHcRank, RT = INJ ? 21 : 20, PF = 4;
  const int kc = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
  const int ks0 = kc * kDnStep;
  uint4 a[PF][3];
#pragma unroll
  for (int p = 0; p < PF; p++) dn_load<INJ>(a[p], Wd, Wi, w, lane, ks0 + p);
  pdl_launch();
  pdl_wait();
  {  // xn for columns [256·kc, +256) (one stream) of all rows; rows ≥ M are zero. Loads first, then math.
    constexpr int RW = kDnWarps, IT = (NT * 8 + RW - 1) / RW;  // rows m = w + 7q
    const int s = kc / (H / 256), k0 = kc * 256, v = lane;
    float gw[8];
    load8f(g + k0 + 8 * v, gw);
    uint4 xv[IT];
    float rr[IT];
#pragma unroll
    for (int q = 0; q < IT; q++) {
      const int m = w + RW * q;
      xv[q] = make_uint4(0u, 0u, 0u, 0u), rr[q] = 0.f;
      if (m < M) xv[q] = *reinterpret_cast<const uint4*>(X + size_t(m) * D + k0 + 8 * v), rr[q] = rrms[m * kStreams + s];
    }
#pragma unroll
    for (int q = 0; q < IT; q++) {
      const int m = w + RW * q;
      if (m >= NT * 8) continue;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) {
        float x[8];
        unpack8(xv[q], x);
#pragma unroll
        for (int e = 0; e < 8; e++) x[e] = hc_norm1(x[e], rr[q], gw[e]);
        o = pack8(x);
      }
      *reinterpret_cast<uint4*>(xs + m * kXsW + 4 * v) = o;
    }
  }
  __syncthreads();
  float c[3][NT][4] = {};
#pragma unroll
  for (int j = 0; j < kDnStep; j++) {
    uint4 cur[3] = {a[j % PF][0], a[j % PF][1], a[j % PF][2]};
    if (j + PF < kDnStep) dn_load<INJ>(a[j % PF], Wd, Wi, w, lane, ks0 + j + PF);
    uint32_t b[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      b[nt][0] = xs[(nt * 8 + gq) * kXsW + j * 8 + tq];
      b[nt][1] = xs[(nt * 8 + gq) * kXsW + j * 8 + tq + 4];
    }
#pragma unroll
    for (int i = 0; i < 3; i++)
      if (w + kDnWarps * i < RT)
#pragma unroll
        for (int nt = 0; nt < NT; nt++) mma(c[i][nt], cur[i], b[nt][0], b[nt][1]);
  }
  float* pk = part + size_t(kc) * N * M;
#pragma unroll
  for (int i = 0; i < 3; i++) {
    const int rt = w + kDnWarps * i;
    if (rt >= RT) continue;
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int n = rt * 16 + gq + 8 * h;
        if (n >= N) continue;
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int m = nt * 8 + 2 * tq + e;
          if (m < M) pk[size_t(n) * M + m] = c[i][nt][2 * h + e];
        }
      }
  }
}

// ---- K2b: fixed-order split-K sum + SiLU epilogue (reference: bf16 GEMM output → silu(x/4) → bf16) ------------
__global__ void __launch_bounds__(256) k_down_reduce(const float* part, bf16* lora, bf16* inj, int M, int N) {
  pdl_launch();
  pdl_wait();
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= N * M) return;
  const int n = idx / M, m = idx - n * M;
  float z = 0.f;
#pragma unroll
  for (int kc = 0; kc < kDnChunks; kc++) z += part[size_t(kc) * N * M + idx];
  const float zb = bfr(z);
  if (n < kHcRank) {
    const float x = zb * 0.25f;
    lora[m * kHcRank + n] = __float2bfloat16_rn(x * sigm(x));
  } else if (inj) {
    inj[m * kStreams + n - kHcRank] = __float2bfloat16_rn(zb);
  }
}

// ---- K3: up projection + sigmoid gate + mean over streams ----------------------------------------------------
// Tile j rows r = 4s + hh ↔ Wup row s·2560 + 4j + hh. Lane (g, t), e = g>>2: rows g, g+8 = streams e, 2+e of hidden
// h = 4j + (g&3); lane ^ 16 holds the other two streams. Lane e outputs column 2t+e (row m = 8nt + 2t + e).
template <int NT, int WU = kUpWarps>
__global__ void __launch_bounds__(WU * 32) k_up(const uint4* __restrict__ Wu, const bf16* lora, const bf16* X,
                                               const float* rrms, const float* g, bf16* u, int M) {
  extern __shared__ __align__(16) uint32_t ls[];  // lora rows of one pass [NG·8][320 (+8)] bf16
  constexpr int NG = NT < 4 ? NT : 4;             // n8 tiles per pass
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
  const int j = blockIdx.x * WU + w, e = gq >> 2, h = 4 * j + (gq & 3);
  uint4 a[kUpKs];
#pragma unroll
  for (int ks = 0; ks < kUpKs; ks++) a[ks] = __ldcs(Wu + (size_t(j) * kUpKs + ks) * 32 + lane);
  pdl_wait();
  float gw[kStreams];
#pragma unroll
  for (int s = 0; s < kStreams; s++) gw[s] = g[s * H + h];
#pragma unroll
  for (int p0 = 0; p0 < NT; p0 += NG) {
    float xn[NG][kStreams];  // this lane's xn (rows 8(p0+nt) + 2t + e, hidden h), loaded before the MMAs
#pragma unroll
    for (int nt = 0; nt < NG; nt++) {
      const int m = (p0 + nt) * 8 + 2 * tq + e;
#pragma unroll
      for (int s = 0; s < kStreams; s++)
        xn[nt][s] = m < M ? hc_norm1(b2f(X[size_t(m) * D + s * H + h]), rrms[m * kStreams + s], gw[s]) : 0.f;
    }
    if (p0) __syncthreads();
    for (int i = threadIdx.x; i < NG * 8 * 40; i += WU * 32) {
      const int r = i / 40, v = i - r * 40, m = p0 * 8 + r;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) o = *reinterpret_cast<const uint4*>(lora + m * kHcRank + 8 * v);
      *reinterpret_cast<uint4*>(ls + r * kLsW + 4 * v) = o;
    }
    __syncthreads();
    float c[NG][4] = {};
#pragma unroll
    for (int ks = 0; ks < kUpKs; ks++)
#pragma unroll
      for (int nt = 0; nt < NG; nt++)
        if (p0 + nt < NT)
          mma(c[nt], a[ks], ls[(nt * 8 + gq) * kLsW + ks * 8 + tq], ls[(nt * 8 + gq) * kLsW + ks * 8 + tq + 4]);
#pragma unroll
    for (int nt = 0; nt < NG; nt++) {
      // e=0 needs streams 1,3 of column 2t from the partner (its c0, c2); e=1 needs streams 0,2 of column 2t+1 (c1, c3)
      const float pa = __shfl_xor_sync(~0u, e ? c[nt][0] : c[nt][1], 16);
      const float pb = __shfl_xor_sync(~0u, e ? c[nt][2] : c[nt][3], 16);
      const float G[4] = {e ? pa : c[nt][0], e ? c[nt][1] : pa, e ? pb : c[nt][2], e ? c[nt][3] : pb};
      const int m = (p0 + nt) * 8 + 2 * tq + e;
      if (m >= M) continue;
      float acc = 0.f;
#pragma unroll
      for (int s = 0; s < kStreams; s++) acc = __fmaf_rn(sigm(bfr(G[s])), xn[nt][s], acc);
      u[size_t(m) * H + h] = __float2bfloat16_rn(acc * 0.25f);
    }
  }
}

// ---- generic bf16 GEMV (PLE kv projection): y[M, N] = x[M, K] · Wᵀ, one warp per 16-row tile, full K ------------
template <int NT>
__global__ void __launch_bounds__(128) k_gemv(const uint4* __restrict__ W, const bf16* x, bf16* y, int M, int N,
                                               int KS) {
  extern __shared__ __align__(16) uint32_t xs[];  // x chunk [NT·8][256 (+8)] bf16
  constexpr int PF = 8;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
  const int tile = blockIdx.x * 4 + w, K = KS * 16;
  const uint4* wp = W + size_t(tile) * KS * 32 + lane;
  uint4 a[PF];
#pragma unroll
  for (int p = 0; p < PF; p++) a[p] = __ldcs(wp + p * 32);
  float c[NT][4] = {};
  for (int c0 = 0; c0 < KS; c0 += 16) {
    __syncthreads();
    for (int i = threadIdx.x; i < NT * 8 * 32; i += 128) {
      const int m = i >> 5, v = i & 31;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) o = *reinterpret_cast<const uint4*>(x + size_t(m) * K + c0 * 16 + 8 * v);
      *reinterpret_cast<uint4*>(xs + m * kXsW + 4 * v) = o;
    }
    __syncthreads();
#pragma unroll
    for (int jj = 0; jj < 16; jj++) {
      const uint4 cur = a[jj % PF];
      if (c0 + jj + PF < KS) a[jj % PF] = __ldcs(wp + (c0 + jj + PF) * 32);
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
        mma(c[nt], cur, xs[(nt * 8 + gq) * kXsW + jj * 8 + tq], xs[(nt * 8 + gq) * kXsW + jj * 8 + tq + 4]);
    }
  }
#pragma unroll
  for (int nt = 0; nt < NT; nt++)
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int m = nt * 8 + 2 * tq + e, n = tile * 16 + gq + 8 * h;
        if (m < M) y[size_t(m) * N + n] = __float2bfloat16_rn(c[nt][2 * h + e]);
      }
}

// ---- small elementwise kernels --------------------------------------------------------------------------------
__global__ void __launch_bounds__(320) k_expand(const bf16* x0, bf16* X) {
  const int m = blockIdx.x, t = threadIdx.x;
  const uint4 v = reinterpret_cast<const uint4*>(x0 + size_t(m) * H)[t];
#pragma unroll
  for (int s = 0; s < kStreams; s++) reinterpret_cast<uint4*>(X + size_t(m) * D + s * H)[t] = v;
}

__global__ void __launch_bounds__(320) k_embed(const bf16* table, int lo, const int* ids, bf16* out) {
  const int m = blockIdx.x, t = threadIdx.x, r = ids[m] - lo;
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (r >= 0 && r < kVocabHalf) v = reinterpret_cast<const uint4*>(table + size_t(r) * H)[t];
  reinterpret_cast<uint4*>(out + size_t(m) * H)[t] = v;
}

// ---- PLE ------------------------------------------------------------------------------------------------------
struct NgCfg {
  unsigned long long mul[3], size[kPleHeads], off[kPleHeads];
  long long lo, hi;  // table rows of this rank [lo, hi)
  float gs;          // NVFP4 global scale (weight_scale_2)
  int eos;
};

// heads 0..7 bigram, 8..15 trigram; predecessors behind the first EOS are EOS (ids ≥ 0, products < 2^63)
__device__ __forceinline__ long long ng_id(const NgCfg& c, int g, int t0, int p1, int p2) {
  if (p1 == c.eos) p2 = c.eos;
  unsigned long long mx = (unsigned long long)t0 * c.mul[0] ^ (unsigned long long)p1 * c.mul[1];
  if (g >= 8) mx ^= (unsigned long long)p2 * c.mul[2];
  return (long long)(mx % c.size[g] + c.off[g]);
}

__global__ void __launch_bounds__(320) k_ple_lookup(NgCfg c, const uint8_t* __restrict__ packed,
                                                    const uint8_t* __restrict__ scales, const int* ids, const int* prev,
                                                    bf16* E, long long* ids_out) {
  __shared__ long long sid[kPleHeads];
  const int m = blockIdx.x, t = threadIdx.x;
  if (t < kPleHeads) {
    const long long id = ng_id(c, t, ids[m], prev[2 * m + 1], prev[2 * m]);
    sid[t] = id;
    if (ids_out) ids_out[m * kPleHeads + t] = id;
  }
  __syncthreads();
  if (!E) return;
  const int g = t / 20, v = t - g * 20;  // head, 8-value group
  const long long id = sid[g];
  uint4 o = make_uint4(0u, 0u, 0u, 0u);
  if (id >= c.lo && id < c.hi) {
    const size_t row = size_t(id - c.lo);
    const uint32_t pk = *reinterpret_cast<const uint32_t*>(packed + row * kPleRowB + 4 * v);
    const float sc = e4m3_to_float(scales[row * kPleSclB + (v >> 1)]) * c.gs;  // reference: (fp32(scale)·global)·LUT
    float f[8];
#pragma unroll
    for (int e = 0; e < 8; e++) f[e] = e2m1_to_float((pk >> (4 * e)) & 15u) * sc;
    o = pack8(f);
  }
  reinterpret_cast<uint4*>(E + size_t(m) * H + g * kPleDim)[v] = o;
}

// gate per (stream, row): the reference's ple_gate materialisation points (kn, qn, prod, dot, d, √|d|, g, gated → bf16)
__global__ void __launch_bounds__(320) k_ple_gate(const bf16* kv, const bf16* X, const float* nk, const float* nq,
                                                  const float* nc, bf16* gated, bf16* cin) {
  __shared__ float red[10];
  const int s = blockIdx.x, m = blockIdx.y, t = threadIdx.x, o = s * H + 8 * t;
  float k[8], q[8], v[8], wk[8], wq[8], wc[8];
  unpack8(*reinterpret_cast<const uint4*>(kv + size_t(m) * kKvN + o), k);
  unpack8(*reinterpret_cast<const uint4*>(kv + size_t(m) * kKvN + D + 8 * t), v);
  unpack8(*reinterpret_cast<const uint4*>(X + size_t(m) * D + o), q);
  load8f(nk + o, wk), load8f(nq + o, wq), load8f(nc + o, wc);
  float sk = 0.f, sq = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) sk = __fmaf_rn(k[e], k[e], sk), sq = __fmaf_rn(q[e], q[e], sq);
  sk = block_sum<10>(sk, red);
  sq = block_sum<10>(sq, red);
  const float rk = rsqrtf(sk / float(H) + kEps), rq = rsqrtf(sq / float(H) + kEps);
  float dp = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const float kn = bfr(__fmul_rn(__fmul_rn(k[e], rk), 1.f + wk[e]));
    const float qn = bfr(__fmul_rn(__fmul_rn(q[e], rq), 1.f + wq[e]));
    dp += bfr(__fmul_rn(kn, qn));
  }
  dp = block_sum<10>(dp, red);
  const float d = bfr(__fdiv_rn(bfr(dp), sqrtf(float(H))));
  const float sg = d > 0.f ? 1.f : (d < 0.f ? -1.f : 0.f);
  const float gg = bfr(sigm(sg * bfr(sqrtf(fmaxf(fabsf(d), 1e-6f)))));
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) v[e] = bfr(__fmul_rn(gg, v[e])), ss = __fmaf_rn(v[e], v[e], ss);
  ss = block_sum<10>(ss, red);
  const float rg = rsqrtf(ss / float(H) + kEps);
  float cv[8];
#pragma unroll
  for (int e = 0; e < 8; e++) cv[e] = __fmul_rn(__fmul_rn(v[e], rg), 1.f + wc[e]);
  *reinterpret_cast<uint4*>(gated + size_t(m) * D + o) = pack8(v);
  *reinterpret_cast<uint4*>(cin + size_t(m) * D + o) = pack8(cv);
}

// dilated short conv (k=4, dilation 3) per (sequence, 256 channels): taps c[t-9], c[t-6], c[t-3], c[t]; then
// X ← bf16(X + bf16(gated + bf16(silu(bf16(conv))))); state ← last 9 conv inputs (oldest first).
__global__ void __launch_bounds__(256) k_ple_conv(bf16* X, const bf16* gated, const bf16* cin, const bf16* cw,
                                                  bf16* const* state, const int* seq_start) {
  constexpr int SL = 9;
  const int b = blockIdx.x, c = blockIdx.y * 256 + threadIdx.x;
  const int j0 = seq_start[b], T = seq_start[b + 1] - j0;
  bf16* st = state[b];
  float w[4];
#pragma unroll
  for (int k = 0; k < 4; k++) w[k] = b2f(cw[c * 4 + k]);
  bf16 old[SL];
#pragma unroll
  for (int i = 0; i < SL; i++) old[i] = st[size_t(i) * D + c];
  auto tap = [&](int h) -> bf16 {  // window position h: state for h < 9, else in-step conv input j0 + h - 9
    if (h < SL) {
      bf16 r = old[0];
#pragma unroll
      for (int i = 1; i < SL; i++) if (h == i) r = old[i];
      return r;
    }
    return cin[size_t(j0 + h - SL) * D + c];
  };
  for (int j = 0; j < T; j++) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) acc = __fmaf_rn(w[k], b2f(tap(j + 3 * k)), acc);
    const float cv = bfr(acc), y = bfr(cv * sigm(cv));
    const size_t o = size_t(j0 + j) * D + c;
    const float p = bfr(b2f(gated[o]) + y);
    X[o] = __float2bfloat16_rn(b2f(X[o]) + p);
  }
  bf16 nw[SL];
#pragma unroll
  for (int i = 0; i < SL; i++) nw[i] = tap(T + i);
#pragma unroll
  for (int i = 0; i < SL; i++) st[size_t(i) * D + c] = nw[i];
}

#include "kernels/hc/hc_prefill.cuh"  // large-M kernels (namespace pre)
#include "kernels/hc/p003_hc_fused_prefill.cuh"  // P003: large-M module, combine fused into down (namespace p3)

// ---- host helpers ---------------------------------------------------------------------------------------------
template <class... P, class... A>
void launch(void (*k)(P...), dim3 g, dim3 b, size_t smem, cudaStream_t s, bool pdl, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g, cfg.blockDim = b, cfg.dynamicSmemBytes = smem, cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at, cfg.numAttrs = pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, args...));
}

template <class T>
T* to_dev(const void* h, size_t bytes) {
  T* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, bytes));
  CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  return d;
}

std::vector<float> to_f32(const Weights& W, const std::string& name, size_t n) {
  const auto& ti = W.info(name);
  auto b = W.slice(name);
  std::vector<float> f(n);
  if (ti.dtype == "F32" && b.size() == n * 4) {
    std::memcpy(f.data(), b.data(), n * 4);
  } else if (ti.dtype == "BF16" && b.size() == n * 2) {
    for (size_t i = 0; i < n; i++) {
      uint32_t u = uint32_t(reinterpret_cast<const uint16_t*>(b.data())[i]) << 16;
      std::memcpy(&f[i], &u, 4);
    }
  } else {
    qlog::fail("hc", "%s: dtype %s · %zu B (expected %zu elements)", name.c_str(), ti.dtype.c_str(), b.size(), n);
  }
  return f;
}

std::vector<uint8_t> bf16_rows(const Weights& W, const std::string& name, int64_t n, int64_t k) {
  const auto& ti = W.info(name);
  if (ti.dtype != "BF16" || ti.shape.size() < 2 || ti.shape[0] != n || ti.shape[1] != k)
    qlog::fail("hc", "%s: %s, expected BF16 [%lld,%lld]", name.c_str(), ti.dtype.c_str(), (long long)n, (long long)k);
  return W.slice(name);
}

// mma A fragments of 16-row tiles: out[((tile·KS + ks)·32 + lane)·8 ..] = rows g, g+8 (g = lane/4) × cols
// 16ks + 2t + {0, 1, 8, 9} (t = lane%4) in register order a0..a3. row(tile, r) → pointer to K bf16 values.
std::vector<uint16_t> pack_tiles(int tiles, int K, const std::function<const uint16_t*(int, int)>& row) {
  const int KS = K / 16;
  std::vector<uint16_t> out(size_t(tiles) * KS * 256);
  for (int tl = 0; tl < tiles; tl++) {
    const uint16_t* r[16];
    for (int i = 0; i < 16; i++) r[i] = row(tl, i);
    for (int ks = 0; ks < KS; ks++)
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane >> 2, k = ks * 16 + 2 * (lane & 3);
        uint16_t* o = out.data() + ((size_t(tl) * KS + ks) * 32 + lane) * 8;
        o[0] = r[g][k], o[1] = r[g][k + 1], o[2] = r[g + 8][k], o[3] = r[g + 8][k + 1];
        o[4] = r[g][k + 8], o[5] = r[g][k + 9], o[6] = r[g + 8][k + 8], o[7] = r[g + 8][k + 9];
      }
  }
  return out;
}

#define Q38_NT_SWITCH(nt, F) \
  switch (nt) {              \
    case 1: F(1); break;     \
    case 2: F(2); break;     \
    case 3: F(3); break;     \
    case 4: F(4); break;     \
    case 5: F(5); break;     \
    case 6: F(6); break;     \
    case 7: F(7); break;     \
    default: F(8); break;    \
  }

#include "kernels/hc/p012_hc_pack12.cuh"  // P012: decode kernels on a 12-bit copy of the weights (namespace p12)

}  // namespace

// ---- HC ---------------------------------------------------------------------------------------------------------
HcW hc_upload(const Weights& W, const std::string& prefix, bool inject) {
  if (!g_ws.part) {  // once per process, sized for kHcMaxRows (no allocation inside forward calls)
    CUDA_CHECK(cudaMalloc(&g_ws.rrms, sizeof(float) * kHcMaxRows * kStreams));
    CUDA_CHECK(cudaMalloc(&g_ws.part, sizeof(float) * kDnChunks * kHcDown * kChunkM));
    CUDA_CHECK(cudaMalloc(&g_ws.lora, sizeof(bf16) * kHcMaxRows * kHcRank));
    auto big = [](auto k, int bytes) { CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)); };
#define Q38_DS(WM, NJ) big(pre::k_down<WM, NJ, true>, pre::down_smem<WM>()), big(pre::k_down<WM, NJ, false>, pre::down_smem<WM>());
    Q38_DS(1, 4) Q38_DS(2, 4) Q38_DS(4, 4)
#undef Q38_DS
    big(pre::k_up, pre::kUbSmem), big(pre::k_gemm, pre::kGbSmem);
  }
  HcW w;
  w.inject = inject;
  w.norm = to_dev<float>(to_f32(W, prefix + ".hc_norm.weight", D).data(), sizeof(float) * D);
  auto dn = bf16_rows(W, prefix + ".input_mix_weight_down.weight", kHcRank, D);
  std::vector<uint16_t> td;
  {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(dn.data());
    auto& t = td = pack_tiles(20, D, [&](int tl, int r) { return p + size_t(tl * 16 + r) * D; });
    size_t main = t.size() * 2, extra = 0;
    std::vector<uint16_t> inj;
    if (inject) {  // compact tile: lanes 0..15 (row g < 4) × cols 16ks + 2t + {0, 1, 8, 9}
      auto ib = bf16_rows(W, prefix + ".block_inject_weight.weight", kStreams, D);
      const uint16_t* q = reinterpret_cast<const uint16_t*>(ib.data());
      inj.resize(size_t(kDnKs) * 16 * 4);
      for (int ks = 0; ks < kDnKs; ks++)
        for (int lane = 0; lane < 16; lane++) {
          const int g = lane >> 2, k = ks * 16 + 2 * (lane & 3);
          uint16_t* o = inj.data() + (size_t(ks) * 16 + lane) * 4;
          o[0] = q[size_t(g) * D + k], o[1] = q[size_t(g) * D + k + 1];
          o[2] = q[size_t(g) * D + k + 8], o[3] = q[size_t(g) * D + k + 9];
        }
      extra = inj.size() * 2;
    }
    uint8_t* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, main + extra));
    CUDA_CHECK(cudaMemcpy(d, t.data(), main, cudaMemcpyHostToDevice));
    if (extra) CUDA_CHECK(cudaMemcpy(d + main, inj.data(), extra, cudaMemcpyHostToDevice));
    w.down = d;
  }
  auto up = bf16_rows(W, prefix + ".input_mix_weight_up.weight", D, kHcRank);
  std::vector<uint16_t> tu;
  {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(up.data());
    auto& t = tu = pack_tiles(kUpTiles, kHcRank, [&](int tl, int r) {
      return p + size_t((r >> 2) * H + 4 * tl + (r & 3)) * kHcRank;
    });
    w.up = to_dev<uint8_t>(t.data(), t.size() * 2);
  }
  if (patch::on(patch::hc_pack12))  // P012: 12-bit copy for the decode kernels, checked against w.down / w.up
    w.pk = p12::pack(reinterpret_cast<const uint16_t*>(dn.data()), reinterpret_cast<const uint16_t*>(up.data()), td, tu,
                     w.down, w.up);
  return w;
}

size_t hc_pack_bytes(const HcW& w) {
  if (!w.pk) return 0;
  uint32_t n = 0;  // entries = last up list offset
  CUDA_CHECK(cudaMemcpy(&n, static_cast<const uint8_t*>(w.pk) + p12::kOEoU + p12::kUpUnits * 4, 4, cudaMemcpyDeviceToHost));
  return p12::kOEnt + size_t(n) * 4;
}

std::string hc_pack_stats() {
  const auto& s = p12::g_st;
  char b[256];
  std::snprintf(b, sizeof b, "modules %d (bf16 %d) · values %lld · exceptions %lld (%.4f%%) · longest list down %d up %d",
                s.modules, s.bf16_modules, s.values, s.exc, s.values ? 100.0 * s.exc / s.values : 0.0, s.max_dn_list,
                s.max_up_list);
  return b;
}

void hc_combine_mix(const HcW& w, bf16* X, const bf16* bo, const bf16* inj_prev, bf16* u, bf16* inj_out, int M,
                    cudaStream_t s) {
  const int N = w.inject ? kHcDown : kHcRank;
  const uint4* Wd = reinterpret_cast<const uint4*>(w.down);
  const uint2* Wi = reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(w.down) + size_t(20) * kDnKs * 512);
  const uint4* Wu = reinterpret_cast<const uint4*>(w.up);
  if (M >= kHcBigRows) {  // prefill: combine over all rows, then the large-M down and up kernels (same per-row values)
    if (M > kHcMaxRows) qlog::fail("hc", "M %d > %d", M, kHcMaxRows);
    if (patch::on(patch::hc_fused_prefill)) return p3::run(w, Wd, Wi, Wu, X, bo, inj_prev, u, inj_out, M, s);
    launch(k_combine, dim3(kStreams, M), dim3(320), 0, s, false, X, bo, bo && inj_prev ? inj_prev : nullptr, g_ws.rrms);
    const int wm = cdiv(M, 128) * 3 >= 96 ? 4 : cdiv(M, 64) * 3 >= 96 ? 2 : 1;  // rows per CTA 32·wm
#define Q38_DB(WM, NJ)                                                                                            \
  if (w.inject)                                                                                                   \
    pre::k_down<WM, NJ, true><<<cdiv(M, 32 * WM) * 3, 512 * WM / NJ, pre::down_smem<WM>(), s>>>(                  \
        Wd, Wi, X, g_ws.rrms, w.norm, g_ws.lora, inj_out, M);                                                     \
  else                                                                                                            \
    pre::k_down<WM, NJ, false><<<cdiv(M, 32 * WM) * 3, 512 * WM / NJ, pre::down_smem<WM>(), s>>>(                 \
        Wd, Wi, X, g_ws.rrms, w.norm, g_ws.lora, nullptr, M);
    if (wm == 4) { Q38_DB(4, 4) } else if (wm == 2) { Q38_DB(2, 4) } else { Q38_DB(1, 4) }
#undef Q38_DB
    // 3 row ranges per hidden group: 240 CTAs = 5 full waves on 48 SMs (one CTA per SM), weights loaded once per CTA
    const int chunks = cdiv(M, pre::kUbRows), nch = cdiv(chunks, std::min(3, chunks));
    pre::k_up<<<pre::kUbGroups * cdiv(chunks, nch), pre::kUbW * 32, pre::kUbSmem, s>>>(Wu, g_ws.lora, X, g_ws.rrms,
                                                                                      w.norm, u, M, nch);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  const bool pk = w.pk && patch::on(patch::hc_pack12);  // P012: down / up read the 12-bit copy
  for (int m0 = 0; m0 < M; m0 += kChunkM) {
    const int mc = std::min(kChunkM, M - m0), nt = cdiv(mc, 8);
    bf16* x = X + size_t(m0) * D;
    // P015 hc_after_ar_pdl: K1 as a programmatic dependent — behind an RDMA all-reduce (which lets dependents launch
    // at its start) K1 waits in place and K2 already streams its weights into L2 while the GPU idles on the network
    launch(k_combine, dim3(kStreams, mc), dim3(320), 0, s, patch::on(patch::hc_after_ar_pdl), x,
           bo ? bo + size_t(m0) * H : nullptr, bo && inj_prev ? inj_prev + m0 * kStreams : nullptr, g_ws.rrms);
    if (pk) {
      const int S = cdiv(N * mc, 32) * 32;  // partials: K-chunk stride in whole 128-byte lines (≤ the workspace's)
      p12::down(w, Wi, x, g_ws.part, mc, nt, S, s);
      launch(p12::k_reduce, dim3(std::max(cdiv(N * mc, 256), p12::kUpPieces)), dim3(256), 0, s, true,
             (const float*)g_ws.part, g_ws.lora + size_t(m0) * kHcRank, inj_out ? inj_out + m0 * kStreams : nullptr, mc,
             N, S, (const uint8_t*)w.pk + p12::kDnB);
      p12::up(w, g_ws.lora + size_t(m0) * kHcRank, x, u + size_t(m0) * H, mc, nt, s);
      continue;
    }
#define Q38_DOWN(NT)                                                                                              \
  if (w.inject)                                                                                                   \
    launch(k_down<NT, true>, dim3(kDnChunks), dim3(kDnWarps * 32), size_t(NT) * 8 * kXsW * 4, s, true, Wd, Wi,    \
           (const bf16*)x, (const float*)g_ws.rrms, w.norm, g_ws.part, mc);                                       \
  else                                                                                                            \
    launch(k_down<NT, false>, dim3(kDnChunks), dim3(kDnWarps * 32), size_t(NT) * 8 * kXsW * 4, s, true, Wd, Wi,   \
           (const bf16*)x, (const float*)g_ws.rrms, w.norm, g_ws.part, mc);
    Q38_NT_SWITCH(nt, Q38_DOWN)
#undef Q38_DOWN
    launch(k_down_reduce, dim3(cdiv(N * mc, 256)), dim3(256), 0, s, true, (const float*)g_ws.part,
           g_ws.lora + size_t(m0) * kHcRank, inj_out ? inj_out + m0 * kStreams : nullptr, mc, N);
#define Q38_UP(NT)                                                                                                \
  launch(k_up<NT>, dim3(kUpTiles / kUpWarps), dim3(kUpWarps * 32), size_t(NT < 4 ? NT : 4) * 8 * kLsW * 4, s, true, Wu, \
         (const bf16*)g_ws.lora + size_t(m0) * kHcRank,                                                          \
         (const bf16*)x, (const float*)g_ws.rrms, w.norm, u + size_t(m0) * H, mc);
    Q38_NT_SWITCH(nt, Q38_UP)
#undef Q38_UP
  }
}

void hc_combine(bf16* X, const bf16* bo, const bf16* inj, int M, cudaStream_t s) {
  if (M > 0) launch(k_combine, dim3(kStreams, M), dim3(320), 0, s, false, X, bo, inj, (float*)nullptr);
}

void hc_expand(const bf16* x0, bf16* X, int M, cudaStream_t s) {
  if (M > 0) k_expand<<<M, 320, 0, s>>>(x0, X);
}

void embed_lookup(const bf16* table_half, int rank, const int* ids, bf16* out, int M, cudaStream_t s) {
  if (M > 0) k_embed<<<M, 320, 0, s>>>(table_half, rank * kVocabHalf, ids, out);
}

// media rows: out[rows[k]] = src[k] (a full embedding row), or 0 where src[k] is null (the other rank)
__global__ void k_embed_media(bf16* out, const int* rows, const bf16* const* src) {
  const int k = blockIdx.x;
  uint4* o = reinterpret_cast<uint4*>(out + size_t(rows[k]) * kHidden);
  const uint4* sp = reinterpret_cast<const uint4*>(src[k]);
  for (int i = threadIdx.x; i < kHidden / 8; i += blockDim.x) o[i] = sp ? sp[i] : make_uint4(0, 0, 0, 0);
}

void embed_media(bf16* out, const int* rows, const bf16* const* src, int K, cudaStream_t s) {
  if (K > 0) k_embed_media<<<K, 320, 0, s>>>(out, rows, src);
}

const bf16* hc_last_lora() { return g_ws.lora; }

// ---- PLE ----------------------------------------------------------------------------------------------------------
struct PleW {
  NgCfg cfg;
  uint8_t* packed = nullptr;  // [rows, 80] e2m1 pairs (low nibble = even value)
  uint8_t* scales = nullptr;  // [rows, 10] e4m3 per 16 values
  uint4* kv = nullptr;        // key_proj ‖ value_proj [12800, 2560], packed fragments
  float *nk = nullptr, *nq = nullptr, *nc = nullptr;  // norm_key/query/conv [10240] fp32
  bf16* conv_w = nullptr;     // [10240, 4]
  int cap = 0;                // rows of the workspaces (kPleChunk)
  bf16 *kvb = nullptr, *gated = nullptr, *cin = nullptr;
};

static void ple_alloc(PleW* w) {  // workspaces once (upload / clone), never inside ple_apply
  w->cap = std::max(kChunkM, kPleChunk);
  CUDA_CHECK(cudaMalloc(&w->kvb, sizeof(bf16) * w->cap * kKvN));
  CUDA_CHECK(cudaMalloc(&w->gated, sizeof(bf16) * w->cap * D));
  CUDA_CHECK(cudaMalloc(&w->cin, sizeof(bf16) * w->cap * D));
}

PleW* ple_upload(const Weights& W, int rank) {
  const std::string p = "model.language_model.layers.1.ple.", e = p + "ple_embedding.";
  auto* w = new PleW();
  NgCfg& c = w->cfg;
  {
    auto i64 = [&](const std::string& n, int cnt) {
      auto b = W.slice(e + n);
      if (W.info(e + n).dtype != "I64" || b.size() != size_t(cnt) * 8) qlog::fail("ple", "%s: expected I64 [%d]", n.c_str(), cnt);
      std::vector<long long> v(cnt);
      std::memcpy(v.data(), b.data(), b.size());
      return v;
    };
    auto mu = i64("layer_multipliers", 3), sz = i64("ngram_heads_vocab_sizes", kPleHeads),
         of = i64("ngram_heads_offsets", kPleHeads);
    // spec §2.3 defaults (seed 1234, dense layer 0) — the checkpoint buffers win, a difference is reported
    const long long mu0[3] = {23703573157769LL, 20109073645365LL, 8052911324071LL};
    if (mu[0] != mu0[0] || mu[1] != mu0[1] || mu[2] != mu0[2]) qlog::warn("ple", "layer_multipliers ≠ default");
    long long tot = 0;
    for (int g = 0; g < kPleHeads; g++) {
      if (of[g] != tot) qlog::warn("ple", "ngram_heads_offsets[%d] %lld ≠ cumulative %lld", g, of[g], tot);
      tot += sz[g];
    }
    const long long padded = (tot + 127) / 128 * 128, half = padded / 2;
    for (int i = 0; i < 3; i++) c.mul[i] = (unsigned long long)mu[i];
    for (int g = 0; g < kPleHeads; g++) c.size[g] = (unsigned long long)sz[g], c.off[g] = (unsigned long long)of[g];
    c.lo = half * rank, c.hi = half * (rank + 1);
    c.eos = 248044;
    float gs;
    auto b = W.slice(e + "ngram_embedding.weight_scale_2");
    std::memcpy(&gs, b.data(), 4);
    c.gs = gs;
    // table half: shards 64r .. 64r+63 (own:r), 2,500,012 rows each
    const long long rows = c.hi - c.lo;
    CUDA_CHECK(cudaMalloc(&w->packed, size_t(rows) * kPleRowB));
    CUDA_CHECK(cudaMalloc(&w->scales, size_t(rows) * kPleSclB));
    long long r0 = 0;
    for (int k = 64 * rank; k < 64 * (rank + 1); k++) {
      const std::string sh = e + "ngram_embedding.shard_" + std::to_string(k);
      std::vector<int64_t> s1, s2;
      auto pk = W.slice(sh + ".weight", &s1);
      auto sc = W.slice(sh + ".weight_scale", &s2);
      if (s1.size() != 2 || s1[1] != kPleRowB || s2.size() != 2 || s2[1] != kPleSclB || s1[0] != s2[0] ||
          r0 + s1[0] > rows)
        qlog::fail("ple", "shard_%d has a wrong shape (rank %d)", k, rank);
      CUDA_CHECK(cudaMemcpy(w->packed + size_t(r0) * kPleRowB, pk.data(), pk.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(w->scales + size_t(r0) * kPleSclB, sc.data(), sc.size(), cudaMemcpyHostToDevice));
      r0 += s1[0];
    }
    if (r0 != rows) qlog::fail("ple", "table rows %lld ≠ %lld (rank %d)", r0, rows, rank);
  }
  {
    auto kb = bf16_rows(W, p + "key_proj.weight", D, H), vb = bf16_rows(W, p + "value_proj.weight", H, H);
    const uint16_t *kp = reinterpret_cast<const uint16_t*>(kb.data()), *vp = reinterpret_cast<const uint16_t*>(vb.data());
    auto t = pack_tiles(kKvN / 16, H, [&](int tl, int r) {
      const int n = tl * 16 + r;
      return n < D ? kp + size_t(n) * H : vp + size_t(n - D) * H;
    });
    w->kv = to_dev<uint4>(t.data(), t.size() * 2);
  }
  w->nk = to_dev<float>(to_f32(W, p + "norm_key.weight", D).data(), sizeof(float) * D);
  w->nq = to_dev<float>(to_f32(W, p + "norm_query.weight", D).data(), sizeof(float) * D);
  w->nc = to_dev<float>(to_f32(W, p + "norm_conv.weight", D).data(), sizeof(float) * D);
  {
    auto cb = bf16_rows(W, p + "conv1d.weight", D, 1);
    if (cb.size() != size_t(D) * 4 * 2) qlog::fail("ple", "expected conv1d.weight [10240,1,4]");
    w->conv_w = to_dev<bf16>(cb.data(), cb.size());
  }
  ple_alloc(w);
  return w;
}

void ple_lookup(const PleW* w, const int* ids, const int* prev, bf16* E, int M, cudaStream_t s) {
  if (M > 0) k_ple_lookup<<<M, 320, 0, s>>>(w->cfg, w->packed, w->scales, ids, prev, E, nullptr);
}

void ple_ngram_ids(const PleW* w, const int* ids, const int* prev, long long* out, int M, cudaStream_t s) {
  if (M > 0) k_ple_lookup<<<M, 320, 0, s>>>(w->cfg, w->packed, w->scales, ids, prev, (bf16*)nullptr, out);
}

void ple_apply(const PleW* w, bf16* X, const bf16* E, bf16* const* conv_state, const int* seq_start, int B, int M,
               cudaStream_t s) {
  if (M <= 0) return;
  if (M > kChunkM) {  // prefill: passes of ≤ kPleChunk rows (kv GEMM, gate, row-parallel conv, state advance)
    if (M > kHcMaxRows) qlog::fail("ple", "M %d > %d", M, kHcMaxRows);
    for (int r0 = 0; r0 < M; r0 += kPleChunk) {
      const int rows = std::min(kPleChunk, M - r0), mt = cdiv(rows, 128), ntl = cdiv(kKvN, 128);
      pre::k_gemm<<<mt * ntl, 256, pre::kGbSmem, s>>>(w->kv, E + size_t(r0) * H, w->kvb, rows, kKvN, H, mt, ntl);
      k_ple_gate<<<dim3(kStreams, rows), 320, 0, s>>>(w->kvb, X + size_t(r0) * D, w->nk, w->nq, w->nc, w->gated, w->cin);
      pre::k_ple_conv_rows<<<dim3(rows, D / 256), 256, 0, s>>>(X, w->gated, w->cin, w->conv_w, conv_state, seq_start, B, r0);
      pre::k_ple_state<<<dim3(B, D / 256), 256, 0, s>>>(w->cin, conv_state, seq_start, r0, r0 + rows);
    }
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  for (int m0 = 0; m0 < M; m0 += kChunkM) {
    const int mc = std::min(kChunkM, M - m0), nt = cdiv(mc, 8);
#define Q38_KV(NT)                                                                                              \
  k_gemv<NT><<<kKvN / 64, 128, size_t(NT) * 8 * kXsW * 4, s>>>(w->kv, E + size_t(m0) * H,                     \
                                                               w->kvb + size_t(m0) * kKvN, mc, kKvN, H / 16);
    Q38_NT_SWITCH(nt, Q38_KV)
#undef Q38_KV
  }
  k_ple_gate<<<dim3(kStreams, M), 320, 0, s>>>(w->kvb, X, w->nk, w->nq, w->nc, w->gated, w->cin);
  k_ple_conv<<<dim3(B, D / 256), 256, 0, s>>>(X, w->gated, w->cin, w->conv_w, conv_state, seq_start);
}

const bf16* ple_last_kv(const PleW* w) { return w->kvb; }
const bf16* ple_last_cin(const PleW* w) { return w->cin; }

PleW* ple_debug_clone(const PleW* w, bool round_norms) {
  auto* c = new PleW(*w);
  ple_alloc(c);
  if (round_norms)
    for (float** p : {&c->nk, &c->nq, &c->nc}) {
      std::vector<float> h(D);
      CUDA_CHECK(cudaMemcpy(h.data(), *p, D * 4, cudaMemcpyDeviceToHost));
      for (auto& v : h) v = __bfloat162float(__float2bfloat16_rn(v));
      *p = to_dev<float>(h.data(), D * 4);
    }
  return c;
}

}  // namespace q38
