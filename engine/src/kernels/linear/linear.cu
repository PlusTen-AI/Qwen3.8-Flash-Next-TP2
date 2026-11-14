// linear.cu — skinny GEMM y[M,N] = x[M,K]·Wᵀ for decode (M ≤ 64): MXFP8 (W8A16) and bf16 weights, tensor cores.
// M > 64 (prefill) runs linear_prefill.cuh: same weights and per-output arithmetic, tiled for throughput; with patch
// P004 gemm_tune on, split-1 shapes at M > 64 run p004_gemm_tune.cuh instead (faster tiles, same bits).
//
// Swap-AB mma.sync.m16n8k16 (bf16 in, fp32 accumulate): weight rows are the MMA M side (16-row tiles), activation rows
// the N side (n8 tiles, zero-padded to a multiple of 8). One CTA = 4 warps = 64 weight rows × one K range (split);
// each warp owns one 16-row tile. A row's arithmetic is fixed by (N, K) alone:
//   · every k32 block is 2 MMA k16 steps with a fixed k → slot permutation (lane t of a quad holds k 8t..8t+7),
//   · the k32 blocks of a split are chained in ascending order into one fp32 accumulator (initial 0),
//   · the S K-splits (S chosen at upload from N, K — never from M) are summed in order s = 0..S-1 through
//     distributed shared memory of a thread-block cluster (no workspace, no atomics),
// and MMA output columns are independent, so a row's result never depends on M or on the other rows.
// MXFP8 dequant is exact: e4m3 → f16 (cvt) → f32 · 2^(s−127) → bf16 (≤ 4 significant bits, in range).
// Weights (kernel layout, packed at upload) and x chunks go through a cp.async multistage pipeline in shared memory;
// chunk size and stage count are scheduling only (chosen per M), they do not change any row's arithmetic.
// P013 lin_pack12 (p013_pack12.cuh): with the patch on, a bf16 Lin that holds the 12-bit copy is read from it here
// (format kPack12); unpacking rebuilds the exact bf16 operands, so every output is bitwise the bf16 copy's.
#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <type_traits>
#include <utility>

#include "kernels/linear/linear.h"
#include "core/log.h"
#include "model/patches.h"

namespace q38 {
namespace cg = cooperative_groups;

namespace {

constexpr int kWarps = 4, kThreads = 128, kRows = 64, kMaxSplit = 8;  // CTA = 4 warps × 16 rows; cluster ≤ 8

// Kernel layout per format: TB bytes per (16-row tile, k32 block), SB scale bytes, CK k32 blocks per upload chunk.
template <int FMT> struct Fmt;
template <> struct Fmt<WMat::MXFP8> { static constexpr int TB = 512, SB = 16, CK = 4; };
template <> struct Fmt<WMat::BF16> { static constexpr int TB = 1024, SB = 0, CK = 2; };
template <> struct Fmt<WMat::NVFP4> { static constexpr int TB = 256, SB = 32, CK = 8; };  // lin_make_nvfp4 (decode only)
constexpr int kPack12 = 3;  // P013: the 12-bit copy of a bf16 Lin (L.pk), decode kernel only
template <> struct Fmt<kPack12> { static constexpr int TB = 768, SB = 0, CK = 4; };  // CK 2 if needed (pick)
constexpr int chunk_blocks(int fmt) {
  return fmt == WMat::MXFP8 ? Fmt<WMat::MXFP8>::CK : fmt == WMat::NVFP4 ? Fmt<WMat::NVFP4>::CK : Fmt<WMat::BF16>::CK;
}

template <class F>
void parallel_for(int n, F f) {
  const int nt = std::max(1, std::min<int>(16, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (int i = 0; i < nt; i++)
    th.emplace_back([&, i] {
      for (int j = i; j < n; j += nt) f(j);
    });
  for (auto& x : th) x.join();
}

#include "kernels/linear/p013_pack12.cuh"  // P013 lin_pack12: format, unpacking, host packer

template <int FMT, int NT, int CK, int BUDGET> struct Cfg {
  static constexpr int PT = CK * Fmt<FMT>::TB / 16;      // 16-B pieces per tile and chunk
  static constexpr int WST = kWarps * CK * Fmt<FMT>::TB;  // weights per stage
  static constexpr int SST = kWarps * CK * Fmt<FMT>::SB;  // scales per stage
  static constexpr int XST = NT * CK * 512;               // x fragments per stage (bf16 [NT·8, CK·32])
  static constexpr int STAGE = WST + SST + XST;
  static constexpr int ST = std::min(8, std::max(2, BUDGET / STAGE));
  static constexpr int PSTRIDE = kRows + 4;               // epilogue partial tile [NT·8][68] fp32 (bank-conflict free)
  static constexpr int SMEM = std::max(ST * STAGE, NT * 8 * PSTRIDE * 4);
};

struct Args {
  const uint8_t* w;  // packed weights
  const uint8_t* s;  // packed scales (MXFP8)
  const bf16* x;     // [M, K]
  void* y;           // [M, N] bf16 | fp32
  int M, N, K, KB, kb_split;  // KB = K/32, kb_split = k32 blocks per split
  float gs;                   // NVFP4: global scale · 2^8, applied to the fp32 sums
  const uint8_t* pb;          // P013: row bases
  const uint32_t *xo, *xe;    // P013: per tile exception offsets, exceptions
};

__device__ __forceinline__ void cp16(void* dst, const void* src) {
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool valid) {  // zero-fill when !valid
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// NVFP4 (as moe.cu): e4m3 scales of rows r / r+8 (low 16 bits) → bf16x2 broadcasts of s·2^118; 8 e2m1 codes in nibble
// order [k6,k4,k2,k0,k7,k5,k3,k1] → bf16x2 (k0,k1) (k2,k3) (k4,k5) (k6,k7) = fp4·s·2^-8 exactly.
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ void sc_fp4(uint32_t two, uint32_t& lo, uint32_t& hi) {
  const uint32_t bits = (__byte_perm(two, 0, 0x4140) << 4) + 0x77007700u;
  lo = __byte_perm(bits, 0, 0x1010);
  hi = __byte_perm(bits, 0, 0x3232);
}
__device__ __forceinline__ void dq_fp4x8(uint32_t q, uint32_t sc, uint32_t (&a)[4]) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const uint32_t w = q << (4 * j);
    a[j] = hmul2((w & 0x80008000u) | ((w & 0x70007000u) >> 6), sc);  // fp4·2^-126 × s·2^118
  }
}

// 4 e4m3 bytes (byte 0 = lowest k) · 2^(s−127) → bf16x2 (k0,k1) and (k2,k3); exact.
__device__ __forceinline__ void dq4(uint32_t u, float sc, uint32_t& lo, uint32_t& hi) {
  const __half2 h0 = __half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(u & 0xffff), __NV_E4M3));
  const __half2 h1 = __half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(u >> 16), __NV_E4M3));
  const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
  const __nv_bfloat162 b0 = __floats2bfloat162_rn(f0.x * sc, f0.y * sc), b1 = __floats2bfloat162_rn(f1.x * sc, f1.y * sc);
  lo = *reinterpret_cast<const uint32_t*>(&b0);
  hi = *reinterpret_cast<const uint32_t*>(&b1);
}

template <int FMT, int NT, bool F32, class C, int CK>
__global__ void __launch_bounds__(kThreads) lin_kernel(Args a) {
  constexpr int TB = Fmt<FMT>::TB;
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int S = gridDim.y, kb0 = blockIdx.y * a.kb_split, nch = a.kb_split / CK;
  const size_t tile0 = size_t(blockIdx.x) * kWarps;

  auto load = [&](int c, int buf) {
    uint8_t* st = sm + buf * C::STAGE;
    const int kb = kb0 + c * CK;
#pragma unroll
    for (int p = tid; p < C::WST / 16; p += kThreads)  // tile p/PT: CK·TB contiguous bytes
      cp16(st + p * 16, a.w + ((tile0 + p / C::PT) * a.KB + kb) * TB + (p % C::PT) * 16);
    if (C::SST > 0 && tid < C::SST / 16) {  // PB 16-B pieces per (tile, block): tile tid/(CK·PB), block kb + …
      constexpr int PB = Fmt<FMT>::SB / 16;
      const int tl = tid / (CK * PB), rm = tid % (CK * PB);
      cp16(st + C::WST + tid * 16, a.s + ((tile0 + tl) * a.KB + kb + rm / PB) * Fmt<FMT>::SB + (rm % PB) * 16);
    }
#pragma unroll
    for (int p = tid; p < NT * CK * 32; p += kThreads) {  // piece p = (kbl·NT + nt)·32 + lane'
      const int ln = p & 31, nt = (p >> 5) % NT, kbl = (p >> 5) / NT, m = nt * 8 + (ln >> 2);
      const bf16* src = a.x + size_t(min(m, a.M - 1)) * a.K + (kb + kbl) * 32 + (ln & 3) * 8;
      cp16z(st + C::WST + C::SST + p * 16, src, m < a.M);
    }
  };

  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

  [[maybe_unused]] uint32_t bb0 = 0, bb1 = 0;  // P013: window bases of rows g, g+8 (× 0x01010101)
  [[maybe_unused]] p013::Exc ex;               // P013: this warp's tile's exceptions
  if constexpr (FMT == kPack12) {
    const int T = int(tile0) + warp;
    bb0 = a.pb[16 * T + g] * 0x01010101u, bb1 = a.pb[16 * T + g + 8] * 0x01010101u;
    ex.init(a.xo, a.xe, T, kb0, lane);
  }

#pragma unroll
  for (int i = 0; i < C::ST - 1; i++) {
    if (i < nch) load(i, i);
    cp_commit();
  }
  for (int c = 0; c < nch; c++) {
    cp_wait<C::ST - 2>();
    __syncthreads();
    if (c + C::ST - 1 < nch) load(c + C::ST - 1, (c + C::ST - 1) % C::ST);
    cp_commit();
    const uint8_t* st = sm + (c % C::ST) * C::STAGE;
#pragma unroll
    for (int kbl = 0; kbl < CK; kbl++) {
      // A fragments of k16 steps j = 0, 1: {row g k(2t,2t+1), row g+8 k(2t,2t+1), row g k(2t+8,2t+9), row g+8 …}
      // with slot 2t+i ↔ k 8t+4j+i and slot 2t+8+i ↔ k 8t+4j+2+i.
      uint32_t A0[4], A1[4];
      if constexpr (FMT == WMat::MXFP8) {
        const uint4 w = *reinterpret_cast<const uint4*>(st + (warp * CK + kbl) * TB + lane * 16);  // row g ‖ row g+8
        const uint32_t sv = *reinterpret_cast<const uint16_t*>(st + C::WST + (warp * CK + kbl) * 16 + 2 * g);
        const float s0 = ue8m0_to_float(sv & 0xff), s1 = ue8m0_to_float(sv >> 8);
        dq4(w.x, s0, A0[0], A0[2]);
        dq4(w.z, s1, A0[1], A0[3]);
        dq4(w.y, s0, A1[0], A1[2]);
        dq4(w.w, s1, A1[1], A1[3]);
      } else if constexpr (FMT == WMat::NVFP4) {
        const uint2 w = *reinterpret_cast<const uint2*>(st + (warp * CK + kbl) * TB + lane * 8);  // row g ‖ row g+8
        const uint32_t sv = *reinterpret_cast<const uint16_t*>(st + C::WST + (warp * CK + kbl) * 32 + (t >> 1) * 16 + 2 * g);
        uint32_t lo, hi, ag[4], ah[4];
        sc_fp4(sv, lo, hi);
        dq_fp4x8(w.x, lo, ag);
        dq_fp4x8(w.y, hi, ah);
        A0[0] = ag[0], A0[1] = ah[0], A0[2] = ag[1], A0[3] = ah[1];
        A1[0] = ag[2], A1[1] = ah[2], A1[2] = ag[3], A1[3] = ah[3];
      } else if constexpr (FMT == kPack12) {  // the bf16 branch's u (row g) = w[0..3], v (row g+8) = w[4..7], exactly
        const uint8_t* b = st + (warp * CK + kbl) * TB;
        const uint4 lo = *reinterpret_cast<const uint4*>(b + lane * 16);
        const uint2 nb = *reinterpret_cast<const uint2*>(b + 512 + lane * 8);
        uint32_t w[8];
        p013::unpack8(lo.x, lo.y, nb.x, bb0, w);
        p013::unpack8(lo.z, lo.w, nb.y, bb1, w + 4);
        const int kb = kb0 + c * CK + kbl;
        if (kb == int(ex.nk)) ex.apply(a.xe, kb, lane, w);
        A0[0] = w[0], A0[1] = w[4], A0[2] = w[1], A0[3] = w[5];
        A1[0] = w[2], A1[1] = w[6], A1[2] = w[3], A1[3] = w[7];
      } else {
        const uint8_t* b = st + (warp * CK + kbl) * TB;
        const uint4 u = *reinterpret_cast<const uint4*>(b + lane * 16), v = *reinterpret_cast<const uint4*>(b + 512 + lane * 16);
        A0[0] = u.x, A0[1] = v.x, A0[2] = u.y, A0[3] = v.y;
        A1[0] = u.z, A1[1] = v.z, A1[2] = u.w, A1[3] = v.w;
      }
      uint4 xv[NT];
#pragma unroll
      for (int nt = 0; nt < NT; nt++)
        xv[nt] = *reinterpret_cast<const uint4*>(st + C::WST + C::SST + ((kbl * NT + nt) * 32 + lane) * 16);
#pragma unroll
      for (int nt = 0; nt < NT; nt++) mma(acc[nt], A0, xv[nt].x, xv[nt].y);  // k16 step j = 0 for all n tiles
#pragma unroll
      for (int nt = 0; nt < NT; nt++) mma(acc[nt], A1, xv[nt].z, xv[nt].w);  // then j = 1
    }
  }
  cp_wait<0>();
  __syncthreads();

  // partial tile P[m][r] (m = activation row, r = weight row in CTA); C fragment: c0 (g, 2t) c1 (g, 2t+1) c2 (g+8, 2t) c3
  float* P = reinterpret_cast<float*>(sm);
  const int r = warp * 16 + g;
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    const int m = nt * 8 + 2 * t;
    P[m * C::PSTRIDE + r] = acc[nt][0];
    P[(m + 1) * C::PSTRIDE + r] = acc[nt][1];
    P[m * C::PSTRIDE + r + 8] = acc[nt][2];
    P[(m + 1) * C::PSTRIDE + r + 8] = acc[nt][3];
  }
  cg::cluster_group cl = cg::this_cluster();
  if (S > 1) cl.sync();
  else __syncthreads();
  const int n0 = blockIdx.x * kRows, rank = S > 1 ? int(cl.block_rank()) : 0;
  for (int i = rank * kThreads + tid; i < a.M * kRows; i += S * kThreads) {  // ranks share the store work
    const int m = i / kRows, rr = i % kRows, o = m * C::PSTRIDE + rr;
    float v;
    if (S > 1) {
      v = cl.map_shared_rank(P, 0)[o];
      for (int s = 1; s < S; s++) v += cl.map_shared_rank(P, s)[o];  // fixed order s = 0..S-1
    } else {
      v = P[o];
    }
    if constexpr (FMT == WMat::NVFP4) v *= a.gs;
    if (n0 + rr < a.N) {
      if constexpr (F32) static_cast<float*>(a.y)[size_t(m) * a.N + n0 + rr] = v;
      else static_cast<bf16*>(a.y)[size_t(m) * a.N + n0 + rr] = __float2bfloat16_rn(v);
    }
  }
  if (S > 1) cl.sync();  // keep shared memory alive until every rank has read it
}

template <int FMT, int NT, bool F32, int CK, int BUDGET>
void launch(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  using C = Cfg<FMT, NT, CK, BUDGET>;
  const uint8_t* w = FMT == kPack12 ? L.pk : static_cast<const uint8_t*>(L.w);
  Args a{w, L.s, x, y, M, L.n, L.k, L.k / 32, L.k / 32 / L.split, L.gs, L.pk_base, L.pk_xo, L.pk_x};
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(L.npad / kRows, L.split);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1, at[0].val.clusterDim.y = L.split, at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = L.split > 1 ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, lin_kernel<FMT, NT, F32, C, CK>, a));
}

#include "kernels/linear/linear_prefill.cuh"  // large-M path (big::forward)
#include "kernels/linear/p004_gemm_tune.cuh"  // P004 gemm_tune: faster large-M path (p004::forward, same bits)

// Scheduling per M (measured on GB10): 48 KB per CTA (2 per SM) by default; for M > 16 with 97–144 CTAs, half-size
// chunks in 32 KB so 3 CTAs fit per SM and the grid is resident in one wave (qkvz, qkv at M = 64: +7–18 %).
template <int FMT, int NT, bool F32, int CK>
void pick_ck(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  const int ctas = L.npad / kRows * L.split;
  if (NT >= 3 && ctas > 96 && ctas <= 144) return launch<FMT, NT, F32, CK / 2, 32 * 1024>(L, x, y, M, s);
  launch<FMT, NT, F32, CK, 48 * 1024>(L, x, y, M, s);
}
template <int FMT, int NT, bool F32>
void pick(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  // P013: 4 k32 blocks per chunk (lm_head: as 2 or 1 at M ≤ 20, −2 % at M = 64, measured), 2 when a split's blocks
  // are not a multiple of 4 (a bf16 Lin's always are of 2)
  if constexpr (FMT == kPack12)
    if ((L.k / 32 / L.split) % Fmt<kPack12>::CK) return pick_ck<FMT, NT, F32, 2>(L, x, y, M, s);
  pick_ck<FMT, NT, F32, Fmt<FMT>::CK>(L, x, y, M, s);
}

// Shared-memory opt-in for every instantiation, once, at upload (not inside a later graph capture).
template <int FMT, int NT, bool F32, int CK, int BUDGET>
void set_smem() {
  using C = Cfg<FMT, NT, CK, BUDGET>;
  CUDA_CHECK(cudaFuncSetAttribute(lin_kernel<FMT, NT, F32, C, CK>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
}
template <int FMT, bool F32, int... I>
void set_smem_fmt(std::integer_sequence<int, I...>) {
  constexpr int CK = Fmt<FMT>::CK;
  (set_smem<FMT, I + 1, F32, CK, 48 * 1024>(), ...);
  (set_smem<FMT, I + 1, F32, CK / 2, 32 * 1024>(), ...);
  if constexpr (FMT == kPack12) {  // pick(): the chunk-2 fallback
    (set_smem<FMT, I + 1, F32, 2, 48 * 1024>(), ...);
    (set_smem<FMT, I + 1, F32, 1, 32 * 1024>(), ...);
  }
}
void init_kernels() {
  static const bool done = [] {
    constexpr auto nts = std::make_integer_sequence<int, 8>{};
    set_smem_fmt<WMat::MXFP8, false>(nts), set_smem_fmt<WMat::MXFP8, true>(nts);
    set_smem_fmt<WMat::BF16, false>(nts), set_smem_fmt<WMat::BF16, true>(nts);
    set_smem_fmt<WMat::NVFP4, false>(nts), set_smem_fmt<WMat::NVFP4, true>(nts);
    set_smem_fmt<kPack12, false>(nts), set_smem_fmt<kPack12, true>(nts);
    big::init();
    p004::init();
    return true;
  }();
  (void)done;
}

template <int FMT, bool F32>
void dispatch(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  switch ((M + 7) / 8) {
    case 1: return pick<FMT, 1, F32>(L, x, y, M, s);
    case 2: return pick<FMT, 2, F32>(L, x, y, M, s);
    case 3: return pick<FMT, 3, F32>(L, x, y, M, s);
    case 4: return pick<FMT, 4, F32>(L, x, y, M, s);
    case 5: return pick<FMT, 5, F32>(L, x, y, M, s);
    case 6: return pick<FMT, 6, F32>(L, x, y, M, s);
    case 7: return pick<FMT, 7, F32>(L, x, y, M, s);
    case 8: return pick<FMT, 8, F32>(L, x, y, M, s);
  }
}

template <bool F32>
void forward(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  if (M <= 0) return;
  if (M > kMaxRows) qlog::fail("linear", "M %d > %d", M, kMaxRows);
  if (L.fmt == WMat::NVFP4) {  // draft copies only: decode kernel, M ≤ 64
    if (M > kDecodeRows) qlog::fail("linear", "NVFP4 weights need M ≤ %d (M %d)", kDecodeRows, M);
    return dispatch<WMat::NVFP4, F32>(L, x, y, M, s);
  }
  if (M <= kDecodeRows && L.pk && patch::on(patch::lin_pack12)) return dispatch<kPack12, F32>(L, x, y, M, s);  // P013
  if (M > kDecodeRows && p004::takes(L, M) && patch::on(patch::gemm_tune)) {  // P004 (same bits as the paths below)
    if (L.fmt == WMat::MXFP8) p004::forward<WMat::MXFP8, F32>(L, x, y, M, s);
    else p004::forward<WMat::BF16, F32>(L, x, y, M, s);
    return;
  }
  if (M > kDecodeRows && big::supports(L) && (M > 2 * kDecodeRows || big::ctas(L, M) >= 48)) {  // prefill (same bits)
    if (L.fmt == WMat::MXFP8) big::forward<WMat::MXFP8, F32>(L, x, y, M, s);
    else big::forward<WMat::BF16, F32>(L, x, y, M, s);
    return;
  }
  if (M > kDecodeRows) {  // 65..128 rows on a sparse grid (out/o, ba, router; faster) or split > 5: 64-row passes
    const size_t es = F32 ? 4 : 2;
    for (int m0 = 0; m0 < M; m0 += kDecodeRows)
      forward<F32>(L, x + size_t(m0) * L.k, static_cast<uint8_t*>(y) + size_t(m0) * L.n * es, std::min(kDecodeRows, M - m0), s);
    return;
  }
  if (L.fmt == WMat::MXFP8) dispatch<WMat::MXFP8, F32>(L, x, y, M, s);
  else dispatch<WMat::BF16, F32>(L, x, y, M, s);
}

// Split count from the shape only (never from M): 1 with ≥ 32 row blocks; else the largest S ≤ 8 that divides the
// chunks, keeps ≥ 4 chunks per split and ≤ 32 CTAs (ba 48×2560 → 5, router 512×2560 → 4; measured).
int pick_split(int fmt, int npad, int k) {
  const int rb = npad / kRows, chunks = k / 32 / chunk_blocks(fmt);
  if (rb >= 32) return 1;
  int best = 1;
  for (int s = 2; s <= kMaxSplit; s++)
    if (chunks % s == 0 && chunks / s >= 4 && rb * s <= 32) best = s;
  return best;
}

}  // namespace

Lin lin_make(int fmt, int n, int k, const uint8_t* w, const uint8_t* s, int split, bool pack12) {
  if (fmt != WMat::MXFP8 && fmt != WMat::BF16) qlog::fail("linear", "format %d not supported", fmt);
  if (n <= 0 || k <= 0 || k % (32 * chunk_blocks(fmt)))
    qlog::fail("linear", "shape %d×%d not supported (K must be a multiple of %d)", n, k, 32 * chunk_blocks(fmt));
  init_kernels();
  Lin L;
  L.fmt = fmt, L.n = n, L.k = k, L.npad = cdiv(n, kRows) * kRows;
  const int KB = k / 32, tiles = L.npad / 16, chunks = KB / chunk_blocks(fmt);
  L.split = split > 0 ? split : pick_split(fmt, L.npad, k);
  if (L.split > kMaxSplit || chunks % L.split) qlog::fail("linear", "split %d not possible (chunks %d)", L.split, chunks);
  const size_t es = fmt == WMat::MXFP8 ? 1 : 2;
  std::vector<uint8_t> pw(size_t(L.npad) * k * es), ps(fmt == WMat::MXFP8 ? size_t(L.npad) * KB : 0);
  // MXFP8 (tile T, block kb): 512 B, lane (g,t) → [row 16T+g, k 32kb+8t..+7 | row 16T+g+8, same k].
  //   scales: 16 B, byte 2g = row 16T+g, byte 2g+1 = row 16T+g+8.
  // bf16 (tile T, block kb): 1024 B, lane → row 16T+g k 32kb+8t..+7 at lane·16, row 16T+g+8 at 512 + lane·16.
  // Padding rows (≥ n) are zero (scale 0 → weight 0).
  parallel_for(tiles, [&](int T) {
    for (int kb = 0; kb < KB; kb++) {
      uint8_t* blk = pw.data() + (size_t(T) * KB + kb) * 512 * es;
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane >> 2, t = lane & 3, r0 = 16 * T + g, r1 = r0 + 8;
        const size_t ko = size_t(kb * 32 + 8 * t) * es;
        uint8_t* d1 = fmt == WMat::MXFP8 ? blk + lane * 16 + 8 : blk + 512 + lane * 16;
        if (r0 < n) std::memcpy(blk + lane * 16, w + size_t(r0) * k * es + ko, 8 * es);
        if (r1 < n) std::memcpy(d1, w + size_t(r1) * k * es + ko, 8 * es);
        if (fmt == WMat::MXFP8 && t == 0) {
          uint8_t* sd = ps.data() + (size_t(T) * KB + kb) * 16 + 2 * g;
          sd[0] = r0 < n ? s[size_t(r0) * KB + kb] : 0;
          sd[1] = r1 < n ? s[size_t(r1) * KB + kb] : 0;
        }
      }
    }
  });
  CUDA_CHECK(cudaMalloc(&L.w, pw.size()));
  CUDA_CHECK(cudaMemcpy(L.w, pw.data(), pw.size(), cudaMemcpyHostToDevice));
  if (!ps.empty()) {
    CUDA_CHECK(cudaMalloc(&L.s, ps.size()));
    CUDA_CHECK(cudaMemcpy(L.s, ps.data(), ps.size(), cudaMemcpyHostToDevice));
  }
  if (pack12) {  // P013: 12-bit copy for the decode kernel (built whatever the switch; bf16 copy kept)
    if (fmt != WMat::BF16) qlog::fail("linear", "pack12 is bf16 only (format %d)", fmt);
    p013::build(L, reinterpret_cast<const uint16_t*>(w));
  }
  return L;
}

void lin_pack12_unpack(const Lin& L, bf16* out, cudaStream_t s) {
  if (!L.pk) qlog::fail("linear", "no pack12 copy");
  const int blocks = L.npad / 16 * (L.k / 32);
  p013::unpack_kernel<<<cdiv(blocks, 4), 128, 0, s>>>(L.pk, L.pk_base, L.pk_xo, L.pk_x, out, L.n, L.k, L.npad / 16);
  CUDA_CHECK(cudaGetLastError());
}

Lin lin_upload(const Weights& W, const std::vector<std::string>& names, bool pack12) {
  if (names.empty()) qlog::fail("linear", "tensor list is empty");
  std::vector<uint8_t> wb, sb;
  int fmt = -1, n = 0, k = -1;
  for (const auto& name : names) {
    const auto& ti = W.info(name);
    const int f = ti.dtype == "F8_E4M3" ? WMat::MXFP8 : ti.dtype == "BF16" ? WMat::BF16 : -1;
    if (f < 0 || (fmt >= 0 && f != fmt)) qlog::fail("linear", "%s: format %s unsupported or mixed", name.c_str(), ti.dtype.c_str());
    std::vector<int64_t> sh;
    auto b = W.slice(name, &sh);
    if (sh.size() != 2 || (k >= 0 && sh[1] != k)) qlog::fail("linear", "%s: shape mismatch", name.c_str());
    fmt = f, k = int(sh[1]), n += int(sh[0]);
    wb.insert(wb.end(), b.begin(), b.end());
    if (f == WMat::MXFP8) {
      std::vector<int64_t> ss;
      auto sc = W.slice(name + "_scale", &ss);
      if (ss.size() != 2 || ss[0] != sh[0] || ss[1] * 32 != sh[1]) qlog::fail("linear", "%s_scale: shape mismatch", name.c_str());
      sb.insert(sb.end(), sc.begin(), sc.end());
    }
  }
  return lin_make(fmt, n, k, wb.data(), sb.empty() ? nullptr : sb.data(), 0, pack12);
}

// NVFP4 copy of a bf16 matrix (OCP NVFP4: tensor scale g = amax / (448·6); per row and 16 columns an e4m3 scale
// s = e4m3(amax16 / (6g)); codes e2m1 = round-to-nearest-even(x / (s·g)), saturating at ±6; a group whose s rounds
// to 0 is stored as zeros). Layout per (16-row tile T, k32 block kb): 256 B, lane (g,t) at lane·8: row 16T+g then
// row 16T+g+8, each a word of the codes of k 32kb+8t … +7 in nibble order [k6,k4,k2,k0,k7,k5,k3,k1]; scales 32 B,
// group j (k 32kb+16j …) at 16j: byte 2g = row 16T+g, 2g+1 = row 16T+g+8.
Lin lin_make_nvfp4(int n, int k, const bf16* w) {
  if (n <= 0 || k <= 0 || k % (32 * Fmt<WMat::NVFP4>::CK)) qlog::fail("linear", "NVFP4 shape %d×%d not supported", n, k);
  init_kernels();
  Lin L;
  L.fmt = WMat::NVFP4, L.n = n, L.k = k, L.npad = cdiv(n, kRows) * kRows, L.split = 1;
  const int KB = k / 32, tiles = L.npad / 16, G = k / 16;
  float amax = 0.f;
  for (size_t i = 0; i < size_t(n) * k; i++) amax = std::max(amax, std::fabs(__bfloat162float(w[i])));
  const float g = amax > 0.f ? amax / (448.f * 6.f) : 1.f;
  L.gs = g * 256.f;
  std::vector<uint8_t> code(size_t(L.npad) * k, 0), sc(size_t(L.npad) * G, 0x38);  // padding rows: zeros, scale 1
  auto e2m1 = [](float v) -> uint8_t {
    const float x = std::fabs(v);
    const uint8_t m = x <= 0.25f ? 0 : x < 0.75f ? 1 : x <= 1.25f ? 2 : x < 1.75f ? 3 : x <= 2.5f ? 4 : x < 3.5f ? 5 : x <= 5.f ? 6 : 7;
    return uint8_t((v < 0.f ? 8 : 0) | m);
  };
  parallel_for(n, [&](int r) {
    for (int j = 0; j < G; j++) {
      const bf16* x = w + size_t(r) * k + 16 * j;
      float a = 0.f;
      for (int e = 0; e < 16; e++) a = std::max(a, std::fabs(__bfloat162float(x[e])));
      const uint8_t sb = __nv_cvt_float_to_fp8(a / (6.f * g), __NV_SATFINITE, __NV_E4M3);
      const float sf = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
      if (sf == 0.f) continue;  // zeros, scale 1
      sc[size_t(r) * G + j] = sb;
      for (int e = 0; e < 16; e++) code[size_t(r) * k + 16 * j + e] = e2m1(__bfloat162float(x[e]) / (sf * g));
    }
  });
  std::vector<uint8_t> pw(size_t(L.npad) * k / 2), ps(size_t(L.npad) * KB * 32);
  static constexpr int ord[8] = {6, 4, 2, 0, 7, 5, 3, 1};
  parallel_for(tiles, [&](int T) {
    for (int kb = 0; kb < KB; kb++) {
      uint8_t* blk = pw.data() + (size_t(T) * KB + kb) * 256;
      for (int lane = 0; lane < 32; lane++) {
        const int gg = lane >> 2, t = lane & 3;
        for (int h = 0; h < 2; h++) {
          const uint8_t* c = code.data() + size_t(16 * T + gg + 8 * h) * k + kb * 32 + 8 * t;
          uint32_t word = 0;
          for (int i = 0; i < 8; i++) word |= uint32_t(c[ord[i]]) << (4 * i);
          std::memcpy(blk + lane * 8 + 4 * h, &word, 4);
        }
      }
      uint8_t* sd = ps.data() + (size_t(T) * KB + kb) * 32;
      for (int j = 0; j < 2; j++)
        for (int gg = 0; gg < 8; gg++) {
          sd[16 * j + 2 * gg] = sc[size_t(16 * T + gg) * G + 2 * kb + j];
          sd[16 * j + 2 * gg + 1] = sc[size_t(16 * T + gg + 8) * G + 2 * kb + j];
        }
    }
  });
  CUDA_CHECK(cudaMalloc(&L.w, pw.size()));
  CUDA_CHECK(cudaMemcpy(L.w, pw.data(), pw.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&L.s, ps.size()));
  CUDA_CHECK(cudaMemcpy(L.s, ps.data(), ps.size(), cudaMemcpyHostToDevice));
  return L;
}

void lin_free(Lin& L) {
  if (L.w) cudaFree(L.w);
  if (L.s) cudaFree(L.s);
  for (void* p : {static_cast<void*>(L.pk), static_cast<void*>(L.pk_base), static_cast<void*>(L.pk_xo), static_cast<void*>(L.pk_x)})
    if (p) cudaFree(p);
  L = Lin{};
}

void lin_forward(const Lin& L, const bf16* x, bf16* y, int M, cudaStream_t s) { forward<false>(L, x, y, M, s); }
void lin_forward_f32(const Lin& L, const bf16* x, float* y, int M, cudaStream_t s) { forward<true>(L, x, y, M, s); }
void lin_p004_tile(int t) { p004::g_force = t; }

}  // namespace q38
