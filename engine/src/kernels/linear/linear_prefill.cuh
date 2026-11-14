// linear_prefill.cuh — large-M path of lin_forward (M > kDecodeRows, prefill): tiled tensor-core GEMM over the same
// packed weights as the decode kernel. Included by linear.cu inside its anonymous namespace (Fmt, cp16z, mma, dq4).
//
// Bitwise contract: every output equals lin_kernel's. Each MMA is the decode kernel's instruction with identical
// operands — swap-AB (weight rows on the MMA M side, activation rows on the N side), the same k → slot permutation
// (lane t of a quad holds k 8t..8t+7 of a k32 block, k16 step 0 then 1), the same chain (k32 blocks ascending into one
// fp32 accumulator from 0) and, for K-split shapes, the same split sum (split s accumulated from 0, then v = P0,
// v += P1, … in order — sp::kernel, one warp row per split). MMA columns are independent, so only scheduling differs:
// a CTA computes a BN × BM tile (weight rows × activation rows) through a cp.async multistage pipeline; activation k32
// blocks sit in shared memory as [block][row][64 B] (a lane's 16-B fragment load is conflict-free), weights in their
// packed (tile, block) order.
#pragma once

namespace big {

constexpr int kKbs = 2;  // k32 blocks per pipeline stage

template <int FMT, int WN, int WM, int ST>
struct Cfg {
  static constexpr int TB = Fmt<FMT>::TB, SB = Fmt<FMT>::SB;
  static constexpr int BN = WN * 32, BM = WM * 64, NTH = WN * WM * 32, TILES = BN / 16;  // warp: 32 w-rows × 64 rows
  static constexpr int WST = TILES * kKbs * TB, SST = TILES * kKbs * SB, XST = BM * kKbs * 64;
  static constexpr int STAGE = WST + SST + XST;
  static constexpr int PB = BN + 8, PF = BN + 4;  // epilogue pitches (bf16 / fp32 elements), conflict-free
  static constexpr int SMEM = std::max({ST * STAGE, BM * PB * 2, BM * PF * 4});
};

struct Args {
  const uint8_t* w;  // packed weights (tile, k32 block)
  const uint8_t* s;  // packed scales (MXFP8)
  const bf16* x;     // [M, K]
  void* y;           // [M, N]
  int M, N, K, KB, kb_split, tiles;  // tiles = npad / 16 (valid weight tiles)
  int MT, NT;                        // CTA tiles along activation rows / weight rows
};

__device__ __forceinline__ void cpz(void* dst, const void* src, bool valid) {  // 16 B, L2 only, zero-fill if !valid
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}

template <int FMT, int WN, int WM, int ST, bool F32>
__global__ void __launch_bounds__(WN * WM * 32, 1) kernel(Args a) {
  using C = Cfg<FMT, WN, WM, ST>;
  constexpr int TB = C::TB;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, wm = warp / WN;
  // grouped raster: GM activation tiles share the sweep over all weight tiles (their activations stay in L2)
  constexpr int GM = 8;
  const int per = GM * a.NT, grp = blockIdx.x / per, fm = grp * GM, gm = min(a.MT - fm, GM);
  const int rem = blockIdx.x - grp * per, mt = fm + rem % gm, nt = rem / gm;
  const int T0 = nt * C::TILES, m0 = mt * C::BM, nch = a.KB / kKbs;

  auto load = [&](int c, int buf) {
    uint8_t* st = sm + buf * C::STAGE;
    const int kb = c * kKbs;
    constexpr int PT = kKbs * TB / 16;  // 16-B pieces per tile and stage (contiguous in global)
#pragma unroll
    for (int p = tid; p < C::WST / 16; p += C::NTH) {
      const int T = T0 + p / PT;
      const bool ok = T < a.tiles;
      cpz(st + p * 16, a.w + (size_t(ok ? T : 0) * a.KB + kb) * TB + (p % PT) * 16, ok);
    }
    if constexpr (C::SST > 0) {
      if (tid < C::SST / 16) {  // piece = (tile, block)
        const int T = T0 + tid / kKbs;
        const bool ok = T < a.tiles;
        cpz(st + C::WST + tid * 16, a.s + (size_t(ok ? T : 0) * a.KB + kb + tid % kKbs) * 16, ok);
      }
    }
#pragma unroll
    for (int p = tid; p < C::BM * kKbs * 4; p += C::NTH) {  // row-major pieces: 8 per row (128 B of global)
      const int r = p / (kKbs * 4), q = p % (kKbs * 4), kbl = q >> 2, ch = q & 3, m = m0 + r;
      const bf16* src = a.x + size_t(min(m, a.M - 1)) * a.K + (kb + kbl) * 32 + ch * 8;
      cpz(st + C::WST + C::SST + ((kbl * C::BM + r) * 4 + ch) * 16, src, m < a.M);
    }
  };

  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

#pragma unroll
  for (int i = 0; i < ST - 1; i++) {
    if (i < nch) load(i, i);
    cp_commit();
  }
  for (int c = 0; c < nch; c++) {
    cp_wait<ST - 2>();
    __syncthreads();
    if (c + ST - 1 < nch) load(c + ST - 1, (c + ST - 1) % ST);
    cp_commit();
    const uint8_t* st = sm + (c % ST) * C::STAGE;
#pragma unroll
    for (int kbl = 0; kbl < kKbs; kbl++) {
      uint32_t A0[2][4], A1[2][4];
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int ti = wn * 2 + i;
        if constexpr (FMT == WMat::MXFP8) {
          const uint4 w = *reinterpret_cast<const uint4*>(st + (ti * kKbs + kbl) * TB + lane * 16);  // row g ‖ g+8
          const uint32_t sv = *reinterpret_cast<const uint16_t*>(st + C::WST + (ti * kKbs + kbl) * 16 + 2 * g);
          const float s0 = ue8m0_to_float(sv & 0xff), s1 = ue8m0_to_float(sv >> 8);
          dq4(w.x, s0, A0[i][0], A0[i][2]);
          dq4(w.z, s1, A0[i][1], A0[i][3]);
          dq4(w.y, s0, A1[i][0], A1[i][2]);
          dq4(w.w, s1, A1[i][1], A1[i][3]);
        } else {
          const uint8_t* b = st + (ti * kKbs + kbl) * TB;
          const uint4 u = *reinterpret_cast<const uint4*>(b + lane * 16), v = *reinterpret_cast<const uint4*>(b + 512 + lane * 16);
          A0[i][0] = u.x, A0[i][1] = v.x, A0[i][2] = u.y, A0[i][3] = v.y;
          A1[i][0] = u.z, A1[i][1] = v.z, A1[i][2] = u.w, A1[i][3] = v.w;
        }
      }
      uint4 xv[8];
#pragma unroll
      for (int j = 0; j < 8; j++)
        xv[j] = *reinterpret_cast<const uint4*>(st + C::WST + C::SST + ((kbl * C::BM + wm * 64 + j * 8 + g) * 4 + t) * 16);
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) mma(acc[i][j], A0[i], xv[j].x, xv[j].y);  // k16 step 0
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) mma(acc[i][j], A1[i], xv[j].z, xv[j].w);  // then step 1
    }
  }
  cp_wait<0>();
  __syncthreads();

  // epilogue: C fragment c0 (w-row g, act-row 2t) c1 (g, 2t+1) c2 (g+8, 2t) c3 (g+8, 2t+1) → smem [act row][w row]
  auto res = [&](int i, int j, int e) -> float { return acc[i][j][e]; };
  constexpr int V = F32 ? 4 : 8, P = F32 ? C::PF : C::PB;
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int wr = wn * 32 + i * 16 + g, ar = wm * 64 + j * 8 + 2 * t;
      if constexpr (F32) {
        float* Ps = reinterpret_cast<float*>(sm);
        Ps[ar * P + wr] = res(i, j, 0), Ps[(ar + 1) * P + wr] = res(i, j, 1);
        Ps[ar * P + wr + 8] = res(i, j, 2), Ps[(ar + 1) * P + wr + 8] = res(i, j, 3);
      } else {
        bf16* Ps = reinterpret_cast<bf16*>(sm);
        Ps[ar * P + wr] = __float2bfloat16_rn(res(i, j, 0)), Ps[(ar + 1) * P + wr] = __float2bfloat16_rn(res(i, j, 1));
        Ps[ar * P + wr + 8] = __float2bfloat16_rn(res(i, j, 2));
        Ps[(ar + 1) * P + wr + 8] = __float2bfloat16_rn(res(i, j, 3));
      }
    }
  __syncthreads();
  using E = typename std::conditional<F32, float, bf16>::type;
  const E* Ps = reinterpret_cast<const E*>(sm);
  E* y = static_cast<E*>(a.y);
  const int n0 = nt * C::BN;
  const bool vec = a.N % V == 0;
  for (int p = tid; p < C::BM * (C::BN / V); p += C::NTH) {
    const int r = p / (C::BN / V), q = p % (C::BN / V), m = m0 + r, n = n0 + q * V;
    if (m >= a.M || n >= a.N) continue;
    const E* src = Ps + r * P + q * V;
    E* dst = y + size_t(m) * a.N + n;
    if (vec) {
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else {
      for (int e = 0; e < V && n + e < a.N; e++) dst[e] = src[e];
    }
  }
}

// ---- split-parallel variant (K-split shapes: ba S=5, router S=4) ------------------------------------------------
// Warp (wn, s) computes the 32 × 64 tile of split s over its own k32 blocks [s·kb_split, +kb_split) from 0; the S
// partials meet in shared memory and are summed v = P0, v += P1, … (the decode cluster's order). One k32 block of every
// split per stage; S is a runtime value (warps = 2·S, BN = 64 weight rows, BM = 64 activation rows).
namespace sp {
constexpr int WN = 2, BN = 64, BM = 64, TILES = 4, PF = BN + 4, kSmemMax = 101376;
template <int FMT> constexpr int stage_bytes(int S) { return S * (TILES * (Fmt<FMT>::TB + Fmt<FMT>::SB) + BM * 64); }
template <int FMT> int stages(int S) {
  const int st = stage_bytes<FMT>(S), ep = S * BM * PF * 4;
  if (ep > kSmemMax) return 0;
  return 3 * st <= kSmemMax ? 3 : 2 * st <= kSmemMax ? 2 : 0;
}

template <int FMT, int ST, bool F32>
__global__ void __launch_bounds__(320, 1) kernel(Args a) {  // S ≤ 5
  constexpr int TB = Fmt<FMT>::TB, SB = Fmt<FMT>::SB;
  extern __shared__ __align__(128) uint8_t sm[];
  const int S = a.KB / a.kb_split, NTH = WN * 32 * S;
  const int WSZ = S * TILES * TB, SSZ = S * TILES * SB, STAGE = WSZ + SSZ + S * BM * 64;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp % WN, ws = warp / WN;
  const int nt = blockIdx.x % a.NT, mt = blockIdx.x / a.NT;  // consecutive CTAs share the activation rows (L2)
  const int T0 = nt * TILES, m0 = mt * BM, nch = a.kb_split;

  auto load = [&](int c, int buf) {  // k32 block s·kb_split + c of every split s
    uint8_t* st = sm + buf * STAGE;
    constexpr int PT = TB / 16;
    for (int p = tid; p < WSZ / 16; p += NTH) {  // [s][tile][TB]
      const int s = p / (TILES * PT), T = T0 + (p / PT) % TILES;
      const bool ok = T < a.tiles;
      cpz(st + p * 16, a.w + (size_t(ok ? T : 0) * a.KB + s * a.kb_split + c) * TB + (p % PT) * 16, ok);
    }
    if constexpr (SB > 0) {
      for (int p = tid; p < SSZ / 16; p += NTH) {  // [s][tile][16]
        const int s = p / TILES, T = T0 + p % TILES;
        const bool ok = T < a.tiles;
        cpz(st + WSZ + p * 16, a.s + (size_t(ok ? T : 0) * a.KB + s * a.kb_split + c) * 16, ok);
      }
    }
    for (int p = tid; p < S * BM * 4; p += NTH) {  // [s][row][64 B]
      const int s = p / (BM * 4), r = (p / 4) % BM, ch = p & 3, m = m0 + r;
      const bf16* src = a.x + size_t(min(m, a.M - 1)) * a.K + (s * a.kb_split + c) * 32 + ch * 8;
      cpz(st + WSZ + SSZ + p * 16, src, m < a.M);
    }
  };

  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
#pragma unroll
  for (int i = 0; i < ST - 1; i++) {
    if (i < nch) load(i, i);
    cp_commit();
  }
  for (int c = 0; c < nch; c++) {
    cp_wait<ST - 2>();
    __syncthreads();
    if (c + ST - 1 < nch) load(c + ST - 1, (c + ST - 1) % ST);
    cp_commit();
    const uint8_t* st = sm + (c % ST) * STAGE;
    uint32_t A0[2][4], A1[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int ti = ws * TILES + wn * 2 + i;
      if constexpr (FMT == WMat::MXFP8) {
        const uint4 w = *reinterpret_cast<const uint4*>(st + ti * TB + lane * 16);
        const uint32_t sv = *reinterpret_cast<const uint16_t*>(st + WSZ + ti * 16 + 2 * g);
        const float s0 = ue8m0_to_float(sv & 0xff), s1 = ue8m0_to_float(sv >> 8);
        dq4(w.x, s0, A0[i][0], A0[i][2]);
        dq4(w.z, s1, A0[i][1], A0[i][3]);
        dq4(w.y, s0, A1[i][0], A1[i][2]);
        dq4(w.w, s1, A1[i][1], A1[i][3]);
      } else {
        const uint8_t* b = st + ti * TB;
        const uint4 u = *reinterpret_cast<const uint4*>(b + lane * 16), v = *reinterpret_cast<const uint4*>(b + 512 + lane * 16);
        A0[i][0] = u.x, A0[i][1] = v.x, A0[i][2] = u.y, A0[i][3] = v.y;
        A1[i][0] = u.z, A1[i][1] = v.z, A1[i][2] = u.w, A1[i][3] = v.w;
      }
    }
    uint4 xv[8];
#pragma unroll
    for (int j = 0; j < 8; j++)
      xv[j] = *reinterpret_cast<const uint4*>(st + WSZ + SSZ + ((ws * BM + j * 8 + g) * 4 + t) * 16);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int j = 0; j < 8; j++) mma(acc[i][j], A0[i], xv[j].x, xv[j].y);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int j = 0; j < 8; j++) mma(acc[i][j], A1[i], xv[j].z, xv[j].w);
  }
  cp_wait<0>();
  __syncthreads();
  float* P = reinterpret_cast<float*>(sm) + ws * BM * PF;  // partial of split ws: [act row][w row]
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int wr = wn * 32 + i * 16 + g, ar = j * 8 + 2 * t;
      P[ar * PF + wr] = acc[i][j][0], P[(ar + 1) * PF + wr] = acc[i][j][1];
      P[ar * PF + wr + 8] = acc[i][j][2], P[(ar + 1) * PF + wr + 8] = acc[i][j][3];
    }
  __syncthreads();
  const float* P0 = reinterpret_cast<const float*>(sm);
  for (int p = tid; p < BM * BN; p += NTH) {
    const int r = p / BN, q = p % BN, m = m0 + r, n = nt * BN + q;
    if (m >= a.M || n >= a.N) continue;
    float v = P0[r * PF + q];
    for (int s = 1; s < S; s++) v += P0[s * BM * PF + r * PF + q];  // fixed order s = 0..S-1
    if constexpr (F32) static_cast<float*>(a.y)[size_t(m) * a.N + n] = v;
    else static_cast<bf16*>(a.y)[size_t(m) * a.N + n] = __float2bfloat16_rn(v);
  }
}

template <int FMT, bool F32>
bool launch(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  const int S = L.split, st = stages<FMT>(S);
  if (st == 0 || S > 5) return false;
  Args a{static_cast<const uint8_t*>(L.w), L.s, x, y, M, L.n, L.k, L.k / 32, L.k / 32 / S, L.npad / 16,
         cdiv(M, BM), cdiv(L.npad, BN)};
  const int smem = std::max(st * stage_bytes<FMT>(S), S * BM * PF * 4);
  if (st == 3) kernel<FMT, 3, F32><<<a.MT * a.NT, WN * 32 * S, smem, s>>>(a);
  else kernel<FMT, 2, F32><<<a.MT * a.NT, WN * 32 * S, smem, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
  return true;
}
template <int FMT, bool F32>
void set_smem() {
  CUDA_CHECK(cudaFuncSetAttribute(kernel<FMT, 3, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax));
  CUDA_CHECK(cudaFuncSetAttribute(kernel<FMT, 2, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax));
}
}  // namespace sp

template <int FMT, int WN, int WM, int ST, bool F32>
void launch(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  using C = Cfg<FMT, WN, WM, ST>;
  Args a{static_cast<const uint8_t*>(L.w), L.s, x, y, M, L.n, L.k, L.k / 32, L.k / 32 / L.split, L.npad / 16,
         cdiv(M, C::BM), cdiv(L.npad, C::BN)};
  kernel<FMT, WN, WM, ST, F32><<<a.MT * a.NT, C::NTH, C::SMEM, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

template <int FMT, int WN, int WM, int ST, bool F32>
void set_smem() {
  using C = Cfg<FMT, WN, WM, ST>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel<FMT, WN, WM, ST, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
}

// MXFP8: 4 stages (24.3 KB each at 128×128), bf16 weights: 3 stages (32 KB). WN = 2 when the weights are one 64-row block.
template <int FMT> constexpr int stages() { return FMT == WMat::MXFP8 ? 4 : 3; }

template <int FMT, bool F32>
void set_smem_all() {
  set_smem<FMT, 4, 2, stages<FMT>(), F32>();
  set_smem<FMT, 2, 2, stages<FMT>(), F32>();
}
void init() {
  set_smem_all<WMat::MXFP8, false>(), set_smem_all<WMat::MXFP8, true>();
  set_smem_all<WMat::BF16, false>(), set_smem_all<WMat::BF16, true>();
  sp::set_smem<WMat::MXFP8, false>(), sp::set_smem<WMat::MXFP8, true>();
  sp::set_smem<WMat::BF16, false>(), sp::set_smem<WMat::BF16, true>();
}

// true if a prefill kernel takes this shape (K-split shapes need S ≤ 5 and their partials in shared memory)
inline bool supports(const Lin& L) { return L.split == 1 || (L.split <= 5 && sp::stages<WMat::MXFP8>(L.split) > 0 &&
                                                                 sp::stages<WMat::BF16>(L.split) > 0); }

// CTAs of the launch forward() would make (for the M ≤ 128 fallback to 64-row decode passes)
inline int ctas(const Lin& L, int M) {
  if (L.split > 1 && L.split <= 5) return cdiv(M, sp::BM) * cdiv(L.npad, sp::BN);
  return cdiv(M, 128) * cdiv(L.npad, L.npad <= 64 ? 64 : 128);
}

template <int FMT, bool F32>
void forward(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  if (L.split > 1) {
    sp::launch<FMT, F32>(L, x, y, M, s);
    return;
  }
  if (L.npad <= 64) return launch<FMT, 2, 2, stages<FMT>(), F32>(L, x, y, M, s);
  launch<FMT, 4, 2, stages<FMT>(), F32>(L, x, y, M, s);
}

}  // namespace big
