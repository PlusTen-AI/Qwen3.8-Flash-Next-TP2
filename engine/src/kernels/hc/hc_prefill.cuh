// hc_prefill.cuh — large-M (prefill, M > kChunkM) kernels of the hc family. Included by hc.cu inside its anonymous
// namespace (uses H, D, kDn*, kUp*, kLsW, mma, hc_norm1, bfr, b2f, sigm, unpack8, pack8, load8f).
//
// Bitwise contract: every value equals the decode kernels' (K2+K2b, K3, k_gemv, k_ple_conv) for the same row — the
// same MMA instructions with identical fragments (weights on the MMA M side, activation rows on the N side) chained in
// the same order, the same split fold and the same epilogue arithmetic. Only the work split differs.
#pragma once

namespace pre {

__device__ __forceinline__ void cpz(void* dst, const void* src, bool valid) {  // 16 B, L2 only, zero-fill if !valid
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// four 8×8 b16 matrices; lane i addresses row i%8 of matrix i/8
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// ---- K2 down (large M) -----------------------------------------------------------------------------------------
// K2 + K2b of one row: z = 0; z += P_kc for kc = 0..39, P_kc = MMA chain over the 16 k16 steps of 256-col chunk kc
// from 0; lora = bf16(silu(bf16(z)/4)), inj = bf16(z). Here a CTA owns 128 weight rows (group grp of 3: tiles
// 8grp..8grp+7; tile 20 = the 4 inject rows, compact fragments; tiles ≥ 21 idle) × BM = 32·WM rows and keeps z in
// registers across K. Warp (wn, wm): tiles 2wn, 2wn+1 × rows 32wm..+31 (4 n8). Stages of 32 columns (2 k16 steps):
// weights [8 tiles][2][512 B], X [BM rows][80 B] and g [32] fp32 by cp.async; after the barrier that opens stage c
// every thread normalises the X piece of stage c+1 it copied (xn = bf16(y + y·g), rrms of its row in registers), and
// the next barrier publishes it. The three groups of a row tile are adjacent CTAs (X comes from DRAM once).
constexpr int kDbSt = 5, kDbXp = 80, kDbW = 8 * 2 * 512;  // stages, X row pitch (B), weight bytes per stage
template <int WM> constexpr int down_smem() { return kDbSt * (kDbW + 32 * WM * kDbXp + 128); }

template <int WM, int NJ, bool INJ>  // NJ = n8 tiles (8 rows) per warp; 4 (16 warps at WM 4) beat 8 on GB10
__global__ void __launch_bounds__(512 * WM / NJ, 1) k_down(const uint4* __restrict__ Wd, const uint2* __restrict__ Wi,
                                                          const bf16* X, const float* rrms, const float* g, bf16* lora,
                                                          bf16* inj, int M) {
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int BM = 32 * WM, NTH = 512 * WM / NJ, PPT = NJ / 4, GOFF = kDbW + BM * kDbXp, STAGE = GOFF + 128;
  constexpr int NST = D / 32, N = INJ ? kHcDown : kHcRank;
  static_assert(BM * 4 == NTH * PPT, "X pieces per thread");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, tq = lane & 3;
  const int wn = warp & 3, wm = warp >> 2, grp = blockIdx.x % 3, m0 = (blockIdx.x / 3) * BM;
  const int T0 = grp * 8 + 2 * wn;                                // this warp's tiles T0, T0 + 1
  const bool act0 = T0 < 20 || (INJ && T0 == 20), act1 = T0 + 1 < 20;  // tile 21+ does not exist

  auto load = [&](int c, int buf) {  // stage c: columns [32c, +32)
    uint8_t* st = sm + buf * STAGE;
    for (int p = tid; p < kDbW / 16; p += NTH) {  // tile i = p/64: 2 k16 blocks = 1024 contiguous bytes
      const int T = grp * 8 + p / 64, q = p % 64;
      if (T < 20) cpz(st + p * 16, reinterpret_cast<const uint8_t*>(Wd) + (size_t(T) * kDnKs + 2 * c) * 512 + q * 16, true);
      else if (INJ && T == 20 && (q & 31) < 8)  // compact inject: 128 B per k16 step → first 128 B of its 512-B slot
        cpz(st + p * 16, reinterpret_cast<const uint8_t*>(Wi) + (2 * c + q / 32) * 128 + (q & 31) * 16, true);
    }
#pragma unroll
    for (int i = 0; i < PPT; i++) {  // X pieces p = tid + i·NTH (row p/4, 16-B column p%4)
      const int p = tid + i * NTH, r = p >> 2, q = p & 3, m = m0 + r;
      cpz(st + kDbW + r * kDbXp + q * 16, X + size_t(min(m, M - 1)) * D + c * 32 + q * 8, m < M);
    }
    if (tid < 8) cpz(st + GOFF + tid * 16, g + c * 32 + tid * 4, true);
  };
  float rr4[PPT][kStreams];  // rrms of the rows of this thread's X pieces
#pragma unroll
  for (int i = 0; i < PPT; i++) {
    const int m = m0 + ((tid + i * NTH) >> 2);
    const float4 v = m < M ? *reinterpret_cast<const float4*>(rrms + size_t(m) * kStreams) : make_float4(0.f, 0.f, 0.f, 0.f);
    rr4[i][0] = v.x, rr4[i][1] = v.y, rr4[i][2] = v.z, rr4[i][3] = v.w;
  }
  auto convert = [&](int c, int buf) {  // the pieces this thread copied; rows ≥ M stay zero (as K2)
    const int sidx = (c * 32) / H;
#pragma unroll
    for (int i = 0; i < PPT; i++) {
      const int p = tid + i * NTH, r = p >> 2, q = p & 3, m = m0 + r;
      if (m >= M) continue;
      uint4* ptr = reinterpret_cast<uint4*>(sm + buf * STAGE + kDbW + r * kDbXp + q * 16);
      float x[8], gw[8];
      unpack8(*ptr, x);
      load8f(reinterpret_cast<const float*>(sm + buf * STAGE + GOFF) + q * 8, gw);
      const float rr = sidx == 0 ? rr4[i][0] : sidx == 1 ? rr4[i][1] : sidx == 2 ? rr4[i][2] : rr4[i][3];
#pragma unroll
      for (int e = 0; e < 8; e++) x[e] = hc_norm1(x[e], rr, gw[e]);
      *ptr = pack8(x);
    }
  };

  float z[2][NJ][4], acc[2][NJ][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) z[i][j][e] = 0.f, acc[i][j][e] = 0.f;
#pragma unroll
  for (int i = 0; i < kDbSt - 1; i++) {
    load(i, i);
    cp_commit();
  }
  cp_wait<kDbSt - 2>();
  __syncthreads();
  convert(0, 0);
  for (int c = 0; c < NST; c++) {
    cp_wait<kDbSt - 3>();  // my copies of stage c + 1 landed
    __syncthreads();       // stage c normalised (and stage c + 1 copied) by everyone; stage c - 1 free
    if (c + 1 < NST) convert(c + 1, (c + 1) % kDbSt);
    if (c + kDbSt - 1 < NST) load(c + kDbSt - 1, (c + kDbSt - 1) % kDbSt);
    cp_commit();
    const uint8_t* st = sm + (c % kDbSt) * STAGE;
    uint4 A[2][2];  // tile i, k16 step s
#pragma unroll
    for (int s = 0; s < 2; s++) {
      if (T0 == 20) {  // inject tile: lanes 0..15 hold rows 0..3 (a1 = a3 = 0)
        const uint2 v = lane < 16 ? *reinterpret_cast<const uint2*>(st + ((2 * wn) * 2 + s) * 512 + lane * 8)
                                  : make_uint2(0u, 0u);
        A[0][s] = make_uint4(v.x, 0u, v.y, 0u);
      } else {
        A[0][s] = *reinterpret_cast<const uint4*>(st + ((2 * wn) * 2 + s) * 512 + lane * 16);
      }
      A[1][s] = *reinterpret_cast<const uint4*>(st + ((2 * wn + 1) * 2 + s) * 512 + lane * 16);
    }
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      uint32_t b[4];  // steps 0, 1 of rows 8·NJ·wm + 8j + gq
      ldsm4(b, st + kDbW + (wm * 8 * NJ + j * 8 + (lane & 7)) * kDbXp + (lane >> 3) * 16);
      if (act0) mma(acc[0][j], A[0][0], b[0], b[1]);
      if (act1) mma(acc[1][j], A[1][0], b[0], b[1]);
      if (act0) mma(acc[0][j], A[0][1], b[2], b[3]);
      if (act1) mma(acc[1][j], A[1][1], b[2], b[3]);
    }
    if ((c & 7) == 7) {  // end of 256-col chunk kc = c/8: z += P_kc
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
          for (int e = 0; e < 4; e++) z[i][j][e] += acc[i][j][e], acc[i][j][e] = 0.f;
    }
  }
#pragma unroll
  for (int i = 0; i < 2; i++) {
    if (!(i ? act1 : act0)) continue;
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int n = (T0 + i) * 16 + gq + 8 * h;
        if (n >= N) continue;
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int m = m0 + wm * 8 * NJ + j * 8 + 2 * tq + e;
          if (m >= M) continue;
          const float zb = bfr(z[i][j][2 * h + e]);
          if (n < kHcRank) {
            const float x = zb * 0.25f;
            lora[size_t(m) * kHcRank + n] = __float2bfloat16_rn(x * sigm(x));
          } else if (inj) {
            inj[m * kStreams + n - kHcRank] = __float2bfloat16_rn(zb);
          }
        }
      }
  }
}

// ---- K3 up + gate mix (large M) ----------------------------------------------------------------------------------
// As K3: warp = 16-row tile j of Wup (rows 4s + hh ↔ stream s, hidden 4j + hh), its 20 A fragments in registers; G =
// chain over the 20 k16 steps from 0, then u = bf16(¼ Σ_s σ(bf16(G_s))·xn_s) (fma chain s = 0..3). CTA = 8 tiles (32
// hidden) × chunks of 32 rows; lora rows, the X slice [32 rows][4 streams][32 hidden] (row pitch 272 B) and rrms of
// the rows through shared memory (cp.async, 3 stages). Each lane normalises its xn after the chunk's MMAs are issued;
// the gate-mix epilogue of chunk c−1 is interleaved with the MMAs of chunk c (in-order issue). Launch: 3 row ranges
// per hidden group (240 CTAs = 5 waves of one CTA per SM).
constexpr int kUbW = 8, kUbRows = 32, kUbGroups = kUpTiles / kUbW, kUbSt = 3, kUbXp = kStreams * kUbW * 4 * 2 + 16;
constexpr int kUbLs = kUbRows * kLsW * 4, kUbXs = kUbRows * kUbXp, kUbStage = kUbLs + kUbXs + kUbRows * 16;
constexpr int kUbSmem = kUbSt * kUbStage;

__global__ void __launch_bounds__(kUbW * 32, 1) k_up(const uint4* __restrict__ Wu, const bf16* lora, const bf16* X,
                                                   const float* rrms, const float* g, bf16* u, int M, int nch) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3, e = gq >> 2, tid = threadIdx.x;
  const int hg = blockIdx.x % kUbGroups, c0 = (blockIdx.x / kUbGroups) * nch;
  const int j = hg * kUbW + w, hl = 4 * w + (gq & 3), h = hg * kUbW * 4 + hl;
  const int n = min(nch, cdiv(M, kUbRows) - c0);
  constexpr int NT = kUbRows / 8;

  auto load = [&](int c, int buf) {
    uint8_t* st = sm + buf * kUbStage;
    const int m0 = (c0 + c) * kUbRows;
    for (int p = tid; p < kUbRows * 40; p += kUbW * 32) {  // lora: 40 pieces per row
      const int r = p / 40, v = p - r * 40, m = m0 + r;
      cpz(st + r * kLsW * 4 + v * 16, lora + size_t(min(m, M - 1)) * kHcRank + 8 * v, m < M);
    }
    for (int p = tid; p < kUbRows * kStreams * 4; p += kUbW * 32) {  // X slice: 4 pieces (32 hidden) per (row, s)
      const int r = p >> 4, s = (p >> 2) & 3, q = p & 3, m = m0 + r;
      cpz(st + kUbLs + r * kUbXp + (p & 15) * 16, X + size_t(min(m, M - 1)) * D + s * H + hg * kUbW * 4 + 8 * q, m < M);
    }
    if (tid < kUbRows) cpz(st + kUbLs + kUbXs + tid * 16, rrms + size_t(min(m0 + tid, M - 1)) * kStreams, m0 + tid < M);
  };

  uint4 a[kUpKs];
#pragma unroll
  for (int ks = 0; ks < kUpKs; ks++) a[ks] = __ldcg(Wu + (size_t(j) * kUpKs + ks) * 32 + lane);
  float gw[kStreams];
#pragma unroll
  for (int s = 0; s < kStreams; s++) gw[s] = g[s * H + h];
#pragma unroll
  for (int i = 0; i < kUbSt - 1; i++) {
    if (i < n) load(i, i);
    cp_commit();
  }
  // gate-mix epilogue of one n8 tile of chunk c (rows 8nt + 2tq + e)
  auto epi = [&](int c, int nt, const float (&cc)[4], const bf16 (&xv)[kStreams]) {
    const float pa = __shfl_xor_sync(~0u, e ? cc[0] : cc[1], 16);
    const float pb = __shfl_xor_sync(~0u, e ? cc[2] : cc[3], 16);
    const float G[4] = {e ? pa : cc[0], e ? cc[1] : pa, e ? pb : cc[2], e ? cc[3] : pb};
    const int m = (c0 + c) * kUbRows + nt * 8 + 2 * tq + e;
    if (m >= M) return;
    float acc = 0.f;
#pragma unroll
    for (int s = 0; s < kStreams; s++) acc = __fmaf_rn(sigm(bfr(G[s])), b2f(xv[s]), acc);
    u[size_t(m) * H + h] = __float2bfloat16_rn(acc * 0.25f);
  };
  // chunk c: MMAs into cc, this lane's xn (rows 8nt + 2tq + e, 4 streams) into xv; the epilogue of chunk pc (< 0:
  // none) is interleaved with the MMAs (in-order issue: it must not wait behind all of them)
  auto stage = [&](int c, float (&cc)[NT][4], bf16 (&xv)[NT][kStreams], int pc, const float (&pcc)[NT][4],
                   const bf16 (&pxv)[NT][kStreams]) {
    cp_wait<kUbSt - 2>();
    __syncthreads();
    if (c + kUbSt - 1 < n) load(c + kUbSt - 1, (c + kUbSt - 1) % kUbSt);
    cp_commit();
    const uint8_t* st = sm + (c % kUbSt) * kUbStage;
    const uint32_t* ls = reinterpret_cast<const uint32_t*>(st);
#pragma unroll
    for (int nt = 0; nt < NT; nt++) cc[nt][0] = cc[nt][1] = cc[nt][2] = cc[nt][3] = 0.f;
#pragma unroll
    for (int kp = 0; kp < kUpKs / 2; kp++) {
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        uint32_t b[4];  // steps 2kp, 2kp+1 of rows nt·8 + gq
        ldsm4(b, ls + (nt * 8 + (lane & 7)) * kLsW + kp * 16 + (lane >> 3) * 4);
        mma(cc[nt], a[2 * kp], b[0], b[1]);
        mma(cc[nt], a[2 * kp + 1], b[2], b[3]);
      }
      if ((kp & 1) && kp / 2 < NT && pc >= 0) epi(pc, kp / 2, pcc[kp / 2], pxv[kp / 2]);
    }
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {  // xn of this lane's row, hidden h, 4 streams (bf16-exact)
      const int r = nt * 8 + 2 * tq + e;
      const bf16* xr = reinterpret_cast<const bf16*>(st + kUbLs + r * kUbXp) + hl;
      const float4 rr = *reinterpret_cast<const float4*>(st + kUbLs + kUbXs + r * 16);
      const float rv[4] = {rr.x, rr.y, rr.z, rr.w};
#pragma unroll
      for (int s = 0; s < kStreams; s++) xv[nt][s] = __float2bfloat16_rn(hc_norm1(b2f(xr[s * kUbW * 4]), rv[s], gw[s]));
    }
  };
  float cc0[NT][4], cc1[NT][4];
  bf16 xv0[NT][kStreams], xv1[NT][kStreams];
  for (int c = 0; c < n; c++) {
    if (c & 1) stage(c, cc1, xv1, c - 1, cc0, xv0);
    else stage(c, cc0, xv0, c - 1, cc1, xv1);
  }
  if (n > 0) {
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      if ((n - 1) & 1) epi(n - 1, nt, cc1[nt], xv1[nt]);
      else epi(n - 1, nt, cc0[nt], xv0[nt]);
    }
  }
}

// ---- natural-layout bf16 GEMM (PLE kv, large M): y[M, N] = x[M, K] · Wᵀ -----------------------------------------
// As k_gemv per output: one MMA chain over all K/16 steps from 0 (pack_tiles fragments, weights on the M side).
// CTA = 128 weight rows (8 tiles) × 128 rows, warp = 32 weight rows × 64 rows; BK = 32 per stage, 4 stages.
constexpr int kGbK = 32, kGbSt = 4, kGbXp = kGbK * 2 + 16;  // x row pitch 80 B: 8 rows hit 8 distinct 16-B bank groups
constexpr int kGbW = 8 * 2 * 512, kGbX = 128 * kGbXp, kGbStage = kGbW + kGbX, kGbPitch = 128 + 8;
constexpr int kGbSmem = std::max(kGbSt * kGbStage, 128 * kGbPitch * 2);

__global__ void __launch_bounds__(256, 1) k_gemm(const uint4* __restrict__ W, const bf16* x, bf16* y, int M, int N,
                                                 int K, int MT, int NTl) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp & 3, wm = warp >> 2;
  constexpr int GM = 8;  // grouped raster as the linear prefill kernel
  const int per = GM * NTl, grp = blockIdx.x / per, fm = grp * GM, gm = min(MT - fm, GM);
  const int rem = blockIdx.x - grp * per, mt = fm + rem % gm, nt = rem / gm;
  const int KS = K / 16, T0 = nt * 8, m0 = mt * 128, nchk = K / kGbK, tiles = N / 16;

  auto load = [&](int c, int buf) {
    uint8_t* st = sm + buf * kGbStage;
    const int ks = c * 2;
    for (int p = tid; p < kGbW / 16; p += 256) {  // tile p/64: 2 k16 blocks = 1024 contiguous bytes
      const int T = T0 + p / 64;
      const bool ok = T < tiles;
      cpz(st + p * 16, reinterpret_cast<const uint8_t*>(W) + (size_t(ok ? T : 0) * KS + ks) * 512 + (p % 64) * 16, ok);
    }
    for (int p = tid; p < 128 * 4; p += 256) {
      const int r = p >> 2, q = p & 3, m = m0 + r;
      cpz(st + kGbW + r * kGbXp + q * 16, x + size_t(min(m, M - 1)) * K + c * kGbK + q * 8, m < M);
    }
  };
  float acc[2][8][4] = {};
#pragma unroll
  for (int i = 0; i < kGbSt - 1; i++) {
    if (i < nchk) load(i, i);
    cp_commit();
  }
  for (int c = 0; c < nchk; c++) {
    cp_wait<kGbSt - 2>();
    __syncthreads();
    if (c + kGbSt - 1 < nchk) load(c + kGbSt - 1, (c + kGbSt - 1) % kGbSt);
    cp_commit();
    const uint8_t* st = sm + (c % kGbSt) * kGbStage;
    uint4 A[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int s = 0; s < 2; s++) A[i][s] = *reinterpret_cast<const uint4*>(st + ((wn * 2 + i) * 2 + s) * 512 + lane * 16);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      uint32_t b[4];  // steps 0, 1 of rows wm·64 + 8j + g
      ldsm4(b, st + kGbW + (wm * 64 + j * 8 + (lane & 7)) * kGbXp + (lane >> 3) * 16);
#pragma unroll
      for (int i = 0; i < 2; i++) mma(acc[i][j], A[i][0], b[0], b[1]);
#pragma unroll
      for (int i = 0; i < 2; i++) mma(acc[i][j], A[i][1], b[2], b[3]);
    }
  }
  cp_wait<0>();
  __syncthreads();
  bf16* Ps = reinterpret_cast<bf16*>(sm);
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int wr = wn * 32 + i * 16 + g, ar = wm * 64 + j * 8 + 2 * t;
      Ps[ar * kGbPitch + wr] = __float2bfloat16_rn(acc[i][j][0]);
      Ps[(ar + 1) * kGbPitch + wr] = __float2bfloat16_rn(acc[i][j][1]);
      Ps[ar * kGbPitch + wr + 8] = __float2bfloat16_rn(acc[i][j][2]);
      Ps[(ar + 1) * kGbPitch + wr + 8] = __float2bfloat16_rn(acc[i][j][3]);
    }
  __syncthreads();
  for (int p = tid; p < 128 * 16; p += 256) {
    const int r = p >> 4, q = p & 15, m = m0 + r, n = nt * 128 + q * 8;
    if (m < M && n < N)
      *reinterpret_cast<uint4*>(y + size_t(m) * N + n) = *reinterpret_cast<const uint4*>(Ps + r * kGbPitch + q * 8);
  }
}

// ---- PLE dilated conv, row-parallel (rows [r0, r1) of this call) -------------------------------------------------
// Thread = (row j, channel c), same arithmetic as k_ple_conv for row j: window start j0 = max(seq_start[b], r0) (the
// state holds the 9 conv inputs before j0), taps c[t-9], c[t-6], c[t-3], c[t]. cin/gated are chunk-local (row - r0).
__device__ __forceinline__ int seq_of(const int* seq_start, int B, int j) {  // largest b with seq_start[b] ≤ j
  int lo = 0, hi = B - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (seq_start[mid] <= j) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

__global__ void __launch_bounds__(256) k_ple_conv_rows(bf16* X, const bf16* gated, const bf16* cin, const bf16* cw,
                                                       bf16* const* state, const int* seq_start, int B, int r0) {
  constexpr int SL = 9;
  const int j = r0 + blockIdx.x, c = blockIdx.y * 256 + threadIdx.x;
  const int b = seq_of(seq_start, B, j), j0 = max(seq_start[b], r0), jj = j - j0;
  const bf16* st = state[b];
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int hh = jj + 3 * k;
    const bf16 v = hh < SL ? st[size_t(hh) * D + c] : cin[size_t(j0 + hh - SL - r0) * D + c];
    acc = __fmaf_rn(b2f(cw[c * 4 + k]), b2f(v), acc);
  }
  const float cv = bfr(acc), yv = bfr(cv * sigm(cv));
  const size_t o = size_t(j - r0) * D + c;
  const float p = bfr(b2f(gated[o]) + yv);
  X[size_t(j) * D + c] = __float2bfloat16_rn(b2f(X[size_t(j) * D + c]) + p);
}

// state ← last 9 conv inputs of each sequence after rows [r0, r1) (unchanged if it has no row there)
__global__ void __launch_bounds__(256) k_ple_state(const bf16* cin, bf16* const* state, const int* seq_start, int r0,
                                                   int r1) {
  constexpr int SL = 9;
  const int b = blockIdx.x, c = blockIdx.y * 256 + threadIdx.x;
  const int j0 = max(seq_start[b], r0), T = min(seq_start[b + 1], r1) - j0;
  if (T <= 0) return;
  bf16* st = state[b];
  bf16 nw[SL];
#pragma unroll
  for (int i = 0; i < SL; i++) {
    const int hh = T + i;
    nw[i] = hh < SL ? st[size_t(hh) * D + c] : cin[size_t(j0 + hh - SL - r0) * D + c];
  }
#pragma unroll
  for (int i = 0; i < SL; i++) st[size_t(i) * D + c] = nw[i];
}

}  // namespace pre
