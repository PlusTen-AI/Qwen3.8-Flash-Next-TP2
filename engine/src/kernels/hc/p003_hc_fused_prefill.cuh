// p003_hc_fused_prefill.cuh — P003 hc_fused_prefill (patches.def): the large-M HC module in two kernels instead of
// three. Included by hc.cu inside its anonymous namespace after hc_prefill.cuh (uses pre::cpz / cp_commit / cp_wait /
// ldsm4 / kUb* and the hc.cu helpers). hc_combine_mix runs it when patch::on(hc_fused_prefill) and M ≥ kHcBigRows;
// otherwise the base path (K1 combine → pre::k_down → pre::k_up) runs unchanged.
//
// Why (GB10, M = 8192, per module): K1 is DRAM-bound (reads X and bo, writes X': 1.58 ms); pre::k_down reads X' back
// from DRAM and is bound by its per-stage barrier — convert, loads and MMAs of all warps run in lock step, so the tensor
// pipe idles while the warps convert (1.35 ms; its MMAs alone take 0.70 ms); pre::k_up likewise (1.42 ms).
//
// K12 = K1 + pre::k_down in one persistent kernel. A cluster of 3 CTAs owns a 64-row tile (CTA rank g = weight group of
// pre::k_down: tiles 8g..8g+7 of the 324 output rows) and walks its tiles cid, cid + NC, … Warp roles per CTA, handing
// 32-column stages around a ring of kSt shared-memory slots (weights, X' rows, g) with mbarriers:
//   loader (2 warps)    — cp.async of a stage's weights, X' rows and g into a free slot (`empty`), completion signalled
//                         by cp.async.mbarrier.arrive (`lded`): it never waits for data, so up to kSt stages are in flight;
//   converter (2 warps) — xn = bf16(y + y·g), y = X'·rrms (hc_norm1), of the slot's X' pieces in place → `full`;
//   MMA (8 warps)       — pre::k_down's warp tile (weight tiles 8g + 2wn, +1 × rows 32wm..+31) with its MMA chains,
//                         split fold (z += P_kc every 8 stages) and epilogue; they only wait for `full` and free the
//                         slot, so their MMAs overlap the loads and conversions;
//   producer (4 warps)  — K1 for this CTA's third of the tile rows, stream by stream (gs = 4·tile + s): per (row,
//                         256-col block) a warp holds 512 B of X and bo (register prefetch, kPd blocks ahead),
//                         X' = bf16(X + bo·w_s) written in place (each piece read and written by one lane), the 8-column
//                         fma chains, K1's xor tree over the 32 lanes and the blocks 0..9 in order → rrms into the
//                         3 CTAs' shared memory (DSMEM) and to global (for K3).
// Stream handoff (mbarriers of the cluster): producers arrive on `rdy` (gs & 1) of all 3 CTAs when their rows of stream
// gs are written; loaders wait for it before the first X' load of the stream, converters before taking the rows' rrms,
// after which they arrive on `fre` of all 3 CTAs; producers wait for `fre` of stream gs − 1 before stream gs (its rrms
// slot is free again and they stay at most one stream ahead).
// L2 (24 MB): 16 clusters × 64 rows = 1024 rows in flight, so X' of ~1.5 streams (8 MB), bo of the tiles (5 MB) and the
// weights (6.6 MB) can stay in L2 while X streams through with an evict-first hint and bo is read with evict-last until
// its 4th read: the loaders' X' read and the 4 bo reads per row hit L2 (128-row tiles, 2048 rows in flight: measured
// 1.7× slower per tile). DRAM traffic ≈ K1's alone. (cp.async with an L2 cache hint is an illegal instruction on GB10;
// ld.global with one is fine.)
//
// K3 = pre::k_up's arithmetic with the same warp specialisation (k_up below): 8 MMA warps, 4 loader warps that also
// convert the X slice to xn.
// Bitwise: per element the same operations as K1 + K2 + K3 (only scheduled differently; bf16 pairs are rounded with one
// cvt.rn.bf16x2 = two cvt.rn.bf16), so u, inj, lora, X and rrms equal the base path for every row and every M.
#pragma once

namespace p3 {

using pre::cp_commit;
using pre::cp_wait;
using pre::cpz;
using pre::ldsm4;

constexpr int kBM = 64;                          // tile rows
constexpr int kNst = D / 32, kSps = H / 32;      // 32-column stages per tile (320) and per stream (80)
constexpr int kMW = 8, kLW = 2, kVW = 2, kPW = 4;  // MMA / loader / converter / producer warps
constexpr int kNTH = 32 * (kMW + kLW + kVW + kPW);
constexpr int kSt = 7;                           // ring slots
constexpr int kW = 8 * 2 * 512, kXp = 80;        // weight bytes per stage (8 tiles × 2 k16 steps), X' row pitch (B)
constexpr int kG = kW + kBM * kXp, kCst = kG + 128;  // slot: weights, X', g (32 fp32)
constexpr int kPd = 5;                           // producer units in flight per warp (registers; divides 10)
constexpr int kXpt = kBM * 4 / (32 * kVW);       // X' pieces (16 B) per loader / converter thread
constexpr int kRrs = kSt * kCst, kBar = kRrs + 2 * kBM * 4;
constexpr int kSmem = kBar + (3 * kSt + 4) * 8;
static_assert(kCst % 128 == 0 && kSmem <= 101376, "layout");
static_assert(kLW == kVW && kW / 16 == 8 * 32 * kLW, "loader: 8 weight pieces (one per tile) per thread");

#ifdef P3_PROF
__device__ unsigned long long g_prof[4][4];  // DEV: role (MMA, loader, converter, producer) × (total, wait a, wait b, wait c)
#define P3_DECL long long p3acc[4] = {0, 0, 0, 0}
#define P3_T0(v) const long long v = clock64()
#define P3_ACC(i, v) (p3acc[i] += clock64() - (v))
#define P3_FLUSH(r) \
  if (lane == 0)    \
    for (int i_ = 0; i_ < 4; i_++) atomicAdd(&g_prof[r][i_], (unsigned long long)p3acc[i_])
#else
#define P3_DECL
#define P3_T0(v)
#define P3_ACC(i, v)
#define P3_FLUSH(r)
#endif

__device__ __forceinline__ uint32_t sa(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {  // same smem offset in CTA `rank`
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ uint32_t cl_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cl_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;\n" ::: "memory");
}
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(sa(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {  // release, CTA scope
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mb_arrive_cp(uint64_t* b) {  // arrives once all prior cp.async of the thread landed
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void mb_arrive_cl(uint32_t rb) {  // barrier in a cluster CTA (mapa address), release
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n" ::"r"(rb) : "memory");
}
// acquire at CTA or cluster scope; the suspend-time hint lets a waiting warp sleep until the phase completes instead of
// spinning (each SM sub-partition runs warps of several roles)
template <bool CL>
__device__ __forceinline__ void mb_wait(const uint64_t* b, uint32_t parity) {
  uint32_t done;
  do {
    if (CL)
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2, %3;\n"
                   " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(sa(b)), "r"(parity), "n"(1000000) : "memory");
    else
      asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n"
                   " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(sa(b)), "r"(parity), "n"(1000000) : "memory");
  } while (!done);
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("barrier.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void st_cl(uint32_t ra, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;\n" ::"r"(ra), "f"(v) : "memory");
}
// L2 eviction priority (createpolicy; the cp.async form with a cache hint is an illegal instruction on GB10, the plain
// ld form is fine): X is read once (evict first), bo 4 times (evict last until its 4th read)
__device__ __forceinline__ uint64_t pol_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t pol_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint4 ldg_hint(const void* p, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ uint32_t bf2(float a, float b, float& ra, float& rb) {  // RN pair (a low) and its values
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  ra = __low2float(v), rb = __high2float(v);
  return *reinterpret_cast<const uint32_t*>(&v);
}

template <bool INJ, bool COMB>
__global__ void __launch_bounds__(kNTH, 1) k_fused(const uint4* __restrict__ Wd, const uint2* __restrict__ Wi,
                                                  bf16* X, const bf16* __restrict__ bo,
                                                  const unsigned short* __restrict__ injp, float* rrms,
                                                  const float* __restrict__ g, bf16* lora, bf16* inj, int M,
                                                  int ntiles) {
  constexpr int NJ = 4, N = INJ ? kHcDown : kHcRank;
  extern __shared__ __align__(128) uint8_t sm[];
  float* rrs = reinterpret_cast<float*>(sm + kRrs);         // [2][kBM] rrms of the tile rows, by stream parity
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + kBar);  // [kSt] slot converted (converter threads)
  uint64_t* empty = full + kSt;                             // [kSt] slot read (MMA threads)
  uint64_t* lded = empty + kSt;                             // [kSt] slot's copies landed (loader threads)
  uint64_t* rdy = lded + kSt;                               // [2] stream published (producer warps of the cluster)
  uint64_t* fre = rdy + 2;                                  // [2] stream's rrms taken (converter warps of the cluster)
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  P3_DECL;
  const int grp = int(cl_rank()), cid = blockIdx.x / 3, NC = gridDim.x / 3;
  const int NS = (ntiles - cid + NC - 1) / NC * kNst, GS = NS / kSps;  // stages and streams of the cluster's tiles
  if (tid == 0) {
    for (int i = 0; i < kSt; i++) mb_init(full + i, 32 * kVW), mb_init(empty + i, 32 * kMW), mb_init(lded + i, 32 * kLW);
    for (int i = 0; i < 2; i++) mb_init(rdy + i, 3 * kPW), mb_init(fre + i, 3 * kVW);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  cl_sync();  // every CTA's barriers are initialised before the first remote arrive

  if (warp < kMW) {  // ---------------- MMA: pre::k_down's warp tile ----------------
    const int gq = lane >> 2, tq = lane & 3, wn = warp & 3, wm = warp >> 2;
    const int T0 = grp * 8 + 2 * wn;                                    // this warp's weight tiles T0, T0 + 1
    const bool act0 = T0 < 20 || (INJ && T0 == 20), act1 = T0 + 1 < 20;  // tile 21+ does not exist
    float z[2][NJ][4], acc[2][NJ][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int j = 0; j < NJ; j++)
#pragma unroll
        for (int e = 0; e < 4; e++) z[i][j][e] = 0.f, acc[i][j][e] = 0.f;
    P3_T0(t0);
    for (int S = 0, c = 0, m0 = cid * kBM; S < NS; S++) {
      P3_T0(tw);
      mb_wait<false>(full + S % kSt, (S / kSt) & 1);
      P3_ACC(1, tw);
      const uint8_t* st = sm + (S % kSt) * kCst;
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
      uint32_t b[NJ][4];  // steps 0, 1 of rows 32wm + 8j + gq
#pragma unroll
      for (int j = 0; j < NJ; j++) ldsm4(b[j], st + kW + (wm * 32 + j * 8 + (lane & 7)) * kXp + (lane >> 3) * 16);
      mb_arrive(empty + S % kSt);  // the slot is in registers: the converters may refill it
#pragma unroll
      for (int j = 0; j < NJ; j++) {
        if (act0) mma(acc[0][j], A[0][0], b[j][0], b[j][1]);
        if (act1) mma(acc[1][j], A[1][0], b[j][0], b[j][1]);
        if (act0) mma(acc[0][j], A[0][1], b[j][2], b[j][3]);
        if (act1) mma(acc[1][j], A[1][1], b[j][2], b[j][3]);
      }
      if ((c & 7) == 7) {  // end of 256-col chunk kc = c/8: z += P_kc
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
          for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) z[i][j][e] += acc[i][j][e], acc[i][j][e] = 0.f;
      }
      if (c == kNst - 1) {  // tile done: epilogue as pre::k_down, z ← 0
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
                const int m = m0 + wm * 32 + j * 8 + 2 * tq + e;
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
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
          for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) z[i][j][e] = 0.f;
      }
      if (++c == kNst) c = 0, m0 += NC * kBM;
    }
    P3_ACC(0, t0);
    P3_FLUSH(0);
  } else if (warp < kMW + kLW) {  // ---------------- loader: weights, X' and g into ring slots ----------------
    const int lt = tid - 32 * kMW, q = lt & 3, r0 = lt >> 2;  // X' pieces: rows r0 + 16k, 16-B column q
    P3_T0(t0);
    for (int L = 0, lc = 0, lm = cid * kBM; L < NS; L++) {
      if (lc % kSps == 0) {  // first stage of stream gs: its X' is written
        P3_T0(tw);
        mb_wait<true>(rdy + ((L / kSps) & 1), ((L / kSps) >> 1) & 1);
        P3_ACC(1, tw);
      }
      if (L >= kSt) {  // the MMA warps are done with the slot's previous stage
        P3_T0(tw);
        mb_wait<false>(empty + L % kSt, (L / kSt - 1) & 1);
        P3_ACC(2, tw);
      }
      uint8_t* st = sm + (L % kSt) * kCst;
#pragma unroll
      for (int k = 0; k < 8; k++) {  // weight tile 8g + k: 2 k16 blocks = 1024 contiguous bytes, 16 B per thread
        const int T = grp * 8 + k;
        if (T < 20) cpz(st + k * 1024 + lt * 16, reinterpret_cast<const uint8_t*>(Wd) + (size_t(T) * kDnKs + 2 * lc) * 512 + lt * 16, true);
        else if (INJ && T == 20 && (lt & 31) < 8)  // compact inject: 128 B per k16 step
          cpz(st + k * 1024 + lt * 16, reinterpret_cast<const uint8_t*>(Wi) + (2 * lc + lt / 32) * 128 + (lt & 31) * 16, true);
      }
#pragma unroll
      for (int k = 0; k < kXpt; k++) {
        const int r = r0 + 16 * k, m = lm + r;
        cpz(st + kW + r * kXp + q * 16, X + size_t(min(m, M - 1)) * D + lc * 32 + q * 8, m < M);
      }
      if (lt < 8) cpz(st + kG + lt * 16, g + lc * 32 + lt * 4, true);
      mb_arrive_cp(lded + L % kSt);
      if (++lc == kNst) lc = 0, lm += NC * kBM;
    }
    P3_ACC(0, t0);
    P3_FLUSH(1);
  } else if (warp < kMW + kLW + kVW) {  // ---------------- converter: xn of the X' pieces in place ----------------
    const int ct = tid - 32 * (kMW + kLW), q = ct & 3, r0 = ct >> 2;  // the loader thread ct's pieces
    float rr[kXpt];
    P3_T0(t0);
    for (int V = 0, vc = 0, vm = cid * kBM; V < NS; V++) {
      if (vc % kSps == 0) {  // first stage of stream gs: take its rrms (published with `rdy`), free the rrms slot
        const int gs = V / kSps;
        P3_T0(tw);
        mb_wait<true>(rdy + (gs & 1), (gs >> 1) & 1);
        P3_ACC(1, tw);
#pragma unroll
        for (int k = 0; k < kXpt; k++) rr[k] = rrs[(gs & 1) * kBM + r0 + 16 * k];
        __syncwarp();
        if (lane == 0)
          for (int k = 0; k < 3; k++) mb_arrive_cl(mapa(sa(fre + (gs & 1)), k));
      }
      P3_T0(tw);
      mb_wait<false>(lded + V % kSt, (V / kSt) & 1);
      P3_ACC(3, tw);
      uint8_t* st = sm + (V % kSt) * kCst;
#pragma unroll
      for (int k = 0; k < kXpt; k++) {  // rows >= M stay zero
        const int r = r0 + 16 * k;
        if (vm + r >= M) continue;
        uint4* ptr = reinterpret_cast<uint4*>(st + kW + r * kXp + q * 16);
        float x[8], gw[8], d0, d1;
        unpack8(*ptr, x);
        load8f(reinterpret_cast<const float*>(st + kG) + q * 8, gw);
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {  // hc_norm1 of a pair
          const float y0 = __fmul_rn(x[2 * e], rr[k]), y1 = __fmul_rn(x[2 * e + 1], rr[k]);
          o[e] = bf2(__fmaf_rn(y0, gw[2 * e], y0), __fmaf_rn(y1, gw[2 * e + 1], y1), d0, d1);
        }
        *ptr = make_uint4(o[0], o[1], o[2], o[3]);
      }
      mb_arrive(full + V % kSt);
      if (++vc == kNst) vc = 0, vm += NC * kBM;
    }
    P3_ACC(0, t0);
    P3_FLUSH(2);
  } else {  // ---------------- producer: K1 for my rows, at most a stream ahead ----------------
    const int pw = warp - kMW - kLW - kVW;
    const int pr0 = grp * kBM / 3, prn = (grp + 1) * kBM / 3 - pr0;  // rows [pr0, pr0 + prn) of the tile
    const int nr = (prn - pw + kPW - 1) / kPW;                       // my rows pr0 + pw + kPW·i, i < nr
    const int NU = GS * nr * 10;  // units (stream, row, 256-col block) in order
    const uint64_t pol_x = pol_first(), pol_b = pol_last();
    int lg = 0, lr = 0, lj = 0;  // next unit to load
    uint4 bx[kPd], bb[kPd];      // units u + 0..kPd−1 in flight: X and bo of my lane (registers)
    auto uload = [&](int u, uint4& vx, uint4& vb) {  // unit u (= lg, lr, lj); rows ≥ M read as zero
      if (u < NU) {
        const int t = lg >> 2, s = lg & 3, m = (cid + t * NC) * kBM + pr0 + pw + kPW * lr;
        vx = vb = make_uint4(0u, 0u, 0u, 0u);
        if (m < M) {
          vx = ldg_hint(X + size_t(m) * D + s * H + lj * 256 + lane * 8, pol_x);
          if (COMB) vb = ldg_hint(bo + size_t(m) * H + lj * 256 + lane * 8, s == 3 ? pol_x : pol_b);
        }
        if (++lj == 10) {
          lj = 0;
          if (++lr == nr) lr = 0, lg++;
        }
      }
    };
    auto ldinj = [&](int gs) -> unsigned {  // raw inj[m, s] of my row `lane` in stream gs (lane < nr)
      const int m = (cid + (gs >> 2) * NC) * kBM + pr0 + pw + kPW * lane;
      return COMB && injp && gs < GS && lane < nr && m < M ? unsigned(__ldg(injp + size_t(m) * kStreams + (gs & 3))) : 0u;
    };
    unsigned inx = ldinj(0);  // one stream ahead
    P3_T0(t0);
#pragma unroll
    for (int k = 0; k < kPd; k++) uload(k, bx[k], bb[k]);
    for (int gs = 0, u = 0; gs < GS; gs++) {
      const int t = gs >> 2, s = gs & 3, m0 = (cid + t * NC) * kBM + pr0 + pw;
      const float wl = injp ? 2.f * sigm(__uint_as_float(inx << 16) * 0.25f) : 1.f;  // w_s of my row `lane` (K1)
      inx = ldinj(gs + 1);
      if (gs >= 1) {  // converters took stream gs − 1: rrms slot gs & 1 is free, and we stay a stream ahead
        P3_T0(tw);
        mb_wait<true>(fre + ((gs - 1) & 1), ((gs - 1) >> 1) & 1);
        P3_ACC(1, tw);
      }
      for (int r = 0; r < nr; r++, u += 10) {
        const int m = m0 + kPW * r;
        const float wc = __shfl_sync(~0u, wl, r);
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 10; j++) {  // unit u + j sits in buffer j % kPd (rows start at multiples of 10)
          float x[8];
          unpack8(bx[j % kPd], x);
          if (COMB) {  // X' = bf16(X + bo·w_s), w_s = 2σ(inj_s/4) (1 without inj) — K1
            float b[8];
            unpack8(bb[j % kPd], b);
            uint32_t o[4];
#pragma unroll
            for (int e = 0; e < 4; e++)
              o[e] = bf2(__fmaf_rn(b[2 * e], wc, x[2 * e]), __fmaf_rn(b[2 * e + 1], wc, x[2 * e + 1]), x[2 * e], x[2 * e + 1]);
            if (m < M) *reinterpret_cast<uint4*>(X + size_t(m) * D + s * H + j * 256 + lane * 8) = make_uint4(o[0], o[1], o[2], o[3]);
          }
          uload(u + j + kPd, bx[j % kPd], bb[j % kPd]);
          float a = 0.f;  // K1: 8-column fma chain, xor tree over the 32 groups of the block, blocks in order
#pragma unroll
          for (int e = 0; e < 8; e++) a = __fmaf_rn(x[e], x[e], a);
#pragma unroll
          for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(~0u, a, o);
          ss += a;  // 0 + red_0 = red_0
        }
        if (lane == 0) {
          const float rr = rsqrtf(ss / float(H) + kEps);
          const uint32_t dst = sa(rrs + (s & 1) * kBM + pr0 + pw + kPW * r);
#pragma unroll
          for (int k = 0; k < 3; k++) st_cl(mapa(dst, k), rr);
          if (m < M) rrms[m * kStreams + s] = rr;
        }
      }
      asm volatile("fence.acq_rel.cluster;\n" ::: "memory");  // my X' and rrms stores before the warp's arrive
      __syncwarp();
      if (lane == 0)
        for (int k = 0; k < 3; k++) mb_arrive_cl(mapa(sa(rdy + (s & 1)), k));
    }
    P3_ACC(0, t0);
    P3_FLUSH(3);
  }
  cl_sync();  // no CTA exits while a peer may still arrive on its barriers or write its rrs
}

// ---- K3: pre::k_up with its loads and the xn conversion in separate warps -------------------------------------------
// Per output the arithmetic of pre::k_up (the same A fragments, chain over the 20 k16 steps from 0, gate-mix epilogue
// with the fma chain over s = 0..3), but its 8 MMA warps no longer wait for each other at a per-chunk barrier: 4 loader
// warps fill a ring of kUSt chunk slots (lora rows, the X slice converted in place to xn = bf16(y + y·g) — k_up's
// hc_norm1 — and rrms) and hand them over by mbarrier (full / empty). Same CTA grid as pre::k_up.
constexpr int kUW = pre::kUbW, kUL = 4, kUTH = 32 * (kUW + kUL), kUSt = 3, kULA = 1;
constexpr int kUSmem = kUSt * pre::kUbStage + 2 * kUSt * 8;
static_assert(pre::kUbStage % 16 == 0 && kUSmem <= 101376, "k_up layout");

// σ(x) = 1/(1 + e^-x) with div.rn's fast path (MUFU.RCP + Newton, operands scaled by 2^-64 when b > 2^64; b = +inf → 0)
// instead of the `/` call: the same bits as sigm() — checked for all 65536 bf16 x (the gate mix takes σ(bf16(G))).
__device__ __forceinline__ float sigm_bf(float x) {
  const float b = 1.f + __expf(-x), sc = b > 0x1p64f ? 0x1p-64f : 1.f, bs = b * sc;
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(bs));
  const float e = fmaf(-bs, r, 1.f), r1 = fmaf(r, e, r), q = sc * r1, rem = fmaf(-bs, q, sc);
  return b == __int_as_float(0x7f800000) ? 0.f : fmaf(r1, rem, q);
}

__global__ void __launch_bounds__(kUTH, 1) k_up(const uint4* __restrict__ Wu, const bf16* lora, const bf16* X,
                                               const float* rrms, const float* __restrict__ g, bf16* u, int M,
                                               int nch) {
  using pre::kUbGroups, pre::kUbLs, pre::kUbRows, pre::kUbStage, pre::kUbXp, pre::kUbXs;
  extern __shared__ __align__(16) uint8_t sm[];
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + kUSt * kUbStage);  // [kUSt] chunk loaded and converted
  uint64_t* empty = full + kUSt;                                         // [kUSt] chunk read by the MMA warps
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int hg = blockIdx.x % kUbGroups, c0 = (blockIdx.x / kUbGroups) * nch;
  const int n = min(nch, cdiv(M, kUbRows) - c0);
  if (tid == 0)
    for (int i = 0; i < kUSt; i++) mb_init(full + i, 32 * kUL), mb_init(empty + i, 32 * kUW);
  __syncthreads();
  if (w < kUW) {  // ---------------- MMA + gate mix (pre::k_up's warp) ----------------
    constexpr int NT = kUbRows / 8;
    const int gq = lane >> 2, tq = lane & 3, e = gq >> 2;
    const int j = hg * kUW + w, hl = 4 * w + (gq & 3), h = hg * kUW * 4 + hl;
    uint4 a[kUpKs];
#pragma unroll
    for (int ks = 0; ks < kUpKs; ks++) a[ks] = __ldcg(Wu + (size_t(j) * kUpKs + ks) * 32 + lane);
    auto epi = [&](int c, int nt, const float (&cc)[4], const bf16 (&xv)[kStreams]) {  // as pre::k_up
      const float pa = __shfl_xor_sync(~0u, e ? cc[0] : cc[1], 16);
      const float pb = __shfl_xor_sync(~0u, e ? cc[2] : cc[3], 16);
      const float G[4] = {e ? pa : cc[0], e ? cc[1] : pa, e ? pb : cc[2], e ? cc[3] : pb};
      const int m = (c0 + c) * kUbRows + nt * 8 + 2 * tq + e;
      if (m >= M) return;
      float acc = 0.f;
#pragma unroll
      for (int s = 0; s < kStreams; s++) acc = __fmaf_rn(sigm_bf(bfr(G[s])), b2f(xv[s]), acc);
      u[size_t(m) * H + h] = __float2bfloat16_rn(acc * 0.25f);
    };
    // chunk c: MMAs into cc, this lane's xn (rows 8nt + 2tq + e, 4 streams) into xv; the epilogue of chunk pc (< 0:
    // none) is interleaved with the MMAs (in-order issue)
    auto stage = [&](int c, float (&cc)[NT][4], bf16 (&xv)[NT][kStreams], int pc, const float (&pcc)[NT][4],
                     const bf16 (&pxv)[NT][kStreams]) {
      mb_wait<false>(full + c % kUSt, (c / kUSt) & 1);
      const uint8_t* st = sm + (c % kUSt) * kUbStage;
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
      for (int nt = 0; nt < NT; nt++) {
        const bf16* xr = reinterpret_cast<const bf16*>(st + kUbLs + (nt * 8 + 2 * tq + e) * kUbXp) + hl;
#pragma unroll
        for (int s = 0; s < kStreams; s++) xv[nt][s] = xr[s * kUW * 4];
      }
      mb_arrive(empty + c % kUSt);
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
  } else {  // ---------------- loader: lora rows, X slice → xn, rrms ----------------
    const int vt = tid - 32 * kUW, xs = (vt >> 2) & 3, xq = vt & 3, xr0 = vt >> 4;  // X pieces: rows xr0 + 8k
    float gx[8];  // g of my 8 hidden columns in stream xs
    load8f(g + xs * H + hg * kUW * 4 + 8 * xq, gx);
    for (int L = 0; L < n + kULA; L++) {
      if (L < n) {
        if (L >= kUSt) mb_wait<false>(empty + L % kUSt, (L / kUSt - 1) & 1);
        uint8_t* st = sm + (L % kUSt) * kUbStage;
        const int m0 = (c0 + L) * kUbRows;
#pragma unroll
        for (int k = 0; k < kUbRows * 40 / (32 * kUL); k++) {  // lora: 40 pieces per row
          const int p = vt + 32 * kUL * k, r = p / 40, v = p - r * 40, m = m0 + r;
          cpz(st + r * kLsW * 4 + v * 16, lora + size_t(min(m, M - 1)) * kHcRank + 8 * v, m < M);
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {  // X slice: 4 pieces (32 hidden) per (row, stream)
          const int r = xr0 + 8 * k, m = m0 + r;
          cpz(st + kUbLs + r * kUbXp + (vt & 15) * 16, X + size_t(min(m, M - 1)) * D + xs * H + hg * kUW * 4 + 8 * xq,
              m < M);
        }
        if (vt < kUbRows) cpz(st + kUbLs + kUbXs + vt * 16, rrms + size_t(min(m0 + vt, M - 1)) * kStreams, m0 + vt < M);
      }
      cp_commit();
      if (L >= kULA) {  // chunk V = L − kULA landed (all loaders: rrms come from threads 0..31) → xn in place
        const int V = L - kULA;
        cp_wait<kULA>();
        bar_sync(1, 32 * kUL);
        uint8_t* st = sm + (V % kUSt) * kUbStage;
#pragma unroll
        for (int k = 0; k < 4; k++) {  // rows ≥ M: zero X and rrms → xn 0 (never stored)
          const int r = xr0 + 8 * k;
          const float rr = reinterpret_cast<const float*>(st + kUbLs + kUbXs + r * 16)[xs];
          uint4* ptr = reinterpret_cast<uint4*>(st + kUbLs + r * kUbXp + (vt & 15) * 16);
          float x[8], d0, d1;
          unpack8(*ptr, x);
          uint32_t o[4];
#pragma unroll
          for (int i = 0; i < 4; i++) {  // hc_norm1 of a pair
            const float y0 = __fmul_rn(x[2 * i], rr), y1 = __fmul_rn(x[2 * i + 1], rr);
            o[i] = bf2(__fmaf_rn(y0, gx[2 * i], y0), __fmaf_rn(y1, gx[2 * i + 1], y1), d0, d1);
          }
          *ptr = make_uint4(o[0], o[1], o[2], o[3]);
        }
        mb_arrive(full + V % kUSt);
      }
    }
  }
}

int g_maxc = 0;  // co-resident 3-CTA clusters (0: not set up yet)

// once, on the first patched call — not in hc_upload, so the base path never depends on the cluster placement
void init_once() {
  if (g_maxc) return;
  auto big = [](auto k) { CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem)); };
  big(k_fused<true, true>), big(k_fused<true, false>), big(k_fused<false, true>), big(k_fused<false, false>);
  CUDA_CHECK(cudaFuncSetAttribute(k_up, cudaFuncAttributeMaxDynamicSharedMemorySize, kUSmem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(3), cfg.blockDim = dim3(kNTH), cfg.dynamicSmemBytes = kSmem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 3, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  cfg.attrs = at, cfg.numAttrs = 1;
  CUDA_CHECK(cudaOccupancyMaxActiveClusters(&g_maxc, k_fused<true, true>, &cfg));
  if (g_maxc < 1) qlog::fail("hc", "P003: cannot launch a 3-CTA cluster");
}

template <bool INJ, bool COMB>
void launch_fused(const uint4* Wd, const uint2* Wi, bf16* X, const bf16* bo, const bf16* injp, const float* g,
                  bf16* inj_out, int M, cudaStream_t s) {
  init_once();
  const int ntiles = cdiv(M, kBM), nc = std::min(ntiles, g_maxc);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(3 * nc), cfg.blockDim = dim3(kNTH), cfg.dynamicSmemBytes = kSmem, cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 3, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  cfg.attrs = at, cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_fused<INJ, COMB>, Wd, Wi, X, bo, reinterpret_cast<const unsigned short*>(injp),
                                g_ws.rrms, g, g_ws.lora, inj_out, M, ntiles));
}

// combine + down (K12), then up + gate mix (K3) — same outputs as the base large-M branch of hc_combine_mix
void run(const HcW& w, const uint4* Wd, const uint2* Wi, const uint4* Wu, bf16* X, const bf16* bo,
         const bf16* inj_prev, bf16* u, bf16* inj_out, int M, cudaStream_t s) {
  const bf16* ip = bo ? inj_prev : nullptr;
  if (w.inject) {
    if (bo) launch_fused<true, true>(Wd, Wi, X, bo, ip, w.norm, inj_out, M, s);
    else launch_fused<true, false>(Wd, Wi, X, bo, ip, w.norm, inj_out, M, s);
  } else {
    if (bo) launch_fused<false, true>(Wd, Wi, X, bo, ip, w.norm, nullptr, M, s);
    else launch_fused<false, false>(Wd, Wi, X, bo, ip, w.norm, nullptr, M, s);
  }
  const int chunks = cdiv(M, pre::kUbRows), nch = cdiv(chunks, std::min(3, chunks));
  k_up<<<pre::kUbGroups * cdiv(chunks, nch), kUTH, kUSmem, s>>>(Wu, g_ws.lora, X, g_ws.rrms, w.norm, u, M, nch);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace p3
