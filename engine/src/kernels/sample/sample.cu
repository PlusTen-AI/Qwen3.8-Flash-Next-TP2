// sample.cu — token sampling and speculative acceptance (contract and numerics: sample.h).
//
// One thread-block cluster per row (sample_rows) or per sequence (spec_accept). The C CTAs of a cluster split the row
// into C contiguous float4 chunks, and each warp of a CTA owns a contiguous run of its chunk, read in rows of 32 float4
// (lane l: float4 l of every row — coalesced). C is scheduling only (up to the device's largest cluster, so that the
// rows of a call cover the SMs): every reduction is exact — max / argmax in the torch.argmax total order, integer weight
// sums — so no result depends on C, on the warp runs, or on the other rows.
// Per row:
//   max pass   argmax (8 float4 in flight per thread) → cluster exchange through distributed shared memory (ranks
//              combined in rank order; no workspace, no atomics);
//   sum pass   weight sums (the row is in L2 by now) → exchange: per-warp totals stay in shared memory, the ranks'
//              chunk sums become an inclusive prefix;
//   pick       the rank whose chunk holds t: a scan of its warp totals finds the warp, which reads its run again
//              (4 rows of 32 float4 per load round trip), finds the row from the row totals, then the lane by a scan.
// spec_accept walks a sequence's rows in order and stops at the first rejection, pipelined: the pass that sums the
// weights of target / draft row i (L2) also takes the maxima of the next rows (pair i+1, or the bonus row; DRAM), so
// one pass and one cluster exchange per draft keep the DRAM streaming while the weights are computed. The residual
// pass runs only for the rejected pair; the bonus row's maximum comes from the last pair's pass (unused on a rejection).
// Cluster exchange by push: each rank stores its partial into its own slot in every rank's shared memory, one cluster
// barrier, then every rank reads only its own shared memory (ranks in order). No remote access follows a barrier, so a
// CTA may leave right after its last exchange. Slots alternate by parity: a rank pushes exchange n after barrier n−1,
// which every rank passes only once it has read exchange n−2's slots. Before the first push every CTA must be running:
// each thread arrives (relaxed) on the cluster barrier at its start and waits right before the first push.
// Loop bodies have no branches (the last group of a walk runs on −inf padding, which has weight 0 and never wins an
// argmax against a real element), so the compiler interleaves the 8 float4 of a group: the weight pass is ~11
// instructions and one MUFU op per token.
// Truncated rows / sequences (top-k / top-p, T > 0) take their own path ("truncation" below); the others run as above.
// P019 sparse_logits (sample_cands, sample_rows_sparse, spec_accept_sparse): p019_sparse.cuh.
#include "kernels/sample/sample.h"

#include <cooperative_groups.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <utility>

#include "core/log.h"
#include "model/patches.h"

namespace q38 {
namespace cg = cooperative_groups;

namespace {

using u64 = unsigned long long;
using u128 = unsigned __int128;

// Threads per CTA: k_sample 1024 (32 warps per SM hide the L2 / DRAM latency of the walks; ≤ 64 registers),
// k_accept 512 (its 4-row pass and 128-bit residual need more registers). Shared arrays are sized for 32 warps.
constexpr int kSampleT = 1024, kAcceptT = 512, kMaxW = 32;
constexpr int kMaxC = 16;               // CTAs per cluster at most (non-portable size where the device runs it)
constexpr int kMinChunk4 = 1024;        // float4 per CTA at least when choosing C (small rows: fewer CTAs)
constexpr int kLoads = 8;               // float4 loads in flight per thread (over the rows of a pass)
constexpr int kNB = 512;                // truncation: histogram buckets = 2^9 (≤ the CTA's threads: find_bucket)
constexpr int kCap = 512;               // truncation: list entries at most (≤ the CTA's threads: one per thread)
constexpr u64 kZTop = 1ull << 52;       // above every rank key
int g_force = 0, g_cap = kCap, g_exact = 0;

// ---- random numbers: Philox4x32-10, key (lo, hi seed), counter (lo, hi ctr, j, tag) → out0 | out1 << 32 ----------
__device__ __forceinline__ u64 philox(u64 seed, u64 ctr, uint32_t j, uint32_t tag) {
  uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
  uint32_t c0 = uint32_t(ctr), c1 = uint32_t(ctr >> 32), c2 = j, c3 = tag;
#pragma unroll
  for (int r = 0; r < 10; r++) {
    const uint32_t h0 = __umulhi(0xD2511F53u, c0), l0 = 0xD2511F53u * c0;
    const uint32_t h1 = __umulhi(0xCD9E8D57u, c2), l1 = 0xCD9E8D57u * c2;
    c0 = h1 ^ c1 ^ k0, c1 = l1, c2 = h0 ^ c3 ^ k1, c3 = l0;
    k0 += 0x9E3779B9u, k1 += 0xBB67AE85u;
  }
  return u64(c1) << 32 | c0;
}

// ---- the distribution of a row (sample.h): weight of logit x given the row max m and c = log2(e) / T ----------------
// e = ex2.approx.ftz((x − m)·c), the MUFU op behind exp2f (a result below 2^-126 flushes to 0: its weight is 0 either
// way); a NaN (degenerate row) → 0. round(e·2^44) without the 64-bit conversion unit and without branches: e·2^22 = h + r
// with h = round(e·2^22) through the 2^23 magic add and r exact (|r| ≤ ½), and round(e·2^44) = h·2^22 + round(r·2^22)
// (magic 1.5·2^23 for the sign; ties to even, h·2^22 being even). Every product is by a power of two, so the FFMAs equal
// the separate rounding steps; explicit intrinsics keep every inlined copy (sum pass, pick, a draft's single logit)
// bit-identical. The parts come back as the magic floats' bits: h = bits − 0x4B000000 ∈ [0, 2^22], l = bits −
// 0x4B400000 ∈ [−2^21, 2^21] — a sum adds them in 32 bits (unsigned: the bias terms wrap) and folds h·2^22 + l per
// float4.
__device__ __forceinline__ float temp_scale(float t) { return __fdiv_rn(1.44269504088896341f, t); }
struct Fix {
  uint32_t h, l;  // bits of the magic floats
};
__device__ __forceinline__ Fix fix(float x, float m, float c) {
  float e;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(__fmul_rn(__fsub_rn(x, m), c)));
  e = fmaxf(e, 0.f);
  const float hf = __fmaf_rn(e, 0x1p22f, 0x1p23f), r = __fmaf_rn(e, 0x1p22f, -__fsub_rn(hf, 0x1p23f));
  return {__float_as_uint(hf), __float_as_uint(__fmaf_rn(r, 0x1p22f, 0x1.8p23f))};
}
__device__ __forceinline__ u64 fold(uint32_t h, uint32_t l) { return (u64(h) << 22) + u64(int64_t(int32_t(l))); }
__device__ __forceinline__ u64 weight(float x, float m, float c) {
  const Fix f = fix(x, m, c);
  return fold(f.h - 0x4B000000u, f.l - 0x4B400000u);
}
__device__ __forceinline__ u64 weight4(const float4& v, float m, float c) {
  const Fix a = fix(v.x, m, c), b = fix(v.y, m, c), d = fix(v.z, m, c), e = fix(v.w, m, c);
  return fold(a.h + b.h + d.h + e.h - 4u * 0x4B000000u, a.l + b.l + d.l + e.l - 4u * 0x4B400000u);
}

// ---- truncation keys (sample.h): Z = key(x)·2^20 + 2^20 − 1 − v, v above u iff Z_v > Z_u ----------------------------
__device__ __forceinline__ uint32_t okey(float x) {
  const uint32_t b = x == 0.f ? 0u : __float_as_uint(x);  // −0 as +0
  return int32_t(b) < 0 ? ~b : b | 0x80000000u;
}
__device__ __forceinline__ u64 zkey(float x, int v) { return u64(okey(x)) << 20 | uint32_t(0xFFFFF - v); }
__device__ __forceinline__ float kval(uint32_t k) { return __uint_as_float(int32_t(k) < 0 ? k & 0x7FFFFFFFu : ~k); }
__device__ __forceinline__ float zval(u64 z) { return kval(uint32_t(z >> 20)); }  // the logit back from its key (a zero
                                                                                     // as +0: the same weight)
// x if token v is kept by the cut zc (Z_v ≥ zc), else −inf: weight 0
__device__ __forceinline__ float kept(float x, int v, u64 zc) { return zkey(x, v) >= zc ? x : -INFINITY; }
__device__ __forceinline__ float4 kept4(const float4& v, int g, u64 zc) {
  return make_float4(kept(v.x, 4 * g, zc), kept(v.y, 4 * g + 1, zc), kept(v.z, 4 * g + 2, zc), kept(v.w, 4 * g + 3, zc));
}
// Level-0 histogram bucket: linear in the logit, 1/8 wide over [−32, 32), the ends clamped (−inf → 0). Bucket b holds
// exactly the keys Z ∈ [zl0(b), zl0(b + 1)): b(x) ≥ b ⟺ ⌊8x⌋ ≥ b − kNB/2 ⟺ x ≥ (b − kNB/2)/8, an exact fp32.
__device__ __forceinline__ int lvl0(float x) {
  const int t = __float2int_rd(x * 8.f);  // saturating: ±inf → INT_MIN / INT_MAX
  return t < -kNB / 2 ? 0 : t >= kNB / 2 ? kNB - 1 : t + kNB / 2;
}
__device__ __forceinline__ float xl0(int b) { return float(b - kNB / 2) * 0.125f; }  // bucket b's lowest logit, b ≥ 1
__device__ __forceinline__ u64 zl0(int b) { return b <= 0 ? 0 : b >= kNB ? kZTop : u64(okey(xl0(b))) << 20; }
// ⌈p·W⌉ exactly for 0 < p < 1: p = mant·2^-s (s ≥ 24), mant·W < 2^88.
__device__ __forceinline__ u64 pthr(float p, u64 W) {
  const uint32_t b = __float_as_uint(p), e = b >> 23;
  const u128 x = u128(e ? (b & 0x7FFFFFu) | 0x800000u : b) * W;
  const int s = e ? 150 - int(e) : 149;
  return s >= 128 ? u64(x != 0) : u64((x + ((u128(1) << s) - 1)) >> s);
}
// Normalised truncation parameters: kk ∈ [1, V) or 0 (no top-k), pp < 1 or 1 (no top-p; NaN too); p ≤ 0 keeps the top
// token: top-1. True if the row is truncated.
__device__ __forceinline__ bool trunc_params(int tk, float tp, int V, int& kk, float& pp) {
  kk = tk > 0 && tk < V ? tk : 0;
  pp = tp < 1.f ? tp : 1.f;
  if (pp <= 0.f) kk = 1, pp = 1.f;
  return kk > 0 || pp < 1.f;
}

// min_p's weight threshold (sample.h): ⌈mp·2^44⌉ for 0 < mp < 1, 2^44 (the top token's weight) for mp ≥ 1, 0 = off
// (mp ≤ 0 or NaN).
__device__ __forceinline__ u64 mp_thr(float mp) {
  return mp > 0.f ? (mp < 1.f ? pthr(mp, 1ull << 44) : 1ull << 44) : 0ull;
}

// ⌊u·x / 2^64⌋ for x < 2^127 (exact: u·hi(x) is an integer, the low limb contributes mulhi(u, lo(x)))
__device__ __forceinline__ u128 scale(u64 u, u128 x) {
  return u128(u) * u64(x >> 64) + __umul64hi(u, u64(x));
}

// ---- argmax in torch.argmax order: a NaN is the maximum, equal values → the lower index -------------------------
struct Arg {
  float v;
  int i;
};
__device__ __forceinline__ bool beats(float v, int i, const Arg& b) {
  if (v != v) return b.v == b.v || i < b.i;
  return b.v == b.v && (v > b.v || (v == b.v && i < b.i));
}
__device__ __forceinline__ void upd(Arg& a, float v, int i) {
  if (beats(v, i, a)) a = {v, i};
}
// One thread's scan in increasing index (start a = {−inf, −1}): the first element, then a larger value or the first
// NaN — the lower-index rule holds by the scan order. Branch-free.
__device__ __forceinline__ void upd_seq(Arg& a, float v, int i) {
  const bool take = (a.i < 0) | ((a.v == a.v) & !(v <= a.v));
  a.v = take ? v : a.v;
  a.i = take ? i : a.i;
}
__device__ __forceinline__ void upd_seq4(Arg& a, const float4& v, int g) {
  upd_seq(a, v.x, 4 * g), upd_seq(a, v.y, 4 * g + 1), upd_seq(a, v.z, 4 * g + 2), upd_seq(a, v.w, 4 * g + 3);
}

__device__ __forceinline__ u64 shfl_xor(u64 v, int o) { return __shfl_xor_sync(~0u, v, o); }
__device__ __forceinline__ u128 shfl_xor(u128 v, int o) { return u128(shfl_xor(u64(v >> 64), o)) << 64 | shfl_xor(u64(v), o); }
__device__ __forceinline__ u64 shfl_up(u64 v, int o) { return __shfl_up_sync(~0u, v, o); }
__device__ __forceinline__ u128 shfl_up(u128 v, int o) { return u128(shfl_up(u64(v >> 64), o)) << 64 | shfl_up(u64(v), o); }

struct Smem {
  Arg wa[kMaxW][2];          // per-warp partials of a block reduction (≤ 2 argmax and ≤ 2 sum rows per exchange)
  u128 ws[kMaxW][2];         // per-warp sums of the last exchange (the pick finds its warp from these)
  Arg xa[2][kMaxC][2];       // cluster exchange slots [parity][source rank][row], written by the ranks
  u128 xs[2][kMaxC][2];
  Arg ra[2];                 // cluster results, broadcast to the CTA
  u128 incl[3][kMaxC];       // inclusive prefix of the ranks' chunk sums: [0] p / the row, [1] q, [2] residual
  u128 pt;                   // pick: t relative to the owner warp's run (warp pw)
  u128 pickw;                // the pick's token and its weight
  int pw, pick;
  // truncation
  u64 z[kCap];               // the list: gathered rank keys (every rank of the cluster writes here), any order
  union {
    struct {
      uint32_t c[kNB];       // histogram counts / weights: this CTA's, then (hist_sum) the cluster's
      u64 w[kNB];
    } h;
    u64 P[kCap];             // the list's weights
  };
  u64 sc[kMaxW][2];          // block scan
  u64 fca, fwa, fcnt;        // find_bucket: count / weight above the bucket, its count
  u64 lz, lw, cz, cw;        // list_cut: the k-th entry's key / prefix, the cut's
  u64 zm;                    // mp_cut's result
  int fb;                    // find_bucket's bucket
  uint32_t nl, gc[kMaxC];    // gather: this rank's tokens, every rank's (pushed by the ranks)
};

// This CTA's place in the row cluster: nw warps; this thread's warp owns float4 [w0, w1) of the row; started: the
// start-up barrier phase has been waited for.
struct Cl {
  Smem& sm;
  int C, c, nw, w0, w1, ph;
  bool started;
};

// Every thread at kernel start (arrives on the cluster barrier: this CTA runs).
__device__ __forceinline__ Cl make_cl(Smem& sm, int V) {
  const cg::cluster_group cl = cg::this_cluster();
  const int C = int(cl.num_blocks()), c = int(cl.block_rank()), n4 = V / 4, nw = int(blockDim.x) / 32;
  const int lo4 = c * n4 / C, hi4 = (c + 1) * n4 / C, L = cdiv(hi4 - lo4, nw);
  const int w0 = min(lo4 + int(threadIdx.x >> 5) * L, hi4);
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  return Cl{sm, C, c, nw, w0, min(w0 + L, hi4), 0, false};
}

// float4 g of a row, or −inf × 4 past the end (weight 0, never an argmax over a real element: its index is larger)
__device__ __forceinline__ float4 ld4(const float* x, int g, int end) {
  return g < end ? __ldg(reinterpret_cast<const float4*>(x) + g) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
}

// This warp's run of NR rows: lane l takes float4 w0 + 32·j + l, kLoads / NR per row in flight. f(g, float4[NR]) per
// float4 in increasing g per thread; the last group runs whole on −inf padding.
template <int NR, class F>
__device__ __forceinline__ void walk(const float* const (&x)[NR], const Cl& k, F f) {
  constexpr int U = kLoads / NR;
  for (int g0 = k.w0 + int(threadIdx.x & 31); g0 < k.w1; g0 += U * 32) {
    float4 v[U][NR];
#pragma unroll
    for (int u = 0; u < U; u++)
#pragma unroll
      for (int r = 0; r < NR; r++) v[u][r] = ld4(x[r], g0 + u * 32, k.w1);
#pragma unroll
    for (int u = 0; u < U; u++) f(g0 + u * 32, v[u]);
  }
}

// One cluster exchange: NA argmax partials (upd_seq form) → the rows' argmax in a (every thread), and NS sum partials
// → sm.incl[at + r][0..C) = inclusive prefix of the ranks' chunk sums (exact integers); sm.ws keeps the warp sums.
// Lane trees within a warp, then over the warps (warp 0), pushed by lanes 0…C−1 of warp 0 to every rank; after the
// barrier warp 0 (argmax) and warp 1 (sums) combine the ranks from this CTA's own slots.
template <int NA, int NS, class Wt>
__device__ void exchange(Arg* a, const Wt* s, int at, Cl& k) {
  const cg::cluster_group cl = cg::this_cluster();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < NA; r++) {
    if (a[r].i < 0) a[r] = {-INFINITY, INT_MAX};  // no element
#pragma unroll
    for (int o = 16; o; o >>= 1) upd(a[r], __shfl_xor_sync(~0u, a[r].v, o), __shfl_xor_sync(~0u, a[r].i, o));
    if (lane == 0) k.sm.wa[warp][r] = a[r];
  }
#pragma unroll
  for (int r = 0; r < NS; r++) {
    Wt v = s[r];
#pragma unroll
    for (int o = 16; o; o >>= 1) v += shfl_xor(v, o);
    if (lane == 0) k.sm.ws[warp][r] = v;
  }
  __syncthreads();
  if (!k.started) {  // every rank of the cluster runs: its shared memory may be written
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    k.started = true;
  }
  if (warp == 0) {
#pragma unroll
    for (int r = 0; r < NA; r++) {
      Arg b = lane < k.nw ? k.sm.wa[lane][r] : Arg{-INFINITY, INT_MAX};
#pragma unroll
      for (int o = 16; o; o >>= 1) upd(b, __shfl_xor_sync(~0u, b.v, o), __shfl_xor_sync(~0u, b.i, o));
      if (lane < k.C) *cl.map_shared_rank(&k.sm.xa[k.ph][k.c][r], lane) = b;
    }
#pragma unroll
    for (int r = 0; r < NS; r++) {
      u128 t = lane < k.nw ? k.sm.ws[lane][r] : u128(0);
#pragma unroll
      for (int o = 16; o; o >>= 1) t += shfl_xor(t, o);
      if (lane < k.C) *cl.map_shared_rank(&k.sm.xs[k.ph][k.c][r], lane) = t;
    }
  }
  cl.sync();
  if (warp == 0)
    for (int r = 0; r < NA; r++) {
      Arg b = lane < k.C ? k.sm.xa[k.ph][lane][r] : Arg{-INFINITY, INT_MAX};
#pragma unroll
      for (int o = 16; o; o >>= 1) upd(b, __shfl_xor_sync(~0u, b.v, o), __shfl_xor_sync(~0u, b.i, o));
      if (lane == 0) k.sm.ra[r] = b;
    }
  if (warp == 1)
    for (int r = 0; r < NS; r++) {
      u128 v = lane < k.C ? k.sm.xs[k.ph][lane][r] : u128(0);
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const u128 y = shfl_up(v, o);
        if (lane >= o) v += y;
      }
      if (lane < k.C) k.sm.incl[at + r][lane] = v;
    }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < NA; r++) a[r] = k.sm.ra[r];
  k.ph ^= 1;
}

// Row maxima / argmax of NR rows (every thread gets them).
template <int NR>
__device__ void row_max(const float* const (&x)[NR], Arg* a, Cl& k) {
#pragma unroll
  for (int r = 0; r < NR; r++) a[r] = {-INFINITY, -1};
  walk<NR>(x, k, [&](int g, const float4* v) {
#pragma unroll
    for (int r = 0; r < NR; r++) upd_seq4(a[r], v[r], g);
  });
  exchange<NR, 0, u64>(a, nullptr, 0, k);
}

// Weight sums of the first NS of NS + NA rows (maxima a, c = temp_scale(T)) → sm.incl[r][·], and in the same pass the
// argmax of the other NA rows → an.
template <int NS, int NA>
__device__ void row_sum(const float* const (&x)[NS + NA], const Arg* a, float c, Arg* an, Cl& k) {
  u64 s[NS] = {};
#pragma unroll
  for (int r = 0; r < NA; r++) an[r] = {-INFINITY, -1};
  walk<NS + NA>(x, k, [&](int g, const float4* v) {
#pragma unroll
    for (int r = 0; r < NS; r++) s[r] += weight4(v[r], a[r].v, c);
#pragma unroll
    for (int r = 0; r < NA; r++) upd_seq4(an[r], v[NS + r], g);
  });
  exchange<NA, NS, u64>(an, s, 0, k);
}

// Owner rank of t ∈ [0, incl[C−1]) and t made relative to its chunk.
__device__ __forceinline__ int owner(const u128* incl, u128& t) {
  int o = 0;
  while (t >= incl[o]) o++;
  if (o) t -= incl[o - 1];
  return o;
}

// The owner CTA's search for the index v of its chunk with Σ_{u<v} w_u ≤ t < Σ_{u≤v} w_u (t chunk-relative;
// sm.ws[·][col] = the warps' sums of these weights from the last exchange; wf(float4[NR], g, Wt[4]) the weights of
// float4 g). Warp 0 scans the warp sums (exact, lanes in order) → the owner warp and t within its run; that warp reads
// its run again in groups of U rows of 32 float4, sums each row (warp xor tree) → the row holding t → a warp scan →
// the lane → the element. → sm.pick / sm.pickw.
template <int NR, class Wt, class WF>
__device__ void warp_pick(const float* const (&x)[NR], Wt t, int col, WF wf, Cl& k) {
  constexpr int U = 4 / NR;  // rows of 32 float4 per load round trip (registers: the pick must not spill)
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp == 0) {
    const Wt v = lane < k.nw ? Wt(k.sm.ws[lane][col]) : Wt(0);
    Wt inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const Wt y = shfl_up(inc, o);
      if (lane >= o) inc += y;
    }
    const unsigned in = __ballot_sync(~0u, t < inc);  // t < total: some lane holds it
    const int wl = __ffs(in) - 1;
    if (lane == wl) k.sm.pw = wl, k.sm.pt = t - (inc - v);
  }
  __syncthreads();
  if (warp == k.sm.pw) {
    t = Wt(k.sm.pt);
    for (int b = k.w0; b < k.w1; b += U * 32) {
      float4 v[U][NR];
      Wt tot[U];  // row sums (every lane)
#pragma unroll
      for (int u = 0; u < U; u++)
#pragma unroll
        for (int r = 0; r < NR; r++) v[u][r] = ld4(x[r], b + 32 * u + lane, k.w1);
#pragma unroll
      for (int u = 0; u < U; u++) {
        Wt w[4];
        wf(v[u], b + 32 * u + lane, w);
        tot[u] = w[0] + w[1] + w[2] + w[3];
#pragma unroll
        for (int o = 16; o; o >>= 1) tot[u] += shfl_xor(tot[u], o);
      }
      int uf = -1;  // the row holding t (the same in every lane)
#pragma unroll
      for (int u = 0; u < U; u++) {
        if (uf < 0 && t < tot[u]) uf = u;
        if (uf < 0) t -= tot[u];
      }
      if (uf < 0) continue;
      float4 sv[NR];
#pragma unroll
      for (int u = 0; u < U; u++)
        if (u == uf)
#pragma unroll
          for (int r = 0; r < NR; r++) sv[r] = v[u][r];
      Wt w[4];
      wf(sv, b + 32 * uf + lane, w);
      const Wt s = w[0] + w[1] + w[2] + w[3];
      Wt inc = s;  // exact lane scan of the row
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const Wt y = shfl_up(inc, o);
        if (lane >= o) inc += y;
      }
      Wt acc = inc - s;
      if (t >= acc && t - acc < s) {
        int e = 0;
        while (t - acc >= w[e]) acc += w[e++];
        k.sm.pick = 4 * (b + 32 * uf + lane) + e, k.sm.pickw = w[e];
      }
      break;
    }
  }
  __syncthreads();
}

// Draw from a row's distribution (maximum a, chunk prefix incl; its sum pass was the last exchange, so sm.ws[·][0] holds
// its warp sums). W = 0 → argmax. CUT: only the tokens kept by the cut zc count (their sum pass was masked_sum).
template <bool CUT>
__device__ void pick_row(const float* row, const Arg& a, float c, const u128* incl, u64 U, int* tok, float* prob, Cl& k,
                         u64 zc = 0) {
  const u64 W = u64(incl[k.C - 1]);
  if (W == 0) {
    if (k.c == 0 && threadIdx.x == 0) {
      *tok = a.i;
      if (prob) *prob = 1.f;
    }
    return;
  }
  u128 r = __umul64hi(U, W);
  if (owner(incl, r) != k.c) return;
  const float* x[1] = {row};
  warp_pick<1, u64>(x, u64(r), 0, [&](const float4* v, int g, u64(&w)[4]) {
    const float4 y = CUT ? kept4(v[0], g, zc) : v[0];
    w[0] = weight(y.x, a.v, c), w[1] = weight(y.y, a.v, c), w[2] = weight(y.z, a.v, c), w[3] = weight(y.w, a.v, c);
  }, k);
  if (threadIdx.x == 0) {
    *tok = k.sm.pick;
    if (prob) *prob = float(double(u64(k.sm.pickw)) / double(W));
  }
}

// ---- truncation (sample.h) -------------------------------------------------------------------------------------------
// A cut z keeps {v : Z_v ≥ z}. Finding one = finding where a running total from the top — count (top-k) or weight
// (top-p) — first reaches a target. The tokens of a Z-range are gathered into every rank's shared memory (the list,
// ≤ cap ≤ one per thread) and the cut read there: each entry's rank and the weight above it are counted over the list
// (exact integers, no sort). Which range:
//   fast path  the thread maxima of the max pass (one per thread, distinct tokens): if ≥ k of them — or thread maxima
//              of weight ≥ ⌈p·W⌉ — lie at or above level-0 bucket b, so do that many tokens / that much weight, and the
//              top segment down to b holds the cut. Top-k takes the highest b over the ranks' own maxima (a CTA-local
//              histogram, one atomic per thread, combined in the max exchange), top-p the cluster's weighted
//              histogram of them (built with the sum pass). Only the threads whose maximum lies in the segment hold
//              tokens of it: they alone read their float4s again (spec_accept's walk also tracks each thread's second
//              largest — below the segment, the maximum is pushed alone; k_sample's walk stays the untruncated one).
//              No pass beyond the untruncated sampler's; on real rows the segment is a few tokens past the cut.
//   general    if the segment holds > cap tokens (flat or tied rows, large k, broad top-p), every token at or above
//              the fast bound is counted: level-0 histogram (lvl0: kNB buckets 1/8 logit wide), then histograms by
//              the top 9 bits of Z − lo narrow a Z-range holding the cut until its tokens fit the list. If the top
//              segment down to the level-0 bucket of the cut fits, it is gathered whole (the list path).
// Histograms are per CTA (shared integer atomics: exact, order-free), summed over the cluster by push-reduction into
// the bucket's owner rank and the owners' push of the totals (hist_sum; remote reductions and stores only, no remote
// load). On the list path the draw walks the list's kept entries in index order; otherwise a masked pass (tokens below
// the cut as −inf: weight 0) sums the kept weights per warp and pick_row<true> draws as for an untruncated row.
// Top-k with top-p: top-p's target ⌈p·W_K⌉ ≤ W_K is reached at or above the top-k cut — on the list it is read from
// the top-k entries, on the general path it is searched over the whole row, no restriction to K needed.
// The cut is a property of the row (Z is a total order, the sums exact): neither the path, cap, C nor the gather order
// changes it. Every rank takes the same decisions from the same totals and lists, so all meet the same barriers.

// A Z-range [lo, hi) that holds a cut; ca / wa: count / weight of the tokens above it (Z ≥ hi); cnt: tokens in it.
struct Sel {
  u64 lo, hi, ca, wa, cnt;
};
// A row's truncation: kept iff Z ≥ zc, W' = W (0: degenerate row). n > 0: the list path — the kept tokens are among
// the n gathered entries sm.z (weights sm.P); n = 0: masked_sum was the last exchange.
struct Cut {
  u64 zc, W;
  int n;
};
// This thread's tokens after the max pass: its maximum tm (its upd_seq partial before the exchange, index ti; −inf
// without tokens) and, if the walk tracked it, its second largest value s2 (+inf: not tracked). It has tokens ≥ some x
// only if tm ≥ x, and then tm alone if s2 < x.
struct Own {
  float tm, s2;
  int ti;
  __device__ Own(const Arg& a, float s2 = INFINITY) : tm(a.i >= 0 ? a.v : -INFINITY), s2(s2), ti(a.i) {}
};

// Exclusive prefix of two values over the CTA's threads in thread order (sm.sc).
__device__ void block_scan(u64& a, u64& b, Smem& sm) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = int(blockDim.x) >> 5;
  u64 ia = a, ib = b;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const u64 ya = shfl_up(ia, o), yb = shfl_up(ib, o);
    if (lane >= o) ia += ya, ib += yb;
  }
  if (lane == 31) sm.sc[warp][0] = ia, sm.sc[warp][1] = ib;
  __syncthreads();
  if (warp == 0) {
    u64 va = lane < nw ? sm.sc[lane][0] : 0, vb = lane < nw ? sm.sc[lane][1] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const u64 ya = shfl_up(va, o), yb = shfl_up(vb, o);
      if (lane >= o) va += ya, vb += yb;
    }
    if (lane < nw) sm.sc[lane][0] = va, sm.sc[lane][1] = vb;
  }
  __syncthreads();
  a = (warp ? sm.sc[warp - 1][0] : 0) + ia - a, b = (warp ? sm.sc[warp - 1][1] : 0) + ib - b;
  __syncthreads();
}

// Clears this CTA's histogram (after a barrier: the list weights sharing its memory may still be being read).
template <bool W>
__device__ void hist_zero(Smem& sm) {
  __syncthreads();
  for (int i = threadIdx.x; i < kNB; i += blockDim.x) {
    sm.h.c[i] = 0;
    if (W) sm.h.w[i] = 0;
  }
  __syncthreads();
}

// After a barrier that follows every rank's histogram: the cluster's totals in every rank. Bucket b belongs to rank
// b·C / kNB. Each rank reduces its nonzero foreign buckets into their owner (remote atomic add, no return) and clears
// them; barrier; each owner stores its nonzero totals into every other rank (whose copy is clear); barrier. Between
// the barriers an owned bucket is written only by reductions and read only by its owner.
template <bool W>
__device__ void hist_sum(Cl& k) {
  const cg::cluster_group cl = cg::this_cluster();
  if (k.C > 1) {
    for (int b = threadIdx.x; b < kNB; b += blockDim.x) {
      const int o = b * k.C / kNB;
      if (o == k.c) continue;
      const uint32_t n = k.sm.h.c[b];
      if (n) atomicAdd(cl.map_shared_rank(&k.sm.h.c[b], o), n), k.sm.h.c[b] = 0;
      if (W) {
        const u64 w = k.sm.h.w[b];
        if (w) atomicAdd(cl.map_shared_rank(&k.sm.h.w[b], o), w), k.sm.h.w[b] = 0;
      }
    }
    cl.sync();
    for (int b = (k.c * kNB + k.C - 1) / k.C + int(threadIdx.x); b < ((k.c + 1) * kNB + k.C - 1) / k.C;
         b += blockDim.x) {
      const uint32_t n = k.sm.h.c[b];
      const u64 w = W ? k.sm.h.w[b] : 0;
      for (int r = 0; r < k.C; r++) {
        if (r == k.c) continue;
        if (n) *cl.map_shared_rank(&k.sm.h.c[b], r) = n;
        if (W && w) *cl.map_shared_rank(&k.sm.h.w[b], r) = w;
      }
    }
  }
  cl.sync();
}

// The histogram scanned from the top bucket down: the bucket at which the running total — counts from s.ca, or weights
// from s.wa (W) — first reaches target; s.ca / s.wa += the buckets above, s.cnt = its count. −1 if the total stays
// below target (s unchanged). Thread t: bucket kNB − 1 − t (none for t ≥ kNB).
template <bool W>
__device__ int find_bucket(u64 target, Sel& s, Smem& sm) {
  const int top = kNB - 1 - int(threadIdx.x), per = top >= 0;
  if (threadIdx.x == 0) sm.fb = -1;
  u64 c = 0, w = 0;
  for (int i = 0; i < per; i++) {
    c += sm.h.c[top - i];
    if (W) w += sm.h.w[top - i];
  }
  block_scan(c, w, sm);
  c += s.ca, w += s.wa;
  for (int i = 0; i < per; i++) {
    const u64 nc = c + sm.h.c[top - i], nw = W ? w + sm.h.w[top - i] : w;
    if (W ? w < target && nw >= target : c < target && nc >= target)
      sm.fb = top - i, sm.fca = c, sm.fwa = w, sm.fcnt = nc - c;
    c = nc, w = nw;
  }
  __syncthreads();
  const int b = sm.fb;
  if (b >= 0) s.ca = sm.fca, s.wa = sm.fwa, s.cnt = sm.fcnt;
  return b;
}

// The max pass's walk (row_max<1>'s, before the exchange): this thread's partial argmax; S2: also its second largest
// value → s2 (upd_seq keeping the value that is not, or no longer, the maximum).
template <bool S2 = false>
__device__ __forceinline__ void walk_max(const float* row, Arg& a, const Cl& k, float* s2 = nullptr) {
  const float* x[1] = {row};
  a = {-INFINITY, -1};
  float t2 = -INFINITY;
  walk<1>(x, k, [&](int g, const float4* v) {
    if (!S2) return upd_seq4(a, v[0], g);
    const float e[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const bool tk = (a.i < 0) | ((a.v == a.v) & !(e[j] <= a.v));
      t2 = fmaxf(t2, tk ? a.v : e[j]);
      a.v = tk ? e[j] : a.v;
      a.i = tk ? 4 * g + j : a.i;
    }
  });
  if (S2) *s2 = t2;
}

// The max exchange after walk_max (a: the row's argmax after it). With kk > 0 also the fast top-k bound, returned: the
// highest level-0 bucket at or above which some rank has ≥ kk thread maxima (distinct tokens: the top segment down to
// it then holds ≥ kk tokens) — per CTA a local histogram (one atomic per thread) scanned before the exchange, the
// ranks' buckets combined by maximum in it (−1: no rank has kk).
__device__ int max_bound(Arg& a, const Own& o, int kk, Cl& k) {
  Arg t[2] = {a, {-INFINITY, -1}};
  if (kk > 0) {
    hist_zero<false>(k.sm);
    if (o.tm > -INFINITY) atomicAdd(&k.sm.h.c[lvl0(o.tm)], 1u);
    __syncthreads();
    Sel s{0, kZTop, 0, 0, 0};
    const int b = find_bucket<false>(u64(kk), s, k.sm);
    if (b >= 0) t[1] = {float(b), 0};
    exchange<2, 0, u64>(t, nullptr, 0, k);
  } else {
    exchange<1, 0, u64>(t, nullptr, 0, k);
  }
  a = t[0];
  return t[1].v >= 0.f ? int(t[1].v) : -1;
}

// W = Σ_v w_v (maximum m, c) → returned, and the level-0 histogram of the thread maxima tm with their weights (cluster
// totals).
__device__ u64 sum_t(const float* row, float m, float c, float tm, Cl& k) {
  hist_zero<true>(k.sm);
  const u64 wt = weight(tm, m, c);
  if (wt) atomicAdd(&k.sm.h.c[lvl0(tm)], 1u), atomicAdd(&k.sm.h.w[lvl0(tm)], wt);
  const float* x[1] = {row};
  u64 s[1] = {0};
  walk<1>(x, k, [&](int, const float4* v) { s[0] += weight4(v[0], m, c); });
  exchange<0, 1, u64>(nullptr, s, 0, k);
  const u64 W = u64(k.sm.incl[0][k.C - 1]);
  hist_sum<true>(k);
  return W;
}

// Level-0 histogram of the tokens with x ≥ xl (cluster totals; exact for the buckets at or above lvl0(xl)): counts,
// and with W the weights (maximum m, c) and the returned W_K = Σ_{Z ≥ zk} w over all tokens (zk = 0: the row's W).
template <bool W>
__device__ u64 hist0(const float* row, float m, float c, u64 zk, float xl, Cl& k) {
  hist_zero<W>(k.sm);
  const float* x[1] = {row};
  u64 s[1] = {0};
  walk<1>(x, k, [&](int g, const float4* v) {
    if (g >= k.w1) return;
    const float e[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const u64 w = W ? weight(e[j], m, c) : 0;
      if (W) s[0] += zkey(e[j], 4 * g + j) >= zk ? w : 0;
      if (e[j] >= xl) {
        const int b = lvl0(e[j]);
        atomicAdd(&k.sm.h.c[b], 1u);
        if (W && w) atomicAdd(&k.sm.h.w[b], w);
      }
    }
  });
  exchange<0, W ? 1 : 0, u64>(nullptr, s, 0, k);
  const u64 WK = W ? u64(k.sm.incl[0][k.C - 1]) : 0;
  hist_sum<W>(k);
  return WK;
}

// The logits of a Z-range [lo, hi) lie in [zx_lo, zx_hi]: the floats of its end keys, ±inf for keys beyond key(±inf)
// (a refined range's end may fall among the keys no float has — NaN patterns — where zval would give NaN).
__device__ __forceinline__ float zx_lo(u64 lo) { return (lo >> 20) <= 0x007FFFFFu ? -INFINITY : zval(lo); }
__device__ __forceinline__ float zx_hi(u64 hi) {
  return hi >= kZTop || ((hi - 1) >> 20) >= 0xFF800000u ? INFINITY : zval(hi - 1);
}

// Histogram of the tokens with Z in s by bucket (Z − lo) >> sh: counts, and weights (maximum m, c) if W → cluster
// totals (a float range test first: only tokens in the range compute their key).
template <bool W>
__device__ void hist_refine(const float* row, const Sel& s, int sh, float m, float c, Cl& k) {
  hist_zero<W>(k.sm);
  const float* x[1] = {row};
  const u64 span = s.hi - s.lo;
  const float xl = zx_lo(s.lo), xh = zx_hi(s.hi);
  walk<1>(x, k, [&](int g, const float4* v) {
    if (g >= k.w1) return;
    const float e[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (!(e[j] >= xl && e[j] <= xh)) continue;
      const u64 d = zkey(e[j], 4 * g + j) - s.lo;
      if (d < span) {
        const int b = int(d >> sh);
        atomicAdd(&k.sm.h.c[b], 1u);
        if (W) {
          const u64 w = weight(e[j], m, c);
          if (w) atomicAdd(&k.sm.h.w[b], w);
        }
      }
    }
  });
  exchange<0, 0, u64>(nullptr, nullptr, 0, k);
  hist_sum<W>(k);
}

// Gathering into the list: rank c owns slots [c·S, (c+1)·S) of every rank's sm.z, S = cap / C; a token takes the next
// local slot (shared atomic) and is stored there in every rank (remote stores only). gather_end publishes the local
// counts, barrier, then every rank compacts the segments in rank order (one slot per thread: C·S ≤ cap ≤ threads).
// Returns the count, or cap + 1 if some rank had more than S (the list is then incomplete).
__device__ __forceinline__ void gather_begin(Smem& sm) {
  if (threadIdx.x == 0) sm.nl = 0;
  __syncthreads();
}
__device__ __forceinline__ void push(u64 z, int S, Cl& k) {
  const uint32_t i = atomicAdd(&k.sm.nl, 1u);
  if (i < uint32_t(S))
    for (int r = 0; r < k.C; r++) *cg::this_cluster().map_shared_rank(&k.sm.z[k.c * S + i], r) = z;
}
__device__ int gather_end(int S, int cap, Cl& k) {
  __syncthreads();
  if (int(threadIdx.x) < k.C) *cg::this_cluster().map_shared_rank(&k.sm.gc[k.c], threadIdx.x) = k.sm.nl;
  cg::this_cluster().sync();
  int n = 0, off = 0;
  bool over = false;
  const int t = threadIdx.x, r = t / S, i = t - r * S;
  for (int q = 0; q < k.C; q++) {
    const int nq = int(k.sm.gc[q]);
    over |= nq > S;
    off += q < r ? nq : 0;
    n += nq;
  }
  if (over) return cap + 1;
  const bool mv = r < k.C && i < int(k.sm.gc[r]);
  const u64 z = mv ? k.sm.z[t] : 0;
  __syncthreads();
  if (mv) k.sm.z[off + i] = z;
  __syncthreads();
  return n;
}

// Every token with Z in [lo, hi) into the list (a pass over the row; float range test first).
__device__ int gather(const float* row, u64 lo, u64 hi, int cap, Cl& k) {
  const int S = max(1, cap / k.C);
  const u64 span = hi - lo;
  const float xl = zx_lo(lo), xh = zx_hi(hi);
  gather_begin(k.sm);
  const float* x[1] = {row};
  walk<1>(x, k, [&](int g, const float4* v) {
    if (g >= k.w1) return;
    const float e[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (e[j] >= xl && e[j] <= xh) {
        const u64 z = zkey(e[j], 4 * g + j);
        if (z - lo < span) push(z, S, k);
      }
  });
  return gather_end(S, cap, k);
}

// The fast path's gather: every token with x ≥ xl into the list without a pass over the row — only the threads whose
// maximum is ≥ xl (≈ the segment's size) hold any: the maximum alone when s2 < xl, else they read their own float4s
// again (L2) and push theirs, stopping once the rank's segment has overflowed (the list is incomplete anyway).
__device__ int gather_own(const float* row, float xl, const Own& o, int cap, Cl& k) {
  const int S = max(1, cap / k.C);
  gather_begin(k.sm);
  if (o.tm >= xl && o.s2 < xl) {
    push(zkey(o.tm, o.ti), S, k);
  } else if (o.tm >= xl) {
    const float4* r4 = reinterpret_cast<const float4*>(row);
    const volatile uint32_t& nl = k.sm.nl;
    for (int g = k.w0 + int(threadIdx.x & 31); g < k.w1 && nl <= uint32_t(S); g += 32) {
      const float4 v = __ldg(r4 + g);
      const float e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; j++)
        if (e[j] >= xl) push(zkey(e[j], 4 * g + j), S, k);
    }
  }
  return gather_end(S, cap, k);
}

// min_p's cut of a row with a finite maximum m (c, weight threshold mt = mp_thr): M = {v : w_v ≥ mt} = {v : Z_v ≥ zm}.
// weight(x, m, c) is non-decreasing in x (sample.h: the ex2 probe), so M = {v : x_v ≥ x*} with x* the lowest float of
// weight ≥ mt, and zm = okey(x*)·2^20 (index bits 0: every token with that logit is kept). x* is found by warp 0 among
// the order-preserving keys in (okey(−inf), okey(m)] — weight 0 at −inf, 2^44 ≥ mt at m: a 32-way search (lane l tests
// lo + ⌊span·(l+1)/32⌋, lane 31 is hi; the first passing lane's point becomes hi, the one before it lo), ≤ 7 rounds.
// 0 (every token kept) if min_p is off or the top token's own weight is below mt (c not finite: every weight is 0 and
// the row is drawn as its argmax anyway). Every thread of the CTA calls it; all get zm.
__device__ u64 mp_cut(float m, float c, u64 mt, Smem& sm) {
  if (mt == 0) return 0;
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    u64 zm = 0;
    if (weight(m, m, c) >= mt) {
      uint32_t lo = okey(-INFINITY), hi = okey(m);  // weight(lo) < mt ≤ weight(hi)
      while (hi - lo > 1) {
        const uint32_t p = lo + uint32_t((u64(hi - lo) * uint32_t(lane + 1)) >> 5);
        const bool ge = lane == 31 || weight(kval(p), m, c) >= mt;
        const int f = __ffs(__ballot_sync(~0u, ge)) - 1;
        const uint32_t pf = __shfl_sync(~0u, p, f), pb = __shfl_sync(~0u, p, f > 0 ? f - 1 : 0);
        lo = f > 0 ? pb : lo, hi = pf;
      }
      zm = u64(hi) << 20;
    }
    if (lane == 0) sm.zm = zm;
  }
  __syncthreads();
  return sm.zm;
}

// The cut in a list of n ≤ threads entries (thread j: entry j). Its weight w (maximum m, c), rank r (entries with a
// larger key) and the weight a above it are exact, O(n) per entry (broadcast reads). kk > 0 (the list is a top
// segment): K = its min(kk, n_M) top entries, n_M = the entries with key ≥ zm (min_p's M — a top part of the list
// holding the row's top token; zm = 0: all n), then with pp < 1 the first entry whose prefix reaches ⌈pp·W_K⌉, else the
// last of K. kk = 0: the entry at which wa + the prefix first reaches thr. Returns its key and wa + its inclusive prefix
// (W' when the list is a top segment). n > cap: incomplete — returned with n, nothing read.
__device__ Cut list_cut(int n, float m, float c, int kk, float pp, u64 thr, u64 wa, u64 zm, int cap, Smem& sm) {
  if (n > cap) return {0, 0, n};
  const int j = threadIdx.x;
  const u64 z = j < n ? sm.z[j] : 0, w = j < n ? weight(zval(z), m, c) : 0;
  if (j < n) sm.P[j] = w;
  __syncthreads();
  int r = 0;
  u64 a = 0;
  if (j < n)
    for (int i = 0; i < n; i++) {
      const u64 y = sm.z[i];
      if (y > z) r++, a += sm.P[i];
    }
  if (kk > 0) {
    if (zm) kk = min(kk, __syncthreads_count(j < n && z >= zm));  // (the entries above one in M are in M)
    if (j < n && r == kk - 1) sm.lz = z, sm.lw = a + w;
    __syncthreads();
    if (!(pp < 1.f)) return {sm.lz, sm.lw, n};
    thr = pthr(pp, sm.lw), wa = 0;
  }
  if (j < n && wa + a < thr && wa + a + w >= thr) sm.cz = z, sm.cw = wa + a + w;
  __syncthreads();
  return {sm.cz, sm.cw, n};
}

// The list path's draw: kept = the entries with key ≥ zc (total W), t = ⌊U·W/2^64⌋ → the kept v with
// Σ_{kept u < v} w_u ≤ t < Σ_{kept u ≤ v} w_u (index order: u < v ⟺ the key's index part is larger). One rank runs it.
__device__ void list_draw(int n, u64 zc, u64 W, u64 U, int* tok, float* prob, const Smem& sm) {
  const int j = threadIdx.x;
  if (j >= n) return;
  const u64 z = sm.z[j], w = sm.P[j], lz = z & 0xFFFFF;
  if (z < zc || w == 0) return;
  const u64 t = __umul64hi(U, W);
  u64 s = 0;
  for (int i = 0; i < n; i++) {
    const u64 y = sm.z[i];
    if (y >= zc && (y & 0xFFFFF) > lz) s += sm.P[i];
  }
  if (s <= t && t - s < w) {
    *tok = 0xFFFFF - int(lz);
    if (prob) *prob = float(double(w) / double(W));
  }
}

// The cut inside s (as find_bucket left it): the token at which the running count — or weight if W — from the top
// reaches target. Histograms by the top 9 bits of Z − lo (kNB = 2^9 buckets) narrow s until its tokens fit the list
// (each pass divides the range by 2^9; a single key holds one token); then they are read.
template <bool W>
__device__ u64 refine(const float* row, u64 target, Sel& s, float m, float c, int cap, Cl& k) {
  for (;;) {
    if (s.cnt <= u64(cap)) {
      const int n = gather(row, s.lo, s.hi, cap, k);
      if (n <= cap)
        return (W ? list_cut(n, m, c, 0, 1.f, target, s.wa, 0, cap, k.sm)
                  : list_cut(n, m, c, int(target - s.ca), 1.f, 0, 0, 0, cap, k.sm)).zc;
    }
    const int bits = 64 - int(__clzll(s.hi - s.lo - 1)), sh = bits > 9 ? bits - 9 : 0;
    hist_refine<W>(row, s, sh, m, c, k);
    const u64 lo = s.lo + (u64(find_bucket<W>(target, s, k.sm)) << sh), hi = lo + (1ull << sh);
    s.hi = hi < s.hi ? hi : s.hi, s.lo = lo;
  }
}

// Σ of the kept weights (Z ≥ zc; maximum m, c) → returned; leaves sm.incl[0] / sm.ws for pick_row<true>.
__device__ u64 masked_sum(const float* row, float m, float c, u64 zc, Cl& k) {
  const float* x[1] = {row};
  u64 s[1] = {0};
  walk<1>(x, k, [&](int g, const float4* v) { s[0] += weight4(kept4(v[0], g, zc), m, c); });
  exchange<0, 1, u64>(nullptr, s, 0, k);
  return u64(k.sm.incl[0][k.C - 1]);
}

// The cut of a row with a finite maximum m (c; kk, pp from trunc_params; mt = mp_thr(min_p); o from walk_max, b0 from
// max_bound). min_p (sample.h) is a floor on every cut: M = {Z ≥ zm} is a top part of the rank order, so K ∩ M keeps
// the cut max(z_K, zm) (on a list: its top min(kk, n_M) entries), and top-p's target ⌈p·W⌉ over that set — W from
// Z ≥ the cut — is reached inside it (its tokens come first in rank order), so its searches need no change; the fast
// top-p path of thread maxima (whose weights would count tokens outside M) is skipped when zm > 0.
__device__ Cut cut_row(const float* row, float m, float c, int kk, float pp, u64 mt, const Own& o, int b0, int cap,
                       Cl& k) {
  Smem& sm = k.sm;
  const u64 zm = mp_cut(m, c, mt, sm);
  u64 zc = zm;
  if (kk > 0) {
    if (b0 > 0) {  // fast: the top segment down to b0 holds ≥ kk tokens
      const Cut ct = list_cut(gather_own(row, xl0(b0), o, cap, k), m, c, kk, pp, 0, 0, zm, cap, sm);
      if (ct.n <= cap) return ct;
    }
    hist0<false>(row, m, c, 0, b0 > 0 ? xl0(b0) : -INFINITY, k);
    Sel s{0, kZTop, 0, 0, 0};
    const int b = find_bucket<false>(u64(kk), s, sm);
    if (s.ca + s.cnt <= u64(cap)) {
      const Cut ct = list_cut(gather(row, zl0(b), kZTop, cap, k), m, c, kk, pp, 0, 0, zm, cap, sm);
      if (ct.n <= cap) return ct;
    }
    s.lo = zl0(b), s.hi = zl0(b + 1);
    zc = max(refine<false>(row, u64(kk), s, m, c, cap, k), zm);
  }
  if (pp < 1.f) {
    float xl = kk > 0 && b0 > 0 ? xl0(b0) : -INFINITY;  // the cut lies at or above xl (top-p's at or above top-k's)
    if (zm) xl = fmaxf(xl, zval(zm));                     // and in M
    if (kk == 0 && zm == 0) {  // fast: thread maxima of weight ≥ ⌈p·W⌉ at or above b, so the tokens there reach it too
      const u64 thr = pthr(pp, sum_t(row, m, c, o.tm, k));
      Sel s{0, kZTop, 0, 0, 0};
      const int b = find_bucket<true>(thr, s, sm);
      if (b > 0) {
        xl = xl0(b);
        if (s.ca + s.cnt <= u64(cap)) {  // (else the segment holds more thread maxima alone than fit)
          const Cut ct = list_cut(gather_own(row, xl, o, cap, k), m, c, 0, 1.f, thr, 0, 0, cap, sm);
          if (ct.n <= cap) return ct;
        }
      }
    }
    const u64 thr = pthr(pp, hist0<true>(row, m, c, zc, xl, k));
    Sel s{0, kZTop, 0, 0, 0};
    const int b = find_bucket<true>(thr, s, sm);
    if (s.ca + s.cnt <= u64(cap)) {
      const Cut ct = list_cut(gather(row, zl0(b), kZTop, cap, k), m, c, 0, 1.f, thr, 0, 0, cap, sm);
      if (ct.n <= cap) return ct;
    }
    s.lo = zl0(b), s.hi = zl0(b + 1);
    zc = refine<true>(row, thr, s, m, c, cap, k);
  }
  return {zc, masked_sum(row, m, c, zc, k), 0};
}

// A truncated row from scratch: maximum pass (+ the fast top-k bound), then its cut. A degenerate row (maximum ±inf
// or NaN) keeps nothing: W' = 0.
__device__ Cut trunc_row(const float* row, Arg& a, float c, int kk, float pp, u64 mt, int cap, Cl& k) {
  float s2;
  walk_max<true>(row, a, k, &s2);
  const Own o(a, s2);
  const int b0 = max_bound(a, o, kk, k);
  if (!isfinite(a.v)) return {kZTop, 0, 0};
  return cut_row(row, a.v, c, kk, pp, mt, o, b0, cap, k);
}

// Draw from a row right after its trunc_row: W' = 0 → argmax; the list, or the pick over the kept weights.
__device__ void draw(const float* row, const Arg& a, float c, const Cut& ct, u64 U, int* tok, float* prob, Cl& k) {
  if (ct.W == 0) {
    if (k.c == 0 && threadIdx.x == 0) {
      *tok = a.i;
      if (prob) *prob = 1.f;
    }
    return;
  }
  if (ct.n) {
    if (k.c == 0) list_draw(ct.n, ct.zc, ct.W, U, tok, prob, k.sm);
    return;
  }
  pick_row<true>(row, a, c, k.sm.incl[0], U, tok, prob, k, ct.zc);
}

// k_sample's truncated row r after its max walk (a: this thread's partial argmax): the bound and the max exchange, the
// cut, then the draw — or (cut) the cut itself. Not inlined, k by value: the untruncated path keeps its registers and
// its Cl out of local memory.
__device__ __noinline__ void sample_trunc(const float* row, Arg a, float t, int kk, float pp, u64 mt, int cap, u64 U,
                                          int* token, float* prob, u64* cut, int r, Cl k) {
  const float c = temp_scale(t);
  const Own o(a);
  const int b0 = max_bound(a, o, kk, k);
  const Cut ct = isfinite(a.v) ? cut_row(row, a.v, c, kk, pp, mt, o, b0, cap, k) : Cut{kZTop, 0, 0};
  if (cut) {
    if (k.c == 0 && threadIdx.x == 0) cut[2 * r] = ct.zc, cut[2 * r + 1] = ct.W;
    return;
  }
  draw(row, a, c, ct, U, token + r, prob ? prob + r : nullptr, k);
}

// grid (C, R), cluster (C): one row per cluster. cut (sample_cut): the row's truncation instead of a draw. The walk
// comes first and the parameters are loaded unconditionally before it, so their latency hides behind it (a null check
// lets the compiler sink a load to its use).
__global__ void __launch_bounds__(kSampleT) k_sample(const float* logits, int V, const float* temp, const int* top_k,
                                               const float* top_p, const float* min_p, const u64* seed, const u64* ctr,
                                               int* token, float* prob, u64* cut, int cap) {
  __shared__ Smem sm;
  Cl k = make_cl(sm, V);
  const int r = blockIdx.y;
  const float* x[1] = {logits + size_t(r) * V};
  const float t = temp[r];
  const u64 sd = seed[r], ct = ctr[r];
  const int tk = top_k[r];
  const float tp = top_p[r], mp = min_p[r];
  Arg a[1];
  walk_max(x[0], a[0], k);
  int kk;
  float pp;
  const u64 mt = mp_thr(mp);
  if (t > 0.f && (trunc_params(tk, tp, V, kk, pp) || mt || cut))
    return sample_trunc(x[0], a[0], t, kk, pp, mt, cap, philox(sd, ct, 0, 0), token, prob, cut, r, k);
  exchange<1, 0, u64>(a, nullptr, 0, k);
  if (cut) {  // temp ≤ 0
    if (k.c == 0 && threadIdx.x == 0) cut[2 * r] = cut[2 * r + 1] = 0;
    return;
  }
  if (t > 0.f) {
    const float c = temp_scale(t);
    row_sum<1, 0>(x, a, c, nullptr, k);
    pick_row<false>(x[0], a[0], c, sm.incl[0], philox(sd, ct, 0, 0), token + r, prob ? prob + r : nullptr, k);
  } else {
    if (k.c == 0 && threadIdx.x == 0) {
      token[r] = a[0].i;
      if (prob) prob[r] = 1.f;
    }
  }
}

// spec_accept of one truncated sequence (T > 0): target row i, then draft row i through trunc_row (no pipelining —
// each needs its own histogram), the test on the kept weights, and on a rejection the residual over the kept weights
// of both (Σr = 0: p_i' == q_i' → p_i drawn, its cut found again). nd drafts dr[i]; target rows P + i·V (i ≤ nd), draft
// rows Q + i·V; out: *n_acc, *nxt. pm / qm: min_p's weight thresholds of the target / draft rows. Not inlined (as
// sample_trunc).
__device__ __noinline__ void accept_trunc(const float* P, const float* Q, const int* dr, int nd, int V, float c, int kk,
                                          float pp, u64 pm, u64 qm, int cap, u64 sd, u64 ct, int* n_acc, int* nxt, Cl k) {
  const bool lead = k.c == 0 && threadIdx.x == 0;
  for (int i = 0; i < nd; i++) {
    const float* p = P + size_t(i) * V;
    const float* q = Q + size_t(i) * V;
    const int d = dr[i];
    const bool in = d >= 0 && d < V;
    Arg ap, aq;
    const Cut cp = trunc_row(p, ap, c, kk, pp, pm, cap, k);
    const Cut cq = trunc_row(q, aq, c, kk, pp, qm, cap, k);
    const u64 wpd = in ? weight(kept(__ldg(p + d), d, cp.zc), ap.v, c) : 0ull;
    const u64 wqd = in ? weight(kept(__ldg(q + d), d, cq.zc), aq.v, c) : 0ull;
    if (scale(philox(sd, ct, 1 + i, 1), u128(wqd) * cp.W) < u128(wpd) * cq.W) continue;  // min(1, p'(d) / q'(d))
    // rejected: residual r_v = (w'p_v·W'q − w'q_v·W'p)⁺ (a degenerate row's kept weights are all 0)
    const float mp = ap.v, mq = aq.v;
    const u64 Wp = cp.W, Wq = cq.W, zp = cp.zc, zq = cq.zc;
    auto res = [&](float y, float z) -> u128 {
      const u128 u = u128(weight(y, mp, c)) * Wq, w = u128(weight(z, mq, c)) * Wp;
      return u > w ? u - w : u128(0);
    };
    auto res4 = [&](const float4* v, int g, u128(&w)[4]) {
      const float4 y = kept4(v[0], g, zp), z = kept4(v[1], g, zq);
      w[0] = res(y.x, z.x), w[1] = res(y.y, z.y), w[2] = res(y.z, z.z), w[3] = res(y.w, z.w);
    };
    const float* x[2] = {p, q};
    u128 rs[1] = {0};
    walk<2>(x, k, [&](int g, const float4* v) {
      u128 w[4];
      res4(v, g, w);
      rs[0] += w[0] + w[1] + w[2] + w[3];
    });
    exchange<0, 1, u128>(nullptr, rs, 2, k);
    if (lead) *n_acc = i;
    const u64 U = philox(sd, ct, 0, 1);
    u128 r = k.sm.incl[2][k.C - 1];
    if (r == 0) {
      Arg a;
      const Cut c2 = trunc_row(p, a, c, kk, pp, pm, cap, k);
      return draw(p, a, c, c2, U, nxt, nullptr, k);
    }
    r = scale(U, r);
    if (owner(k.sm.incl[2], r) != k.c) return;
    warp_pick<2, u128>(x, r, 0, res4, k);
    if (threadIdx.x == 0) *nxt = k.sm.pick;
    return;
  }
  // every draft accepted: the bonus token from the truncated p_nd
  Arg a;
  const float* p = P + size_t(nd) * V;
  const Cut cb = trunc_row(p, a, c, kk, pp, pm, cap, k);
  if (lead) *n_acc = nd;
  draw(p, a, c, cb, philox(sd, ct, 0, 1), nxt, nullptr, k);
}

#include "kernels/sample/p016_block_verify.cuh"

// grid (C, B), cluster (C): one sequence per cluster. Every rank takes the same decisions from the same values, so
// all of them meet every cluster barrier. BV (P016 block_verify): T > 0 with 1 ≤ k ≤ kBvMax drafts by block
// verification (p016_block_verify.cuh; bvx: every classification unsure — test hook), the rest as the base.
template <bool BV>
__global__ void __launch_bounds__(kAcceptT) k_accept(const float* pl, const int* p_start, const float* ql, const int* q_start,
                                               const int* drafts, int V, const float* temp, const int* top_k,
                                               const float* top_p, const float* min_p, const float* q_min_p,
                                               const u64* seed, const u64* ctr, int* n_acc, int* next, int cap, int bvx) {
  __shared__ Smem sm;
  const int b = blockIdx.y, p0 = p_start[b], q0 = q_start[b], nd = q_start[b + 1] - q0;
  const bool lead = cg::this_cluster().block_rank() == 0 && threadIdx.x == 0;
  if (nd < 0 || p_start[b + 1] - p0 != nd + 1) {  // the same in every rank: the cluster leaves before touching a barrier
    if (lead) n_acc[b] = -1, next[b] = -1;
    return;
  }
  Cl k = make_cl(sm, V);
  const float t = temp[b];
  const u64 sd = seed[b], ct = ctr[b];
  const int tk = top_k[b];
  const float tp = top_p[b];
  auto prow = [&](int i) { return pl + size_t(p0 + i) * V; };
  auto qrow = [&](int i) { return ql + size_t(q0 + i) * V; };
  if (!(t > 0.f)) {  // greedy: accept while the draft is the target's argmax
    int i = 0;
    Arg a[1];
    for (;; i++) {
      const float* x[1] = {prow(i)};
      row_max<1>(x, a, k);
      if (i == nd || drafts[q0 + i] != a[0].i) break;
    }
    if (lead) n_acc[b] = i, next[b] = a[0].i;
    return;
  }
  const float c = temp_scale(t);
  int kk;
  float pp;
  const u64 pm = mp_thr(min_p[b]), qm = mp_thr(q_min_p[b]);
  const bool tr = trunc_params(tk, tp, V, kk, pp) || pm || qm;
  if constexpr (BV) {
    if (nd >= 1 && nd <= p016::kBvMax) {
      __shared__ p016::Bv bv;
      if (tr)
        return p016::accept_cut(prow(0), qrow(0), drafts + q0, nd, V, c, kk, pp, pm, qm, cap, sd, ct, bvx != 0,
                                n_acc + b, next + b, bv, k);
      return p016::accept_full(prow(0), qrow(0), drafts + q0, nd, V, c, sd, ct, bvx != 0, n_acc + b, next + b, bv, k);
    }
  }
  if (tr)
    return accept_trunc(prow(0), qrow(0), drafts + q0, nd, V, c, kk, pp, pm, qm, cap, sd, ct, n_acc + b, next + b, k);
  Arg a[2];  // maxima of the current pair (target, draft)
  if (nd > 0) {
    const float* x[2] = {prow(0), qrow(0)};
    row_max<2>(x, a, k);
  } else {
    const float* x[1] = {prow(0)};
    row_max<1>(x, a, k);
  }
  for (int i = 0; i < nd; i++) {
    const float* p = prow(i);
    const float* q = qrow(i);
    const int d = drafts[q0 + i];
    const bool in = d >= 0 && d < V;
    const float xp = in ? __ldg(p + d) : 0.f, xq = in ? __ldg(q + d) : 0.f;
    Arg an[2];  // maxima of the next rows
    if (i + 1 < nd) {
      const float* x[4] = {p, q, prow(i + 1), qrow(i + 1)};
      row_sum<2, 2>(x, a, c, an, k);
    } else {
      const float* x[3] = {p, q, prow(nd)};
      row_sum<2, 1>(x, a, c, an, k);
    }
    const u64 Wp = u64(sm.incl[0][k.C - 1]), Wq = u64(sm.incl[1][k.C - 1]);
    const u128 A = u128(in ? weight(xp, a[0].v, c) : 0ull) * Wq, B = u128(in ? weight(xq, a[1].v, c) : 0ull) * Wp;
    if (scale(philox(sd, ct, 1 + i, 1), B) < A) {  // accepted with probability min(1, A/B)
      a[0] = an[0], a[1] = an[1];
      continue;
    }
    // rejected: residual r_v = (wp_v·Wq − wq_v·Wp)⁺
    const float mp = a[0].v, mq = a[1].v;
    auto res = [&](float y, float z) -> u128 {
      const u128 u = u128(weight(y, mp, c)) * Wq, w = u128(weight(z, mq, c)) * Wp;
      return u > w ? u - w : u128(0);
    };
    auto res4 = [&](const float4* v, int, u128(&w)[4]) {
      w[0] = res(v[0].x, v[1].x), w[1] = res(v[0].y, v[1].y), w[2] = res(v[0].z, v[1].z), w[3] = res(v[0].w, v[1].w);
    };
    const float* x[2] = {p, q};
    u128 rs[1] = {0};
    walk<2>(x, k, [&](int g, const float4* v) {
      u128 w[4];
      res4(v, g, w);
      rs[0] += w[0] + w[1] + w[2] + w[3];
    });
    exchange<0, 1, u128>(nullptr, rs, 2, k);
    if (lead) n_acc[b] = i;
    const u64 U = philox(sd, ct, 0, 1);
    u128 r = sm.incl[2][k.C - 1];
    if (r == 0) {  // p_i == q_i exactly: draw from p_i (its sums again: the residual exchange replaced its warp sums)
      const float* xp1[1] = {p};
      row_sum<1, 0>(xp1, a, c, nullptr, k);
      return pick_row<false>(p, a[0], c, sm.incl[0], U, next + b, nullptr, k);
    }
    r = scale(U, r);
    if (owner(sm.incl[2], r) != k.c) return;
    warp_pick<2, u128>(x, r, 0, res4, k);
    if (threadIdx.x == 0) next[b] = sm.pick;
    return;
  }
  // every draft accepted: the bonus token from p_nd (its maximum came with the last pair's pass)
  const float* x[1] = {prow(nd)};
  row_sum<1, 0>(x, a, c, nullptr, k);
  if (lead) n_acc[b] = nd;
  pick_row<false>(x[0], a[0], c, sm.incl[0], philox(sd, ct, 0, 1), next + b, nullptr, k);
}

// sample_weights: one CTA per row — its maximum (exact in any order on finite rows), then w_v.
__global__ void __launch_bounds__(1024) k_weights(const float* logits, int V, const float* temp, u64* w) {
  __shared__ float red[kMaxW];
  const float* x = logits + size_t(blockIdx.x) * V;
  float m = -INFINITY;
  for (int v = threadIdx.x; v < V; v += blockDim.x) m = fmaxf(m, x[v]);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  for (int i = 0; i < int(blockDim.x >> 5); i++) m = fmaxf(m, red[i]);
  const float c = temp_scale(temp[blockIdx.x]);
  for (int v = threadIdx.x; v < V; v += blockDim.x) w[size_t(blockIdx.x) * V + v] = weight(x[v], m, c);
}

// The ex2 probe (sample_weight_monotone): every fp32 y ≤ 0 (bits 0x80000000 = −0 … 0xFF800000 = −inf, increasing bits =
// decreasing y) → the count of y whose weight(y, 0, 1) = round(ex2.approx.ftz(y)·2^44) is below the weight of the
// next lower float (weight(y − ulp) > weight(y): not non-decreasing), and in cnt[1] the total descent Σ (w(y) −
// w(y − ulp)) over the decreasing pairs — with no violation it telescopes to w(−0) − w(−inf) = 2^44 exactly iff the
// chunks covered the whole range. Thread chunks of 4096 floats, each compared with its predecessor (the chunk's first
// with the float above it).
__global__ void k_ex2_probe(unsigned long long* cnt) {
  constexpr uint32_t kN = 0xFF800000u - 0x80000000u;  // pairs (b, b + 1)
  const uint64_t i0 = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4096;
  if (i0 >= kN) return;
  const uint32_t n = kN - i0 < 4096 ? uint32_t(kN - i0) : 4096u;
  uint32_t b = 0x80000000u + uint32_t(i0);
  u64 prev = weight(__uint_as_float(b), 0.f, 1.f);
  unsigned long long up = 0, down = 0;
  for (uint32_t j = 0; j < n; j++) {
    const u64 w = weight(__uint_as_float(++b), 0.f, 1.f);  // the next lower float
    up += w > prev, down += w < prev ? prev - w : 0ull;
    prev = w;
  }
  if (up) atomicAdd(cnt, up);
  if (down) atomicAdd(cnt + 1, down);
}

// SM count and the largest cluster both kernels run (16 needs the non-portable size), set once — the first call must
// not be inside a graph capture. zeros: kMaxRows fp32 zeros, the min_p arrays of calls without min_p.
constexpr int kMaxRows = 65536;
struct Dev {
  int sms, cap;
  float* zeros;
};
const Dev& dev() {
  static const Dev d = [] {
    Dev r{0, kMaxC, nullptr};
    CUDA_CHECK(cudaMalloc(&r.zeros, kMaxRows * sizeof(float)));
    CUDA_CHECK(cudaMemset(r.zeros, 0, kMaxRows * sizeof(float)));
    CUDA_CHECK(cudaDeviceGetAttribute(&r.sms, cudaDevAttrMultiProcessorCount, 0));
    const std::pair<const void*, int> fs[3] = {{reinterpret_cast<const void*>(k_sample), kSampleT},
                                               {reinterpret_cast<const void*>(k_accept<false>), kAcceptT},
                                               {reinterpret_cast<const void*>(k_accept<true>), kAcceptT}};
    for (const auto& [f, nt] : fs) {
      CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(kMaxC, 1);
      cfg.blockDim = dim3(nt);
      cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = kMaxC, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
      cfg.attrs = at;
      cfg.numAttrs = 1;
      int n = 0;
      CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&n, f, &cfg));
      r.cap = std::min(r.cap, n);
    }
    if (r.cap < 1) qlog::fail("sample", "cluster size %d", r.cap);
    return r;
  }();
  return d;
}
// a min_p array, or zeros (min_p = 0: off) for n ≤ kMaxRows rows / sequences
const float* mp_or_zero(const float* mp, int n) {
  if (mp) return mp;
  if (n > kMaxRows) qlog::fail("sample", "%d rows without min_p (at most %d)", n, kMaxRows);
  return dev().zeros;
}

// CTAs per row: spread `rows` clusters over the SMs (one CTA per SM), at least kMinChunk4 float4 per CTA.
int cluster_size(int V, int rows) {
  const Dev& d = dev();
  const int c = g_force ? g_force : std::min(std::max(1, d.sms / rows), cdiv(V / 4, kMinChunk4));
  return std::max(1, std::min(c, d.cap));
}

void check(const float* x, int rows, int V, const char* what) {
  if (V <= 0 || V % 4 || V >= (1 << 20) || reinterpret_cast<uintptr_t>(x) % 16 || rows > 65535)
    qlog::fail("sample", "%s: V %d (multiple of 4, < 2^20), rows 16-B aligned, row count %d ≤ 65535", what, V, rows);
}

template <class K, class... A>
void launch(K kern, int nt, int C, int rows, cudaStream_t s, A... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C, rows);
  cfg.blockDim = dim3(nt);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

#include "kernels/sample/p019_sparse.cuh"

}  // namespace

void sample_rows(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                 const float* min_p, const uint64_t* seed, const uint64_t* ctr, int* token, float* prob, cudaStream_t s) {
  if (R <= 0) return;
  check(logits, R, V, "sample_rows");
  launch(k_sample, kSampleT, cluster_size(V, R), R, s, logits, V, temp, top_k, top_p, mp_or_zero(min_p, R),
         reinterpret_cast<const u64*>(seed), reinterpret_cast<const u64*>(ctr), token, prob, static_cast<u64*>(nullptr),
         std::min(g_cap, kSampleT));
}

void sample_rows(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                 const uint64_t* seed, const uint64_t* ctr, int* token, float* prob, cudaStream_t s) {
  sample_rows(logits, R, V, temp, top_k, top_p, nullptr, seed, ctr, token, prob, s);
}

void spec_accept(const float* p_logits, const int* p_start, const float* q_logits, const int* q_start,
                 const int* drafts, int B, int V, const float* temp, const int* top_k, const float* top_p,
                 const float* min_p, const float* q_min_p, const uint64_t* seed, const uint64_t* ctr, int* n_acc,
                 int* next, cudaStream_t s) {
  if (B <= 0) return;
  check(p_logits, B, V, "spec_accept p");
  check(q_logits, B, V, "spec_accept q");
  launch(patch::on(patch::block_verify) ? k_accept<true> : k_accept<false>, kAcceptT, cluster_size(V, B), B, s, p_logits,
         p_start, q_logits, q_start, drafts, V, temp, top_k, top_p, mp_or_zero(min_p, B), mp_or_zero(q_min_p, B),
         reinterpret_cast<const u64*>(seed), reinterpret_cast<const u64*>(ctr), n_acc, next, std::min(g_cap, kAcceptT),
         g_exact);
}

void spec_accept(const float* p_logits, const int* p_start, const float* q_logits, const int* q_start,
                 const int* drafts, int B, int V, const float* temp, const int* top_k, const float* top_p,
                 const uint64_t* seed, const uint64_t* ctr, int* n_acc, int* next, cudaStream_t s) {
  spec_accept(p_logits, p_start, q_logits, q_start, drafts, B, V, temp, top_k, top_p, nullptr, nullptr, seed, ctr, n_acc,
              next, s);
}

int sample_force_cluster(int c) {
  const int prev = g_force;
  g_force = std::max(0, std::min(c, kMaxC));
  return prev;
}

int sample_force_exact(int on) {
  const int prev = g_exact;
  g_exact = on != 0;
  return prev;
}

int sample_force_cap(int cap) {
  const int prev = g_cap;
  g_cap = cap <= 0 ? kCap : std::min(cap, kCap);
  return prev;
}

void sample_cut(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                const float* min_p, uint64_t* cut, cudaStream_t s) {
  if (R <= 0) return;
  check(logits, R, V, "sample_cut");
  const u64* any = reinterpret_cast<const u64*>(cut);  // seed / ctr: read, unused ([R] readable entries)
  launch(k_sample, kSampleT, cluster_size(V, R), R, s, logits, V, temp, top_k, top_p, mp_or_zero(min_p, R), any, any,
         static_cast<int*>(nullptr), static_cast<float*>(nullptr), reinterpret_cast<u64*>(cut), std::min(g_cap, kSampleT));
}

void sample_cut(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                uint64_t* cut, cudaStream_t s) {
  sample_cut(logits, R, V, temp, top_k, top_p, nullptr, cut, s);
}

uint64_t sample_weight_monotone(uint64_t* descent) {
  unsigned long long* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 2 * sizeof(*d)));
  CUDA_CHECK(cudaMemset(d, 0, 2 * sizeof(*d)));
  constexpr unsigned kThreads = (0xFF800000u - 0x80000000u) / 4096;  // exact: 0x7F800000 is a multiple of 4096
  k_ex2_probe<<<cdiv(int(kThreads), 256), 256>>>(d);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long h[2] = {0, 0};
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  if (descent) *descent = h[1];
  return h[0];
}

void sample_weights(const float* logits, int R, int V, const float* temp, uint64_t* w, cudaStream_t s) {
  if (R <= 0) return;
  k_weights<<<R, 1024, 0, s>>>(logits, V, temp, reinterpret_cast<u64*>(w));
  CUDA_CHECK(cudaGetLastError());
}

// ---- P019 sparse_logits (p019_sparse.cuh) ----------------------------------------------------------------------------
void sample_cands(const float* x, int R, int Vh, int off, SparseCand* out, cudaStream_t s) {
  if (R <= 0) return;
  if (Vh < kSparseK || Vh % 4 || off < 0 || off + Vh >= (1 << 20) || reinterpret_cast<uintptr_t>(x) % 16 || R > 65535)
    qlog::fail("sample", "sample_cands: Vh %d (multiple of 4, ≥ %d), off %d (off + Vh < 2^20), 16-B aligned, row count %d ≤ 65535", Vh,
               kSparseK, off, R);
  launch(p019::k_cands, p019::kCT, p019::cands_cluster(Vh, R), R, s, x, Vh, off, out);
}

void sample_rows_sparse(const SparseCand* cands, int ld, int R, int V, const float* temp, const int* top_k,
                        const float* top_p, const float* min_p, const uint64_t* seed, const uint64_t* ctr, int* token,
                        float* prob, cudaStream_t s) {
  if (R <= 0) return;
  if (ld < R || V <= 0 || V >= (1 << 20)) qlog::fail("sample", "sample_rows_sparse: ld %d ≥ R %d, V %d < 2^20", ld, R, V);
  p019::k_sample_sparse<<<R, 2 * kSparseK, 0, s>>>(cands, ld, V, temp, top_k, top_p, mp_or_zero(min_p, R),
                                                   reinterpret_cast<const u64*>(seed), reinterpret_cast<const u64*>(ctr),
                                                   token, prob);
  CUDA_CHECK(cudaGetLastError());
}

void sample_rows_sparse(const SparseCand* cands, int ld, int R, int V, const float* temp, const int* top_k,
                        const float* top_p, const uint64_t* seed, const uint64_t* ctr, int* token, float* prob,
                        cudaStream_t s) {
  sample_rows_sparse(cands, ld, R, V, temp, top_k, top_p, nullptr, seed, ctr, token, prob, s);
}

void spec_accept_sparse(const SparseCand* p_cands, int p_ld, const int* p_start, const SparseCand* q_cands, int q_ld,
                        const int* q_start, const int* drafts, int B, int V, const float* temp, const int* top_k,
                        const float* top_p, const float* min_p, const float* q_min_p, const uint64_t* seed,
                        const uint64_t* ctr, int* n_acc, int* next, cudaStream_t s) {
  if (B <= 0) return;
  if (V <= 0 || V >= (1 << 20)) qlog::fail("sample", "spec_accept_sparse: V %d < 2^20", V);
  const bool bv = patch::on(patch::block_verify);
  auto k = bv ? p019::k_accept_sparse<true> : p019::k_accept_sparse<false>;
  bv ? p019::accept_attr<true>() : p019::accept_attr<false>();
  k<<<B, p019::kAT, sizeof(p019::ASm), s>>>(p_cands, p_ld, p_start, q_cands, q_ld, q_start, drafts, V, temp, top_k, top_p,
                                            mp_or_zero(min_p, B), mp_or_zero(q_min_p, B),
                                            reinterpret_cast<const u64*>(seed), reinterpret_cast<const u64*>(ctr), n_acc,
                                            next, g_exact);
  CUDA_CHECK(cudaGetLastError());
}

void spec_accept_sparse(const SparseCand* p_cands, int p_ld, const int* p_start, const SparseCand* q_cands, int q_ld,
                        const int* q_start, const int* drafts, int B, int V, const float* temp, const int* top_k,
                        const float* top_p, const uint64_t* seed, const uint64_t* ctr, int* n_acc, int* next,
                        cudaStream_t s) {
  spec_accept_sparse(p_cands, p_ld, p_start, q_cands, q_ld, q_start, drafts, B, V, temp, top_k, top_p, nullptr, nullptr,
                     seed, ctr, n_acc, next, s);
}

int sample_cands_cluster(int Vh, int R) { return p019::cands_cluster(Vh, R); }

unsigned sample_cands_slow(bool reset) {
  unsigned n = 0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&n, p019::g_slow, sizeof(n)));
  if (reset) {
    const unsigned z = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(p019::g_slow, &z, sizeof(z)));
  }
  return n;
}

}  // namespace q38
