// gdn.cu — Gated DeltaNet core (arch spec §2.4), one fused launch per call.
//
// Grid (8 k-heads × 3·NC, B), cluster 3·NC = the 3 v-heads of one k-head × NC V-slices of R = 128/NC rows.
// Each CTA: cp.async its small inputs (conv window of the k-head's q|k channels and its own v rows, z, b|a, weights)
// and wait for them, then issue its S slice [R,128] fp32 loads into registers. Per chunk of ≤16 tokens: conv1d (k=4)
// + silu → l2norm, q·128^-½, g, β → delta rule per token with S in registers → bf16 o → gated RMSNorm (Σo² of the head
// summed over the NC slices through DSMEM, fixed order) → core. S is read once and written once per call (stored after
// the norm's cluster barrier). The call is bound by the S traffic plus ~2-3 µs of launch/latency.
// Numerics follow vLLM's decode path (causal_conv1d_update + fused_recurrent packed decode + layernorm_guard):
// conv products rounded to bf16, summed in fp32, conv output bf16; fp32 recurrence; o rounded to bf16 before the norm.
// Conv state race: q|k channels are shared by the cluster; every CTA has its copy of the old state before the first
// cluster barrier, cluster rank 0 writes the new q|k state after it. v channels have a single owner.
// Alignment: qkvz/ba/core rows, S and conv state 16-B aligned (cudaMalloc'd arrays; row sizes are multiples of 16 B).
// Speculative rows (gdn.h): gdn_core_spec = the same kernel with SPEC set (no S / conv stores, a per-row record
// instead); gdn_commit_kernel replays the accepted rows of the record into S and shifts the conv state.
// P020 gdn_fused_commit: gdn_core_spec_pending = the same kernel with SPEC and PEND set — the previous verify step's
// commit is applied to S in registers between its load and the new rows (S read once and written once per round).
#include "kernels/gdn/gdn.h"

#include <cooperative_groups.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "core/log.h"
#include "model/patches.h"

namespace cg = cooperative_groups;

namespace q38 {
namespace {

constexpr int kNC = 2;                 // CTAs per v-head
constexpr int kThreads = 256, kWarps = kThreads / 32;
constexpr int kTC = 16;                // tokens per on-chip chunk
constexpr int kPendSmem = kGdnRecRows * (128 + 4) * sizeof(float);  // P020: dynamic smem kp [16][128] | ep [16][4]
constexpr int kQKVZ = 8192, kQKV = 5120, kZ0 = 5120, kV0 = 2048, kCore = 3072;
constexpr float kQScale = 0.08838834764831845f;  // 128^-0.5

__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);  // identical result in every lane
  return v;
}
// split cluster barrier (all threads, converged): arrive has release semantics unless relaxed; wait is acquire
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); }
__device__ __forceinline__ float row_sum8(float v) {  // over the 8 lanes of a row group, same result in each
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 4);
}
__device__ __forceinline__ float dot4(float4 a, float4 b) {
  return fmaf(a.w, b.w, fmaf(a.z, b.z, fmaf(a.y, b.y, a.x * b.x)));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {  // async 16-B global → smem (L2 only)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// More rows than a record holds: `say` threads print, every thread traps. Not inlined and noreturn, so the printf call
// (stack frame, call ABI) stays out of the kernels' register allocation: nothing is live across the call.
[[noreturn]] __device__ __noinline__ void rec_overflow(const char* fn, int seq, int rows, bool say) {
  if (say) printf("[gdn] %s: sequence %d: %d rows > record capacity %d\n", fn, seq, rows, kGdnRecRows);
  __trap();
}

// SPEC (gdn_core_spec): same arithmetic and outputs, but S and the conv state are not stored; instead the record of
// sequence b (recs[b]) receives per row e^g, δ, normalised k and pre-conv x for gdn_commit. The base instantiation
// (SPEC = false) compiles to the same code as before the flag existed (recs unused, the extra smem array dropped).
// PEND (P020 gdn_fused_commit, with SPEC; gdn_core_spec_pending): the pending commit of sequence b — rows 0 … np−1 of
// pend[b], np = npend[b] — is applied first: δ of the CTA's rows (into dls, unused until the delta rule), k of the
// k-head and e^g (into 8.25 KiB of dynamic smem: kp | ep) are fetched with the inputs; the conv window of the new rows
// is taken from (old conv ‖ pending x rows) — the committed conv state — and stored after the first cluster barrier
// (like the base kernel's conv state); S is replayed in registers just before the delta rule (gdn_commit_kernel's
// arithmetic: S ← fma(δ, k, S·e^g), row after row; S loads still overlap the conv and l2norm as in SPEC) and stored at
// once (streaming stores, overlapping the new rows). The new rows then run as SPEC on the committed state.
// np = 0: nothing is read from pend[b] or stored (= SPEC). In place (pend[b] == recs[b]) is safe: every value a CTA
// reads from the pending record is shared at most within its cluster (δ: its own rows; k, q|k x: the k-head's cluster;
// e^g: the v-head's two CTAs; v x: its own rows), all of it has landed in smem before the first cluster barrier, and
// every record store comes after that barrier.
template <int NC, bool SPEC, bool PEND = false>
__global__ void __launch_bounds__(kThreads) gdn_core_kernel(const bf16* __restrict__ qkvz, const bf16* __restrict__ ba,
                                                            bf16* __restrict__ core, bf16* __restrict__ o_out,
                                                            GdnState* const* __restrict__ states,
                                                            const int* __restrict__ seq_start, GdnW w,
                                                            GdnRec* const* __restrict__ recs,
                                                            GdnRec* const* __restrict__ pend,
                                                            const int* __restrict__ npend) {
  static_assert(SPEC || !PEND, "PEND needs SPEC");
  constexpr int R = 128 / NC, RW = R / kWarps, CH = 256 + R;  // rows per CTA / per warp, conv channels per CTA
  static_assert(RW % 4 == 0 && R % kWarps == 0 && R % 8 == 0, "split");
  __shared__ __align__(16) bf16 xs[kTC + 3][CH];  // pre-conv x(t0−3 … t0+n−1) of the CTA's channels (q | k | v rows)
  __shared__ __align__(16) bf16 zs[kTC][R], bas[kTC][48], cws[CH][4], nws[R], os[kTC][R];
  __shared__ __align__(16) float qs[kTC][128], ks[kTC][128];
  __shared__ __align__(16) float adt[2][24];  // A, dt_bias
  __shared__ float vs[kTC][R], eg_s[kTC], beta_s[kTC], part[kTC], rstd[kTC];
  __shared__ __align__(16) float dls[SPEC ? kTC : 1][R];  // SPEC: δ of the CTA's rows (static smem 41.4 → 45.4 KiB)

  cg::cluster_group cl = cg::this_cluster();
  const int cr = int(cl.block_rank()), hk = blockIdx.x / (3 * NC), hl = cr / NC, v0 = (cr % NC) * R;
  const int h = hk * 3 + hl, b = blockIdx.y, tid = threadIdx.x, wp = tid >> 5, lane = tid & 31;
  const int row0 = seq_start[b], T = seq_start[b + 1] - row0;
  if (T <= 0) return;  // uniform over the cluster: no barrier is reached
  if (SPEC && T > kGdnRecRows) rec_overflow("gdn_core_spec", b, T, tid == 0 && cr == 0 && hk == 0);
  const GdnState st = *states[b];
  GdnRec rc;
  if constexpr (SPEC) rc = *recs[b];
  int np = 0;  // PEND: pending rows of sequence b (≤ 0: none)
  GdnRec pr;
  if constexpr (PEND) {
    np = max(npend[b], 0);
    // (the existing name string: a new one would move the format string's constant slot in every kernel)
    if (np > kGdnRecRows) rec_overflow("gdn_commit", b, np, tid == 0 && cr == 0 && hk == 0);
    if (np) pr = *pend[b];
  }
  auto chan = [&](int c) {  // local conv channel → channel of the rank's q|k|v (8-aligned groups stay contiguous)
    return c < 128 ? hk * 128 + c : c < 256 ? 1024 + hk * 128 + (c - 128) : kV0 + h * 128 + v0 + (c - 256);
  };
  auto fetch = [&](int t0, int n) {
    constexpr int XC = CH / 8, ZC = R / 8;
    const int nx = (n + 3) * XC, nz = n * ZC;
    for (int i = tid; i < nx + nz + n * 6; i += kThreads) {
      if (i < nx) {
        const int r = i / XC, c = (i % XC) * 8, t = t0 - 3 + r;
        if constexpr (PEND) {  // rows before the first: committed conv state = (old conv ‖ pending x) rows np … np+2
          const int k = 3 + t + np;
          const bf16* x = t >= 0 ? qkvz + size_t(row0 + t) * kQKVZ : k < 3 ? st.conv + k * kQKV : pr.x + (k - 3) * kQKV;
          cp16(&xs[r][c], x + chan(c));
        } else {  // (kept as before the flag: same SASS for the base and SPEC kernels)
          cp16(&xs[r][c], (t < 0 ? st.conv + (3 + t) * kQKV : qkvz + size_t(row0 + t) * kQKVZ) + chan(c));
        }
      } else if (i < nx + nz) {
        const int r = (i - nx) / ZC, c = ((i - nx) % ZC) * 8;
        cp16(&zs[r][c], qkvz + size_t(row0 + t0 + r) * kQKVZ + kZ0 + h * 128 + v0 + c);
      } else {
        const int r = (i - nx - nz) / 6, c = ((i - nx - nz) % 6) * 8;
        cp16(&bas[r][c], ba + size_t(row0 + t0 + r) * 48 + c);
      }
    }
    cp_commit();
  };
  for (int i = tid; i < CH / 2 + R / 8 + 12; i += kThreads) {  // weights: conv | norm | A | dt_bias
    if (i < CH / 2) cp16(&cws[2 * i][0], w.conv_w + size_t(chan(2 * i)) * 4);
    else if (i < CH / 2 + R / 8) cp16(&nws[(i - CH / 2) * 8], w.norm_w + v0 + (i - CH / 2) * 8);
    else if (i < CH / 2 + R / 8 + 6) cp16(&adt[0][(i - CH / 2 - R / 8) * 4], w.A + (i - CH / 2 - R / 8) * 4);
    else cp16(&adt[1][(i - CH / 2 - R / 8 - 6) * 4], w.dt_bias + (i - CH / 2 - R / 8 - 6) * 4);
  }
  if constexpr (PEND) {  // pending rows (same group as the inputs): δ of the CTA's S rows → dls; k of the k-head → kp,
    // the 16-B chunk of e^g holding v-head h → ep (dynamic smem kp [16][128] | ep [16][4], kPendSmem bytes)
    extern __shared__ __align__(16) float pend_sm[];
    constexpr int PC = R / 4 + 33;
    for (int i = tid; i < np * PC; i += kThreads) {
      const int t = i / PC, c = i % PC;
      if (c < R / 4) cp16(&dls[t][c * 4], pr.dl + (size_t(t) * 24 + h) * 128 + v0 + c * 4);
      else if (c < R / 4 + 32)
        cp16(pend_sm + t * 128 + (c - R / 4) * 4, pr.k + (size_t(t) * 8 + hk) * 128 + (c - R / 4) * 4);
      else
        cp16(pend_sm + kGdnRecRows * 128 + t * 4, pr.eg + t * 24 + (h & ~3));
    }
  }
  fetch(0, min(kTC, T));
  cp_wait();  // inputs land before the S stream starts (issued alongside, they would arrive only after it)
  __syncthreads();

  // S in registers: warp wp owns rows v0 + wp·RW + [0, RW); lane (g = lane/8, p = lane%8) holds rows g + 4m
  // (m < RM) at float4 columns p + 8j (j < 4) — every load/store instruction covers whole 128-B lines, and a row's
  // dot product reduces over 8 lanes only.
  constexpr int RM = RW / 4;
  const int g = lane >> 3, p = lane & 7;
  float* Sg = st.S + (size_t(h) * 128 + v0 + wp * RW + g) * 128 + p * 4;
  float4 S[RM][4];
#pragma unroll
  for (int m = 0; m < RM; m++)
#pragma unroll
    for (int j = 0; j < 4; j++) S[m][j] = __ldcs(reinterpret_cast<const float4*>(Sg + m * 512 + j * 32));

  for (int t0 = 0; t0 < T; t0 += kTC) {
    const int n = min(kTC, T - t0);
    if (t0) {
      fetch(t0, n);
      cp_wait();
      __syncthreads();
    }
    // conv1d (k=4) + silu: products rounded to bf16, fp32 sum, bf16 output
    for (int c = tid; c < CH; c += kThreads) {
      const float w0 = bf(cws[c][0]), w1 = bf(cws[c][1]), w2 = bf(cws[c][2]), w3 = bf(cws[c][3]);
      float x0 = bf(xs[0][c]), x1 = bf(xs[1][c]), x2 = bf(xs[2][c]);
      for (int i = 0; i < n; i++) {
        const float x3 = bf(xs[3 + i][c]);
        float acc = bfr(w0 * x0);
        acc += bfr(w1 * x1);
        acc += bfr(w2 * x2);
        acc += bfr(w3 * x3);
        const float y = bfr(acc / (1.f + __expf(-acc)));
        if (c < 128) qs[i][c] = y;
        else if (c < 256) ks[i][c - 128] = y;
        else vs[i][c - 256] = y;
        x0 = x1, x1 = x2, x2 = x3;
      }
    }
    if (tid < n) {  // e^g with g = A·softplus(a + dt_bias), A = −exp(A_log); β = sigmoid(b)
      const float xa = bf(bas[tid][24 + h]) + adt[1][h];
      const float sp = xa <= 20.f ? logf(1.f + expf(xa)) : xa;
      eg_s[tid] = expf(adt[0][h] * sp);
      beta_s[tid] = 1.f / (1.f + expf(-bf(bas[tid][h])));
    }
    __syncthreads();
    for (int i = wp; i < n; i += kWarps) {  // l2norm (eps inside sqrt), q·128^-½
      float4 q = reinterpret_cast<float4*>(qs[i])[lane], k = reinterpret_cast<float4*>(ks[i])[lane];
      const float nq = sqrtf(warp_sum(dot4(q, q)) + 1e-6f), nk = sqrtf(warp_sum(dot4(k, k)) + 1e-6f);
      q.x = q.x / nq * kQScale, q.y = q.y / nq * kQScale, q.z = q.z / nq * kQScale, q.w = q.w / nq * kQScale;
      k.x /= nk, k.y /= nk, k.z /= nk, k.w /= nk;
      reinterpret_cast<float4*>(qs[i])[lane] = q;
      reinterpret_cast<float4*>(ks[i])[lane] = k;
    }
    __syncthreads();
    if constexpr (PEND) {
      if (t0 == 0 && np) {  // pending commit: gdn_commit_kernel's replay (two roundings, S·e^g kept separate), S stored
        extern __shared__ __align__(16) float pend_sm[];
        for (int t = 0; t < np; t++) {
          float4 k[4];
#pragma unroll
          for (int j = 0; j < 4; j++) k[j] = reinterpret_cast<const float4*>(pend_sm + t * 128)[p + 8 * j];
          const float eg = pend_sm[kGdnRecRows * 128 + t * 4 + (h & 3)];
#pragma unroll
          for (int m = 0; m < RM; m++) {
            const float dl = dls[t][wp * RW + g + 4 * m];
#pragma unroll
            for (int j = 0; j < 4; j++) {
              S[m][j].x = fmaf(dl, k[j].x, __fmul_rn(S[m][j].x, eg));
              S[m][j].y = fmaf(dl, k[j].y, __fmul_rn(S[m][j].y, eg));
              S[m][j].z = fmaf(dl, k[j].z, __fmul_rn(S[m][j].z, eg));
              S[m][j].w = fmaf(dl, k[j].w, __fmul_rn(S[m][j].w, eg));
            }
          }
        }
#pragma unroll
        for (int m = 0; m < RM; m++)
#pragma unroll
          for (int j = 0; j < 4; j++) __stcs(reinterpret_cast<float4*>(Sg + m * 512 + j * 32), S[m][j]);
        __syncwarp();  // the delta rule below rewrites dls rows of this warp (row group's lane 0)
      }
    }
    // delta rule: S ← S·e^g; δ = β(v − S k); S ← S + δ kᵀ; o = S q
    for (int i = 0; i < n; i++) {
      float4 k[4], q[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        k[j] = reinterpret_cast<const float4*>(ks[i])[p + 8 * j];
        q[j] = reinterpret_cast<const float4*>(qs[i])[p + 8 * j];
      }
      const float eg = eg_s[i], beta = beta_s[i];
      float d[RM];
#pragma unroll
      for (int m = 0; m < RM; m++) {
#pragma unroll
        for (int j = 0; j < 4; j++) S[m][j].x *= eg, S[m][j].y *= eg, S[m][j].z *= eg, S[m][j].w *= eg;
        d[m] = row_sum8((dot4(S[m][0], k[0]) + dot4(S[m][1], k[1])) + (dot4(S[m][2], k[2]) + dot4(S[m][3], k[3])));
      }
#pragma unroll
      for (int m = 0; m < RM; m++) {
        const float dl = (vs[i][wp * RW + g + 4 * m] - d[m]) * beta;
        if (SPEC && p == 0) dls[i][wp * RW + g + 4 * m] = dl;  // same value in the row's 8 lanes
#pragma unroll
        for (int j = 0; j < 4; j++) {
          S[m][j].x = fmaf(dl, k[j].x, S[m][j].x), S[m][j].y = fmaf(dl, k[j].y, S[m][j].y);
          S[m][j].z = fmaf(dl, k[j].z, S[m][j].z), S[m][j].w = fmaf(dl, k[j].w, S[m][j].w);
        }
        d[m] = row_sum8((dot4(S[m][0], q[0]) + dot4(S[m][1], q[1])) + (dot4(S[m][2], q[2]) + dot4(S[m][3], q[3])));
      }
      if (p == 0) {
#pragma unroll
        for (int m = 0; m < RM; m++) os[i][wp * RW + g + 4 * m] = __float2bfloat16_rn(d[m]);
      }
    }
    __syncthreads();
    for (int i = wp; i < n; i += kWarps) {  // Σ bf16(o)² over this CTA's rows
      float s = 0.f;
      for (int r = lane; r < R; r += 32) {
        const float o = bf(os[i][r]);
        s = fmaf(o, o, s);
      }
      s = warp_sum(s);
      if (lane == 0) part[i] = s;
    }
    // After this barrier the head's partial sums are visible and every CTA of the cluster holds the old conv state.
    // No global stores are in flight here (a release arrive would wait for them), so S is stored after it.
    cl_arrive(), cl_wait();
    if (tid < n) {
      float tot = 0.f;
      for (int j = 0; j < NC; j++) tot += *cl.map_shared_rank(&part[tid], hl * NC + j);
      rstd[tid] = rsqrtf(tot / 128.f + kEps);
    }
    cl_arrive_relaxed();  // done reading remote smem; matched by the wait at the end of the chunk
    __syncthreads();
    const bool last = t0 + n == T;
    if (!SPEC && last) {
#pragma unroll
      for (int m = 0; m < RM; m++)
#pragma unroll
        for (int j = 0; j < 4; j++) __stcs(reinterpret_cast<float4*>(Sg + m * 512 + j * 32), S[m][j]);
    }
    // gated RMSNorm: ((o·rstd)·w)·sigmoid(z)
    for (int e = tid; e < n * R; e += kThreads) {
      const int i = e / R, r = e % R;
      const size_t at = size_t(row0 + t0 + i) * kCore + h * 128 + v0 + r;
      const float o = bf(os[i][r]), z = bf(zs[i][r]);
      core[at] = __float2bfloat16_rn(o * rstd[i] * bf(nws[r]) * (1.f / (1.f + expf(-z))));
      if (o_out) o_out[at] = os[i][r];
    }
    if (!SPEC && last) {  // new conv state = x(T−3 … T−1) = xs rows n … n+2 (q|k by cluster rank 0, v rows by owner)
      for (int i = tid; i < 3 * (CH / 8); i += kThreads) {
        const int s = i / (CH / 8), c = (i % (CH / 8)) * 8;
        if (c >= 256 || cr == 0)
          *reinterpret_cast<uint4*>(st.conv + s * kQKV + chan(c)) = *reinterpret_cast<const uint4*>(&xs[n + s][c]);
      }
    }
    if constexpr (PEND) {  // committed conv state = x(−3 … −1) = xs rows 0 … 2 of the first chunk, same owners
      if (t0 == 0 && np) {
        for (int i = tid; i < 3 * (CH / 8); i += kThreads) {
          const int s = i / (CH / 8), c = (i % (CH / 8)) * 8;
          if (c >= 256 || cr == 0)
            *reinterpret_cast<uint4*>(st.conv + s * kQKV + chan(c)) = *reinterpret_cast<const uint4*>(&xs[s][c]);
        }
      }
    }
    if constexpr (SPEC) {  // record rows t0 … t0+n−1: δ of the CTA's rows, e^g of its v-head (V-slice 0), k of the
      // k-head and x of the q|k channels (cluster rank 0), x of its own v rows — each value written by one CTA
      for (int e = tid; e < n * (R / 4); e += kThreads) {
        const int i = e / (R / 4), c = (e % (R / 4)) * 4;
        *reinterpret_cast<float4*>(rc.dl + (size_t(t0 + i) * 24 + h) * 128 + v0 + c) =
            *reinterpret_cast<const float4*>(&dls[i][c]);
      }
      if (v0 == 0 && tid < n) rc.eg[(t0 + tid) * 24 + h] = eg_s[tid];
      if (cr == 0) {
        for (int e = tid; e < n * 32; e += kThreads)
          reinterpret_cast<float4*>(rc.k + (size_t(t0 + e / 32) * 8 + hk) * 128)[e % 32] =
              reinterpret_cast<const float4*>(ks[e / 32])[e % 32];
      }
      for (int e = tid; e < n * (CH / 8); e += kThreads) {
        const int i = e / (CH / 8), c = (e % (CH / 8)) * 8;
        if (c >= 256 || cr == 0)
          *reinterpret_cast<uint4*>(rc.x + size_t(t0 + i) * kQKV + chan(c)) =
              *reinterpret_cast<const uint4*>(&xs[3 + i][c]);
      }
    }
    if (!last) __syncthreads();  // smem reads of this chunk done before the next chunk overwrites it
    cl_wait();  // peers are done with this CTA's part[] before reuse / exit
  }
}

// gdn_commit: grid (24 · 2, L·B), pair p = blockIdx.y. Block x: v-head x/2, V-half x%2 = S rows [64, 128] fp32; thread
// (warp wp, lane) holds rows wp + 8j (j < 8) at float4 column `lane`, so every load/store covers a whole 512-B row.
// The block stages e^g, δ of its rows and k of the k-head for the n rows in smem, then replays row after row
// S ← fma(δ_t[r], k_t[c], S·e^g_t): the kernel's two roundings in its order (__fmul_rn keeps the product a separate
// rounding, as `S *= eg` is in gdn_core_kernel). Blocks x < 3 also shift the conv state, one 16-B column chunk per
// thread (all three rows read before any is written: in place). n = 0: the block exits without touching memory.
constexpr int kCmRows = 64, kCmJ = kCmRows / kWarps;
__global__ void __launch_bounds__(kThreads) gdn_commit_kernel(GdnState* const* __restrict__ states,
                                                              GdnRec* const* __restrict__ recs,
                                                              const int* __restrict__ nacc, int B) {
  __shared__ __align__(16) float egs[kGdnRecRows], dls[kGdnRecRows][kCmRows], kss[kGdnRecRows][128];
  const int pr = blockIdx.y, n = nacc[pr % B], tid = threadIdx.x, wp = tid >> 5, lane = tid & 31;
  if (n <= 0) return;
  if (n > kGdnRecRows) rec_overflow("gdn_commit", pr, n, tid == 0 && blockIdx.x == 0 && pr < B);
  const GdnState st = *states[pr];
  const GdnRec rc = *recs[pr];
  const int h = blockIdx.x >> 1, r0 = (blockIdx.x & 1) * kCmRows;
  float* Sg = st.S + (size_t(h) * 128 + r0 + wp) * 128 + lane * 4;
  float4 S[kCmJ];
#pragma unroll
  for (int j = 0; j < kCmJ; j++) S[j] = __ldcs(reinterpret_cast<const float4*>(Sg + j * kWarps * 128));
  constexpr int DC = kCmRows / 4, RC = DC + 32;  // float4 per row: δ | k
  const float4* dl0 = reinterpret_cast<const float4*>(rc.dl + h * 128 + r0);  // row t at + t·24·128 floats
  const float4* k0 = reinterpret_cast<const float4*>(rc.k + (h / 3) * 128);   // row t at + t·8·128 floats
  for (int e = tid; e < n * RC; e += kThreads) {
    const int t = e / RC, c = e % RC;
    if (c < DC) reinterpret_cast<float4*>(dls[t])[c] = dl0[t * 24 * 32 + c];
    else reinterpret_cast<float4*>(kss[t])[c - DC] = k0[t * 8 * 32 + c - DC];
  }
  if (tid < n) egs[tid] = rc.eg[tid * 24 + h];
  if (blockIdx.x < 3) {
    const int c = (blockIdx.x * kThreads + tid) * 8;
    if (c < kQKV) {
      uint4 x[3];
#pragma unroll
      for (int s = 0; s < 3; s++) {
        const bf16* src = s + n < 3 ? st.conv + (s + n) * kQKV : rc.x + (s + n - 3) * kQKV;
        x[s] = *reinterpret_cast<const uint4*>(src + c);
      }
#pragma unroll
      for (int s = 0; s < 3; s++) *reinterpret_cast<uint4*>(st.conv + s * kQKV + c) = x[s];
    }
  }
  __syncthreads();
  for (int t = 0; t < n; t++) {
    const float4 k = reinterpret_cast<const float4*>(kss[t])[lane];
    const float eg = egs[t];
#pragma unroll
    for (int j = 0; j < kCmJ; j++) {
      const float dl = dls[t][wp + j * kWarps];
      S[j].x = fmaf(dl, k.x, __fmul_rn(S[j].x, eg)), S[j].y = fmaf(dl, k.y, __fmul_rn(S[j].y, eg));
      S[j].z = fmaf(dl, k.z, __fmul_rn(S[j].z, eg)), S[j].w = fmaf(dl, k.w, __fmul_rn(S[j].w, eg));
    }
  }
#pragma unroll
  for (int j = 0; j < kCmJ; j++) __stcs(reinterpret_cast<float4*>(Sg + j * kWarps * 128), S[j]);
}

template <bool SPEC, bool PEND = false>
void launch_core(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, bf16* o, GdnState* const* states,
                 GdnRec* const* recs, const int* seq_start, int B, cudaStream_t s, GdnRec* const* pend = nullptr,
                 const int* npend = nullptr) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(8 * 3 * kNC, B);
  cfg.blockDim = dim3(kThreads);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 3 * kNC, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if constexpr (PEND) {  // P020: pending k | e^g in dynamic smem (static 45.4 KiB + 8.25 KiB > 48 KiB: opt-in once)
    static const bool attr = [] {
      CUDA_CHECK(cudaFuncSetAttribute(gdn_core_kernel<kNC, SPEC, PEND>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      kPendSmem));
      return true;
    }();
    (void)attr;
    cfg.dynamicSmemBytes = kPendSmem;
  }
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, gdn_core_kernel<kNC, SPEC, PEND>, qkvz, ba, core, o, states, seq_start, w, recs,
                                pend, npend));
}

#include "kernels/gdn/p005_gdn_fast_prefill.cuh"  // P005 gdn_fast_prefill: warp-specialised long-T variant

// P005: rows per sequence (M / B) from which the patch kernel is used (crossover on GB10: B=1 ~10 rows, B=2…8 ~32)
int fast_min_rows = 32;

float bf16_host(const uint8_t* p) {
  uint16_t u;
  std::memcpy(&u, p, 2);
  uint32_t x = uint32_t(u) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

std::vector<float> to_f32(const Weights& W, const std::string& name, const std::vector<uint8_t>& b) {
  const std::string& dt = W.info(name).dtype;
  std::vector<float> f;
  if (dt == "BF16") {
    for (size_t i = 0; i < b.size(); i += 2) f.push_back(bf16_host(b.data() + i));
  } else if (dt == "F32") {
    f.resize(b.size() / 4);
    std::memcpy(f.data(), b.data(), b.size());
  } else {
    qlog::fail("gdn", "%s dtype %s", name.c_str(), dt.c_str());
  }
  return f;
}

}  // namespace

GdnW gdn_upload(const Weights& W, int layer) {
  gdn_p005_prepare();
  const std::string p = "model.language_model.layers." + std::to_string(layer) + ".linear_attn.";
  auto get = [&](const std::string& n, size_t numel, bool bf16_only) {
    std::vector<int64_t> sh;
    auto b = W.slice(p + n, &sh);
    size_t e = 1;
    for (auto d : sh) e *= size_t(d);
    if (e != numel || (bf16_only && W.info(p + n).dtype != "BF16"))
      qlog::fail("gdn", "%s%s: %zu elements, dtype %s", p.c_str(), n.c_str(), e, W.info(p + n).dtype.c_str());
    return b;
  };
  const auto cw = get("conv1d.weight", 5120 * 4, true), nw = get("norm.weight", 128, true);
  const auto al = to_f32(W, p + "A_log", get("A_log", 24, false));
  const auto dt = to_f32(W, p + "dt_bias", get("dt_bias", 24, false));
  std::vector<float> A(24);
  for (int h = 0; h < 24; h++) A[h] = -std::exp(al[h]);
  // one block: conv_w [5120,4] bf16 | norm_w [128] bf16 | A [24] f32 | dt_bias [24] f32
  const size_t oc = 0, on = cw.size(), oa = on + nw.size(), od = oa + 96, tot = od + 96;
  uint8_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, tot));
  std::vector<uint8_t> h(tot);
  std::memcpy(h.data() + oc, cw.data(), cw.size());
  std::memcpy(h.data() + on, nw.data(), nw.size());
  std::memcpy(h.data() + oa, A.data(), 96);
  std::memcpy(h.data() + od, dt.data(), 96);
  CUDA_CHECK(cudaMemcpy(d, h.data(), tot, cudaMemcpyHostToDevice));
  GdnW g;
  g.conv_w = reinterpret_cast<const bf16*>(d + oc);
  g.norm_w = reinterpret_cast<const bf16*>(d + on);
  g.A = reinterpret_cast<const float*>(d + oa);
  g.dt_bias = reinterpret_cast<const float*>(d + od);
  return g;
}

void gdn_core_tap(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, bf16* o, GdnState* const* states,
                  const int* seq_start, int B, int M, cudaStream_t s) {
  if (B <= 0) return;
  if (M >= B * fast_min_rows && patch::on(patch::gdn_fast_prefill)) {
    gdn_core_p005(w, qkvz, ba, core, o, states, seq_start, B, s);
    return;
  }
  launch_core<false>(w, qkvz, ba, core, o, states, nullptr, seq_start, B, s);
}

size_t gdn_rec_bytes() {
  return size_t(kGdnRecRows) * ((24 + 24 * 128 + 8 * 128) * sizeof(float) + kQKV * sizeof(bf16));
}

GdnRec gdn_rec_at(void* base) {  // eg | dl | k | x; every offset a multiple of 16 B
  GdnRec r;
  r.eg = static_cast<float*>(base);
  r.dl = r.eg + kGdnRecRows * 24;
  r.k = r.dl + kGdnRecRows * 24 * 128;
  r.x = reinterpret_cast<bf16*>(r.k + kGdnRecRows * 8 * 128);
  return r;
}

void gdn_core_spec(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, GdnState* const* states,
                   GdnRec* const* recs, const int* seq_start, int B, int M, cudaStream_t s) {
  if (B <= 0) return;
  if (M > B * kGdnRecRows) qlog::fail("gdn", "gdn_core_spec: %d rows > %d sequences × %d", M, B, kGdnRecRows);
  launch_core<true>(w, qkvz, ba, core, nullptr, states, recs, seq_start, B, s);
}

void gdn_core_spec_pending(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, GdnState* const* states,
                           GdnRec* const* recs, GdnRec* const* pend, const int* npend, const int* seq_start, int B,
                           int M, cudaStream_t s) {
  if (B <= 0) return;
  if (M > B * kGdnRecRows) qlog::fail("gdn", "gdn_core_spec_pending: %d rows > %d sequences × %d", M, B, kGdnRecRows);
  launch_core<true, true>(w, qkvz, ba, core, nullptr, states, recs, seq_start, B, s, pend, npend);
}

void gdn_commit(GdnState* const* states, GdnRec* const* recs, const int* n, int L, int B, cudaStream_t s) {
  if (L <= 0 || B <= 0) return;
  gdn_commit_kernel<<<dim3(24 * (128 / kCmRows), L * B), kThreads, 0, s>>>(states, recs, n, B);
  CUDA_CHECK(cudaGetLastError());
}

int gdn_set_fast_min_rows(int rows) {
  const int old = fast_min_rows;
  fast_min_rows = rows;
  return old;
}

void gdn_core(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, GdnState* const* states,
              const int* seq_start, int B, int M, cudaStream_t s) {
  gdn_core_tap(w, qkvz, ba, core, nullptr, states, seq_start, B, M, s);
}

}  // namespace q38
