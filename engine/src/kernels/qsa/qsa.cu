// qsa.cu — QSA full-attention core (spec §2.5) with the numerics of vLLM's qsa_prepare / qsa_indexer / qsa attention.
//   k_prepare : per row — main Q/K Gemma RMSNorm + NeoX RoPE (fp32 rotate of bf16 normed values), K‖V → cache,
//               indexer q (bf16 RoPE as Triton compiles it), raw ik → cache, compressed key when (p+1)%4 == 0
//               (bf16 mean of 4 raw keys → norm → RoPE at p−3); groups completing inside the call use its own rows
// Decode path (M ≤ kDecRows = 64):
//   k_score   : per sequence, compressed keys streamed once for all its rows: score = Σ_h ReLU(q_h·k̂) — one tensor-core
//               chain per (key, q-vector) (idx_dot, qsa_dev.cuh; the prefill GEMM runs the same chain) + a
//               bf16-granular histogram per row; only for rows that see > 512 blocks
//   k_select  : per row — exact top-512 blocks (score desc, ties → lower block), listed ascending + open tail, in
//               vLLM's packed format [2051 tokens | count]; ≤ 512 visible blocks ⇒ dense 0..p. Base tier (calls that
//               see ≤ 65,536 blocks): static smem bitsets; wide tier (up to 262,144 blocks = 1M positions): the same
//               code on bitsets in dynamic smem sized for the call (qsa_dev.cuh kMaxBlk)
//   k_attn    : per (row, 32-token tile) — tensor-core S = Q·Kᵀ, ·(1/16)·log2e, tile max, P = exp2(S−m) → bf16,
//               O = P·V (fp32) → normalized partial + LSE (vLLM decode split-K layout: BLOCK_N 32, one tile per split)
//   k_merge   : per (row, head) — LSE merge in tile order, o → bf16, out = bf16(o·sigmoid(gate))
// P021 qsa_decode_attn (p021_qsa_decode_attn.cuh) replaces k_attn + k_merge for calls with more work (rows × tiles ≥
//   280 + 6·rows and some sequence with ≥ 2 rows, e.g. MTP verify at long context): one kernel, a 4-CTA cluster per
//   row (CTA = tile group), partials on chip.
// Prefill path (M > 64, qsa_prefill.cu): k_pscore (indexer GEMM) → k_select<true> per 1,024-row sub-chunk, then
//   k_pattn (k_attn's tile math + k_merge's merge per row, on chip). P009 qsa_fast_index (p009_qsa_fast_index.cuh)
//   replaces the indexer part for calls of ≥ 512 rows: per-row threshold from sampled keys, GEMM keeping only scores
//   ≥ it, exact top-512 over those.
// Every value a row produces depends only on that row and the caches (fixed reduction orders, no float atomics), so
// results are bitwise independent of M, B, the path and of how a sequence's rows are split over calls. Kernels are
// launched with programmatic dependent launch (each waits for its predecessor in griddepcontrol.wait).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "kernels/qsa/qsa_dev.cuh"
#include "model/patches.h"

namespace q38 {
namespace {
using namespace qsa_dev;

// RoPE (qsa.h QsaRope; config rope_parameters: θ 1e7, rotary 64). Default YaRN factor 1.5625, correction range from
// 262,144 (HF original_max_position_embeddings; vLLM: max(original, max_model_len)) → lo 14, hi 22, mscale
// 0.1·ln(1.5625)+1 folded into cos/sin.
constexpr float kTheta = 1e7f;

struct RopeK {  // table parameters (host: rope_k)
  int rows, plain;           // plain: vLLM RotaryEmbedding (no YaRN terms, no mscale)
  float lo, rinv;            // YaRN ramp: mask = 1 − clamp((f − lo)·rinv, 0, 1), rinv = fp32 1/(hi − lo)
  float factor, mscale;
};

// ---- RoPE table (vLLM RotaryEmbedding / YaRNScalingRotaryEmbedding, torch fp32 op by op on the GPU, then bf16) -----
// torch: pos_freqs = θ ** (arange(0, 64, 2) / 64); 1.0 / x = reciprocal (IEEE division); tensor ÷ Python scalar on the
// GPU = × its fp32 reciprocal; tensor × Python scalar = × fp32 scalar; t·inv_freq (einsum outer product) one product.
__global__ void k_rope_table(bf16* tab, RopeK k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= k.rows * 32) return;
  const int p = i >> 5, f = i & 31;
  const float pf = powf(kTheta, __fmul_rn(float(2 * f), 1.f / 64.f));
  const float extra = __fdiv_rn(1.f, pf);
  float inv = extra;
  if (!k.plain) {
    const float inter = __fdiv_rn(1.f, __fmul_rn(k.factor, pf));
    const float mask = 1.f - fminf(fmaxf(__fmul_rn(float(f) - k.lo, k.rinv), 0.f), 1.f);
    inv = __fadd_rn(__fmul_rn(inter, 1.f - mask), __fmul_rn(extra, mask));
  }
  const float fr = __fmul_rn(float(p), inv);
  float c = cosf(fr), sn = sinf(fr);
  if (!k.plain) c = __fmul_rn(c, k.mscale), sn = __fmul_rn(sn, k.mscale);
  tab[size_t(p) * 64 + f] = __float2bfloat16_rn(c);
  tab[size_t(p) * 64 + 32 + f] = __float2bfloat16_rn(sn);
}

// ---- prepare -----------------------------------------------------------------------------------------------------
// mRoPE (config mrope_section [11, 11, 10], interleaved; vLLM MRotaryEmbedding / HF Qwen3-VL): rotary frequency f
// takes its angle from the h position if f % 3 == 1, from w if f % 3 == 2 and f < 30, else from t. A row's cos/sin
// is gathered from the table rows of its (t, h, w) positions; t == h == w (text) is the plain table row, same values.
struct RopeAt {
  int t, h, w;
};
__device__ __forceinline__ int rope_row(const RopeAt& ra, int f) {
  const int m = f % 3;
  return m == 1 ? ra.h : (m == 2 && f < 30) ? ra.w : ra.t;
}
// cos and sin of frequencies f0 … f0 + N − 1 (bf16, as stored)
template <int N>
__device__ __forceinline__ void cs_load(const bf16* tab, const RopeAt& ra, int f0, bf16* c, bf16* s) {
  if (ra.t == ra.h && ra.t == ra.w) {  // one row: vector loads (f0 is a multiple of N)
    const bf16* row = tab + size_t(ra.t) * 64;
    if constexpr (N == 8) {
      const uint4 uc = *reinterpret_cast<const uint4*>(row + f0), us = *reinterpret_cast<const uint4*>(row + 32 + f0);
      memcpy(c, &uc, 16), memcpy(s, &us, 16);
    } else {
      const uint2 uc = *reinterpret_cast<const uint2*>(row + f0), us = *reinterpret_cast<const uint2*>(row + 32 + f0);
      memcpy(c, &uc, 8), memcpy(s, &us, 8);
    }
  } else {
#pragma unroll
    for (int i = 0; i < N; i++) {
      const bf16* row = tab + size_t(rope_row(ra, f0 + i)) * 64;
      c[i] = row[f0 + i], s[i] = row[32 + f0 + i];
    }
  }
}
// Main head (256): lane holds dims [8l, 8l+8). Rotated dims 0..63: x1 = lanes 0..3, x2 = lanes 4..7 (partner xor 4).
__device__ __forceinline__ void norm_rope_main(const bf16* src, const bf16* w, const bf16* tab, const RopeAt& ra,
                                               bf16* dst, int lane) {
  float x[8], wv[8], y[8], o[8];
  ld8(src + lane * 8, x);
  ld8(w + lane * 8, wv);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(x[i], x[i], ss);
  const float inv = rsqrtf(warp_sum(ss) * (1.f / 256.f) + kEps);
#pragma unroll
  for (int i = 0; i < 8; i++) y[i] = bfr(x[i] * inv * (wv[i] + 1.f));
  float py[8];
#pragma unroll
  for (int i = 0; i < 8; i++) py[i] = __shfl_xor_sync(kFull, y[i], 4);
  if (lane < 8) {
    float c[8], s[8];
    bf16 cb[8], sb[8];
    cs_load<8>(tab, ra, (lane & 3) * 8, cb, sb);
#pragma unroll
    for (int i = 0; i < 8; i++) c[i] = __bfloat162float(cb[i]), s[i] = __bfloat162float(sb[i]);
    // Triton contracts x1·c − x2·s → fma(x1, c, −x2·s) and x2·c + x1·s → fma(x2, c, x1·s)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float t = __fmul_rn(py[i], s[i]);
      o[i] = __fmaf_rn(y[i], c[i], lane < 4 ? -t : t);
    }
    st8(dst + lane * 8, o);
  } else {
    st8(dst + lane * 8, y);
  }
}
// Indexer head (128): lane holds dims [4l, 4l+4) in x (fp32). Normed values rounded to bf16, RoPE in bf16 arithmetic
// as Triton 3.7 compiles vLLM's _norm_rope on sm_121 (checked bitwise against that kernel): o1 = fma_bf16(x1, c,
// −bf16(x2·s)); o2 = fma_bf16(x2, c, bf16(x1·s)) for the query tiles but fma_bf16(x1, s, bf16(x2·c)) for the 1-row
// compressed-key tile (the contraction picks the other product). Rotated: x1 = lanes 0..7, x2 = lanes 8..15.
template <bool kComp>
__device__ __forceinline__ void norm_rope_idx(const float* x, const bf16* w, const bf16* tab, const RopeAt& ra,
                                              bf16* dst, int lane) {
  float wv[4], y[4], py[4];
  ld4(w + lane * 4, wv);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) ss = fmaf(x[i], x[i], ss);
  const float inv = rsqrtf(warp_sum(ss) * (1.f / 128.f) + kEps);
#pragma unroll
  for (int i = 0; i < 4; i++) y[i] = bfr(x[i] * inv * (wv[i] + 1.f));
#pragma unroll
  for (int i = 0; i < 4; i++) py[i] = __shfl_xor_sync(kFull, y[i], 8);
  if (lane < 16) {
    bf16 c[4], s[4];
    cs_load<4>(tab, ra, (lane & 7) * 4, c, s);
    bf16 o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const bf16 a = __float2bfloat16_rn(y[i]), b = __float2bfloat16_rn(py[i]);
      if (lane < 8) o[i] = __hfma(a, c[i], __hneg(__hmul_rn(b, s[i])));
      else o[i] = kComp ? __hfma(b, s[i], __hmul_rn(a, c[i])) : __hfma(a, c[i], __hmul_rn(b, s[i]));
    }
    *reinterpret_cast<uint2*>(dst + lane * 4) = *reinterpret_cast<const uint2*>(o);
  } else {
    st4(dst + lane * 4, y);
  }
}

struct PrepArgs {
  const bf16 *qkv, *rope, *q_norm, *k_norm, *iq_norm, *ik_norm;
  const int *pos, *seq_start;
  const int* rp;  // [M, 6] RoPE positions (t, h, w of the row, t, h, w of position pos − 3), or null: the row positions
  QsaCache* const* caches;
  int B;
  bf16 *q, *qi;
  int* hist;
};

// grid M, 18 warps: 0..11 main Q heads, 12 K (+V copy), 13..16 indexer q heads, 17 raw ik (+ compressed key).
__global__ void __launch_bounds__(576) k_prepare(PrepArgs a) {
  pdl_wait();
  const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int b = seq_of(a.seq_start, a.B, r), p = a.pos[r];
  const QsaCache* c = a.caches[b];
  const bf16* row = a.qkv + size_t(r) * kQkv;
  const int* rr = a.rp ? a.rp + size_t(r) * 6 : nullptr;
  const RopeAt ra = rr ? RopeAt{rr[0], rr[1], rr[2]} : RopeAt{p, p, p};
  if (a.hist && threadIdx.x < kBins / 4)  // score histogram of this row (decode k_score)
    reinterpret_cast<int4*>(a.hist + size_t(r) * kBins)[threadIdx.x] = make_int4(0, 0, 0, 0);
  if (warp < kQH) {
    norm_rope_main(row + warp * 512, a.q_norm, a.rope, ra, a.q + (size_t(r) * kQH + warp) * 256, lane);
  } else if (warp == kQH) {
    bf16* kv = c->kv + size_t(p) * 512;
    norm_rope_main(row + kOffK, a.k_norm, a.rope, ra, kv, lane);
    *reinterpret_cast<uint4*>(kv + 256 + lane * 8) = *reinterpret_cast<const uint4*>(row + kOffV + lane * 8);
  } else if (warp < kQH + 5) {
    const int h = warp - kQH - 1;
    float x[4];
    ld4(row + kOffIq + h * 128 + lane * 4, x);
    norm_rope_idx<false>(x, a.iq_norm, a.rope, ra, a.qi + (size_t(r) * kIdxHeads + h) * 128, lane);
  } else {
    *reinterpret_cast<uint2*>(c->ik_raw + size_t(p) * 128 + lane * 4) =
        *reinterpret_cast<const uint2*>(row + kOffIk + lane * 4);
    if ((p & 3) == 3) {  // group p/4 complete: members p-3..p from this call's rows or (older) from the cache
      const int p0 = a.pos[a.seq_start[b]];
      float acc[4] = {0.f, 0.f, 0.f, 0.f}, v[4];
      for (int q = p - 3; q <= p; q++) {
        const bf16* src = q >= p0 ? a.qkv + size_t(r - (p - q)) * kQkv + kOffIk : c->ik_raw + size_t(q) * 128;
        ld4(src + lane * 4, v);
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i] += v[i];
      }
#pragma unroll
      for (int i = 0; i < 4; i++) acc[i] = bfr(acc[i] * 0.25f);
      const RopeAt g = rr ? RopeAt{rr[3], rr[4], rr[5]} : RopeAt{p - 3, p - 3, p - 3};  // the group's first position
      norm_rope_idx<true>(acc, a.ik_norm, a.rope, g, c->ik_comp + size_t(p >> 2) * 128, lane);
    }
  }
}

// ---- indexer scores (decode path) ----------------------------------------------------------------------------------
// grid (G, B), 8 warps: warp batches of 16 compressed keys (A fragments loaded directly, coalesced 64-B row pieces),
// batches interleaved over all warps of sequence b; rows in groups of 8 (32 q-vectors, B fragments in registers). For
// each row of b that sees > 512 blocks: score = Σ_h ReLU(q_h·k̂) by idx_dot/idx_relu_sum (qsa_dev.cuh — the same chain
// as the prefill GEMM) + bf16-granular histogram.
constexpr int kSW = 8;   // warps per CTA
constexpr int kQG = 8;   // query rows per register group (32 q-vectors = 4 n-tiles)

struct ScoreArgs {
  const bf16* qi;
  const int *pos, *seq_start;
  QsaCache* const* caches;
  float* score;
  int* hist;
  int stride;  // score row stride of the wide tier (base tier: kMaxBlk)
};

template <bool kWide>
__global__ void __launch_bounds__(kSW * 32) k_score(ScoreArgs a) {
  const size_t stride = kWide ? a.stride : kMaxBlk;
  pdl_wait();
  const int b = blockIdx.y, s0 = a.seq_start[b], s1 = a.seq_start[b + 1];
  const int nvis = (a.pos[s1 - 1] + 1) >> 2;
  if (nvis <= kBudget) return;
  const int rs = s0 + max(0, kSelCols - a.pos[s0]);  // first row with pos ≥ 2051
  const bf16* comp = a.caches[b]->ik_comp;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int nbat = (nvis + 15) >> 4, wid = blockIdx.x * kSW + warp, nwarp = gridDim.x * kSW;
  for (int g0 = rs; g0 < s1; g0 += kQG) {
    const int ng = min(kQG, s1 - g0), ntile = (ng + 1) >> 1;
    uint4 qf[kQG / 2][4];  // n-tile nt: q-vector 8nt + g = row g0 + 2nt + (g>>2), head g&3
#pragma unroll
    for (int nt = 0; nt < kQG / 2; nt++) {
      const int rr = 2 * nt + (g >> 2);
      idx_load(qf[nt], a.qi + (size_t(g0 + rr) * 4 + (g & 3)) * 128, rr < ng, c);
    }
    for (int bt = wid; bt < nbat; bt += nwarp) {
      const int j0 = bt * 16;
      uint4 ka[4], kb[4];
      idx_load(ka, comp + size_t(j0 + g) * 128, j0 + g < nvis, c);
      idx_load(kb, comp + size_t(j0 + g + 8) * 128, j0 + g + 8 < nvis, c);
#pragma unroll
      for (int nt = 0; nt < kQG / 2; nt++) {
        if (nt >= ntile) break;
        float acc[4];
        idx_dot(acc, ka, kb, qf[nt]);
        const float2 sc = idx_relu_sum(acc);
        const int r = g0 + nt * 2 + ((lane >> 1) & 1);
        if ((lane & 1) == 0 && r < s1) {
          const int vis = (a.pos[r] + 1) >> 2, j = j0 + g;
          if (j < vis) {
            a.score[size_t(r) * stride + j] = sc.x;
            atomicAdd(a.hist + size_t(r) * kBins + score_bin(sc.x), 1);
          }
          if (j + 8 < vis) {
            a.score[size_t(r) * stride + j + 8] = sc.y;
            atomicAdd(a.hist + size_t(r) * kBins + score_bin(sc.y), 1);
          }
        }
      }
    }
  }
}

// ---- top-512 selection + expand ------------------------------------------------------------------------------------
// One CTA (1024 threads) per row r = r0 + blockIdx.x; its scores at score + blockIdx.x·stride. Bf16-granular
// histogram (bin = top 16 bits, 2048 bins over [2^-4, 2^12); the end bins catch under/overflow): counted by k_score
// (decode, kOwnHist = false) or here from the scores (prefill, smem atomics). Then: the bin holding the 512th score,
// one sweep that marks every block in a higher bin (ballot → bitmask) and collects the threshold bin's scores, an
// exact rank inside that bin (score desc, block asc — ties go to the lower block), then an ordered popcount scan
// writes the selected blocks ascending. Fallback (threshold in an end bin or > kCand candidates): exact 11/11/10-bit
// radix select streaming the scores. Base tier: SelSmem (static, bitsets for kMaxBlk blocks); wide tier: SelHead +
// gt [W] + eq [W] in dynamic smem (W words for the call's blocks) — the same code on the bitset pointers.
constexpr int kSelT = 1024, kSelW32 = kSelT / 32, kCand = 2048;

__device__ int block_excl_scan(int v, int* sh) {  // kSelT threads; sh ≥ 32 ints
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(kFull, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) sh[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int y = lane < kSelW32 ? sh[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int z = __shfl_up_sync(kFull, y, o);
      if (lane >= o) y += z;
    }
    sh[lane] = y;
  }
  __syncthreads();
  const int res = x - v + (warp ? sh[warp - 1] : 0);
  __syncthreads();
  return res;
}

struct SelSmem {
  int hist[kBins];
  uint32_t gt[kMaxBlk / 32], eq[kMaxBlk / 32];
  uint32_t ck[kCand];
  int ci[kCand];
  int sh[32];
  int digit, krem, nc;
};
struct SelHead {  // wide tier: SelSmem without the bitsets (gt [W], eq [W] follow it)
  int hist[kBins];
  uint32_t ck[kCand];
  int ci[kCand];
  int sh[32];
  int digit, krem, nc;
};
size_t sel_smem(int W) { return sizeof(SelHead) + sizeof(uint32_t) * 2 * size_t(W); }
static_assert(sizeof(SelHead) + 8 * (kMaxBlkW / 32) <= 99 * 1024, "wide selection smem ≤ 99 KB (sm_121 opt-in)");

// hist: bins in increasing key order. Finds the bin holding the krem-th largest → S.digit, S.krem (rank inside).
template <class SM>
__device__ void find_bin(const int* hist, int nb, int krem, SM& S) {
  const int per = nb / kSelT, o0 = threadIdx.x * per;  // thread owns bins nb−1−o0 … downwards
  int loc = 0;
  for (int k = 0; k < per; k++) loc += hist[nb - 1 - o0 - k];
  const int before = block_excl_scan(loc, S.sh);
  if (before < krem && krem <= before + loc) {
    int cum = before;
    for (int k = 0; k < per; k++) {
      const int b = nb - 1 - o0 - k, h = hist[b];
      if (cum + h >= krem) {
        S.digit = b, S.krem = krem - cum;
        break;
      }
      cum += h;
    }
  }
  __syncthreads();
}

// Fallback: exact radix select of the 512th largest key (bits + 1; 0 = padding), then u > T and the lowest-index
// u == T into gt.
template <class SM>
__device__ void select_radix(const float* sc, int vis, SM& S, uint32_t* gt, uint32_t* eq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nk = (vis + kSelT - 1) / kSelT;
  auto key = [&](int k) -> uint32_t {
    const int i = k * kSelT + tid;
    return i < vis ? __float_as_uint(sc[i]) + 1u : 0u;
  };
  uint32_t prefix = 0, mask = 0;
  int krem = kBudget;
  for (int pass = 0; pass < 3; pass++) {
    const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0, nb = pass == 2 ? 1024 : 2048;
    __syncthreads();
    for (int i = tid; i < nb; i += kSelT) S.hist[i] = 0;
    __syncthreads();
    for (int k = 0; k < nk; k++) {
      const uint32_t u = key(k);
      if (u && (u & mask) == prefix) atomicAdd(&S.hist[(u >> shift) & uint32_t(nb - 1)], 1);
    }
    __syncthreads();
    find_bin(S.hist, nb, krem, S);
    prefix |= uint32_t(S.digit) << shift;
    mask |= uint32_t(nb - 1) << shift;
    krem = S.krem;
  }
  const int nw = (vis + 31) / 32;
  for (int k = 0; k < nk; k++) {  // word k·32 + warp covers blocks [32·word, +32): lane = bit
    const uint32_t u = key(k);
    const uint32_t g = __ballot_sync(kFull, u > prefix), e = __ballot_sync(kFull, u == prefix);
    if (lane == 0 && k * kSelW32 + warp < nw) gt[k * kSelW32 + warp] = g, eq[k * kSelW32 + warp] = e;
  }
  __syncthreads();
  const int wpt = (nw + kSelT - 1) / kSelT, w0 = min(nw, tid * wpt), w1 = min(nw, w0 + wpt);
  int ce = 0;
  for (int w = w0; w < w1; w++) ce += __popc(eq[w]);
  int take = max(0, min(ce, krem - block_excl_scan(ce, S.sh)));
  for (int w = w0; w < w1; w++) {
    uint32_t e = eq[w], g = gt[w];
    for (; take > 0 && e; take--) g |= e & (0u - e), e &= e - 1;  // lowest-index ties
    gt[w] = g;
  }
  __syncthreads();
}

// k_select's body: bitsets gt / eq of W words, score rows `stride` floats apart.
template <bool kOwnHist, class SM>
__device__ __forceinline__ void select_row(SM& S, uint32_t* gt, uint32_t* eq, int W, size_t stride, const int* pos,
                                           const float* __restrict__ score, const int* __restrict__ hist, int* sel,
                                           int r0) {
  constexpr int kWB = 16;  // words per warp per sweep batch (loads in flight)
  pdl_wait();
  const int r = r0 + blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int p = pos[r], vis = (p + 1) >> 2, tail0 = vis * kIdxRatio, ntail = p + 1 - tail0;
  int* out = sel + size_t(r) * kSelW;
  if (vis <= kBudget) {  // dense: tokens 0..p
    for (int i = tid; i < kSelCols; i += kSelT) out[i] = i <= p ? i : -1;
    if (tid == 0) out[kSelCols] = p + 1;
    return;
  }
  const float* sc = score + size_t(blockIdx.x) * stride;
  const int nw = (vis + 31) / 32;
  if (nw > W) __trap();  // the call was sized for fewer blocks (max_pos hint below a row's position)
  float v[kWB];
  auto fetch = [&](int w0) {  // words w0 + k·(warps) (k < kWB) of this warp
#pragma unroll
    for (int k = 0; k < kWB; k++) {
      const int i = (w0 + k * kSelW32) * 32 + lane;
      v[k] = i < vis ? __ldg(sc + i) : -1.f;
    }
  };
  if constexpr (kOwnHist) {
    for (int i = tid; i < kBins; i += kSelT) S.hist[i] = 0;
    __syncthreads();
    for (int i0 = 0; i0 < vis; i0 += kSelT * 8) {
      float x[8];
#pragma unroll
      for (int k = 0; k < 8; k++) x[k] = i0 + k * kSelT + tid < vis ? __ldg(sc + i0 + k * kSelT + tid) : -1.f;
#pragma unroll
      for (int k = 0; k < 8; k++)
        if (x[k] >= 0.f) atomicAdd(&S.hist[score_bin(x[k])], 1);
    }
    fetch(warp);
  } else {
    fetch(warp);  // first batch in flight while the threshold bin is found
    for (int i = tid; i < kBins; i += kSelT) S.hist[i] = hist[size_t(blockIdx.x) * kBins + i];
  }
  if (tid == 0) S.nc = 0;
  __syncthreads();
  find_bin(S.hist, kBins, kBudget, S);
  const int bt = S.digit, krem = S.krem;
  bool fb = bt == 0 || bt == kBins - 1;  // under/overflow bin: not exact
  if (!fb) {
    for (int w0 = warp;;) {
#pragma unroll
      for (int k = 0; k < kWB; k++) {
        const int w = w0 + k * kSelW32, i = w * 32 + lane, b = v[k] < 0.f ? -1 : score_bin(v[k]);
        const uint32_t g = __ballot_sync(kFull, b > bt);
        if (lane == 0 && w < nw) gt[w] = g;
        if (b == bt) {
          const int c = atomicAdd(&S.nc, 1);
          if (c < kCand) S.ck[c] = __float_as_uint(v[k]), S.ci[c] = i;
        }
      }
      w0 += kSelW32 * kWB;
      if (w0 >= nw) break;
      fetch(w0);
    }
    __syncthreads();
    const int nc = S.nc;
    if (nc <= kCand) {
      for (int c = warp; c < nc; c += kSelW32) {  // rank by (score desc, block asc): one warp per candidate
        const uint32_t u = S.ck[c];
        const int ix = S.ci[c];
        int rank = 0;
        for (int d = lane; d < nc; d += 32) rank += S.ck[d] > u || (S.ck[d] == u && S.ci[d] < ix);
        rank = __reduce_add_sync(kFull, rank);
        if (lane == 0 && rank < krem) atomicOr(&gt[ix >> 5], 1u << (ix & 31));
      }
      __syncthreads();
    } else {
      fb = true;
    }
  }
  if (fb) select_radix(sc, vis, S, gt, eq);
  const int wpt = (nw + kSelT - 1) / kSelT, w0 = min(nw, tid * wpt), w1 = min(nw, w0 + wpt);
  int cnt = 0;
  for (int w = w0; w < w1; w++) cnt += __popc(gt[w]);
  int off = block_excl_scan(cnt, S.sh);
  for (int w = w0; w < w1; w++) {
    for (uint32_t g = gt[w]; g; g &= g - 1) {
      const int t0 = (w * 32 + __ffs(g) - 1) * kIdxRatio;
      *reinterpret_cast<int4*>(out + off * kIdxRatio) = make_int4(t0, t0 + 1, t0 + 2, t0 + 3);
      off++;
    }
  }
  if (tid < kIdxRatio - 1) out[kBudget * kIdxRatio + tid] = tid < ntail ? tail0 + tid : -1;
  if (tid == 0) out[kSelCols] = kBudget * kIdxRatio + ntail;
}

// stride, W: wide tier only (score row stride, bitset words; dynamic smem sel_smem(W)).
template <bool kOwnHist, bool kWide>
__global__ void __launch_bounds__(kSelT, 1) k_select(const int* pos, const float* __restrict__ score,
                                                     const int* __restrict__ hist, int* sel, int r0, int stride, int W) {
  if constexpr (kWide) {
    extern __shared__ __align__(16) unsigned char sel_sm[];
    SelHead& S = *reinterpret_cast<SelHead*>(sel_sm);
    uint32_t* gt = reinterpret_cast<uint32_t*>(sel_sm + sizeof(SelHead));
    select_row<kOwnHist>(S, gt, gt + W, W, stride, pos, score, hist, sel, r0);
  } else {
    __shared__ SelSmem S;
    select_row<kOwnHist>(S, S.gt, S.eq, kMaxBlk / 32, kMaxBlk, pos, score, hist, sel, r0);
  }
}

// ---- sparse attention: partial per (row, tile) ---------------------------------------------------------------------
// Tensor cores (mma.sync m16n8k16 bf16 → fp32, k in ascending order as Triton's tl.dot): S = Q·Kᵀ for 16 (12) heads ×
// 32 tokens, P = exp2(S·scale − m) → bf16, O = P·V. 4 warps: S n-tile = warp (8 tokens), O dims [64w, 64w+64).
struct AttnArgs {
  const bf16* q;
  const int *sel, *seq_start;
  QsaCache* const* caches;
  int B;
  float *po, *plse;
};

__global__ void __launch_bounds__(kAT) k_attn(AttnArgs a) {
  __shared__ __align__(16) bf16 qs[16 * kRS];
  __shared__ __align__(16) bf16 ks[kTileN * kRS];
  __shared__ __align__(16) bf16 vs[kTileN * kRS];
  __shared__ __align__(16) bf16 ps[16 * kPS];
  __shared__ float ss[kQH][kTileN + 1];
  __shared__ int tok[kTileN];
  __shared__ float inv_l[16];
  pdl_wait();
  const int t = blockIdx.x, r = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int* srow = a.sel + size_t(r) * kSelW;
  const int n = min(kTileN, srow[kSelCols] - t * kTileN);
  if (n <= 0) return;
  const bf16* kv = a.caches[seq_of(a.seq_start, a.B, r)]->kv;
  if (tid < kTileN) tok[tid] = tid < n ? srow[t * kTileN + tid] : 0;
  __syncthreads();
  uint4 x[16];  // K‖V rows of the tile: all 16 loads per thread in flight (faster than cp.async here)
#pragma unroll
  for (int k = 0; k < 16; k++) {
    const int i = tid + k * kAT, j = i >> 6, part = i & 63;
    x[k] = j < n ? __ldg(reinterpret_cast<const uint4*>(kv + size_t(tok[j]) * 512 + part * 8)) : make_uint4(0, 0, 0, 0);
  }
  for (int i = tid; i < 16 * 32; i += kAT) {  // Q rows 0..11, zero rows 12..15
    const int h = i >> 5, c = i & 31;
    *reinterpret_cast<uint4*>(qs + h * kRS + c * 8) =
        h < kQH ? *reinterpret_cast<const uint4*>(a.q + (size_t(r) * kQH + h) * 256 + c * 8) : make_uint4(0, 0, 0, 0);
  }
  for (int i = tid; i < 16 * kPS / 2; i += kAT) reinterpret_cast<uint32_t*>(ps)[i] = 0;
#pragma unroll
  for (int k = 0; k < 16; k++) {
    const int i = tid + k * kAT, j = i >> 6, part = i & 63;
    *reinterpret_cast<uint4*>((part < 32 ? ks : vs) + j * kRS + (part & 31) * 8) = x[k];
  }
  __syncthreads();
  const float scale = 0.0625f * 1.4426950408889634f;  // 1/√256 in log2 units (the reference's exp2 softmax)
  {  // S tile n = warp: tokens 8w..8w+7
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const bf16* qa = qs + (lane & 15) * kRS + (lane >> 4) * 8;
    const bf16* kb = ks + (warp * 8 + (lane & 7)) * kRS + (lane >> 3) * 8;
#pragma unroll
    for (int kk = 0; kk < 256; kk += 32) {
      uint32_t A0[4], A1[4], Bf[4];
      ldsm_x4(A0, qa + kk);
      ldsm_x4(A1, qa + kk + 16);
      ldsm_x4(Bf, kb + kk);  // dims kk..kk+7 | +8 | +16 | +24 of tokens 8w..8w+7
      mma_bf16(c, A0, Bf[0], Bf[1]);
      mma_bf16(c, A1, Bf[2], Bf[3]);
    }
    const int h = lane >> 2, j = warp * 8 + (lane & 3) * 2;
    if (h < kQH) ss[h][j] = c[0] * scale, ss[h][j + 1] = c[1] * scale;
    if (h + 8 < kQH) ss[h + 8][j] = c[2] * scale, ss[h + 8][j + 1] = c[3] * scale;
  }
  __syncthreads();
#pragma unroll
  for (int hh = 0; hh < 3; hh++) {  // softmax: warp w → heads 3w..3w+2, lane → token
    const int h = warp * 3 + hh;
    const float s = lane < n ? ss[h][lane] : -1.0e20f;
    const float m = warp_max(s);
    const float pr = lane < n ? exp2f(s - m) : 0.f;
    const float l = warp_sum(pr);
    ps[h * kPS + lane] = __float2bfloat16_rn(pr);
    if (lane == 0) {
      inv_l[h] = 1.f / fmaxf(l, 1.0e-20f);
      a.plse[(size_t(r) * kMaxTiles + t) * kQH + h] = m + log2f(fmaxf(l, 1.0e-20f));
    }
  }
  __syncthreads();
  float o[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < kTileN; kk += 16) {
    uint32_t A[4];
    ldsm_x4(A, ps + (lane & 15) * kPS + kk + (lane >> 4) * 8);
#pragma unroll
    for (int nt = 0; nt < 8; nt += 2) {  // V rows kk..kk+15, dims d0..d0+15 (two n-tiles), transposed
      uint32_t Bf[4];
      ldsm_x4_t(Bf, vs + (kk + (lane & 15)) * kRS + warp * 64 + nt * 8 + (lane >> 4) * 8);
      mma_bf16(o[nt], A, Bf[0], Bf[1]);
      mma_bf16(o[nt + 1], A, Bf[2], Bf[3]);
    }
  }
  float* dst = a.po + (size_t(r) * kMaxTiles + t) * kQH * 256;
  const int h = lane >> 2;
#pragma unroll
  for (int nt = 0; nt < 8; nt++) {
    const int d = warp * 64 + nt * 8 + (lane & 3) * 2;
    if (h < kQH) *reinterpret_cast<float2*>(dst + h * 256 + d) = make_float2(o[nt][0] * inv_l[h], o[nt][1] * inv_l[h]);
    if (h + 8 < kQH)
      *reinterpret_cast<float2*>(dst + (h + 8) * 256 + d) =
          make_float2(o[nt][2] * inv_l[h + 8], o[nt][3] * inv_l[h + 8]);
  }
}

// ---- merge + gate ---------------------------------------------------------------------------------------------------
// grid (12·4, M): head h, dims [64q, 64q+64). Thread (g, d): tile group g sums tiles [17g, 17g+17) in order, groups
// combined in order: out = Σ w_t·o_t / Σ w_t, w_t = exp2(lse_t − max); o → bf16; × sigmoid(gate) → bf16.
__global__ void __launch_bounds__(256) k_merge(const int* sel, const float* __restrict__ po,
                                               const float* __restrict__ plse, const bf16* qkv, bf16* attn) {
  __shared__ float wt[kMG * kMT];
  __shared__ float part[kMG][2][64];
  pdl_wait();
  const int h = blockIdx.x / 4, d0 = (blockIdx.x % 4) * 64, r = blockIdx.y, tid = threadIdx.x;
  const int g = tid >> 6, dl = tid & 63;
  const int nt = (sel[size_t(r) * kSelW + kSelCols] + kTileN - 1) / kTileN;
  const float* lse = plse + size_t(r) * kMaxTiles * kQH + h;
  const float* src = po + size_t(r) * kMaxTiles * kQH * 256 + h * 256 + d0 + dl;
  float x[kMT];  // this group's partials, all in flight
#pragma unroll
  for (int k = 0; k < kMT; k++) {
    const int t = g * kMT + k;
    x[k] = t < nt ? __ldg(src + size_t(t) * kQH * 256) : 0.f;
  }
  if (tid < 32) {
    float l[3], mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < 3; k++) l[k] = tid + 32 * k < nt ? lse[(tid + 32 * k) * kQH] : -INFINITY, mx = fmaxf(mx, l[k]);
    mx = warp_max(mx);
#pragma unroll
    for (int k = 0; k < 3; k++)
      if (tid + 32 * k < kMG * kMT) wt[tid + 32 * k] = tid + 32 * k < nt ? exp2f(l[k] - mx) : 0.f;
  }
  __syncthreads();
  float num = 0.f, den = 0.f;
#pragma unroll
  for (int k = 0; k < kMT; k++) num = fmaf(wt[g * kMT + k], x[k], num), den += wt[g * kMT + k];
  part[g][0][dl] = num, part[g][1][dl] = den;
  __syncthreads();
  if (g == 0) {
    for (int k = 1; k < kMG; k++) num += part[k][0][dl], den += part[k][1][dl];
    const float o = bfr(den > 0.f ? num / den : 0.f);
    const int d = d0 + dl;
    const float gt = __bfloat162float(qkv[size_t(r) * kQkv + h * 512 + 256 + d]);
    attn[size_t(r) * kQH * 256 + h * 256 + d] = __float2bfloat16_rn(o * (1.f / (1.f + __expf(-gt))));
  }
}

#include "kernels/qsa/p021_qsa_decode_attn.cuh"  // P021 qsa_decode_attn: decode attention + merge, one cluster per row

// ---- workspace ---------------------------------------------------------------------------------------------------
// Rows of a call: q, qi, sel. Decode path (≤ kDecRows rows): fp32 scores + histograms + attention partials + LSE.
// Prefill path: fp32 scores for kPfRows rows at a time (k_pscore → k_select<true> per sub-chunk). Score rows are
// g_stride floats apart in memory (kMaxBlk for windows ≤ 262,144); the base tier indexes them kMaxBlk apart.
constexpr int kPfRows = 1024;  // 1024 × 64K blocks × 4 B = 256 MB (window ≤ 256K)

struct Ws {
  int rows = 0, pscore_rows = 0;
  bf16 *q = nullptr, *qi = nullptr;
  float *score = nullptr, *po = nullptr, *plse = nullptr, *pscore = nullptr;
  int *sel = nullptr, *hist = nullptr;
} g_ws;
int g_sms = 0;
const bf16* g_rope = nullptr;
QsaRope g_rcfg;          // parameters of g_rope (qsa.h defaults until qsa_rope_set)
int g_blk = kMaxBlk;     // blocks of the window: ⌈rows / 4⌉
int g_stride = kMaxBlk;  // score row stride of the workspace: max(kMaxBlk, g_blk rounded up to 64)

template <class T>
T* dmalloc(size_t n) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

// Selection tier of a call (qsa.h, max_pos): wide when its rows may see > kMaxBlk blocks; W = bitset words (wide).
struct Tier {
  bool wide;
  int W;
};
Tier tier_of(int max_pos, cudaStream_t s) {
  if (g_blk <= kMaxBlk) return {false, kMaxBlk / 32};  // window ≤ 262,144: every position fits the base tier
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  CUDA_CHECK(cudaStreamIsCapturing(s, &cs));  // a captured graph replays at later positions: size for the window
  const int v = max_pos >= 0 && cs == cudaStreamCaptureStatusNone ? (max_pos + 1) >> 2 : g_blk;
  return {v > kMaxBlk, (v + 31) / 32};
}

#include "kernels/qsa/p009_qsa_fast_index.cuh"  // P009 qsa_fast_index: prefill indexer via candidates

void pscore_reserve(int rows) {  // base prefill indexer's score buffer (grow-only)
  if (rows <= g_ws.pscore_rows) return;
  if (g_ws.pscore) {
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(g_ws.pscore);
  }
  g_ws.pscore = dmalloc<float>(size_t(rows) * g_stride);
  g_ws.pscore_rows = rows;
}

// RoPE parameters → table kernel parameters. vLLM _yarn_find_correction_dim / _range in Python doubles (beta_fast 32,
// beta_slow 1, floor / ceil, clamp to [0, 63]); _yarn_linear_ramp_mask divides by (hi − lo) (+ 0.001 if equal), which
// torch on the GPU runs as a multiply by the fp32 reciprocal; mscale = yarn_get_mscale(factor) · attn_factor 1.
RopeK rope_k(const QsaRope& r) {
  if (r.rows < 1 || r.rows > kQsaMaxPos) qlog::fail("qsa", "RoPE rows %d outside 1 … %d", r.rows, kQsaMaxPos);
  RopeK k{r.rows, !(r.factor > 1.0), 0.f, 0.f, 1.f, 1.f};
  if (!k.plain) {
    const double L = r.ramp_len > 0 ? r.ramp_len : r.original, pi = 3.141592653589793;
    if (!(L > 0)) qlog::fail("qsa", "YaRN factor %g needs original / ramp_len > 0", r.factor);
    auto dim = [&](double rot) { return (64 * std::log(L / (rot * 2 * pi))) / (2 * std::log(10000000.0)); };
    const int lo = std::max(int(std::floor(dim(32))), 0), hi = std::min(int(std::ceil(dim(1))), 63);
    const double d = lo == hi ? (hi + 0.001) - lo : double(hi - lo);
    k.lo = float(lo), k.rinv = 1.f / float(d), k.factor = float(r.factor);
    k.mscale = float(0.1 * std::log(r.factor) + 1.0);
  }
  return k;
}

void rope_build(const RopeK& k, bf16* dst) {
  k_rope_table<<<cdiv(k.rows * 32, 256), 256>>>(dst, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void ws_free() {  // all of it (qsa_rope_set: score stride and candidate width follow the window)
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(g_ws.q), cudaFree(g_ws.qi), cudaFree(g_ws.score), cudaFree(g_ws.po), cudaFree(g_ws.plse);
  cudaFree(g_ws.pscore), cudaFree(g_ws.sel), cudaFree(g_ws.hist);
  g_ws = Ws{};
  p9_free();
}

}  // namespace

void qsa_reserve(int M) {
  if (!g_sms) CUDA_CHECK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
  if (M <= g_ws.rows) return;
  if (g_ws.rows) {
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(g_ws.q), cudaFree(g_ws.qi), cudaFree(g_ws.score), cudaFree(g_ws.po), cudaFree(g_ws.plse);
    cudaFree(g_ws.pscore), cudaFree(g_ws.sel), cudaFree(g_ws.hist);
    g_ws.pscore = nullptr, g_ws.pscore_rows = 0;
  }
  const size_t R = size_t(M), Rd = std::min<size_t>(R, kDecRows);
  g_ws.q = dmalloc<bf16>(R * kQH * 256);
  g_ws.qi = dmalloc<bf16>(R * kIdxHeads * 128);
  g_ws.sel = dmalloc<int>(R * kSelW);
  g_ws.score = dmalloc<float>(Rd * g_stride);
  g_ws.po = dmalloc<float>(Rd * kMaxTiles * kQH * 256);
  g_ws.plse = dmalloc<float>(Rd * kMaxTiles * kQH);
  g_ws.hist = dmalloc<int>(Rd * kBins);
  if (M > kDecRows) {
    if (patch::on(patch::qsa_fast_index)) {  // P009 for calls of ≥ kP9MinRows rows, the base indexer below
      if (M >= kP9MinRows) p9_reserve(M);
      pscore_reserve(std::min(M, kP9MinRows - 1));
    } else {
      pscore_reserve(kPfRows);
    }
  }
  qsa_dev::pattn_init();  // prefill attention
  p21_init();             // P021 decode attention
  if (g_blk > kMaxBlk) {  // wide-tier selection: up to the window's bitsets in dynamic smem
    const int sm = int(sel_smem((g_blk + 31) / 32));
    CUDA_CHECK(cudaFuncSetAttribute(k_select<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    CUDA_CHECK(cudaFuncSetAttribute(k_select<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
  }
  g_ws.rows = M;
}

const bf16* qsa_rope_set(const QsaRope& r) {
  const RopeK k = rope_k(r);
  const int keep = g_ws.rows;
  if (keep) ws_free();
  if (g_rope) {
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(const_cast<bf16*>(g_rope));
    g_rope = nullptr;
  }
  g_rcfg = r;
  g_blk = cdiv(r.rows, kIdxRatio);
  g_stride = std::max(kMaxBlk, (g_blk + 63) / 64 * 64);
  bf16* t = dmalloc<bf16>(size_t(r.rows) * 64);
  rope_build(k, t);
  g_rope = t;
  if (keep) qsa_reserve(keep);
  return g_rope;
}

const QsaRope& qsa_rope() { return g_rcfg; }

void qsa_rope_fill(const QsaRope& r, bf16* dst) { rope_build(rope_k(r), dst); }

void qsa_last_q(const bf16** q, const bf16** qi) {
  *q = g_ws.q;
  *qi = g_ws.qi;
}

void qsa_p009_last(const int** cnt, const int** fb, int* splits) {
  *cnt = g_p9.cnt;
  *fb = g_p9.fb;
  *splits = kP9KS;
}

QsaW qsa_upload(const Weights& W, int layer) {
  return qsa_upload(W, "model.language_model.layers." + std::to_string(layer) + ".self_attn.");
}

QsaW qsa_upload(const Weights& W, const std::string& a) {
  QsaW w;
  auto up = [&](const std::string& name, int n) {
    std::vector<int64_t> shape;
    auto b = W.slice(name, &shape);
    if (b.size() != size_t(n) * 2) qlog::fail("qsa", "%s: %zu B (expected %d)", name.c_str(), b.size(), n * 2);
    bf16* d = dmalloc<bf16>(n);
    CUDA_CHECK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice));
    return static_cast<const bf16*>(d);
  };
  w.q_norm = up(a + "q_norm.weight", 256);
  w.k_norm = up(a + "k_norm.weight", 256);
  w.iq_norm = up(a + "indexer.q_layernorm.weight", 128);
  w.ik_norm = up(a + "indexer.k_layernorm.weight", 128);
  if (!g_rope) qsa_rope_set(g_rcfg);  // defaults (or the parameters of the last qsa_rope_set)
  w.rope = g_rope;
  return w;
}

void qsa_prepare(const QsaW& w, const bf16* qkv, const int* pos, QsaCache* const* caches, const int* seq_start, int B,
                 int M, cudaStream_t s, const int* rp) {
  if (M <= 0) return;
  qsa_reserve(M);
  launch(k_prepare, M, 576, s,
         PrepArgs{qkv, w.rope, w.q_norm, w.k_norm, w.iq_norm, w.ik_norm, pos, seq_start, rp, caches, B, g_ws.q, g_ws.qi,
                  nullptr});
  CUDA_CHECK(cudaGetLastError());
}

void qsa_core(const QsaW& w, const bf16* qkv, const int* pos, bf16* attn, QsaCache* const* caches, const int* seq_start,
              int B, int M, int* sel_out, cudaStream_t s, int max_pos, const int* rp) {
  if (M <= 0) return;
  qsa_reserve(M);
  if (max_pos >= g_rcfg.rows) qlog::fail("qsa", "position %d ≥ RoPE table rows %d (the window)", max_pos, g_rcfg.rows);
  int* sel = sel_out ? sel_out : g_ws.sel;
  const bool dec = M <= kDecRows;
  const Tier t = tier_of(max_pos, s);
  launch(k_prepare, M, 576, s,
         PrepArgs{qkv, w.rope, w.q_norm, w.k_norm, w.iq_norm, w.ik_norm, pos, seq_start, rp, caches, B, g_ws.q, g_ws.qi,
                  dec ? g_ws.hist : nullptr});
  if (dec) {
    const ScoreArgs sa{g_ws.qi, pos, seq_start, caches, g_ws.score, g_ws.hist, g_stride};
    const dim3 sg(std::max(1, 2 * g_sms / B), B);
    if (!t.wide) {
      launch(k_score<false>, sg, kSW * 32, s, sa);
      launch(k_select<false, false>, M, kSelT, s, pos, g_ws.score, g_ws.hist, sel, 0, g_stride, t.W);
    } else {
      launch(k_score<true>, sg, kSW * 32, s, sa);
      launch_smem(k_select<false, true>, M, kSelT, sel_smem(t.W), s, pos, g_ws.score, g_ws.hist, sel, 0, g_stride,
                  t.W);
    }
    if (patch::on(patch::qsa_decode_attn) && !p21_base(M, B, max_pos)) {  // P021: a 4-CTA cluster per row
      p21_attn(qsa_dev::PAttnArgs{g_ws.q, qkv, sel, seq_start, caches, B, attn}, M, s);
    } else {
      launch(k_attn, dim3(kMaxTiles, M), kAT, s, AttnArgs{g_ws.q, sel, seq_start, caches, B, g_ws.po, g_ws.plse});
      launch(k_merge, dim3(kQH * 4, M), 256, s, sel, g_ws.po, g_ws.plse, qkv, attn);
    }
  } else if (M >= kP9MinRows && patch::on(patch::qsa_fast_index)) {  // P009: same selection, candidate scores only
    p9_reserve(M);  // no-op once reserved (a patch switched on in a test)
    p9_indexer(g_ws.qi, pos, seq_start, caches, B, M, sel, t, s);
    qsa_dev::pattn(qsa_dev::PAttnArgs{g_ws.q, qkv, sel, seq_start, caches, B, attn}, M, s);
  } else {  // prefill path (qsa_prefill.cu): same per-row values, row-tiled kernels
    pscore_reserve(std::min(M, kPfRows));  // no-op once reserved (a patch switched off in a test)
    for (int ra = 0; ra < M; ra += kPfRows) {
      const int rb = std::min(M, ra + kPfRows);
      qsa_dev::pscore(qsa_dev::PScoreArgs{g_ws.qi, pos, seq_start, caches, B, ra, rb, g_ws.pscore, g_stride, t.wide}, s);
      const int* nohist = nullptr;
      if (!t.wide)
        launch(k_select<true, false>, rb - ra, kSelT, s, pos, g_ws.pscore, nohist, sel, ra, g_stride, t.W);
      else
        launch_smem(k_select<true, true>, rb - ra, kSelT, sel_smem(t.W), s, pos, g_ws.pscore, nohist, sel, ra,
                    g_stride, t.W);
    }
    qsa_dev::pattn(qsa_dev::PAttnArgs{g_ws.q, qkv, sel, seq_start, caches, B, attn}, M, s);
  }
  CUDA_CHECK(cudaGetLastError());
}

void qsa_decode_attn_stage(int variant, const bf16* q, const bf16* qkv, const int* sel, QsaCache* const* caches,
                           const int* seq_start, int B, int M, bf16* attn, cudaStream_t s, int max_pos) {
  if (M <= 0 || M > kDecRows) qlog::fail("qsa", "decode attention stage: M %d", M);
  qsa_reserve(M);
  const qsa_dev::PAttnArgs pa{q, qkv, sel, seq_start, caches, B, attn};
  if (variant == 0 || (variant == 2 && p21_base(M, B, max_pos))) {
    launch(k_attn, dim3(kMaxTiles, M), kAT, s, AttnArgs{q, sel, seq_start, caches, B, g_ws.po, g_ws.plse});
    launch(k_merge, dim3(kQH * 4, M), 256, s, sel, static_cast<const float*>(g_ws.po),
           static_cast<const float*>(g_ws.plse), qkv, attn);
  } else if (variant == 1) {
    qsa_dev::pattn(pa, M, s);
  } else {
    p21_attn(pa, M, s);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace q38
