// qsa_prefill.cu — QSA prefill path (qsa_core with M > kDecRows rows, e.g. one prompt chunk of up to 8,192 rows).
// Every per-row value is computed with exactly the operations of the decode path (qsa.cu), so a row is bitwise the
// same whichever path or chunking ran it:
//   k_pscore : indexer scores as a tensor-core GEMM — CTA = 16 rows (64 q-vectors, B fragments held in registers by
//              each warp) × 16-key batches strided over the tile's warps; per (key, q-vector) the 8-step mma chain of
//              k_score (idx_dot), Σ_h ReLU (idx_relu_sum); rows' causal limit; fp32 scores (rows that see > 512
//              blocks) written coalesced. Selection: k_select<true> (qsa.cu).
//   k_pattn  : per row (8 warps) — every tile's S, softmax and P exactly as k_attn (pass 1, tiles over warps, P kept
//              in smem), then k_merge's LSE merge on chip (pass 2, dims over warps: O = P·V, o_t = O·(1/l), per 17-tile
//              group fma(w_t, o_t, ·) in tile order, groups added in order), o → bf16, × sigmoid(gate). No per-tile
//              partials in memory.
#include "kernels/qsa/qsa_dev.cuh"

namespace q38::qsa_dev {
namespace {

// ---- indexer GEMM ------------------------------------------------------------------------------------------------
// grid (row tiles of 16, kPSplit), 4 warps. Each warp holds the tile's 64 q-vectors as B fragments (8 n-tiles) and
// walks 16-key batches (A fragments loaded directly, next batch prefetched) strided over the kPSplit·4 warps of the
// tile; idx_dot per n-tile, Σ_h ReLU, scores staged in the warp's smem slice → 64-B coalesced row writes.
constexpr int kPR = 16;     // rows per CTA (64 q-vectors = 8 n-tiles)
constexpr int kPSplit = 8;  // CTAs per row tile (key batches strided over kPSplit·4 warps)

template <bool kWide>  // score row stride kMaxBlk (base tier) or a.stride (wide)
__global__ void __launch_bounds__(128) k_pscore(PScoreArgs a) {
  __shared__ __align__(16) float st[4][kPR][20];
  pdl_wait();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int t0 = a.ra + blockIdx.x * kPR, t1 = min(a.rb, t0 + kPR);
  for (int r0 = t0; r0 < t1;) {  // rows of one sequence at a time
    const int b = seq_of(a.seq_start, a.B, r0), r1 = min(t1, a.seq_start[b + 1]);
    const int vmax = (a.pos[r1 - 1] + 1) >> 2, nbat = (vmax + 15) >> 4;
    const int bt0 = blockIdx.y * 4 + warp, bstep = kPSplit * 4;
    if (vmax > kBudget && bt0 < nbat) {
      const bf16* comp = a.caches[b]->ik_comp;
      uint4 qf[8][4];
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        const int rr = 2 * nt + (g >> 2);
        idx_load(qf[nt], a.qi + (size_t(r0 + rr) * 4 + (g & 3)) * 128, r0 + rr < r1, c);
      }
      const int wr = lane >> 1;  // write phase: row wr of the tile, keys (lane&1)·8 .. +8 of the batch
      const int vis_w = r0 + wr < r1 ? (a.pos[r0 + wr] + 1) >> 2 : 0;
      float* dst_row = a.score + size_t(r0 - a.ra + wr) * (kWide ? a.stride : kMaxBlk);
      uint4 ka[4], kb[4], na[4], nb[4];
      idx_load(na, comp + size_t(bt0 * 16 + g) * 128, bt0 * 16 + g < vmax, c);
      idx_load(nb, comp + size_t(bt0 * 16 + g + 8) * 128, bt0 * 16 + g + 8 < vmax, c);
      for (int bt = bt0; bt < nbat; bt += bstep) {
        const int j0 = bt * 16, jn = j0 + bstep * 16;
#pragma unroll
        for (int p = 0; p < 4; p++) ka[p] = na[p], kb[p] = nb[p];
        if (jn < vmax) {
          idx_load(na, comp + size_t(jn + g) * 128, jn + g < vmax, c);
          idx_load(nb, comp + size_t(jn + g + 8) * 128, jn + g + 8 < vmax, c);
        }
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
          float acc[4];
          idx_dot(acc, ka, kb, qf[nt]);
          const float2 sc = idx_relu_sum(acc);
          if ((lane & 1) == 0) {
            const int rr = nt * 2 + ((lane >> 1) & 1);
            st[warp][rr][g] = sc.x, st[warp][rr][g + 8] = sc.y;
          }
        }
        __syncwarp();
        const int kc = (lane & 1) * 8, j = j0 + kc;
        if (vis_w > kBudget && j < vis_w) {
          const float* sv = &st[warp][wr][kc];
          if (j + 8 <= vis_w) {
            *reinterpret_cast<float4*>(dst_row + j) = *reinterpret_cast<const float4*>(sv);
            *reinterpret_cast<float4*>(dst_row + j + 4) = *reinterpret_cast<const float4*>(sv + 4);
          } else {
            for (int i = 0; j + i < vis_w; i++) dst_row[j + i] = sv[i];
          }
        }
        __syncwarp();
      }
    }
    r0 = r1;
  }
}

// ---- attention -----------------------------------------------------------------------------------------------------
// One CTA (8 warps) per row. Q's 16 k-step A fragments (heads 0..11 + zero rows, k_attn's ldmatrix mapping) in every
// warp's registers.
// Pass 1 — warp w takes tiles w, w+8, …: S (4 n-tiles × 16 k-steps, k_attn's chain per element) with K streamed in
//   32-dim steps through the warp's smem slice (register ring 3 steps ahead), then k_attn's softmax in registers:
//   s = S·scale, tile max, pr = exp2(s − m), l = Σ pr in warp_sum's tree (pairs j^16, j^8, j^4, j^2, j^1 of the token
//   index j = 8·nt + 2c + e), P = bf16(pr) → the row's P store (smem, all tiles), 1/l and LSE → smem.
// Merge weights — k_merge: mx = max LSE, w_t = exp2(LSE_t − mx), den per head in its group order.
// Pass 2 — warp w owns dims [32w, 32w+32): per tile O = P·V (k_attn's chain), o_t = O·(1/l),
//   num = fma(w_t, o_t, num) per 17-tile group, groups added in order; out = bf16(o)·sigmoid(gate).
constexpr int kPW = 8;                // warps
constexpr int kSlS = 40;              // slice row stride (bf16): 80 B → conflict-free ldmatrix
constexpr int kSlice = 32 * kSlS;     // one 32-row slice buffer (bf16)
constexpr int kPSlot = kQH * 32;      // P of one tile: 12 heads × 32 tokens (bf16), 16-B chunks swizzled

struct PAttnSmem {
  bf16 pst[kMaxTiles * kPSlot];       // 49,920 B
  bf16 slice[kPW][2][kSlice];         // 40,960 B (Q staging at row start)
  float il[kMaxTiles][kQH];           // 1/l per tile, head
  float w[kMaxTiles][kQH];            // LSE, then merge weight
  float mx[kQH], dpart[kQH][kMG];
  uint4 zero[4];                      // 64-B zero row (P rows 12..15)
};
constexpr int kPAttnSmem = int(sizeof(PAttnSmem));
static_assert(16 * kRS <= kPW * 2 * kSlice, "Q staging fits in the slices");

__device__ __forceinline__ int pst_off(int h, int j) {  // element (head h, token j) of a P slot
  return h * 32 + (((j >> 3) ^ ((h >> 1) & 3)) << 3) + (j & 7);
}

__global__ void __launch_bounds__(kPW * 32, 1) k_pattn(PAttnArgs a) {
  extern __shared__ __align__(16) unsigned char smraw[];
  PAttnSmem& S = *reinterpret_cast<PAttnSmem*>(smraw);
  pdl_wait();
  const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
  const int* srow = a.sel + size_t(r) * kSelW;
  const int cnt = srow[kSelCols], ntl = (cnt + kTileN - 1) / kTileN;
  const bf16* kv = a.caches[seq_of(a.seq_start, a.B, r)]->kv;
  __nv_bfloat162 gate[2][4];  // this thread's output gates (heads g, g+8 × dims 32w + 8nt + 2c), loaded early
#pragma unroll
  for (int hh = 0; hh < 2; hh++)
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const int h = min(g + 8 * hh, kQH - 1);
      gate[hh][nt] = *reinterpret_cast<const __nv_bfloat162*>(a.qkv + size_t(r) * kQkv + h * 512 + 256 + 32 * warp +
                                                               8 * nt + 2 * c);
    }
  {  // Q → staging → A fragments
    bf16* qs = &S.slice[0][0][0];
    for (int i = tid; i < 16 * 32; i += kPW * 32) {
      const int h = i >> 5, cc = i & 31;
      *reinterpret_cast<uint4*>(qs + h * kRS + cc * 8) =
          h < kQH ? __ldg(reinterpret_cast<const uint4*>(a.q + (size_t(r) * kQH + h) * 256 + cc * 8))
                  : make_uint4(0, 0, 0, 0);
    }
    if (tid < 4) S.zero[tid] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  uint32_t Qa[16][4];
#pragma unroll
  for (int kk = 0; kk < 16; kk++) ldsm_x4(Qa[kk], &S.slice[0][0][0] + (lane & 15) * kRS + kk * 16 + (lane >> 4) * 8);
  __syncthreads();
  bf16* sl = &S.slice[warp][0][0];
  auto tok_of = [&](int t) -> int {  // this lane's token (row `lane`) of tile t, −1 beyond the selection
    const int j = t * kTileN + lane;
    return t < ntl && j < cnt ? __ldg(srow + j) : -1;
  };

  // ---- pass 1 ----
  {
    const float scale = 0.0625f * 1.4426950408889634f;  // 1/√256 in log2 units (the reference's exp2 softmax)
    const int nmy = warp < ntl ? (ntl - 1 - warp) / kPW + 1 : 0;
    auto issue = [&](uint4* f, int tk, int pp) {  // K dims [32pp, 32pp+32) of tokens 8i + lane/4 (tk = lane's token)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int t = __shfl_sync(kFull, tk, 8 * i + (lane >> 2));
        f[i] = t >= 0 ? __ldg(reinterpret_cast<const uint4*>(kv + size_t(t) * 512 + 32 * pp + 8 * (lane & 3)))
                      : make_uint4(0, 0, 0, 0);
      }
    };
    uint4 ring[4][4];
    int tk_cur = tok_of(warp), tk_nxt = tok_of(warp + kPW);
    if (nmy > 0) issue(ring[0], tk_cur, 0), issue(ring[1], tk_cur, 1), issue(ring[2], tk_cur, 2);
    for (int k = 0; k < nmy; k++) {
      const int T = warp + kPW * k, n = min(kTileN, cnt - T * kTileN);
      float acc[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
      for (int P = 0; P < 8; P++) {  // k-steps 2P, 2P+1 (dims 32P..32P+31)
        if (P + 3 < 8) issue(ring[(P + 3) & 3], tk_cur, P + 3);
        else if (k + 1 < nmy) issue(ring[(P + 3) & 3], tk_nxt, P - 5);
        bf16* buf = sl + (P & 1) * kSlice;
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 4; i++)
          *reinterpret_cast<uint4*>(buf + (8 * i + (lane >> 2)) * kSlS + 8 * (lane & 3)) = ring[P & 3][i];
        __syncwarp();
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          uint32_t Bf[4];
          ldsm_x4(Bf, buf + (8 * nt + (lane & 7)) * kSlS + (lane >> 3) * 8);
          mma_bf16(acc[nt], Qa[2 * P], Bf[0], Bf[1]);
          mma_bf16(acc[nt], Qa[2 * P + 1], Bf[2], Bf[3]);
        }
      }
      // softmax of heads g (e = 0,1) and g+8 (e = 2,3) over tokens j = 8nt + 2c + (e&1)
      float s[4][4], m[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int j = 8 * nt + 2 * c + (e & 1);
          s[nt][e] = j < n ? __fmul_rn(acc[nt][e], scale) : -1.0e20f;
          m[e >> 1] = fmaxf(m[e >> 1], s[nt][e]);
        }
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        m[hh] = fmaxf(m[hh], __shfl_xor_sync(kFull, m[hh], 1));
        m[hh] = fmaxf(m[hh], __shfl_xor_sync(kFull, m[hh], 2));
      }
      float pr[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int j = 8 * nt + 2 * c + (e & 1);
          pr[nt][e] = j < n ? exp2f(s[nt][e] - m[e >> 1]) : 0.f;
        }
      float l[2];
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {  // warp_sum's tree: j^16 (nt^2), j^8 (nt^1), j^4 (c^2), j^2 (c^1), j^1 (e)
        float b[2];
#pragma unroll
        for (int e1 = 0; e1 < 2; e1++) {
          const int e = 2 * hh + e1;
          b[e1] = __fadd_rn(__fadd_rn(pr[0][e], pr[2][e]), __fadd_rn(pr[1][e], pr[3][e]));
          b[e1] = __fadd_rn(b[e1], __shfl_xor_sync(kFull, b[e1], 2));
          b[e1] = __fadd_rn(b[e1], __shfl_xor_sync(kFull, b[e1], 1));
        }
        l[hh] = __fadd_rn(b[0], b[1]);
      }
      bf16* pt = S.pst + T * kPSlot;
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int h = g + 8 * hh;
        if (h < kQH) {
#pragma unroll
          for (int nt = 0; nt < 4; nt++)
            *reinterpret_cast<__nv_bfloat162*>(pt + pst_off(h, 8 * nt + 2 * c)) =
                __floats2bfloat162_rn(pr[nt][2 * hh], pr[nt][2 * hh + 1]);
          if (c == 0) {
            S.il[T][h] = 1.f / fmaxf(l[hh], 1.0e-20f);
            S.w[T][h] = m[hh] + log2f(fmaxf(l[hh], 1.0e-20f));
          }
        }
      }
      tk_cur = tk_nxt;
      tk_nxt = tok_of(T + 2 * kPW);
    }
  }
  __syncthreads();
  // ---- merge weights and denominators (k_merge) ----
  if (warp < 4) {  // k_merge: max over the row's tile LSEs, warp → heads 3w..3w+2, lanes over tiles
#pragma unroll
    for (int hh = 0; hh < 3; hh++) {
      const int h = warp * 3 + hh;
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < 3; k++) mx = fmaxf(mx, lane + 32 * k < ntl ? S.w[lane + 32 * k][h] : -INFINITY);
      mx = warp_max(mx);
      if (lane == 0) S.mx[h] = mx;
    }
  }
  __syncthreads();
  for (int i = tid; i < ntl * kQH; i += kPW * 32) {
    const int t = i / kQH, h = i - t * kQH;
    S.w[t][h] = exp2f(S.w[t][h] - S.mx[h]);
  }
  __syncthreads();
  const int ngr = (ntl + kMT - 1) / kMT;
  const bool pad = ntl % kMT != 0, more = ngr < kMG;  // k_merge's zero terms (identities except for −0)
  if (tid < kQH * kMG) {  // k_merge's den: per group Σ w in tile order (+0 pads), groups added in order
    const int h = tid >> 2, gq = tid & 3;
    float dg = 0.f;
    for (int t = gq * kMT; t < min(ntl, gq * kMT + kMT); t++) dg += S.w[t][h];
    if (gq == ngr - 1 && pad) dg = __fadd_rn(dg, 0.f);
    S.dpart[h][gq] = dg;
  }

  // ---- pass 2 ----
  const int d0 = 32 * warp;
  float num[4][4], tot[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) num[nt][e] = tot[nt][e] = 0.f;
  {
    auto issue = [&](uint4* f, int tk) {  // V dims [d0, d0+32) of tokens 8i + lane/4 (tk = lane's token)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int t = __shfl_sync(kFull, tk, 8 * i + (lane >> 2));
        f[i] = t >= 0 ? __ldg(reinterpret_cast<const uint4*>(kv + size_t(t) * 512 + 256 + d0 + 8 * (lane & 3)))
                      : make_uint4(0, 0, 0, 0);
      }
    };
    uint4 rv[4][4];
    int tq[4];
#pragma unroll
    for (int u = 0; u < 4; u++) tq[u] = tok_of(u);
    issue(rv[0], tq[0]), issue(rv[1], tq[1]), issue(rv[2], tq[2]);
    const bf16* zr = reinterpret_cast<const bf16*>(S.zero);
    for (int t0 = 0; t0 < ntl; t0 += 4) {
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int t = t0 + u;
        if (t >= ntl) break;
        issue(rv[(u + 3) & 3], tq[(u + 3) & 3]);  // tile t+3
        tq[u] = tok_of(t + 4);
        bf16* buf = sl + (u & 1) * kSlice;
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 4; i++)
          *reinterpret_cast<uint4*>(buf + (8 * i + (lane >> 2)) * kSlS + 8 * (lane & 3)) = rv[u][i];
        __syncwarp();
        const bf16* pt = S.pst + t * kPSlot;
        float o[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < kTileN; kk += 16) {
          const int pr_ = lane & 15, ch = (kk >> 3) + (lane >> 4);
          uint32_t A[4];
          ldsm_x4(A, pr_ < kQH ? pt + pr_ * 32 + ((ch ^ ((pr_ >> 1) & 3)) << 3) : zr);
#pragma unroll
          for (int np = 0; np < 2; np++) {
            uint32_t Bf[4];
            ldsm_x4_t(Bf, buf + (kk + (lane & 15)) * kSlS + 16 * np + (lane >> 4) * 8);
            mma_bf16(o[2 * np], A, Bf[0], Bf[1]);
            mma_bf16(o[2 * np + 1], A, Bf[2], Bf[3]);
          }
        }
        if (t > 0 && t % kMT == 0) {  // k_merge: num = num_0, then num += num_g in group order
#pragma unroll
          for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) tot[nt][e] = t == kMT ? num[nt][e] : tot[nt][e] + num[nt][e], num[nt][e] = 0.f;
        }
        const bool hi = g + 8 < kQH;
        const float il0 = S.il[t][g], w0 = S.w[t][g], il1 = hi ? S.il[t][g + 8] : 0.f, w1 = hi ? S.w[t][g + 8] : 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {  // k_attn's partial o·(1/l), k_merge's fma(w_t, o_t, num)
          num[nt][0] = fmaf(w0, __fmul_rn(o[nt][0], il0), num[nt][0]);
          num[nt][1] = fmaf(w0, __fmul_rn(o[nt][1], il0), num[nt][1]);
          num[nt][2] = fmaf(w1, __fmul_rn(o[nt][2], il1), num[nt][2]);
          num[nt][3] = fmaf(w1, __fmul_rn(o[nt][3], il1), num[nt][3]);
        }
      }
    }
  }
#pragma unroll
  for (int nt = 0; nt < 4; nt++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      float v = pad ? __fadd_rn(num[nt][e], 0.f) : num[nt][e];
      v = ngr == 1 ? v : tot[nt][e] + v;
      tot[nt][e] = more ? __fadd_rn(v, 0.f) : v;
    }
  __syncthreads();  // S.dpart
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    const int h = g + 8 * hh;
    if (h >= kQH) continue;
    float den = S.dpart[h][0];
    for (int k = 1; k < ngr; k++) den = den + S.dpart[h][k];
    if (more) den = __fadd_rn(den, 0.f);
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
      const int d = d0 + nt * 8 + c * 2;
      const float2 gt2 = __bfloat1622float2(gate[hh][nt]);
      const float gts[2] = {gt2.x, gt2.y};
      float y[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const float ov = bfr(den > 0.f ? tot[nt][2 * hh + e] / den : 0.f);
        const float gt = gts[e];
        y[e] = ov * (1.f / (1.f + __expf(-gt)));
      }
      *reinterpret_cast<__nv_bfloat162*>(a.attn + size_t(r) * kQH * 256 + h * 256 + d) = __floats2bfloat162_rn(y[0], y[1]);
    }
  }
}

}  // namespace

void pscore(const PScoreArgs& a, cudaStream_t s) {
  launch(a.wide ? k_pscore<true> : k_pscore<false>, dim3(cdiv(a.rb - a.ra, kPR), kPSplit), 128, s, a);
}

void pattn_init() {
  CUDA_CHECK(cudaFuncSetAttribute(k_pattn, cudaFuncAttributeMaxDynamicSharedMemorySize, kPAttnSmem));
}

void pattn(const PAttnArgs& a, int M, cudaStream_t s) {
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{dim3(M), dim3(kPW * 32), size_t(kPAttnSmem), s, at, 1};
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_pattn, a));
}

}  // namespace q38::qsa_dev
