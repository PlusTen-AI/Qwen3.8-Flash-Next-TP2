// p021_qsa_decode_attn.cuh — P021 qsa_decode_attn: decode QSA attention + LSE merge + gate (M ≤ 64 rows) in one kernel,
// a 4-CTA thread-block cluster per row, partials on chip. Included by qsa.cu inside its anonymous namespace (after k_attn
// / k_merge). Bitwise the same as k_attn + k_merge (and k_pattn): every rounding point is theirs.
//   CTA g (cluster rank) owns k_merge's tile group g: tiles [17g, 17g+17) ∩ [0, ntl) of the row. Grid (4, M), 8 warps,
//   ≈ 46 KB smem, 128 registers → 2 CTAs per SM (M ≤ 24 rows in one wave).
//   Prologue — the row's cache (seq_start / cache table are written by the host before the step: read before the PDL
//     wait); the count and this warp's candidate tile tokens at once, then its first K loads; Q, the group's token list
//     and the output gates load meanwhile.
//   Pass 1 — group tile i < 16 → warp i mod 8 (≤ 2 tiles); the 17th tile of groups 0..2 → warps 5..7 of CTA 3 (group 3
//     has ≤ 14 tiles, its last ≤ 3 tokens), staged in CTA 3's unused slot 14..16 and copied to its owner (16-B DSMEM
//     stores). Per tile: S with k_attn's mma chain per element (K streamed in 32-dim steps through the warp's smem
//     slice, register ring 3 steps ahead), softmax in registers as k_pattn (k_attn's tile max, exp2, warp_sum's tree),
//     P = bf16(pr), 1/l and LSE → the P store; running max of the LSEs per warp.
//   Max — over warps, then over the cluster (max is exact: k_merge's mx in any order); pass 2's first V tiles go in flight
//     while the cluster barrier completes.
//   Weights w_t = exp2(LSE_t − mx); den_g = Σ w_t in tile order (k_merge's group sum; its zero terms as one + 0).
//   Pass 2 — warp w: dims [32w, 32w+32); per tile O = P·V (k_attn's chain), o_t = O·(1/l), num_g = fma(w_t, o_t, num_g)
//     in tile order (V ring 3 tiles ahead); k_merge's zero terms as one + 0.
//   Combine — CTA g finishes heads 3g..3g+2 (a thread: 4 dims of one head): num = ((num_0 + num_1) + num_2) + num_3
//     over the cluster (DSMEM), den likewise, o = bf16(num/den), out = bf16(o · sigmoid(gate)) — k_merge's expressions.
// Every value depends only on the row and the caches (fixed orders, no atomics) ⇒ independent of M, B and of the other
// rows of the call.
namespace p21 {

constexpr int kC = 4;              // CTAs per row (cluster): one per tile group
constexpr int kW = 8;              // warps per CTA
constexpr int kSlS = 40;           // staging row stride (bf16): 80 B → conflict-free ldmatrix
constexpr int kSlice = 32 * kSlS;  // one warp's staging buffer: 32 rows (bf16)
constexpr int kPSlot = kQH * 32;   // P of one tile: 12 heads × 32 tokens (bf16), 16-B chunks swizzled
constexpr int kP1 = 3;             // pass-1 tiles per warp at most (CTA 3 warp 5: one tile, the tail tile, a 17th)
constexpr int kQ4 = 64, kNO = 3 * kQ4;  // combine: float4 per head row, threads with outputs
static_assert(kMG == kC && kMT == 17 && kMaxTiles == 65, "tile groups: 3 × 17 + 14, last tile ≤ 3 tokens");

struct Smem {
  bf16 qs[16 * kRS];               // Q (rows 12..15 zero)                   8,448 B
  bf16 sl[kW][kSlice];             // per-warp K / V staging                20,480 B
  bf16 pst[kMT * kPSlot];          // P of the group's 17 tiles             13,056 B
  float il[kMT][kQH], lw[kMT][kQH];  // 1/l; LSE, then merge weight (48-B rows)
  int tok[kMT * kTileN];           // the group's token list (raw)
  float wmx[kW][kQH];              // per-warp running max of the LSEs
  float mxl[kQH], mx[kQH], den[kQH];
  uint4 zero[4];                   // 64-B zero row (P rows 12..15)
};
// num of this CTA's group [12][256] fp32, written after pass 2 over qs + sl
static_assert(kQH * 256 * 4 <= (16 * kRS + kW * kSlice) * 2, "part fits over qs + sl");
static_assert(offsetof(Smem, il) % 16 == 0 && offsetof(Smem, lw) % 16 == 0 && offsetof(Smem, pst) % 16 == 0, "16-B rows");
constexpr int kSmem = int(sizeof(Smem));

__device__ __forceinline__ int poff(int h, int j) {  // element (head h, token j) of a P slot (k_pattn's swizzle)
  return h * 32 + (((j >> 3) ^ ((h >> 1) & 3)) << 3) + (j & 7);
}
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_addr(const void* p, int rank) {  // shared::cluster address of p in CTA `rank`
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(a) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))), "r"(rank));
  return a;
}
__device__ __forceinline__ void cl_st16(uint32_t a, uint4 v) {
  asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ float4 cl_ld16f(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ float cl_ldf(uint32_t a) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(a) : "memory");
  return v;
}

__global__ void __launch_bounds__(kW * 32, 2) k_dattn(qsa_dev::PAttnArgs a) {
  extern __shared__ __align__(16) unsigned char smraw[];
  Smem& S = *reinterpret_cast<Smem*>(smraw);
  const int g = blockIdx.x, r = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gq = lane >> 2, cq = lane & 3;
  const int t0 = g * kMT;  // the group's first tile
  // seq_start and the cache table are written by the host before the step (stream order) → read before the PDL wait
  int b = 0;  // sequence of row r (seq_of by ballot)
  for (int k0 = 1; k0 < a.B; k0 += 32)
    b += __popc(__ballot_sync(kFull, k0 + lane < a.B && __ldg(a.seq_start + k0 + lane) <= r));
  const bf16* kv = a.caches[b]->kv;
  if (tid < 4) S.zero[tid] = make_uint4(0, 0, 0, 0);
  cl_arrive();  // every CTA runs before the first P copy (waited below)
  pdl_wait();
  // ---- prologue: count and this warp's pass-1 tokens at once, its first K loads, then Q, token list, gates ----
  const int* srow = a.sel + size_t(r) * kSelW;
  const int gx = g == 3 && warp >= 5 ? warp - 5 : -1;  // helper for the 17th tile of group gx
  const int C0 = t0 + warp, C1 = t0 + warp + kW, CX = gx >= 0 ? kMT * gx + kMT - 1 : 0;
  const int cnt = __ldg(srow + kSelCols);
  const int u0 = C0 * kTileN + lane < kSelCols ? __ldg(srow + C0 * kTileN + lane) : -1;
  const int u1 = C1 * kTileN + lane < kSelCols ? __ldg(srow + C1 * kTileN + lane) : -1;
  const int ux = gx >= 0 ? __ldg(srow + CX * kTileN + lane) : -1;
  const int ntl = (cnt + kTileN - 1) / kTileN, ng = max(0, min(kMT, ntl - t0));  // group's tiles t0 .. t0+ng−1
  const bool v0 = warp < min(ng, kMT - 1), v1 = warp + kW < min(ng, kMT - 1), vx = gx >= 0 && ntl > CX;
  const int nmy = int(v0) + int(v1) + int(vx);
  int Tl[kP1], tk1[kP1];  // this warp's pass-1 row tiles; this lane's token of each (−1 beyond the selection)
  Tl[0] = v0 ? C0 : vx ? CX : -1, tk1[0] = v0 ? u0 : ux;
  Tl[1] = v1 ? C1 : v0 && vx ? CX : -1, tk1[1] = v1 ? u1 : ux;
  Tl[2] = v1 && vx ? CX : -1, tk1[2] = ux;
#pragma unroll
  for (int k = 0; k < kP1; k++)
    if (Tl[k] < 0 || Tl[k] * kTileN + lane >= cnt) tk1[k] = -1;
  auto issue = [&](uint4* f, int tk, int pp) {  // K dims [32pp, 32pp+32) of tokens 8i + lane/4 (tk = lane's token)
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int t = __shfl_sync(kFull, tk, 8 * i + (lane >> 2));
      f[i] = t >= 0 ? __ldg(reinterpret_cast<const uint4*>(kv + size_t(t) * 512 + 32 * pp + 8 * (lane & 3)))
                    : make_uint4(0, 0, 0, 0);
    }
  };
  uint4 ring[4][4];
  if (nmy > 0) issue(ring[0], tk1[0], 0), issue(ring[1], tk1[0], 1), issue(ring[2], tk1[0], 2);
  for (int i = tid; i < kMT * kTileN; i += kW * 32)
    S.tok[i] = t0 * kTileN + i < kSelCols ? __ldg(srow + t0 * kTileN + i) : -1;
  for (int i = tid; i < 16 * 32; i += kW * 32) {
    const int h = i >> 5, cc = i & 31;
    *reinterpret_cast<uint4*>(S.qs + h * kRS + cc * 8) =
        h < kQH ? __ldg(reinterpret_cast<const uint4*>(a.q + (size_t(r) * kQH + h) * 256 + cc * 8))
                : make_uint4(0, 0, 0, 0);
  }
  const int cto = tid < kNO ? tid : 0, h4 = 3 * g + cto / kQ4, d4 = (cto % kQ4) * 4;  // combine: head, 4 dims
  const uint2 gate4 = *reinterpret_cast<const uint2*>(a.qkv + size_t(r) * kQkv + h4 * 512 + 256 + d4);
  __syncthreads();
  bf16* sl = S.sl[warp];
  auto tok2 = [&](int i) -> int {  // this lane's token of group tile i (pass 2), −1 beyond the selection
    const int j = (t0 + i) * kTileN + lane;
    return j < cnt ? S.tok[i * kTileN + lane] : -1;
  };

  // ---- pass 1 ----
  float wm[2] = {-INFINITY, -INFINITY};  // running max of the LSEs (heads gq, gq+8)
  {
    const float scale = 0.0625f * 1.4426950408889634f;  // 1/√256 in log2 units (the reference's exp2 softmax)
    bool waited = false;
#pragma unroll 1
    for (int k = 0; k < nmy; k++) {
      const int T = k == 0 ? Tl[0] : k == 1 ? Tl[1] : Tl[2], n = min(kTileN, cnt - T * kTileN);
      const int gT = T / kMT, it = T - gT * kMT, ls = gT == g ? it : kMT - 3 + gT;  // owner group, slot, local slot
      const int tk_cur = k == 0 ? tk1[0] : k == 1 ? tk1[1] : tk1[2];
      const int tk_nxt = k == 0 ? tk1[1] : tk1[2];
      float acc[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
      for (int P = 0; P < 8; P++) {  // k-steps 2P, 2P+1 (dims 32P..32P+31)
        if (P + 3 < 8) issue(ring[(P + 3) & 3], tk_cur, P + 3);
        else if (k + 1 < nmy) issue(ring[(P + 3) & 3], tk_nxt, P - 5);
#pragma unroll
        for (int i = 0; i < 4; i++)
          *reinterpret_cast<uint4*>(sl + (8 * i + (lane >> 2)) * kSlS + 8 * (lane & 3)) = ring[P & 3][i];
        __syncwarp();
        uint32_t A0[4], A1[4];
        ldsm_x4(A0, S.qs + (lane & 15) * kRS + 32 * P + (lane >> 4) * 8);
        ldsm_x4(A1, S.qs + (lane & 15) * kRS + 32 * P + 16 + (lane >> 4) * 8);
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          uint32_t Bf[4];
          ldsm_x4(Bf, sl + (8 * nt + (lane & 7)) * kSlS + (lane >> 3) * 8);
          mma_bf16(acc[nt], A0, Bf[0], Bf[1]);
          mma_bf16(acc[nt], A1, Bf[2], Bf[3]);
        }
        __syncwarp();
      }
      // softmax of heads gq (e = 0,1) and gq+8 (e = 2,3) over tokens j = 8nt + 2cq + (e&1)
      float s[4][4], m[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int j = 8 * nt + 2 * cq + (e & 1);
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
          const int j = 8 * nt + 2 * cq + (e & 1);
          pr[nt][e] = j < n ? exp2f(s[nt][e] - m[e >> 1]) : 0.f;
        }
      float l[2];
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {  // warp_sum's tree: j^16 (nt^2), j^8 (nt^1), j^4 (c^2), j^2 (c^1), j^1 (e)
        float bb[2];
#pragma unroll
        for (int e1 = 0; e1 < 2; e1++) {
          const int e = 2 * hh + e1;
          bb[e1] = __fadd_rn(__fadd_rn(pr[0][e], pr[2][e]), __fadd_rn(pr[1][e], pr[3][e]));
          bb[e1] = __fadd_rn(bb[e1], __shfl_xor_sync(kFull, bb[e1], 2));
          bb[e1] = __fadd_rn(bb[e1], __shfl_xor_sync(kFull, bb[e1], 1));
        }
        l[hh] = __fadd_rn(bb[0], bb[1]);
      }
      bf16* pt = S.pst + ls * kPSlot;
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int h = gq + 8 * hh;
        const float lse = m[hh] + log2f(fmaxf(l[hh], 1.0e-20f));
        wm[hh] = fmaxf(wm[hh], lse);
        if (h < kQH) {
#pragma unroll
          for (int nt = 0; nt < 4; nt++)
            *reinterpret_cast<__nv_bfloat162*>(pt + poff(h, 8 * nt + 2 * cq)) =
                __floats2bfloat162_rn(pr[nt][2 * hh], pr[nt][2 * hh + 1]);
          if (cq == 0) S.il[ls][h] = 1.f / fmaxf(l[hh], 1.0e-20f), S.lw[ls][h] = lse;
        }
      }
      if (gT != g) {  // a 17th tile: slot → its owner's slot 16 (P 48 × 16 B, 1/l 3, LSE 3) — DSMEM
        if (!waited) cl_wait(), waited = true;
        __syncwarp();
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int c = lane + 32 * u;
          if (c < 54) {
            const int off = c < 48 ? c * 16 : c < 51 ? (c - 48) * 16 : (c - 51) * 16;
            const unsigned char* src = c < 48 ? reinterpret_cast<const unsigned char*>(S.pst + ls * kPSlot)
                                              : reinterpret_cast<const unsigned char*>(c < 51 ? S.il[ls] : S.lw[ls]);
            const unsigned char* dst = c < 48 ? reinterpret_cast<const unsigned char*>(S.pst + it * kPSlot)
                                              : reinterpret_cast<const unsigned char*>(c < 51 ? S.il[it] : S.lw[it]);
            cl_st16(cl_addr(dst + off, gT), *reinterpret_cast<const uint4*>(src + off));
          }
        }
      }
    }
    if (!waited) cl_wait();
  }
  if (cq == 0) {
    S.wmx[warp][gq] = wm[0];
    if (gq + 8 < kQH) S.wmx[warp][gq + 8] = wm[1];
  }
  __syncthreads();
  if (tid < kQH) {  // max over this CTA's warps
    float mx = -INFINITY;
#pragma unroll
    for (int w = 0; w < kW; w++) mx = fmaxf(mx, S.wmx[w][tid]);
    S.mxl[tid] = mx;
  }
  cl_arrive();  // this CTA's mxl and 17th-tile copies released; pass 2's first V tiles go in flight meanwhile
  const int d0 = warp * 32;
  auto issue2 = [&](uint4* f, int tk) {  // V dims [d0, d0+32) of tokens 8i + lane/4 (tk = lane's token)
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
  for (int u = 0; u < 4; u++) tq[u] = u < ng ? tok2(u) : -1;
#pragma unroll
  for (int u = 0; u < 3; u++)
    if (u < ng) issue2(rv[u], tq[u]);
  cl_wait();  // every CTA's mxl and the group's P / 1/l / LSE in place
  if (tid < kQH) {
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < kC; k++) mx = fmaxf(mx, cl_ldf(cl_addr(&S.mxl[tid], k)));
    S.mx[tid] = mx;
  }
  __syncthreads();
  for (int i = tid; i < ng * kQH; i += kW * 32) {
    const int t = i / kQH, h = i - t * kQH;
    S.lw[t][h] = exp2f(S.lw[t][h] - S.mx[h]);
  }
  __syncthreads();
  if (tid < kQH) {  // k_merge's den of group g: Σ w in tile order, its zero terms (tiles ≥ ntl) as one + 0
    float dg = 0.f;
    for (int i = 0; i < ng; i++) dg += S.lw[i][tid];
    S.den[tid] = ng < kMT ? __fadd_rn(dg, 0.f) : dg;
  }

  // ---- pass 2: dims [d0, d0 + 32) of every tile of the group ----
  float num[4][4];
#pragma unroll
  for (int nt = 0; nt < 4; nt++) num[nt][0] = num[nt][1] = num[nt][2] = num[nt][3] = 0.f;
  {
    const bf16* zr = reinterpret_cast<const bf16*>(S.zero);
#pragma unroll 1
    for (int ib = 0; ib < ng; ib += 4) {
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int i = ib + u;
        if (i >= ng) break;
        if (i + 3 < ng) issue2(rv[(u + 3) & 3], tq[(u + 3) & 3]);  // tile i + 3
        tq[u] = i + 4 < ng ? tok2(i + 4) : -1;
#pragma unroll
        for (int k = 0; k < 4; k++)
          *reinterpret_cast<uint4*>(sl + (8 * k + (lane >> 2)) * kSlS + 8 * (lane & 3)) = rv[u][k];
        __syncwarp();
        const bf16* pt = S.pst + i * kPSlot;
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
            ldsm_x4_t(Bf, sl + (kk + (lane & 15)) * kSlS + 16 * np + (lane >> 4) * 8);
            mma_bf16(o[2 * np], A, Bf[0], Bf[1]);
            mma_bf16(o[2 * np + 1], A, Bf[2], Bf[3]);
          }
        }
        __syncwarp();
        const bool hi = gq + 8 < kQH;
        const float il0 = S.il[i][gq], w0 = S.lw[i][gq], il1 = hi ? S.il[i][gq + 8] : 0.f, w1 = hi ? S.lw[i][gq + 8] : 0.f;
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
  __syncthreads();  // staging and Q free → part
  float* part = reinterpret_cast<float*>(smraw);  // [kQH][256]
#pragma unroll
  for (int nt = 0; nt < 4; nt++) {
    const int dl = d0 + nt * 8 + cq * 2;
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int h = gq + 8 * hh;
      if (h < kQH) {
        float x = num[nt][2 * hh], y = num[nt][2 * hh + 1];
        if (ng < kMT) x = __fadd_rn(x, 0.f), y = __fadd_rn(y, 0.f);  // k_merge's zero terms
        *reinterpret_cast<float2*>(part + h * 256 + dl) = make_float2(x, y);
      }
    }
  }
  cl_arrive();
  cl_wait();  // every group's num / den in place

  // ---- combine: heads 3g..3g+2; a thread → 4 dims of one head; DSMEM loads first ----
  float4 xn[kC];
  float xd[kC];
#pragma unroll
  for (int k = 0; k < kC; k++) {
    xn[k] = cl_ld16f(cl_addr(part + h4 * 256 + d4, k));
    xd[k] = cl_ldf(cl_addr(&S.den[h4], k));
  }
  float nm[4] = {xn[0].x, xn[0].y, xn[0].z, xn[0].w}, dn = xd[0];
#pragma unroll
  for (int k = 1; k < kC; k++)  // k_merge: num_0, then += num_g in group order (den likewise)
    nm[0] = nm[0] + xn[k].x, nm[1] = nm[1] + xn[k].y, nm[2] = nm[2] + xn[k].z, nm[3] = nm[3] + xn[k].w, dn = dn + xd[k];
  cl_arrive();  // this CTA's reads of its siblings are done (values consumed); they may leave after the wait below
  if (tid < kNO) {
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gate4);
    const float2 ga = __bfloat1622float2(g2[0]), gb = __bfloat1622float2(g2[1]);
    const float gts[4] = {ga.x, ga.y, gb.x, gb.y};
    bf16 y[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float ov = bfr(dn > 0.f ? nm[e] / dn : 0.f);
      y[e] = __float2bfloat16_rn(ov * (1.f / (1.f + __expf(-gts[e]))));
    }
    *reinterpret_cast<uint2*>(a.attn + size_t(r) * kQH * 256 + h4 * 256 + d4) = *reinterpret_cast<const uint2*>(y);
  }
  cl_wait();  // no CTA leaves while a sibling may still read its smem
}

}  // namespace p21

void p21_init() {
  CUDA_CHECK(cudaFuncSetAttribute(p21::k_dattn, cudaFuncAttributeMaxDynamicSharedMemorySize, p21::kSmem));
}

// Calls where k_attn + k_merge stay faster (one CTA per tile, every load in flight at once, small partials): rows ×
// tiles < 280 + 6·rows (tiles from the max_pos hint — a dense row holds p+1 tokens; unknown → 65, i.e. M ≤ 4; the
// cluster kernel's latency floor grows with its clusters), or one row per sequence up to 6 (MTP draft passes, plain
// decode). Measured on C: results/runs/20261113-p021.
bool p21_base(int M, int B, int max_pos) {
  const int tiles = max_pos < 0 ? kMaxTiles : std::min(kMaxTiles, (max_pos + kTileN) / kTileN);
  return M * tiles < 280 + 6 * M || (M == B && M <= 6);
}

void p21_attn(const qsa_dev::PAttnArgs& a, int M, cudaStream_t s) {
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  at[1].id = cudaLaunchAttributeClusterDimension;
  at[1].val.clusterDim.x = p21::kC, at[1].val.clusterDim.y = 1, at[1].val.clusterDim.z = 1;
  cudaLaunchConfig_t cfg{dim3(p21::kC, M), dim3(p21::kW * 32), size_t(p21::kSmem), s, at, 2};
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, p21::k_dattn, a));
}
