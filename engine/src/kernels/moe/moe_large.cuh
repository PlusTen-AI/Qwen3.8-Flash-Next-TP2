// moe_large.cuh — large-M (prefill) path of moe_forward, 64 < M ≤ 8192. Included once by moe.cu (one translation unit:
// same device helpers, same weight layout (decode A-fragment blocks, read in place — no second copy), same MoeW).
//
// Launches (stream-ordered, graph-capturable, grids depend on M only):
//   k_router  : logits = bf16 router GEMM, CTA 128 experts × 64 rows.
//   k_topk    : per 64-row block: softmax/top-10/renormalise per row (warp per row, decode code), per-expert row masks.
//   k_scan    : per expert: row count, rows in earlier blocks, pair offset; routed 64-row token tiles.
//   k_place   : pairs sorted by (expert, row): row, weight; per row its pairs in expert-id order.
//   k_up      : routed gate/up (NVFP4) per (expert, 64-pair tile, 128 rows = 64 gate ‖ 64 up) → h_r [pairs][320]
//               = bf16(bf16(silu(g))·u).
//   k_down_sup: routed down per (expert, 64-pair tile, 640 outputs): y[pair] = bf16(acc·(w·g2)) (DRAM-bound), with the
//               shared gate/up items (MXFP8, same code as k_up, → h_s) interleaved so they run in its shadow; the first
//               feature block of a row tile also forms the shared gate σ = bf16(sigmoid(bf16(x·wg))) from the staged rows.
//   k_sdown   : shared down per (32 rows, 640 outputs): se = bf16(σ·bf16(acc)); then out = bf16(bf16(Σ_e y_e) + se),
//               Σ in fp32 over the row's pairs in expert-id order.
// Bitwise equal to the decode path row by row: every output element is computed with the same mma.m16n8k16 instructions
// on the same operand values in the same order, and the same fp32 fold points —
//   router: 3 K parts (56/56/48 k-steps) × 8 chains (7/7/6 k-steps): part = Σ chains in order, logit = bf16(Σ parts);
//   gate/up: 5 chains of 32 k-steps, summed in order; down: one chain of 20 k-steps.
// No float atomics; integer atomics only build per-block expert masks (order-free OR). Results do not depend on M, on
// the other rows, or on where a pair sits in the work lists.
// Workspace: one process-wide allocation for M = 8192 (refcounted by moe_upload/moe_free), shared by every MoeW — so
// large-M calls must be stream-ordered with each other (one layer at a time), as the per-layer decode scratch already is.

namespace {
constexpr int kMaxML = 8192;
constexpr int LBT = 64;                                  // token tile (router, gate/up, routed down)
constexpr int kMaxPairsL = kMaxML * kTopK;               // 81,920
constexpr int kMaxTilesL = kExperts + kMaxPairsL / LBT;  // ≥ Σ_e ceil(rows_e / 64)
constexpr int kMaxBlkL = kMaxML / 64;                    // 64-row plan blocks
constexpr int UP_ST = 3;                                 // gate/up cp.async stages (4 k-steps each)
constexpr int DN_NS = 4;                                 // routed down: CTAs per token tile (640 outputs each)
constexpr int SD_BT = 32, SD_NS = 4;                     // shared down: rows per CTA, CTAs per row tile

struct LWs {
  bf16* logits;               // [M][512]
  int* tki;                   // [M][10] expert ids, top-k order
  float* tkw;                 // [M][10] renormalised weights
  float* sig;                 // [M] bf16(sigmoid(bf16 logit))
  unsigned long long* bmsk;   // [blk][512] rows of 64-row block blk using expert e
  int *bcnt, *bbase;          // [blk][512] popcount / rows of e in earlier blocks
  int *ecnt, *eoff;           // [512] rows per expert, first pair
  int* tt;                    // [kMaxTilesL] routed token tiles: e·256 + tile index within e
  int* meta;                  // [0] routed token tiles, [1] distinct experts
  int* utok;                  // [pairs] row of each pair (pairs sorted by expert, rows ascending)
  float* uw;                  // [pairs] its top-k weight
  int* tpair;                 // [M][10] pairs of a row in expert-id order
  bf16* hr;                   // [pairs][320]
  bf16* hs;                   // [M][320]
  bf16* y;                    // [pairs][2560]
  void* base;
  size_t bytes;
};
LWs g_lw{};
int g_lw_ref = 0;

struct LArgs {
  const bf16* v;
  bf16* out;
  int M;
  const uint8_t* ex;
  const float* gs;
  const uint4* rt;
  const uint8_t *sw13, *ss13, *sw2, *ss2;
  const bf16* sg;
  LWs w;
  int* topk_out;
  float* topw_out;
  bf16 *dbg_routed, *dbg_shared;
};

// ---- device helpers --------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* s, const void* g, bool pred = true) {  // !pred → 16 zero bytes
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(s)), "l"(g), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp8(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void cp4(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(s)), "l"(g) : "memory");
}
__device__ __forceinline__ void pf_l2(const void* g, uint32_t bytes) {  // bulk prefetch into L2 (16-byte multiples)
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(g), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ uint32_t pick4(const uint4& w, int i) { return i == 0 ? w.x : i == 1 ? w.y : i == 2 ? w.z : w.w; }

// B tiles in shared memory: token rows of 16-byte chunks, chunk c of row t stored at (c & ~7) | ((c ^ t) & 7) (ldmatrix
// conflict-free). For an ldmatrix.x4 over n8 tiles (2jp, 2jp+1) × chunks (c, c+1): lane L addresses row
// 16jp + 8·((L>>3)>>1) + (L&7), chunk c + ((L>>3)&1) → regs = b[2jp][0], b[2jp][1], b[2jp+1][0], b[2jp+1][1].
__device__ __forceinline__ int swz(int c, int t) { return (c & ~7) | ((c ^ t) & 7); }

// ---- k_router -----------------------------------------------------------------------------------------------------------
// grid 4 × 64-row tiles (the 4 expert blocks of a row tile adjacent: rows come from DRAM once). CTA 128 experts × 64 rows,
// warp w = router tiles 8·by + 2w, +1 × 64 rows. A (bf16 fragments, decode layout) and rows through a 4-stage cp.async
// ring of 2 k-steps (rows of 64 B, chunk c of row t at c ^ ((t >> 1) & 3)). Registers hold the chain and the part sum;
// the running total of the parts is parked in fp32 in h_r's memory (unused until the routed gate/up), each thread
// re-reading only what it wrote. Fold points are compile-time: the loop runs over periods of 7 stages (14 k-steps,
// chains of 7) and 3 stages (6 k-steps, chains of 6).
constexpr int RT_ST = 4, RT_A = 8 * 2 * 512, RT_B = LBT * 64, RT_STAGE = RT_A + RT_B, RT_SMEM = RT_ST * RT_STAGE;
__global__ void __launch_bounds__(128, 2) k_router(LArgs a) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, q = lane & 3;
  const int by = blockIdx.x & 3, t0 = (blockIdx.x >> 2) * LBT, M = a.M, nrow = min(LBT, M - t0), nj = (nrow + 7) >> 3;
  float* stash = reinterpret_cast<float*>(a.w.hr);
  auto load = [&](int s) {
    uint8_t* sA = sm + (s % RT_ST) * RT_STAGE;
    uint8_t* sB = sA + RT_A;
#pragma unroll
    for (int c = tid; c < RT_A / 16; c += 128) {
      const int t8 = c >> 6, o = c & 63;
      cp16(sA + c * 16, reinterpret_cast<const uint8_t*>(a.rt + (size_t(8 * by + t8) * KS1 + 2 * s) * 32) + o * 16);
    }
#pragma unroll
    for (int c = tid; c < LBT * 4; c += 128) {
      const int t = c >> 2, ch = c & 3;
      const bool ok = t < nrow;
      cp16(sB + t * 64 + ((ch ^ ((t >> 1) & 3)) << 4), a.v + size_t(t0 + (ok ? t : 0)) * H + s * 32 + ch * 8, ok);
    }
  };
  const int lm = lane >> 3, lr = lane & 7;
  const uint32_t lrow = (8 * (lm >> 1) + lr) * 64;
  float acc[2][8][4], prt[2][8][4];
#pragma unroll
  for (int m = 0; m < 2; m++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int x = 0; x < 4; x++) acc[m][j][x] = prt[m][j][x] = 0.f;
#pragma unroll
  for (int s = 0; s < RT_ST - 1; s++) {
    load(s);
    cp_commit();
  }
  auto stage = [&](int s) {  // wait for stage s, start loading s + 3
    cp_wait<RT_ST - 2>();
    __syncthreads();
    if (s + RT_ST - 1 < KS1 / 2) load(s + RT_ST - 1);
    cp_commit();
  };
  auto kstep = [&](int s, int kk) {
    const uint8_t* sA = sm + (s % RT_ST) * RT_STAGE;
    const uint32_t bb = smem_u32(sA + RT_A) + lrow + (((2 * kk + (lm & 1)) ^ ((lr >> 1) & 3)) << 4);
    uint4 w[2];
#pragma unroll
    for (int m = 0; m < 2; m++) w[m] = *reinterpret_cast<const uint4*>(sA + (2 * warp + m) * 1024 + kk * 512 + lane * 16);
    uint32_t b[8][2];
#pragma unroll
    for (int jp = 0; jp < 4; jp++)
      if (2 * jp < nj) ldsm4(b[2 * jp][0], b[2 * jp][1], b[2 * jp + 1][0], b[2 * jp + 1][1], bb + jp * 1024);
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (j < nj)
#pragma unroll
        for (int m = 0; m < 2; m++) mma(acc[m][j], w[m].x, w[m].y, w[m].z, w[m].w, b[j][0], b[j][1]);
  };
  auto fold = [&]() {  // chain end: part += chain
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int x = 0; x < 4; x++) {
          prt[m][j][x] += acc[m][j][x];
          acc[m][j][x] = 0.f;
        }
  };
  auto part_end = [&](int pe) {  // total = ((0 + p0) + p1) + p2
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
      for (int j = 0; j < 8; j++)
        if (j < nj)
#pragma unroll
          for (int x = 0; x < 4; x++) {
            const int t = 8 * j + 2 * q + (x & 1);
            if (t < nrow) {
              const size_t o = size_t(t0 + t) * kExperts + 16 * (8 * by + 2 * warp + m) + r + 8 * (x >> 1);
              const float tot = (pe == 0 ? 0.f : __ldcg(stash + o)) + prt[m][j][x];
              if (pe == 2) a.w.logits[o] = __float2bfloat16_rn(tot);
              else __stcg(stash + o, tot);
            }
            prt[m][j][x] = 0.f;
          }
  };
  // parts 0, 1 (k-steps 0..111): chains of 7, period 14 k-steps = 7 stages, chain ends after local k-steps 7 and 14
  for (int p = 0; p < 8; p++) {
#pragma unroll
    for (int sp = 0; sp < 7; sp++) {
      const int s = 7 * p + sp;
      stage(s);
#pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        kstep(s, kk);
        if ((sp == 3 && kk == 0) || (sp == 6 && kk == 1)) fold();
      }
    }
    if ((p & 3) == 3) part_end(p >> 2);
  }
  // part 2 (k-steps 112..159): chains of 6, period 3 stages
  for (int p = 0; p < 8; p++) {
#pragma unroll
    for (int sp = 0; sp < 3; sp++) {
      const int s = 56 + 3 * p + sp;
      stage(s);
#pragma unroll
      for (int kk = 0; kk < 2; kk++) kstep(s, kk);
    }
    fold();
  }
  part_end(2);
}

// ---- k_topk: per 64-row block --------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_topk(LArgs a) {
  __shared__ int tk_i[64][kTopK];
  __shared__ unsigned long long msk[kExperts];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, b = blockIdx.x, row0 = 64 * b;
  const int nr = min(64, a.M - row0);
  for (int e = tid; e < kExperts; e += 256) msk[e] = 0;
  for (int lr = warp; lr < nr; lr += 8) {
    const int tok = row0 + lr;
    // softmax / top-10 / renormalise: same code as route_tail
    float p[16];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      p[i] = __bfloat162float(a.w.logits[size_t(tok) * kExperts + lane + 32 * i]);
      mx = fmaxf(mx, p[i]);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, o));
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      p[i] = expf(p[i] - mx);
      s += p[i];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(~0u, s, o);
    const float rs = 1.0f / s;
#pragma unroll
    for (int i = 0; i < 16; i++) p[i] *= rs;
    unsigned taken = 0, bp = 0;
    int bi = 0;
    auto lane_best = [&]() {
      bp = 0;
      bi = 16;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const unsigned u = __float_as_uint(p[i]);
        if (!((taken >> i) & 1) && (bi == 16 || u > bp)) { bp = u; bi = i; }
      }
    };
    lane_best();
    float tot = 0.f, wk = 0.f;
    int ik = 0;
#pragma unroll
    for (int k = 0; k < kTopK; k++) {
      const unsigned best = __reduce_max_sync(~0u, bp);
      const unsigned id = __reduce_min_sync(~0u, (bp == best && bi < 16) ? unsigned(lane + 32 * bi) : 0xffffffffu);
      if (lane == int(id & 31)) {
        taken |= 1u << (id >> 5);
        lane_best();
      }
      if (lane == k) { ik = int(id); wk = __uint_as_float(best); }
      tot += __uint_as_float(best);
    }
    if (lane < kTopK) {
      const float w = wk / tot;
      tk_i[lr][lane] = ik;
      a.w.tki[tok * kTopK + lane] = ik;
      a.w.tkw[tok * kTopK + lane] = w;
      if (a.topk_out) a.topk_out[tok * kTopK + lane] = ik;
      if (a.topw_out) a.topw_out[tok * kTopK + lane] = w;
    }
  }
  __syncthreads();
  for (int i = tid; i < nr * kTopK; i += 256) atomicOr(&msk[tk_i[i / kTopK][i % kTopK]], 1ull << (i / kTopK));
  __syncthreads();
  for (int e = tid; e < kExperts; e += 256) {
    a.w.bmsk[b * kExperts + e] = msk[e];
    a.w.bcnt[b * kExperts + e] = __popcll(msk[e]);
  }
}

// ---- k_scan: one CTA, thread e = expert ---------------------------------------------------------------------------------
__global__ void __launch_bounds__(kExperts) k_scan(LArgs a, int nb) {
  __shared__ int ws[kExperts / 32][3];
  const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
  int run = 0;
  for (int b0 = 0; b0 < nb; b0 += 16) {
    int c[16];
#pragma unroll
    for (int u = 0; u < 16; u++) c[u] = b0 + u < nb ? __ldcg(a.w.bcnt + (b0 + u) * kExperts + e) : 0;
#pragma unroll
    for (int u = 0; u < 16; u++)
      if (b0 + u < nb) {
        a.w.bbase[(b0 + u) * kExperts + e] = run;
        run += c[u];
      }
  }
  const int nt = (run + LBT - 1) / LBT, used = run > 0;
  int sc = run, st = nt, su = used;  // inclusive scans
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int xc = __shfl_up_sync(~0u, sc, o), xt = __shfl_up_sync(~0u, st, o), xu = __shfl_up_sync(~0u, su, o);
    if (lane >= o) { sc += xc; st += xt; su += xu; }
  }
  if (lane == 31) { ws[warp][0] = sc; ws[warp][1] = st; ws[warp][2] = su; }
  __syncthreads();
  int off = sc - run, toff = st - nt;
  for (int w = 0; w < warp; w++) { off += ws[w][0]; toff += ws[w][1]; }
  a.w.ecnt[e] = run;
  a.w.eoff[e] = off;
  for (int j = 0; j < nt; j++) a.w.tt[toff + j] = e * 256 + j;
  if (e == kExperts - 1) {
    int u = 0;
    for (int w = 0; w < kExperts / 32; w++) u += ws[w][2];
    a.w.meta[0] = toff + nt;
    a.w.meta[1] = u;
  }
}

// ---- k_place: per 64-row block --------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) k_place(LArgs a) {
  const int b = blockIdx.x, row0 = 64 * b, nr = min(64, a.M - row0);
  for (int i = threadIdx.x; i < nr * kTopK; i += 256) {
    const int lr = i / kTopK, tok = row0 + lr;
    const int e = a.w.tki[tok * kTopK + i % kTopK];
    const int pair = a.w.eoff[e] + a.w.bbase[b * kExperts + e] + __popcll(a.w.bmsk[b * kExperts + e] & ((1ull << lr) - 1));
    int pos = 0;
#pragma unroll
    for (int j = 0; j < kTopK; j++) pos += a.w.tki[tok * kTopK + j] < e;
    a.w.utok[pair] = tok;
    a.w.uw[pair] = a.w.tkw[tok * kTopK + i % kTopK];
    a.w.tpair[tok * kTopK + pos] = pair;
  }
}

// ---- gate/up GEMM + silu·mul -----------------------------------------------------------------------------------
// item = (token tile T, feature block fb of 8 gate/up m16 tiles = 64 gate + 64 up rows); 4 warps × (2 m16 tiles × 64
// tokens). A (weight fragments, decode layout), scales and gathered token rows go through a 3-stage cp.async ring of 4
// k-steps. Five chains of 32 k-steps (the decode K parts), each started from 0 and added in order.
template <bool FP4>
struct UpL {
  static constexpr int A = FP4 ? 8 * 512 : 8 * 1024;  // 8 m16 tiles × 4 k-steps of fragments
  static constexpr int S = FP4 ? 64 * 8 : 64 * 4;     // 8 tiles × 8 rows × (4 k-steps | 2 groups) × (gate, up)
  static constexpr int B = LBT * 128;                 // 64 token rows × 64 k
  static constexpr int STAGE = A + S + B;
  static constexpr int SMEM = UP_ST * STAGE;
};

template <bool FP4>
__device__ __forceinline__ void lup_item(const LArgs& a, int T, int fb, uint8_t* sm, int* stok) {
  using L = UpL<FP4>;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, q = lane & 3;
  int e = 0, cnt, p0;
  if constexpr (FP4) {
    if (T >= __ldg(a.w.meta)) return;
    const int code = __ldg(a.w.tt + T);
    e = code >> 8;
    cnt = min(LBT, __ldg(a.w.ecnt + e) - LBT * (code & 255));
    p0 = __ldg(a.w.eoff + e) + LBT * (code & 255);
  } else {
    p0 = LBT * T;
    cnt = min(LBT, a.M - p0);
  }
  if (tid < LBT) stok[tid] = tid < cnt ? (FP4 ? __ldg(a.w.utok + p0 + tid) : p0 + tid) : -1;
  __syncthreads();
  const int nj = (cnt + 7) >> 3;
  const uint8_t* wA = FP4 ? a.ex + size_t(e) * kEB : a.sw13;
  const uint8_t* wS = FP4 ? wA + kW13 : a.ss13;
  auto load = [&](int s) {
    uint8_t* sA = sm + (s % UP_ST) * L::STAGE;
    uint8_t* sS = sA + L::A;
    uint8_t* sB = sS + L::S;
    constexpr int CPT = L::A / 128;  // 16-byte chunks per m16 tile per stage
#pragma unroll
    for (int c = tid; c < L::A / 16; c += 128) {
      const int i = 8 * fb + c / CPT, o = c % CPT;
      cp16(sA + c * 16, FP4 ? wA + (size_t(i) * (KS1 / 4) + s) * 512 + o * 16 : wA + (size_t(i) * (KS1 / 2) + 2 * s) * 512 + o * 16);
    }
    if (tid < 64) {
      const int i = 8 * fb + (tid >> 3), rr = tid & 7;
      if constexpr (FP4) cp8(sS + tid * 8, wS + ((size_t(i) * 8 + rr) * KS1 + 4 * s) * 2);
      else cp4(sS + tid * 4, wS + ((size_t(i) * 8 + rr) * (KS1 / 2) + 2 * s) * 2);
    }
#pragma unroll
    for (int c = tid; c < LBT * 8; c += 128) {
      const int t = c >> 3, ch = c & 7, tok = stok[t];
      cp16(sB + t * 128 + (swz(ch, t) << 4), a.v + size_t(tok < 0 ? 0 : tok) * H + s * 64 + ch * 8, tok >= 0);
    }
  };
  // small tiles (≤ 32 rows) only stream weights: L2 prefetch of the next 8 stages of A every 8 stages, scales once
  const bool pf = cnt <= 32;
  auto prefetch = [&](int s0) {
    if (tid < 8) {
      const int i = 8 * fb + tid;
      if constexpr (FP4) pf_l2(wA + (size_t(i) * (KS1 / 4) + s0) * 512, 8 * 512);
      else pf_l2(wA + (size_t(i) * (KS1 / 2) + 2 * s0) * 512, 8 * 1024);
    }
  };
  if (pf && tid < 8) pf_l2(wS + size_t(8 * fb + tid) * 8 * (FP4 ? KS1 : KS1 / 2) * 2, 8 * (FP4 ? KS1 : KS1 / 2) * 2);
  if (pf) {
    prefetch(0);
    prefetch(8);
  }
  const int lm = lane >> 3, lr = lane & 7;
  const uint32_t lrow = (8 * (lm >> 1) + lr) * 128;
  float acc[2][8][4], sum[2][8][4];
#pragma unroll
  for (int m = 0; m < 2; m++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int x = 0; x < 4; x++) acc[m][j][x] = sum[m][j][x] = 0.f;
#pragma unroll
  for (int s = 0; s < UP_ST - 1; s++) {
    load(s);
    cp_commit();
  }
  // shared expert gate (shared gate/up, fb 0): x·wg with the decode warp's partition — lane L adds chunks L, L+32, ...
  // (8 elements each, fmaf in element order); thread (row tid/2, half h) keeps lanes 16h..16h+15 (dA: +0..7, dB: +8..15).
  float dA[8], dB[8];
#pragma unroll
  for (int i = 0; i < 8; i++) dA[i] = dB[i] = 0.f;
  const bool do_sig = !FP4 && fb == 0 && (tid >> 1) < cnt;
  for (int s = 0; s < KS1 / 4; s++) {
    cp_wait<UP_ST - 2>();
    __syncthreads();
    if (s + UP_ST - 1 < KS1 / 4) load(s + UP_ST - 1);
    cp_commit();
    if (pf && (s & 7) == 0 && s + 16 < KS1 / 4) prefetch(s + 16);
    const uint8_t* sA = sm + (s % UP_ST) * L::STAGE;
    if (do_sig && (tid & 1) == ((s >> 1) & 1)) {  // chunks 8s..8s+7 = lanes 8(s%4) + ch
      const int t = tid >> 1;
      const uint8_t* xb = sA + L::A + L::S + t * 128;
#pragma unroll
      for (int ch = 0; ch < 8; ch++) {
        const uint4 x = *reinterpret_cast<const uint4*>(xb + (swz(ch, t) << 4));
        const uint4 g = __ldg(reinterpret_cast<const uint4*>(a.sg) + 8 * s + ch);
        const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, gw[4] = {g.x, g.y, g.z, g.w};
        float d = (s & 1) ? dB[ch] : dA[ch];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xs[j]));
          const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[j]));
          d = fmaf(xf.x, gf.x, d);
          d = fmaf(xf.y, gf.y, d);
        }
        if (s & 1) dB[ch] = d;
        else dA[ch] = d;
      }
    }
    const uint8_t* sS = sA + L::A;
    const uint32_t bb = smem_u32(sS + L::S) + lrow;
    uint4 wa[2][FP4 ? 1 : 2];
    uint32_t sc[2][2];
#pragma unroll
    for (int m = 0; m < 2; m++) {
      const int t8 = 2 * warp + m;
      if constexpr (FP4) {
        wa[m][0] = *reinterpret_cast<const uint4*>(sA + t8 * 512 + lane * 16);
        const uint2 s2 = *reinterpret_cast<const uint2*>(sS + (t8 * 8 + r) * 8);
        sc[m][0] = s2.x;
        sc[m][1] = s2.y;
      } else {
        wa[m][0] = *reinterpret_cast<const uint4*>(sA + t8 * 1024 + lane * 16);
        wa[m][1] = *reinterpret_cast<const uint4*>(sA + t8 * 1024 + 512 + lane * 16);
        sc[m][0] = *reinterpret_cast<const uint32_t*>(sS + (t8 * 8 + r) * 4);
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      uint32_t af[2][4];
#pragma unroll
      for (int m = 0; m < 2; m++) {
        uint32_t lo, hi;
        if constexpr (FP4) {
          sc_fp4(sc[m][kk >> 1] >> (16 * (kk & 1)), lo, hi);
          dq_fp4(pick4(wa[m][0], kk), lo, hi, af[m]);
        } else {
          const int c = kk >> 1, h = kk & 1;
          const uint4& w = wa[m][c];
          sc_fp8(sc[m][0] >> (16 * c), lo, hi);
          dq_fp8(h ? w.z : w.x, h ? w.w : w.y, lo, hi, af[m]);
        }
      }
      uint32_t b[8][2];
#pragma unroll
      for (int jp = 0; jp < 4; jp++)
        if (2 * jp < nj) ldsm4(b[2 * jp][0], b[2 * jp][1], b[2 * jp + 1][0], b[2 * jp + 1][1], bb + jp * 2048 + (((2 * kk + (lm & 1)) ^ lr) << 4));
#pragma unroll
      for (int j = 0; j < 8; j++)
        if (j < nj)
#pragma unroll
          for (int m = 0; m < 2; m++) mma(acc[m][j], af[m][0], af[m][1], af[m][2], af[m][3], b[j][0], b[j][1]);
    }
    if ((s & 7) == 7) {  // end of a 32-k-step chain
#pragma unroll
      for (int m = 0; m < 2; m++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
          for (int x = 0; x < 4; x++) {
            sum[m][j][x] += acc[m][j][x];
            acc[m][j][x] = 0.f;
          }
    }
  }
  if (!FP4 && fb == 0) {  // xor-shuffle tree of k_route over the 32 lane partials: level o adds lanes L and L + o
    float v[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      v[i] = dA[i] + __shfl_xor_sync(~0u, dA[i], 1);
      v[8 + i] = dB[i] + __shfl_xor_sync(~0u, dB[i], 1);
    }
#pragma unroll
    for (int o = 8; o; o >>= 1)
#pragma unroll
      for (int i = 0; i < o; i++) v[i] += v[i + o];
    if (do_sig && (tid & 1) == 0) a.w.sig[p0 + (tid >> 1)] = bf16r(1.0f / (1.0f + expf(-bf16r(v[0]))));
  }
  cp_wait<0>();
  __syncthreads();
  // h tile [64 tokens][64 features] in shared memory, then 128-byte rows out
  bf16* sh = reinterpret_cast<bf16*>(sm);
  const float gg = FP4 ? a.gs[3 * e] : 1.f, gu = FP4 ? a.gs[3 * e + 1] : 1.f;
#pragma unroll
  for (int m = 0; m < 2; m++)
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (j < nj)
#pragma unroll
        for (int x = 0; x < 2; x++) {
          const int t = 8 * j + 2 * q + x, f = 8 * (2 * warp + m) + r;
          sh[t * 64 + f] = FP4 ? silu_mul(bf16r(sum[m][j][x] * gg), bf16r(sum[m][j][2 + x] * gu))
                               : silu_mul(bf16r(sum[m][j][x]), bf16r(sum[m][j][2 + x]));
        }
  __syncthreads();
  bf16* dst = FP4 ? a.w.hr : a.w.hs;
  for (int c = tid; c < cnt * 8; c += 128) {
    const int t = c >> 3, ch = c & 7;
    *reinterpret_cast<uint4*>(dst + size_t(p0 + t) * I + 64 * fb + 8 * ch) = *reinterpret_cast<const uint4*>(sh + t * 64 + 8 * ch);
  }
}

__global__ void __launch_bounds__(128, 2) k_up(LArgs a) {  // routed gate/up: item = (token tile, feature block)
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ int stok[LBT];
  lup_item<true>(a, blockIdx.x / 5, blockIdx.x % 5, sm, stok);
}

// ---- routed down, item = (token tile T, 640 outputs) -------------------------------------------------------------
// h tile [64 pairs][320] in shared memory (40 KB); 5 output blocks of 128 rows, warp = 2 m16 tiles × 64 pairs, A
// fragments straight from global into registers (chunk c of the next block is requested once chunk c is used).
__device__ __forceinline__ void ldown_item(const LArgs& a, int T, int part, uint8_t* sm) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, q = lane & 3;
  if (T >= __ldg(a.w.meta)) return;
  const int code = __ldg(a.w.tt + T), e = code >> 8;
  const int cnt = min(LBT, __ldg(a.w.ecnt + e) - LBT * (code & 255)), p0 = __ldg(a.w.eoff + e) + LBT * (code & 255);
  const int nj = (cnt + 7) >> 3;
  for (int c = tid; c < LBT * (I / 8); c += 128) {
    const int t = c / (I / 8), ch = c % (I / 8);
    const bool ok = t < cnt;
    cp16(sm + t * (I * 2) + (swz(ch, t) << 4), a.w.hr + size_t(p0 + (ok ? t : 0)) * I + ch * 8, ok);
  }
  cp_commit();
  const uint8_t* eb = a.ex + size_t(e) * kEB + kW13 + kS13;
  const float g2 = a.gs[3 * e + 2];
  float wsc[8][2];
#pragma unroll
  for (int j = 0; j < 8; j++)
#pragma unroll
    for (int x = 0; x < 2; x++) {
      const int t = 8 * j + 2 * q + x;
      wsc[j][x] = t < cnt ? __ldg(a.w.uw + p0 + t) * g2 : 0.f;
    }
  constexpr int NB = H / 128 / DN_NS;  // 5 output blocks
  const int nb0 = part * NB;
  uint4 wa[2][5];
  uint2 sa[2][5];
  auto ldA = [&](int nb, int m, int c) {
    const int n = 8 * nb + 2 * warp + m;
    wa[m][c] = ld_stream(reinterpret_cast<const uint4*>(eb) + (size_t(n) * (KS2 / 4) + c) * 32 + lane);
    sa[m][c] = ld_stream2(reinterpret_cast<const uint2*>(eb + kW2 + (size_t(n) * 8 + r) * KS2 * 2) + c);
  };
#pragma unroll
  for (int m = 0; m < 2; m++)
#pragma unroll
    for (int c = 0; c < 5; c++) ldA(nb0, m, c);
  const int lm = lane >> 3, lr = lane & 7;
  const uint32_t bb = smem_u32(sm) + (8 * (lm >> 1) + lr) * (I * 2);
  cp_wait<0>();
  __syncthreads();
  for (int nb = nb0; nb < nb0 + NB; nb++) {
    float acc[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int x = 0; x < 4; x++) acc[m][j][x] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS2; ks++) {
      const int c = ks >> 2, s4 = ks & 3;
      uint32_t af[2][4];
#pragma unroll
      for (int m = 0; m < 2; m++) {
        uint32_t lo, hi;
        sc_fp4((s4 < 2 ? sa[m][c].x : sa[m][c].y) >> (16 * (s4 & 1)), lo, hi);
        dq_fp4(pick4(wa[m][c], s4), lo, hi, af[m]);
      }
      if (s4 == 3 && nb + 1 < nb0 + NB)
#pragma unroll
        for (int m = 0; m < 2; m++) ldA(nb + 1, m, c);
      const int ch = 2 * ks + (lm & 1);
      const uint32_t co = ((ch & ~7) | ((ch & 7) ^ lr)) << 4;
      uint32_t b[8][2];
#pragma unroll
      for (int jp = 0; jp < 4; jp++)
        if (2 * jp < nj) ldsm4(b[2 * jp][0], b[2 * jp][1], b[2 * jp + 1][0], b[2 * jp + 1][1], bb + jp * 16 * (I * 2) + co);
#pragma unroll
      for (int j = 0; j < 8; j++)
        if (j < nj)
#pragma unroll
          for (int m = 0; m < 2; m++) mma(acc[m][j], af[m][0], af[m][1], af[m][2], af[m][3], b[j][0], b[j][1]);
    }
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
      for (int j = 0; j < 8; j++)
        if (j < nj)
#pragma unroll
          for (int x = 0; x < 4; x++) {
            const int t = 8 * j + 2 * q + (x & 1);
            if (t < cnt)
              a.w.y[size_t(p0 + t) * H + 16 * (8 * nb + 2 * warp + m) + r + 8 * (x >> 1)] = __float2bfloat16_rn(acc[m][j][x] * wsc[j][x & 1]);
          }
  }
}

// routed down (DRAM-bound: writes y) with the shared gate/up items (tensor-bound) spread evenly among them: item b is a
// shared gate/up item iff ⌊(b+1)·ns/n⌋ > ⌊b·ns/n⌋ (index ⌊b·ns/n⌋), else routed down item b − ⌊b·ns/n⌋.
constexpr int DS_SMEM = UpL<false>::SMEM > LBT * I * 2 ? UpL<false>::SMEM : LBT * I * 2;
__global__ void __launch_bounds__(128, 2) k_down_sup(LArgs a, int nd, int ns) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ int stok[LBT];
  const long long n = nd + ns, b = blockIdx.x;
  const int c0 = int(b * ns / n), c1 = int((b + 1) * ns / n);
  if (c1 > c0) lup_item<false>(a, c0 / 5, c0 % 5, sm, stok);
  else ldown_item(a, int(b - c0) / DN_NS, int(b - c0) % DN_NS, sm);
}

// ---- k_sdown: shared down + combine, item = (32 rows, 640 outputs) --------------------------------------------------------
constexpr int SD_SMEM = SD_BT * I * 2 + SD_BT * 128 * 2;  // h_s tile 20 KB + se tile 8 KB
__global__ void __launch_bounds__(128, 2) k_sdown(LArgs a) {
  extern __shared__ __align__(128) uint8_t sm[];
  bf16* sse = reinterpret_cast<bf16*>(sm + SD_BT * I * 2);  // [32 rows][128 outputs]
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, r = lane >> 2, q = lane & 3;
  const int T = blockIdx.x / SD_NS, part = blockIdx.x % SD_NS, t0 = SD_BT * T, cnt = min(SD_BT, a.M - t0);
  const int nj = (cnt + 7) >> 3;
  for (int c = tid; c < SD_BT * (I / 8); c += 128) {
    const int t = c / (I / 8), ch = c % (I / 8);
    const bool ok = t < cnt;
    cp16(sm + t * (I * 2) + (swz(ch, t) << 4), a.w.hs + size_t(t0 + (ok ? t : 0)) * I + ch * 8, ok);
  }
  cp_commit();
  float sg[4][2];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int x = 0; x < 2; x++) {
      const int t = 8 * j + 2 * q + x;
      sg[j][x] = t < cnt ? __ldg(a.w.sig + t0 + t) : 0.f;
    }
  const int lm = lane >> 3, lr = lane & 7;
  const uint32_t bb = smem_u32(sm) + (8 * (lm >> 1) + lr) * (I * 2);
  constexpr int NB = H / 128 / SD_NS;
  cp_wait<0>();
  __syncthreads();
  for (int nb = part * NB; nb < part * NB + NB; nb++) {
    uint4 w[2][10];
    uint32_t sw[2][5];
#pragma unroll
    for (int m = 0; m < 2; m++) {
      const int n = 8 * nb + 2 * warp + m;
#pragma unroll
      for (int c = 0; c < 10; c++) w[m][c] = ld_stream(reinterpret_cast<const uint4*>(a.sw2) + (size_t(n) * 10 + c) * 32 + lane);
#pragma unroll
      for (int j = 0; j < 5; j++) sw[m][j] = ld_stream1(a.ss2 + size_t(n) * 8 * 20 + r * 20 + 4 * j);
    }
    float acc[2][4][4];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int x = 0; x < 4; x++) acc[m][j][x] = 0.f;
#pragma unroll
    for (int c = 0; c < 10; c++) {
      uint32_t lo[2], hi[2];
#pragma unroll
      for (int m = 0; m < 2; m++) sc_fp8(sw[m][c >> 1] >> (16 * (c & 1)), lo[m], hi[m]);
#pragma unroll
      for (int h = 0; h < 2; h++) {
        uint32_t af[2][4];
#pragma unroll
        for (int m = 0; m < 2; m++) dq_fp8(h ? w[m][c].z : w[m][c].x, h ? w[m][c].w : w[m][c].y, lo[m], hi[m], af[m]);
        const int ch = 2 * (2 * c + h) + (lm & 1);
        const uint32_t co = ((ch & ~7) | ((ch & 7) ^ lr)) << 4;
        uint32_t b[4][2];
#pragma unroll
        for (int jp = 0; jp < 2; jp++)
          if (2 * jp < nj) ldsm4(b[2 * jp][0], b[2 * jp][1], b[2 * jp + 1][0], b[2 * jp + 1][1], bb + jp * 16 * (I * 2) + co);
#pragma unroll
        for (int j = 0; j < 4; j++)
          if (j < nj)
#pragma unroll
            for (int m = 0; m < 2; m++) mma(acc[m][j], af[m][0], af[m][1], af[m][2], af[m][3], b[j][0], b[j][1]);
      }
    }
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        if (j < nj)
#pragma unroll
          for (int x = 0; x < 4; x++) {
            const int t = 8 * j + 2 * q + (x & 1);
            sse[t * 128 + 16 * (2 * warp + m) + r + 8 * (x >> 1)] = __float2bfloat16_rn(sg[j][x & 1] * bf16r(acc[m][j][x]));
          }
    __syncthreads();
    // out = bf16(bf16(Σ_k y[pair_k]) + se), pairs in expert-id order; 8 outputs per item
    for (int it = tid; it < cnt * 16; it += 128) {
      const int t = it >> 4, ch = it & 15, tok = t0 + t, col = 128 * nb + 8 * ch;
      uint4 yv[kTopK];
#pragma unroll
      for (int k = 0; k < kTopK; k++)
        yv[k] = __ldcs(reinterpret_cast<const uint4*>(a.w.y + size_t(__ldg(a.w.tpair + tok * kTopK + k)) * H + col));
      float s[8];
#pragma unroll
      for (int i = 0; i < 8; i++) s[i] = 0.f;
#pragma unroll
      for (int k = 0; k < kTopK; k++) {
        const uint32_t u[4] = {yv[k].x, yv[k].y, yv[k].z, yv[k].w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[i]));
          s[2 * i] += f.x;
          s[2 * i + 1] += f.y;
        }
      }
      const uint4 sev = *reinterpret_cast<const uint4*>(sse + t * 128 + 8 * ch);
      const uint32_t su[4] = {sev.x, sev.y, sev.z, sev.w};
      uint32_t o[4], ro[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float2 se = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&su[i]));
        const float r0 = bf16r(s[2 * i]), r1 = bf16r(s[2 * i + 1]);
        const __nv_bfloat162 ov = __floats2bfloat162_rn(r0 + se.x, r1 + se.y);
        const __nv_bfloat162 rv = __floats2bfloat162_rn(r0, r1);
        o[i] = *reinterpret_cast<const uint32_t*>(&ov);
        ro[i] = *reinterpret_cast<const uint32_t*>(&rv);
      }
      *reinterpret_cast<uint4*>(a.out + size_t(tok) * H + col) = make_uint4(o[0], o[1], o[2], o[3]);
      if (a.dbg_routed) *reinterpret_cast<uint4*>(a.dbg_routed + size_t(tok) * H + col) = make_uint4(ro[0], ro[1], ro[2], ro[3]);
      if (a.dbg_shared) *reinterpret_cast<uint4*>(a.dbg_shared + size_t(tok) * H + col) = sev;
    }
    __syncthreads();
  }
}

#include "kernels/moe/p010_moe_down_tune.cuh"  // P010: k_down_sup / k_sdown variants (namespace p010)

// tests: h_s → B-fragment order (the decode path's debug layout)
__global__ void k_hperm(const bf16* hs, bf16* dst, int M) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M * I; i += gridDim.x * blockDim.x)
    dst[(i / I) * I + frag_off(i % I)] = hs[i];
}

// ---- host ------------------------------------------------------------------------------------------------------------------
void lw_acquire() {
  if (g_lw_ref++) return;
  size_t off = 0;
  auto take = [&](size_t bytes) {
    const size_t o = off;
    off += (bytes + 255) & ~size_t(255);
    return o;
  };
  const size_t o_lg = take(sizeof(bf16) * kMaxML * kExperts), o_tki = take(4 * kMaxPairsL), o_tkw = take(4 * kMaxPairsL),
               o_sig = take(4 * kMaxML), o_bm = take(8 * kMaxBlkL * kExperts), o_bc = take(4 * kMaxBlkL * kExperts),
               o_bb = take(4 * kMaxBlkL * kExperts), o_ec = take(4 * kExperts), o_eo = take(4 * kExperts),
               o_tt = take(4 * kMaxTilesL), o_meta = take(16), o_ut = take(4 * kMaxPairsL), o_uw = take(4 * kMaxPairsL),
               o_tp = take(4 * kMaxPairsL), o_hr = take(sizeof(bf16) * kMaxPairsL * I), o_hs = take(sizeof(bf16) * kMaxML * I),
               o_y = take(sizeof(bf16) * kMaxPairsL * H);
  uint8_t* b = nullptr;
  CUDA_CHECK(cudaMalloc(&b, off));
  CUDA_CHECK(cudaMemset(b, 0, off));
  LWs& w = g_lw;
  w.base = b;
  w.bytes = off;
  w.logits = reinterpret_cast<bf16*>(b + o_lg);
  w.tki = reinterpret_cast<int*>(b + o_tki);
  w.tkw = reinterpret_cast<float*>(b + o_tkw);
  w.sig = reinterpret_cast<float*>(b + o_sig);
  w.bmsk = reinterpret_cast<unsigned long long*>(b + o_bm);
  w.bcnt = reinterpret_cast<int*>(b + o_bc);
  w.bbase = reinterpret_cast<int*>(b + o_bb);
  w.ecnt = reinterpret_cast<int*>(b + o_ec);
  w.eoff = reinterpret_cast<int*>(b + o_eo);
  w.tt = reinterpret_cast<int*>(b + o_tt);
  w.meta = reinterpret_cast<int*>(b + o_meta);
  w.utok = reinterpret_cast<int*>(b + o_ut);
  w.uw = reinterpret_cast<float*>(b + o_uw);
  w.tpair = reinterpret_cast<int*>(b + o_tp);
  w.hr = reinterpret_cast<bf16*>(b + o_hr);
  w.hs = reinterpret_cast<bf16*>(b + o_hs);
  w.y = reinterpret_cast<bf16*>(b + o_y);
  CUDA_CHECK(cudaFuncSetAttribute(k_router, cudaFuncAttributeMaxDynamicSharedMemorySize, RT_SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(k_up, cudaFuncAttributeMaxDynamicSharedMemorySize, UpL<true>::SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(k_down_sup, cudaFuncAttributeMaxDynamicSharedMemorySize, DS_SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(k_sdown, cudaFuncAttributeMaxDynamicSharedMemorySize, SD_SMEM));
  p010::setup();
}

void lw_release() {
  if (--g_lw_ref) return;
  CUDA_CHECK(cudaFree(g_lw.base));
  g_lw = LWs{};
}

// routed token tiles: Σ_e ceil(rows_e/64) ≤ (10M + 63·min(512, 10M)) / 64
int max_tiles(int M) { return (kTopK * M + (LBT - 1) * std::min(kExperts, kTopK * M)) / LBT + 1; }

// stage = -1: whole call; else only launches [0, stage] (tests: per-kernel timing)
void moe_large(const MoeW* m, const bf16* v, bf16* out, int M, int* topk_out, float* topw_out, const MoeDbg& dbg,
               cudaStream_t s, cudaEvent_t* ev = nullptr) {
  const LArgs a{v, out, M, m->ex, m->gs, m->rt, m->sw13, m->ss13, m->sw2, m->ss2, m->sg, g_lw, topk_out, topw_out,
                dbg.routed, dbg.shared};
  const int nb = cdiv(M, 64), mt = max_tiles(M);
  int k = 0;
  auto mark = [&] {
    if (ev) CUDA_CHECK(cudaEventRecord(ev[k++], s));
  };
  mark();
  k_router<<<nb * (kExperts / 128), 128, RT_SMEM, s>>>(a);
  mark();
  k_topk<<<nb, 256, 0, s>>>(a);
  mark();
  k_scan<<<1, kExperts, 0, s>>>(a, nb);
  mark();
  k_place<<<nb, 256, 0, s>>>(a);
  mark();
  k_up<<<mt * 5, 128, UpL<true>::SMEM, s>>>(a);
  mark();
  const bool p10 = patch::on(patch::moe_down_tune);
  if (p10) p010::down_sup(a, mt * DN_NS, nb * 5, s);
  else k_down_sup<<<mt * DN_NS + nb * 5, 128, DS_SMEM, s>>>(a, mt * DN_NS, nb * 5);
  mark();
  if (p10) p010::sdown(a, M, s);
  else k_sdown<<<cdiv(M, SD_BT) * SD_NS, 128, SD_SMEM, s>>>(a);
  mark();
  CUDA_CHECK(cudaGetLastError());
  if (dbg.logits) CUDA_CHECK(cudaMemcpyAsync(dbg.logits, g_lw.logits, sizeof(bf16) * M * kExperts, cudaMemcpyDeviceToDevice, s));
  if (dbg.h_shared) {
    k_hperm<<<cdiv(M * I, 256), 256, 0, s>>>(g_lw.hs, dbg.h_shared, M);
    CUDA_CHECK(cudaGetLastError());
  }
  if (dbg.n_experts) CUDA_CHECK(cudaMemcpyAsync(dbg.n_experts, g_lw.meta + 1, sizeof(int), cudaMemcpyDeviceToDevice, s));
}
}  // namespace
