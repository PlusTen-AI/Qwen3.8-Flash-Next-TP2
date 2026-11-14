// p009_qsa_fast_index.cuh — P009 qsa_fast_index (patches.def): prefill indexer without the fp32 score round trip.
// Included by qsa.cu inside its anonymous namespace (qsa_dev's idx_* chain, score bins and constants).
//
// Base prefill indexer: k_pscore writes every score of 1,024 rows (fp32, up to 256 MB) and every 16-row tile reads all
// keys from L2; k_select<true> reads each row's scores twice. At 128K that is 1 GB written + 1 GB read per 8,192-row
// call. P009 (prefill calls of ≥ kP9MinRows rows; smaller ones keep the base indexer, which spreads them over more
// CTAs) computes the same scores and the same exact top-512, but only keeps the scores that can be selected:
//   k_tau    : per row a threshold τ ≤ its 512th score with high probability: the scores of ≤ 1,024 evenly spaced
//              sample keys, τ = the k-th largest (bf16-truncated) with k = kP9Target·samples/visible (~1,280 keys
//              expected ≥ τ). Rows that see ≤ kP9Dense blocks: τ = 0 (every block); ≤ 512 blocks: none.
//   k_cscore : indexer GEMM, CTA = 64 rows (8 warps × 8 rows, q B fragments in registers, pair layout) × key split ks:
//              128-key stages s ≡ ks (mod 3) by cp.async into smem, 16-B chunks swizzled so each lane's ld.shared gives
//              exactly idx_load's registers; per (key, q-vector) idx_dot's mma chain, per (key, row) idx_relu_sum's;
//              scores ≥ τ appended to the region (row, ks) — one warp owns a row, offsets by ballot (no atomics), the
//              block stored with it (every-block rows: block implied by the slot). Software-pipelined: the mma chains
//              of the next 16-key batch are issued with the epilogue of the current one.
//   k_cselect: per row (256 threads, candidates in registers): exact top-512 as k_select (bf16-bin histogram →
//              threshold bin → rank by (score desc, block asc); radix select for an end bin or a crowded bin), blocks
//              ascending + open tail. < 512 candidates or a full region → the row goes to the fallback list.
//   k_cfall  : per fallback row: all its scores (same chain) into a per-CTA slice, then the same exact select.
// Exactness: when ≥ 512 scores are ≥ τ, every block left out has a score < τ ≤ the 512th largest, so the candidates
// hold the whole top-512 (ties included); each score is the base chain on the same fragment values → identical
// selection for any M, B and chunking; k_pattn is unchanged → identical output. τ itself needs no exactness.
// Layer 3, 8,192-row call, µs (k_pscore + k_select → k_tau + k_cscore + k_cselect): 8K 1,130 → 670, 32K 2,980 →
// 1,620, 128K 12,010 → 4,180. Real rows (12 QSA layers, 16K–128K): 817–2,038 candidates, no fallback.
// Tiers (qsa_dev.cuh): calls that see ≤ kMaxBlk blocks keep 16-bit candidate blocks and the static P9SelSmem; wider
// calls (up to 1M positions) store 32-bit candidate blocks and select on bitsets in dynamic smem (P9SelW), same code.

constexpr int kP9KS = 3;                  // key splits per row group (grid x)
constexpr int kP9E = 1024;                // candidate slots per (row, split)
constexpr int kP9Dense = kP9KS * kP9E;    // rows seeing ≤ this many blocks keep every block (fits the regions)
constexpr int kP9Target = 1280;           // expected candidates of a sampled row
constexpr int kP9Samp = 1024;             // max sample keys per row
constexpr int kP9TR = 16;                 // rows per k_tau CTA (4 warps, 8 n-tiles each)
constexpr int kP9W = 8;                   // k_cscore warps
constexpr int kP9NT = 4;                  // k_cscore n-tiles per warp (2 pairs = 8 rows)
constexpr int kP9R = kP9W * 2 * kP9NT;    // k_cscore rows per CTA
constexpr int kP9SK = 128;                // keys per stage (32 KB)
constexpr int kP9NS = 3;                  // stages in flight
constexpr int kP9Smem = kP9NS * kP9SK * 256;
constexpr int kP9ST = 256;                // k_cselect threads
constexpr int kP9FT = 512;                // k_cfall threads
constexpr int kP9Cand = 1024;             // threshold-bin candidates ranked directly (more → radix select)
constexpr int kP9Sub = 8192;              // rows per k_tau/k_cscore/k_cselect round (candidate buffer)
constexpr uint32_t kP9None = 0xffffffffu; // τ of a row without candidates (above every score's bits)
constexpr int kP9MinRows = 512;           // smaller prefill calls keep the base indexer (more CTAs: faster there)
static_assert(kP9E % kP9SK == 0 && 2 * kP9Cand >= kMaxBlk / 32, "P009 sizes");  // base tier: eq fits the union

struct P9Args {
  const bf16* qi;
  const int *pos, *seq_start;
  QsaCache* const* caches;
  int B, ra, rb, reset;
  uint32_t* tau;    // [M] score bits of τ (absolute rows)
  uint32_t* cs;     // [rows of the round][kP9KS][kP9E] candidate score bits
  void* cj;         // [rows of the round][kP9KS][kP9E] candidate block (sampled rows only): uint16_t base tier, uint32_t wide
  int* cnt;         // [rows of the round][kP9KS] candidates found (may exceed kP9E: overflow)
  int* fb;          // [0] fallback rows, [1 + i] row
  float* scratch;   // [k_cfall CTAs][stride]
  int* sel;
  int stride, W;    // wide tier: scratch row stride, selection bitset words (base tier: kMaxBlk, kMaxBlk / 32)
};
template <bool kWide>
using P9J = std::conditional_t<kWide, uint32_t, uint16_t>;  // candidate block type

__device__ __forceinline__ void p9_cp16(void* dst, const void* src, bool ok) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void p9_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void p9_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ int p9_bin(uint32_t u) { return min(max(int(u >> 16) - kBinBase, 0), kBins - 1); }

// idx_dot for N n-tiles with the same key fragments: each acc[nt] gets idx_dot's chain (k-steps 0..7 in order, same
// operands); only the independent chains are interleaved (k-step outer, n-tile inner).
template <int N>
__device__ __forceinline__ void p9_dot(float (*acc)[4], const uint4* ka, const uint4* kb, const uint4 (*q)[4]) {
#pragma unroll
  for (int nt = 0; nt < N; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
  for (int p = 0; p < 4; p++) {
    const uint32_t a0[4] = {ka[p].x, kb[p].x, ka[p].y, kb[p].y}, a1[4] = {ka[p].z, kb[p].z, ka[p].w, kb[p].w};
#pragma unroll
    for (int nt = 0; nt < N; nt++) mma_bf16(acc[nt], a0, q[nt][p].x, q[nt][p].y);
#pragma unroll
    for (int nt = 0; nt < N; nt++) mma_bf16(acc[nt], a1, q[nt][p].z, q[nt][p].w);
  }
}

// Pair layout of the q-vectors (k_tau, k_cscore): n-tiles 2P and 2P+1 hold rows 4P..4P+3 of the warp, column n of
// n-tile 2P+h = (row 4P + n/2, head 2h + n%2). Lane (g, c) then holds heads 0,1 (n-tile 2P) and 2,3 (n-tile 2P+1) of
// row 4P + c for keys g and g+8, and forms idx_relu_sum's score (h0+h1)+(h2+h3) itself: no shuffle, no score computed
// twice. Each (key, q-vector) element is still idx_dot's chain (an output element depends only on its A row and B
// column), and the sum has idx_relu_sum's operands and order (fadd is commutative).
__device__ __forceinline__ float p9_sum4(const float* x, const float* y, int e) {  // e = 0: key g, 2: key g+8
  return __fadd_rn(__fadd_rn(relu0(x[e]), relu0(x[e + 1])), __fadd_rn(relu0(y[e]), relu0(y[e + 1])));
}
__device__ __forceinline__ const bf16* p9_qvec(const bf16* qi, int row4p, int nt, int g) {  // B column g of n-tile nt
  return qi + (size_t(row4p + 4 * (nt >> 1) + (g >> 1)) * 4 + 2 * (nt & 1) + (g & 1)) * 128;
}

// ---- k_tau -------------------------------------------------------------------------------------------------------
// grid (row tiles of 16), 4 warps. Per sequence part of the tile: sample keys j = D/2 + i·D (D = power of 2, ≤ 1,024
// samples below the part's last visible block), warps stride over 16-sample batches (next batch prefetched) holding the
// 16 rows' q-vectors (pair layout); bf16-truncated sample scores → smem; per row (warp) the k-th largest by bisection
// over the 15 value bits. (τ only steers how many scores are kept: any value gives the exact selection.)
__global__ void __launch_bounds__(128) k_tau(P9Args a) {
  __shared__ __align__(16) uint16_t sv[kP9TR][kP9Samp];
  pdl_wait();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  if (a.reset && blockIdx.x == 0 && threadIdx.x == 0) a.fb[0] = 0;
  const int t0 = a.ra + blockIdx.x * kP9TR, t1 = min(a.rb, t0 + kP9TR);
  for (int r0 = t0; r0 < t1;) {  // rows of one sequence at a time
    const int b = seq_of(a.seq_start, a.B, r0), r1 = min(t1, a.seq_start[b + 1]);
    const int vmax = (a.pos[r1 - 1] + 1) >> 2;
    int D = 1, off = 0, ns = 0;
    if (vmax > kP9Dense) {
      while (D * kP9Samp < vmax) D <<= 1;
      off = D >> 1;
      ns = (vmax - off + D - 1) / D;
      const bf16* comp = a.caches[b]->ik_comp;
      uint4 qf[8][4];
#pragma unroll
      for (int nt = 0; nt < 8; nt++) {
        const int r = t0 + 4 * (nt >> 1) + (g >> 1);
        idx_load(qf[nt], p9_qvec(a.qi, t0, nt, g), r >= r0 && r < r1, c);
      }
      const int nbt = (ns + 15) >> 4;
      uint4 na[4], nb[4];
      if (warp < nbt) {
        idx_load(na, comp + size_t(off + (warp * 16 + g) * D) * 128, warp * 16 + g < ns, c);
        idx_load(nb, comp + size_t(off + (warp * 16 + g + 8) * D) * 128, warp * 16 + g + 8 < ns, c);
      }
      for (int bt = warp; bt < nbt; bt += 4) {
        const int i0 = bt * 16 + g, i1 = i0 + 8, in = i0 + 64;
        uint4 ka[4], kb[4];
#pragma unroll
        for (int p = 0; p < 4; p++) ka[p] = na[p], kb[p] = nb[p];
        if (bt + 4 < nbt) {
          idx_load(na, comp + size_t(off + in * D) * 128, in < ns, c);
          idx_load(nb, comp + size_t(off + (in + 8) * D) * 128, in + 8 < ns, c);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float acc[4][4];
          p9_dot<4>(acc, ka, kb, qf + 4 * h);
#pragma unroll
          for (int P = 0; P < 2; P++) {
            const int rr = 4 * (2 * h + P) + c;
            if (i0 < ns) sv[rr][i0] = uint16_t(__float_as_uint(p9_sum4(acc[2 * P], acc[2 * P + 1], 0)) >> 16);
            if (i1 < ns) sv[rr][i1] = uint16_t(__float_as_uint(p9_sum4(acc[2 * P], acc[2 * P + 1], 2)) >> 16);
          }
        }
      }
      __syncthreads();
    }
    for (int i = warp; i < kP9TR; i += 4) {
      const int r = t0 + i;
      if (r < r0 || r >= r1) continue;
      const int vis = (a.pos[r] + 1) >> 2;
      uint32_t tau = kP9None;
      if (vis > kBudget && vis <= kP9Dense) {
        tau = 0;
      } else if (vis > kBudget) {
        const int n = vis > off ? min(ns, (vis - off + D - 1) / D) : 0;
        const int k = max(1, (kP9Target * n + vis / 2) / vis);
        uint32_t v[16];  // samples lane·32 .. +31 (two per word), 0 beyond n
        const uint4* src = reinterpret_cast<const uint4*>(&sv[i][lane * 32]);
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint4 x = src[q];
          v[4 * q] = x.x, v[4 * q + 1] = x.y, v[4 * q + 2] = x.z, v[4 * q + 3] = x.w;
        }
#pragma unroll
        for (int q = 0; q < 16; q++) {
          const int i0 = lane * 32 + 2 * q;
          if (i0 >= n) v[q] = 0;
          else if (i0 + 1 >= n) v[q] &= 0xffffu;
        }
        uint32_t t = 0;
        for (int bit = 14; bit >= 0; bit--) {
          const uint32_t cand = t | (1u << bit);
          int cn = 0;
#pragma unroll
          for (int q = 0; q < 16; q++) cn += ((v[q] & 0xffffu) >= cand) + ((v[q] >> 16) >= cand);
          if (__reduce_add_sync(kFull, cn) >= k) t = cand;
        }
        tau = max(t << 16, 1u);  // 0 marks every-block rows
      }
      if (lane == 0) a.tau[r] = tau;
    }
    __syncthreads();
    r0 = r1;
  }
}

// ---- k_cscore ----------------------------------------------------------------------------------------------------
// kP9W warps × kP9NT n-tiles (2·kP9NT rows per warp, pair layout). Stage: kP9SK keys × 16 chunks of 16 B, chunk q
// of key k at slot q ^ 4(k&1) → lane (g, c) reads chunk 4p+c of keys g / g+8 conflict-free; the registers equal
// idx_load's (zero beyond the part's last visible block). Lane (g, c) scores keys g, g+8 for row 4P + c of each pair P.
// Every-block rows (τ = 0): block j at slot i·kP9SK + j mod kP9SK of the split's i-th stage; sampled rows: scores ≥ τ
// appended (offsets by ballot over the row's 8 lanes) with their block. The chains of batch b+1 (p9_dot) are issued in
// the same block as the epilogue of batch b (software pipeline: the epilogue's ALU work fills the mma latency).
// Grid (kP9KS, row tiles): the splits of a tile run together, last tile first (largest contexts first; k_cselect then
// starts on candidates still in L2).
template <bool kWide>
__global__ void __launch_bounds__(kP9W * 32, 1) k_cscore(P9Args a) {
  constexpr int W = kP9W, NT = kP9NT, R = kP9R, NP = NT / 2;
  using J = P9J<kWide>;
  extern __shared__ __align__(16) unsigned char p9sm[];
  J* const cj = static_cast<J*>(a.cj);
  pdl_wait();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3, ks = blockIdx.x;
  const int t0 = a.ra + (gridDim.y - 1 - blockIdx.y) * R, t1 = min(a.rb, t0 + R), wr = t0 + 2 * NT * warp;
  for (int r0 = t0; r0 < t1;) {
    const int b = seq_of(a.seq_start, a.B, r0), r1 = min(t1, a.seq_start[b + 1]);
    const int vmax = (a.pos[r1 - 1] + 1) >> 2;
    if (vmax > kBudget) {
      const bf16* comp = a.caches[b]->ik_comp;
      const int nst = (vmax + kP9SK - 1) / kP9SK, nmy = ks < nst ? (nst - 1 - ks) / kP9KS + 1 : 0;
      uint4 qf[NT][4];
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        const int r = wr + 4 * (nt >> 1) + (g >> 1);
        idx_load(qf[nt], p9_qvec(a.qi, wr, nt, g), r >= r0 && r < r1, c);
      }
      uint32_t tau[NP];
      int vis[NP], cnt[NP], ro[NP];
#pragma unroll
      for (int P = 0; P < NP; P++) {
        const int r = wr + 4 * P + c;
        const bool ok = r >= r0 && r < r1;
        tau[P] = ok ? a.tau[r] : kP9None;
        vis[P] = ok ? (a.pos[r] + 1) >> 2 : 0;
        cnt[P] = 0;
        ro[P] = ((ok ? r - a.ra : 0) * kP9KS + ks) * kP9E;
      }
      auto issue = [&](int i) {  // this split's i-th stage → buffer i % kP9NS
        if (i < nmy) {
          const int key0 = (ks + i * kP9KS) * kP9SK;
          unsigned char* buf = p9sm + (i % kP9NS) * (kP9SK * 256);
#pragma unroll
          for (int q = tid; q < kP9SK * 16; q += W * 32) {
            const int k = q >> 4, ch = q & 15, j = key0 + k;
            p9_cp16(buf + k * 256 + ((ch ^ ((k & 1) << 2)) << 4), comp + size_t(j < vmax ? j : 0) * 128 + ch * 8,
                    j < vmax);
          }
        }
        p9_commit();
      };
#pragma unroll
      for (int i = 0; i < kP9NS - 1; i++) issue(i);
      const int lofs = g * 256 + c * 16, sw = g & 1;
      const unsigned rowm = 0x11111111u << c, lt = (1u << lane) - 1u;
      // scores of pair P (x: n-tile 2P, y: 2P+1) for keys j0+g, j0+g+8 → candidates
      auto epi = [&](int P, const float* x, const float* y, int j0, int so0) {
        const uint32_t ua = __float_as_uint(p9_sum4(x, y, 0)), ub = __float_as_uint(p9_sum4(x, y, 2));
        const int ja = j0 + g, jb = ja + 8;
        const bool ev = tau[P] == 0, ia = ja < vis[P], ib = jb < vis[P];  // ev: every-block row
        const bool pa = !ev && ia && ua >= tau[P], pb = !ev && ib && ub >= tau[P];
        const unsigned ba = __ballot_sync(kFull, pa) & rowm, bbl = __ballot_sync(kFull, pb) & rowm;
        const int na = __popc(ba);
        const int oa = ev ? so0 + g : cnt[P] + __popc(ba & lt), ob = ev ? so0 + g + 8 : cnt[P] + na + __popc(bbl & lt);
        const bool sa = pa && oa < kP9E, sb = pb && ob < kP9E;
        if ((ev && ia) || sa) a.cs[ro[P] + oa] = ua;
        if ((ev && ib) || sb) a.cs[ro[P] + ob] = ub;
        if (sa) cj[ro[P] + oa] = J(ja);
        if (sb) cj[ro[P] + ob] = J(jb);
        cnt[P] += ev ? min(16, max(0, vis[P] - j0)) : na + __popc(bbl);
      };
      float accP[NT][4];
      int j0P = 1 << 30, soP = 0;  // pending batch (none yet: nothing stored, nothing counted)
#pragma unroll
      for (int nt = 0; nt < NT; nt++) accP[nt][0] = accP[nt][1] = accP[nt][2] = accP[nt][3] = 0.f;
      for (int i = 0; i < nmy; i++) {
        p9_wait<kP9NS - 2>();
        __syncthreads();
        issue(i + kP9NS - 1);
        const unsigned char* buf = p9sm + (i % kP9NS) * (kP9SK * 256) + lofs;
        const int key0 = (ks + i * kP9KS) * kP9SK;
#pragma unroll 1
        for (int bb = 0; bb < kP9SK / 16; bb++) {
          uint4 ka[4], kb[4];
#pragma unroll
          for (int p = 0; p < 4; p++) {
            ka[p] = *reinterpret_cast<const uint4*>(buf + bb * 4096 + ((p ^ sw) << 6));
            kb[p] = *reinterpret_cast<const uint4*>(buf + bb * 4096 + 2048 + ((p ^ sw) << 6));
          }
          float accN[NT][4];
          p9_dot<NT>(accN, ka, kb, qf);
#pragma unroll
          for (int P = 0; P < NP; P++) epi(P, accP[2 * P], accP[2 * P + 1], j0P, soP);
#pragma unroll
          for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) accP[nt][e] = accN[nt][e];
          j0P = key0 + bb * 16, soP = i * kP9SK + bb * 16;
        }
      }
#pragma unroll
      for (int P = 0; P < NP; P++) epi(P, accP[2 * P], accP[2 * P + 1], j0P, soP);
      if (g == 0) {
#pragma unroll
        for (int P = 0; P < NP; P++) {
          const int r = wr + 4 * P + c;
          if (r >= r0 && r < r1) a.cnt[(r - a.ra) * kP9KS + ks] = cnt[P];
        }
      }
      p9_wait<0>();
      __syncthreads();
    }
    r0 = r1;
  }
}

// ---- exact top-512 over a candidate source -------------------------------------------------------------------------
// k_select's method for T threads: bf16-bin histogram, bin of the 512th score, bins above → bitmask, the threshold
// bin ranked by (score desc, block asc); radix select (11/11/10 bits) when that bin is an end bin or holds > kP9Cand.
// Then the bitmask → selected blocks ascending (+ open tail, count). Base tier: P9SelSmem (static; gt = S.gt, eq =
// S.u.eq); wide tier: P9SelW in dynamic smem — eq [W] over the union's region (≥ W words), gt [W] after it.
struct P9SelSmem {
  uint32_t gt[kMaxBlk / 32];
  union {  // histogram, then the threshold bin's candidates or the radix select's ties
    int hist[kBins];
    struct {
      uint32_t ck[kP9Cand];
      int ci[kP9Cand];
    } c;
    uint32_t eq[kMaxBlk / 32];
  } u;
  int sh[32];
  int digit, krem, nc;
};
struct P9SelW {
  int sh[32];
  int digit, krem, nc;
  union {
    int hist[kBins];
    struct {
      uint32_t ck[kP9Cand];
      int ci[kP9Cand];
    } c;
  } u;
};
constexpr int kP9U = int(sizeof(P9SelW::u) / 4);  // union words (2,048)
__host__ __device__ constexpr int p9_region(int W) { return W > kP9U ? W : kP9U; }
size_t p9_sel_smem(int W) { return offsetof(P9SelW, u) + sizeof(uint32_t) * size_t(p9_region(W) + W); }
static_assert(offsetof(P9SelW, u) + 8 * (kMaxBlkW / 32) <= 99 * 1024, "wide P009 selection smem ≤ 99 KB");

template <int T>
__device__ int p9_scan(int v, int* sh) {  // exclusive block scan over T threads
  constexpr int W = T / 32;
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
    int y = lane < W ? sh[lane] : 0;
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

// hist: nb bins in increasing key order → S.digit = bin holding the krem-th largest, S.krem = rank inside it.
template <int T, class SM>
__device__ void p9_find_bin(SM& S, int nb, int krem) {
  const int per = nb / T, o0 = threadIdx.x * per;
  int loc = 0;
  for (int k = 0; k < per; k++) loc += S.u.hist[nb - 1 - o0 - k];
  const int before = p9_scan<T>(loc, S.sh);
  if (before < krem && krem <= before + loc) {
    int cum = before;
    for (int k = 0; k < per; k++) {
      const int b = nb - 1 - o0 - k, h = S.u.hist[b];
      if (cum + h >= krem) {
        S.digit = b, S.krem = krem - cum;
        break;
      }
      cum += h;
    }
  }
  __syncthreads();
}

template <int T, int NPT>
struct P9RegSrc {  // one row's candidates, NPT per thread in registers (entry e = tid + k·T)
  uint32_t u[NPT];
  int j[NPT];
  int n;
  template <class F>
  __device__ __forceinline__ void each(int tid, int, F f) const {
#pragma unroll
    for (int k = 0; k < NPT; k++)
      if (tid + k * T < n) f(u[k], j[k]);
  }
};
struct P9DenseSrc {  // all scores of one row (written by this CTA: L2 loads, 8 per thread in flight)
  const float* sc;
  int vis;
  template <class F>
  __device__ __forceinline__ void each(int tid, int T, F f) const {
    for (int i0 = tid; i0 < vis; i0 += 8 * T) {
      uint32_t u[8];
#pragma unroll
      for (int k = 0; k < 8; k++) u[k] = i0 + k * T < vis ? __float_as_uint(__ldcg(sc + i0 + k * T)) : 0u;
#pragma unroll
      for (int k = 0; k < 8; k++)
        if (i0 + k * T < vis) f(u[k], i0 + k * T);
    }
  }
};

template <int T, class SM, class Src>
__device__ void p9_radix(SM& S, uint32_t* gt, uint32_t* eq, const Src& src, int nw) {
  const int tid = threadIdx.x;
  uint32_t prefix = 0, mask = 0;
  int krem = kBudget;
  for (int pass = 0; pass < 3; pass++) {
    const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0, nb = pass == 2 ? 1024 : 2048;
    __syncthreads();
    for (int i = tid; i < nb; i += T) S.u.hist[i] = 0;
    __syncthreads();
    src.each(tid, T, [&](uint32_t u, int) {
      if ((u & mask) == prefix) atomicAdd(&S.u.hist[(u >> shift) & uint32_t(nb - 1)], 1);
    });
    __syncthreads();
    p9_find_bin<T>(S, nb, krem);
    prefix |= uint32_t(S.digit) << shift;
    mask |= uint32_t(nb - 1) << shift;
    krem = S.krem;
  }
  for (int i = tid; i < nw; i += T) gt[i] = 0, eq[i] = 0;
  __syncthreads();
  src.each(tid, T, [&](uint32_t u, int j) {
    if (u > prefix) atomicOr(&gt[j >> 5], 1u << (j & 31));
    else if (u == prefix) atomicOr(&eq[j >> 5], 1u << (j & 31));
  });
  __syncthreads();
  const int wpt = (nw + T - 1) / T, w0 = min(nw, tid * wpt), w1 = min(nw, w0 + wpt);
  int ce = 0;
  for (int w = w0; w < w1; w++) ce += __popc(eq[w]);
  int take = max(0, min(ce, krem - p9_scan<T>(ce, S.sh)));
  for (int w = w0; w < w1; w++) {
    uint32_t e = eq[w], gg = gt[w];
    for (; take > 0 && e; take--) gg |= e & (0u - e), e &= e - 1;  // lowest-index ties
    gt[w] = gg;
  }
  __syncthreads();
}

// bitsets gt / eq of W words (eq shares the histogram / candidate region)
template <int T, class SM, class Src>
__device__ void p9_select(SM& S, uint32_t* gt, uint32_t* eq, int W, const Src& src, int vis, int* out, int tail0,
                          int ntail) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (vis + 31) >> 5;
  if (nw > W) __trap();  // the call was sized for fewer blocks (max_pos hint below a row's position)
  for (int i = tid; i < kBins; i += T) S.u.hist[i] = 0;
  for (int i = tid; i < nw; i += T) gt[i] = 0;
  if (tid == 0) S.nc = 0;
  __syncthreads();
  src.each(tid, T, [&](uint32_t u, int) { atomicAdd(&S.u.hist[p9_bin(u)], 1); });
  __syncthreads();
  p9_find_bin<T>(S, kBins, kBudget);
  const int bt = S.digit, krem = S.krem;
  bool fb = bt == 0 || bt == kBins - 1;  // under/overflow bin: not exact
  if (!fb) {
    src.each(tid, T, [&](uint32_t u, int j) {
      const int b = p9_bin(u);
      if (b > bt) {
        atomicOr(&gt[j >> 5], 1u << (j & 31));
      } else if (b == bt) {
        const int k = atomicAdd(&S.nc, 1);
        if (k < kP9Cand) S.u.c.ck[k] = u, S.u.c.ci[k] = j;
      }
    });
    __syncthreads();
    const int nc = S.nc;
    if (nc <= kP9Cand) {
      for (int k = warp; k < nc; k += T / 32) {  // rank by (score desc, block asc): one warp per candidate
        const uint32_t u = S.u.c.ck[k];
        const int ix = S.u.c.ci[k];
        int rank = 0;
        for (int d = lane; d < nc; d += 32) rank += S.u.c.ck[d] > u || (S.u.c.ck[d] == u && S.u.c.ci[d] < ix);
        rank = __reduce_add_sync(kFull, rank);
        if (lane == 0 && rank < krem) atomicOr(&gt[ix >> 5], 1u << (ix & 31));
      }
      __syncthreads();
    } else {
      fb = true;
    }
  }
  if (fb) p9_radix<T>(S, gt, eq, src, nw);
  const int wpt = (nw + T - 1) / T, w0 = min(nw, tid * wpt), w1 = min(nw, w0 + wpt);
  int cnt = 0;
  for (int w = w0; w < w1; w++) cnt += __popc(gt[w]);
  int off = p9_scan<T>(cnt, S.sh);
  for (int w = w0; w < w1; w++) {
    for (uint32_t gg = gt[w]; gg; gg &= gg - 1) {
      const int q0 = (w * 32 + __ffs(gg) - 1) * kIdxRatio;
      *reinterpret_cast<int4*>(out + off * kIdxRatio) = make_int4(q0, q0 + 1, q0 + 2, q0 + 3);
      off++;
    }
  }
  if (tid < kIdxRatio - 1) out[kBudget * kIdxRatio + tid] = tid < ntail ? tail0 + tid : -1;
  if (tid == 0) out[kSelCols] = kBudget * kIdxRatio + ntail;
}

// ---- k_cselect: one CTA (256 threads) per row of the round ----------------------------------------------------------
template <bool kWide, class SM>
__device__ __forceinline__ void p9_cselect_row(SM& S, uint32_t* gt, uint32_t* eq, int W, const P9Args& a) {
  constexpr int T = kP9ST, NPT = kP9KS * kP9E / T;
  pdl_wait();
  const int r = a.ra + blockIdx.x, tid = threadIdx.x;
  const int p = a.pos[r], vis = (p + 1) >> 2, tail0 = vis * kIdxRatio, ntail = p + 1 - tail0;
  int* out = a.sel + size_t(r) * kSelW;
  if (vis <= kBudget) {  // dense: tokens 0..p
    for (int i = tid; i < kSelCols; i += T) out[i] = i <= p ? i : -1;
    if (tid == 0) out[kSelCols] = p + 1;
    return;
  }
  int pre[kP9KS + 1];  // region s holds candidates pre[s] .. pre[s+1] of the row
  bool over = false;
  pre[0] = 0;
#pragma unroll
  for (int s = 0; s < kP9KS; s++) {
    const int n = a.cnt[(r - a.ra) * kP9KS + s];
    pre[s + 1] = pre[s] + n, over |= n > kP9E;
  }
  const int tot = pre[kP9KS];
  if (over || tot < kBudget) {  // τ too high or a region full: k_cfall
    if (tid == 0) a.fb[1 + atomicAdd(a.fb, 1)] = r;
    return;
  }
  const bool every = vis <= kP9Dense;  // block implied by the slot
  const size_t rb = size_t(r - a.ra) * kP9KS * kP9E;
  P9RegSrc<T, NPT> src;
  src.n = tot;
#pragma unroll
  for (int k = 0; k < NPT; k++) {  // all loads in flight
    const int e = tid + k * T;
    int s = 0, o = e;
#pragma unroll
    for (int q = 1; q < kP9KS; q++)
      if (e >= pre[q]) s = q, o = e - pre[q];
    const size_t at = rb + size_t(s) * kP9E + o;
    src.u[k] = e < tot ? __ldg(a.cs + at) : 0u;
    src.j[k] = e >= tot ? 0
               : every ? (s + kP9KS * (o / kP9SK)) * kP9SK + o % kP9SK
                       : int(__ldg(static_cast<const P9J<kWide>*>(a.cj) + at));
  }
  p9_select<T>(S, gt, eq, W, src, vis, out, tail0, ntail);
}
template <bool kWide>
__global__ void __launch_bounds__(kP9ST) k_cselect(P9Args a) {
  if constexpr (kWide) {
    extern __shared__ __align__(16) unsigned char p9sel_sm[];
    P9SelW& S = *reinterpret_cast<P9SelW*>(p9sel_sm);
    uint32_t* eq = reinterpret_cast<uint32_t*>(&S.u);
    p9_cselect_row<true>(S, eq + p9_region(a.W), eq, a.W, a);
  } else {
    __shared__ P9SelSmem S;
    p9_cselect_row<false>(S, S.gt, S.u.eq, kMaxBlk / 32, a);
  }
}

// ---- k_cfall: fallback rows (persistent over the list, 2 CTAs per SM), 16 warps -------------------------------------
// All scores of the row as k_pscore (row in n-tile columns 0..3, zero q-vectors in 4..7) → this CTA's slice → select.
template <bool kWide, class SM>
__device__ __forceinline__ void p9_cfall_rows(SM& S, uint32_t* gt, uint32_t* eq, int W, const P9Args& a) {
  pdl_wait();
  const int n = a.fb[0], warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  float* scr = a.scratch + size_t(blockIdx.x) * (kWide ? a.stride : kMaxBlk);
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    const int r = a.fb[1 + i], b = seq_of(a.seq_start, a.B, r);
    const int p = a.pos[r], vis = (p + 1) >> 2, tail0 = vis * kIdxRatio, ntail = p + 1 - tail0;
    const bf16* comp = a.caches[b]->ik_comp;
    uint4 qf[4];
    idx_load(qf, a.qi + (size_t(r) * 4 + (g & 3)) * 128, g < 4, c);
    constexpr int NW = kP9FT / 32;
    for (int b0 = warp; b0 < (vis + 15) >> 4; b0 += 2 * NW) {  // two batches in flight per warp
      uint4 ka[2][4], kb[2][4];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int j = (b0 + h * NW) * 16 + g;
        idx_load(ka[h], comp + size_t(j) * 128, j < vis, c);
        idx_load(kb[h], comp + size_t(j + 8) * 128, j + 8 < vis, c);
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int j = (b0 + h * NW) * 16 + g;
        float acc[4];
        idx_dot(acc, ka[h], kb[h], qf);
        const float2 sc = idx_relu_sum(acc);
        if ((lane & 3) == 0) {
          if (j < vis) scr[j] = sc.x;
          if (j + 8 < vis) scr[j + 8] = sc.y;
        }
      }
    }
    __syncthreads();
    p9_select<kP9FT>(S, gt, eq, W, P9DenseSrc{scr, vis}, vis, a.sel + size_t(r) * kSelW, tail0, ntail);
    __syncthreads();
  }
}
template <bool kWide>
__global__ void __launch_bounds__(kP9FT, 1) k_cfall(P9Args a) {
  if constexpr (kWide) {
    extern __shared__ __align__(16) unsigned char p9fall_sm[];
    P9SelW& S = *reinterpret_cast<P9SelW*>(p9fall_sm);
    uint32_t* eq = reinterpret_cast<uint32_t*>(&S.u);
    p9_cfall_rows<true>(S, eq + p9_region(a.W), eq, a.W, a);
  } else {
    __shared__ P9SelSmem S;
    p9_cfall_rows<false>(S, S.gt, S.u.eq, kMaxBlk / 32, a);
  }
}

// ---- host -----------------------------------------------------------------------------------------------------------
// Workspace: τ [M], candidates of one round (≤ 8,192 rows × 3 × 1,024 × 6 B = 151 MB; windows > 262,144: 8 B, 201 MB),
// counts, fallback list, fallback score slices (2 per SM: 25 MB; 1M window 101 MB). Replaces the base path's 268 MB
// score buffer when the patch is on.
struct P9Ws {
  int rows = 0, sub = 0;
  uint32_t *tau = nullptr, *cs = nullptr;
  void* cj = nullptr;
  int *cnt = nullptr, *fb = nullptr;
  float* scratch = nullptr;
} g_p9;

void p9_free() {
  cudaFree(g_p9.tau), cudaFree(g_p9.cs), cudaFree(g_p9.cj), cudaFree(g_p9.cnt), cudaFree(g_p9.fb);
  cudaFree(g_p9.scratch);
  g_p9 = P9Ws{};
}

void p9_reserve(int M) {
  if (M <= g_p9.rows) return;
  if (g_p9.rows) {
    CUDA_CHECK(cudaDeviceSynchronize());
    p9_free();
  }
  const int sub = std::min(M, kP9Sub);
  g_p9.tau = dmalloc<uint32_t>(M);
  g_p9.cs = dmalloc<uint32_t>(size_t(sub) * kP9KS * kP9E);
  g_p9.cj = dmalloc<unsigned char>(size_t(sub) * kP9KS * kP9E * (g_blk > kMaxBlk ? 4 : 2));
  g_p9.cnt = dmalloc<int>(size_t(sub) * kP9KS);
  g_p9.fb = dmalloc<int>(size_t(M) + 1);
  g_p9.scratch = dmalloc<float>(size_t(2 * g_sms) * g_stride);
  CUDA_CHECK(cudaFuncSetAttribute(k_cscore<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kP9Smem));
  CUDA_CHECK(cudaFuncSetAttribute(k_cscore<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kP9Smem));
  if (g_blk > kMaxBlk) {  // wide tier: up to the window's bitsets in dynamic smem
    const int sm = int(p9_sel_smem((g_blk + 31) / 32));
    CUDA_CHECK(cudaFuncSetAttribute(k_cselect<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    CUDA_CHECK(cudaFuncSetAttribute(k_cfall<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
  }
  g_p9.rows = M, g_p9.sub = sub;
}

template <bool kWide>
void p9_cscore(const P9Args& a, cudaStream_t s) {
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{dim3(kP9KS, cdiv(a.rb - a.ra, kP9R)), dim3(kP9W * 32), size_t(kP9Smem), s, at, 1};
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_cscore<kWide>, a));
}

void p9_indexer(const bf16* qi, const int* pos, const int* seq_start, QsaCache* const* caches, int B, int M, int* sel,
                const Tier& t, cudaStream_t s) {
  P9Args a{qi, pos, seq_start, caches, B, 0, 0, 1, g_p9.tau, g_p9.cs, g_p9.cj, g_p9.cnt, g_p9.fb, g_p9.scratch, sel,
           g_stride, t.W};
  for (int ra = 0; ra < M; ra += g_p9.sub) {
    a.ra = ra, a.rb = std::min(M, ra + g_p9.sub), a.reset = ra == 0;
    launch(k_tau, cdiv(a.rb - a.ra, kP9TR), 128, s, a);
    if (t.wide) {
      p9_cscore<true>(a, s);
      launch_smem(k_cselect<true>, a.rb - a.ra, kP9ST, p9_sel_smem(t.W), s, a);
    } else {
      p9_cscore<false>(a, s);
      launch(k_cselect<false>, a.rb - a.ra, kP9ST, s, a);
    }
  }
  if (t.wide) launch_smem(k_cfall<true>, 2 * g_sms, kP9FT, p9_sel_smem(t.W), s, a);
  else launch(k_cfall<false>, 2 * g_sms, kP9FT, s, a);
}
