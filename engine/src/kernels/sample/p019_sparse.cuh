// p019_sparse.cuh — P019 sparse_logits: sampling and the speculative accept test from each vocab half's top candidates
// (contract and conditions: sample.h, "P019"). Included by sample.cu inside its anonymous namespace after
// p016_block_verify.cuh (uses philox, temp_scale, weight, okey / zkey / zval, pthr, trunc_params, scale, the shuffle
// helpers and p016's big integers and per-row tests).
//
// sample_cands (one rank's half, one thread-block cluster of C CTAs per row; C is scheduling only):
//   pass     each thread walks its float4s (strided over the CTA's chunk: g ≡ lo + t mod 512), keeping its maximum
//            (torch.argmax order, first index — the same token as the largest Z among its own), its second largest
//            value s2 and whether any value is NaN / ±inf (→ the dense flag);
//   bound    every warp sorts its 32 thread maxima (bitonic, registers); the top 8 of each warp (128 distinct tokens)
//            → θ = the 64th largest of them: at least 64 tokens of the chunk have Z ≥ θ, so the chunk's top 64 do;
//   gather   threads whose maximum is ≥ θ push their tokens with Z ≥ θ into a shared list — the maximum alone when
//            s2 < x(θ), else their float4s read again (L2). On logits rows ~64–70 entries; more than 1024 (adversarial
//            layouts: the top tokens packed into few threads) → slow path: θ raised by bisection on Z (a counting pass
//            over the chunk per step, Z unique per token) until 64 ≤ count ≤ 1024, then gathered again;
//   local    each entry's rank = the entries above it (exact, O(n) per entry) → the chunk's top 64, sorted;
//   merge    the CTAs push their lists into CTA 0 (distributed shared memory, one cluster barrier), whose threads rank
//            the entries ≥ max_c list_c[63] (the others cannot reach the row's top 64) by branch-free binary searches
//            in the other lists → the row's top 64, written with the flag.
//   C        the largest ≤ 8 whose R clusters are co-resident (cands_cluster; measured: larger clusters cost more in the
//            merge than they save in the pass, and a size that leaves R clusters for a second wave doubles the time).
// The result is a property of the half (Z is a total order): neither C, the cap, the path nor the gather order changes it.
// sample_rows_sparse: one CTA (128 threads) per row: thread j holds candidate j of the two records, its merged rank
//   = its index + the other list's entries above it; the top kk in rank order → weights (row maximum = the top one),
//   W_K, the top-p cut (the first rank whose inclusive prefix reaches ⌈p·W_K⌉), W', the draw in index order
//   (list_draw's rule) — sample_rows' truncated list path step by step.
// spec_accept_sparse: one CTA (16 warps) per sequence: the warps cut every row of the sequence (2k + 1) into its kept
//   set in index order (ids, weights, W'); warp 0 then runs accept_trunc's rule (or P016's accept_cut + rounds with
//   p016's big-integer chain and tests) over those sets: a draft's weight is a lookup, a residual / P016 row pass runs
//   over the target row's kept tokens (r_v > 0 only there; tokens with r_v = 0 add nothing to any sum), the draws walk
//   them in index order.
#pragma once

namespace p019 {

constexpr int kK = kSparseK, kRec = kSparseRec;
static_assert(kSparseMaxK == p016::kBvMax, "P019 rows per sequence follow P016's depth limit");

// ---- sample_cands --------------------------------------------------------------------------------------------------
constexpr int kCT = 512, kCW = kCT / 32;  // threads / warps per CTA
constexpr int kJ = 8;                     // thread maxima per warp in the bound's set (kJ·kCW = 128 ≥ 2·kK)
constexpr int kCCap = 1024;               // shared list entries at most
constexpr int kCLd = 4;                   // float4 loads in flight per thread
constexpr int kCMinChunk4 = 512;          // float4 per CTA at least when choosing C
constexpr int kCMaxC = 16;
static_assert(kJ * kCW >= kK && kJ <= 32, "bound set");

__device__ unsigned g_slow;  // test hook: CTAs that took the slow path

struct CSm {
  u64 tm[kJ * kCW];      // the warps' top thread maxima
  u64 list[kCCap];       // gathered keys (any order)
  u64 loc[kK];           // this CTA's top kK, sorted from the top
  u64 all[kCMaxC][kK];   // CTA 0: every CTA's list (pushed; 0 past its entries)
  uint32_t fall[kCMaxC]; // CTA 0: every CTA's flag
  u64 theta;
  uint32_t cnt;
};

// One u64 per lane → sorted descending over the warp (lane 0 the largest).
__device__ __forceinline__ u64 warp_sort_desc(u64 v) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const u64 o = shfl_xor(v, j);
      const bool keep_max = ((lane & j) == 0) == ((lane & k) == 0);
      v = keep_max ? (v > o ? v : o) : (v < o ? v : o);
    }
  return v;
}

// Entries of a descending list a[0, kK) above z (branch-free): binary lifting finds p ≤ kK − 1 with a[0, p) > z, the
// last entry decides kK.
__device__ __forceinline__ int count_gt64(const u64* a, u64 z) {
  int p = 0;
#pragma unroll
  for (int s = kK / 2; s; s >>= 1) p += a[p + s - 1] > z ? s : 0;
  return p + (a[kK - 1] > z);
}

__device__ __forceinline__ SparseCand cand(u64 z) { return {int(0xFFFFF - (z & 0xFFFFF)), zval(z)}; }
__device__ __forceinline__ u64 ckey(const SparseCand& e) { return zkey(e.x, e.id); }

__device__ __forceinline__ void cpush(u64 z, CSm& sm) {
  const uint32_t i = atomicAdd(&sm.cnt, 1u);
  if (i < uint32_t(kCCap)) sm.list[i] = z;
}

// grid (C, R), cluster (C): row r of x [R, Vh] (ids off + v) → out[r] (kRec entries).
__global__ void __launch_bounds__(kCT) k_cands(const float* x, int Vh, int off, SparseCand* out) {
  __shared__ CSm sm;
  const cg::cluster_group cl = cg::this_cluster();
  const int C = int(cl.num_blocks()), c = int(cl.block_rank()), r = blockIdx.y;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // this CTA runs (before any remote store)
  const int n4 = Vh / 4, lo4 = c * n4 / C, hi4 = (c + 1) * n4 / C;
  const float4* row = reinterpret_cast<const float4*>(x + size_t(r) * Vh);
  // pass: maximum (first index), second largest value, non-finite values
  float mv = -INFINITY, s2 = -INFINITY;
  int mi = -1;
  bool bad = false;
  for (int g0 = lo4 + t; g0 < hi4; g0 += kCT * kCLd) {
    float4 v[kCLd];
#pragma unroll
    for (int u = 0; u < kCLd; u++) {
      const int g = g0 + u * kCT;
      v[u] = g < hi4 ? __ldg(row + g) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < kCLd; u++) {
      const int g = g0 + u * kCT;
      const bool ok = g < hi4;
      const float e[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        bad |= ok & !(fabsf(e[j]) < INFINITY);
        const bool tk = ok & ((mi < 0) | ((mv == mv) & !(e[j] <= mv)));
        s2 = ok ? fmaxf(s2, tk ? mv : e[j]) : s2;
        mv = tk ? e[j] : mv;
        mi = tk ? 4 * g + j : mi;
      }
    }
  }
  const u64 zm = mi >= 0 ? zkey(mv, off + mi) : 0ull;  // real keys are > 0
  bad = __syncthreads_or(bad);
  // bound θ: the 64th largest of the warps' top kJ thread maxima (0: fewer than 64 — every token kept)
  const u64 srt = warp_sort_desc(zm);
  if (lane < kJ) sm.tm[warp * kJ + lane] = srt;
  if (t == 0) sm.theta = 0, sm.cnt = 0;
  __syncthreads();
  if (t < kJ * kCW) {
    const u64 z = sm.tm[t];
    if (z) {
      int rk = 0;
      for (int i = 0; i < kJ * kCW; i++) rk += sm.tm[i] > z;
      if (rk == kK - 1) sm.theta = z;
    }
  }
  __syncthreads();
  u64 th = sm.theta;
  // gather Z ≥ θ
  if (zm && zm >= th) {
    const float xt = th ? zval(th) : -INFINITY;
    if (s2 < xt) {
      cpush(zm, sm);
    } else {
      for (int g = lo4 + t; g < hi4; g += kCT) {
        const float4 v = __ldg(row + g);
        const float e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const u64 z = zkey(e[j], off + 4 * g + j);
          if (z >= th) cpush(z, sm);
        }
      }
    }
  }
  __syncthreads();
  uint32_t n = sm.cnt;
  if (n > uint32_t(kCCap)) {  // slow path: raise θ (count(Z ≥ lo) = nl > cap ≥ kK, count(Z ≥ hi) < kK)
    if (t == 0) atomicAdd(&g_slow, 1u);
    u64 lo = th, hi = kZTop;
    uint32_t nl = n;
    while (nl > uint32_t(kCCap)) {
      const u64 mid = lo + (hi - lo) / 2;  // > lo: hi − lo ≥ 2 while nl > kK (one token per key)
      __syncthreads();
      if (t == 0) sm.cnt = 0;
      __syncthreads();
      uint32_t cn = 0;
      for (int g = lo4 + t; g < hi4; g += kCT) {
        const float4 v = __ldg(row + g);
        const float e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; j++) cn += zkey(e[j], off + 4 * g + j) >= mid;
      }
      if (cn) atomicAdd(&sm.cnt, cn);
      __syncthreads();
      const uint32_t nm = sm.cnt;
      if (nm < uint32_t(kK)) hi = mid;
      else lo = mid, nl = nm;
    }
    th = lo;
    __syncthreads();
    if (t == 0) sm.cnt = 0;
    __syncthreads();
    for (int g = lo4 + t; g < hi4; g += kCT) {
      const float4 v = __ldg(row + g);
      const float e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const u64 z = zkey(e[j], off + 4 * g + j);
        if (z >= th) cpush(z, sm);
      }
    }
    __syncthreads();
    n = sm.cnt;
  }
  // this CTA's top kK by rank among the list (keys distinct)
  for (int j = t; j < int(n); j += kCT) {
    const u64 z = sm.list[j];
    int rk = 0;
    for (int i = 0; i < int(n); i++) rk += sm.list[i] > z;
    if (rk < kK) sm.loc[rk] = z;
  }
  const int nloc = min(int(n), kK);
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // every CTA of the cluster runs
  __syncthreads();
  SparseCand* o = out + size_t(r) * kRec;
  if (C == 1) {
    for (int j = t; j < nloc; j += kCT) o[j] = cand(sm.loc[j]);
    if (t == 0) o[kK] = {bad ? 1 : 0, 0.f};
    return;
  }
  for (int j = t; j < kK; j += kCT) *cl.map_shared_rank(&sm.all[c][j], 0) = j < nloc ? sm.loc[j] : 0ull;  // 0: none
  if (t == 0) *cl.map_shared_rank(&sm.fall[c], 0) = bad;
  cl.sync();
  if (c != 0) return;
  // the row's 64th largest key is ≥ every full list's 64th entry: only entries ≥ the largest of those can rank < kK
  u64 tg = 0;
  for (int c2 = 0; c2 < C; c2++) tg = max(tg, sm.all[c2][kK - 1]);
  for (int e = t; e < C * kK; e += kCT) {
    const int cc = e / kK, i = e - cc * kK;
    const u64 z = sm.all[cc][i];
    if (z == 0 || z < tg) continue;
    int rk = i;
#pragma unroll
    for (int c2 = 0; c2 < kCMaxC; c2++)
      if (c2 < C && c2 != cc) rk += count_gt64(sm.all[c2], z);  // (padding zeros are below every key)
    if (rk < kK) o[rk] = cand(z);
  }
  if (t == 0) {
    uint32_t f = 0;
    for (int c2 = 0; c2 < C; c2++) f |= sm.fall[c2];
    o[kK] = {f ? 1 : 0, 0.f};
  }
}

// ---- the truncated row from its two records --------------------------------------------------------------------------
// st[0, 2kK): the two lists' keys; after rank_top: st[0, kk) the merged top kk in rank order (warp-level).
__device__ __forceinline__ void rank_top(const SparseCand* r0, const SparseCand* r1, int kk, u64* st) {
  const int lane = threadIdx.x & 31;
  u64 z[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int idx = lane + 32 * j;  // j < 2: rank 0's list, else rank 1's
    z[j] = ckey((j < 2 ? r0 : r1)[idx & (kK - 1)]);
    st[idx] = z[j];
  }
  __syncwarp();
  int rk[4];
#pragma unroll
  for (int j = 0; j < 4; j++) rk[j] = ((lane + 32 * j) & (kK - 1)) + count_gt64(st + (j < 2 ? kK : 0), z[j]);
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 4; j++)
    if (rk[j] < kk) st[rk[j]] = z[j];
  __syncwarp();
}

// After rank_top: the weights of ranks 2·lane, 2·lane + 1 (w0, w1), their inclusive prefixes in rank order, and the
// cut: jc = the last kept rank, Wc = W'. mt: min_p's weight threshold (0: off) — K ∩ M = the ranks j < kk with
// w_j ≥ mt, a prefix (weights are non-increasing in rank order; sample.h), ≥ 1 (the top token's weight is 2^44 ≥ mt
// for a finite c): the dense kernels' max(z_K, zm) cut exactly. Why min_p keeps the sparse conditions as they are: it
// only removes tokens — the kept set lies inside the top kk ≤ kSparseK, which the two records hold (sample.h, P019) —
// and M is decided by each token's own weight against the row maximum, the top candidate.
struct Cut2 {
  u64 w0, w1;
  int jc;
  u64 Wc;
};
__device__ __forceinline__ Cut2 cut_top(const u64* st, float c, int kk, float pp, u64 mt) {
  const int lane = threadIdx.x & 31, j0 = 2 * lane, j1 = j0 + 1;
  const float m = zval(st[0]);
  u64 w0 = j0 < kk ? weight(zval(st[j0]), m, c) : 0ull, w1 = j1 < kk ? weight(zval(st[j1]), m, c) : 0ull;
  if (mt) {
    kk = __popc(__ballot_sync(~0u, j0 < kk && w0 >= mt)) + __popc(__ballot_sync(~0u, j1 < kk && w1 >= mt));
    w0 = j0 < kk ? w0 : 0ull, w1 = j1 < kk ? w1 : 0ull;
  }
  u64 inc = w0 + w1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const u64 y = shfl_up(inc, o);
    if (lane >= o) inc += y;
  }
  const u64 p0 = inc - w1, p1 = inc;  // inclusive prefixes at j0, j1
  const u64 WK = __shfl_sync(~0u, inc, 31);
  if (!(pp < 1.f)) return {w0, w1, kk - 1, WK};
  const u64 thr = pthr(pp, WK);  // ≤ W_K: some rank < kk reaches it
  const int jc = __popc(__ballot_sync(~0u, j0 < kk && p0 < thr)) + __popc(__ballot_sync(~0u, j1 < kk && p1 < thr));
  const u64 Wc = __shfl_sync(~0u, jc & 1 ? p1 : p0, jc >> 1);
  return {w0, w1, jc, Wc};
}

// ---- sample_rows_sparse ------------------------------------------------------------------------------------------
struct SSm {
  u64 key[2][kK];
  u64 w[kK];
};

// grid R, 128 threads: row r.
__global__ void __launch_bounds__(2 * kK) k_sample_sparse(const SparseCand* cands, int ld, int V, const float* temp,
                                                          const int* top_k, const float* top_p, const float* min_p,
                                                          const u64* seed, const u64* ctr, int* token, float* prob) {
  __shared__ SSm sm;
  const int r = blockIdx.x, t = threadIdx.x, q = t / kK, i = t - q * kK;
  const SparseCand* rec = cands + (size_t(q) * ld + r) * kRec;
  const u64 z = ckey(rec[i]);
  sm.key[q][i] = z;
  const int fl = i == 0 ? rec[kK].id : 0;
  const float T = temp[r];
  const int tk = top_k[r];
  const float tp = top_p[r], mp = min_p[r];
  const u64 sd = seed[r], ct = ctr[r];
  if (__syncthreads_or(fl != 0)) {
    if (t == 0) {
      token[r] = kSparseDense;
      if (prob) prob[r] = 0.f;
    }
    return;
  }
  if (!(T > 0.f)) {  // greedy: the row's argmax = its top candidate
    if (t == 0) {
      token[r] = cand(max(sm.key[0][0], sm.key[1][0])).id;
      if (prob) prob[r] = 1.f;
    }
    return;
  }
  int kk;
  float pp;
  trunc_params(tk, tp, V, kk, pp);
  const float c = temp_scale(T);
  if (!(kk >= 1 && kk <= kK && isfinite(c))) {
    if (t == 0) {
      token[r] = kSparseDense;
      if (prob) prob[r] = 0.f;
    }
    return;
  }
  const int rk = i + count_gt64(sm.key[1 - q], z);
  __syncthreads();
  if (rk < kk) sm.key[0][rk] = z;  // reuses rank 0's slots: every rank is known
  __syncthreads();
  if (t >= 32) return;
  const u64* st = sm.key[0];
  const Cut2 ct2 = cut_top(st, c, kk, pp, mp_thr(mp));
  const int lane = t, j0 = 2 * lane, j1 = j0 + 1;
  sm.w[j0] = ct2.w0, sm.w[j1] = ct2.w1;
  __syncwarp();
  // the draw over the kept ranks 0 … jc in index order (list_draw)
  const u64 U = philox(sd, ct, 0, 0), tt = __umul64hi(U, ct2.Wc);
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int j = j0 + h;
    if (j > ct2.jc) continue;
    const u64 lz = st[j] & 0xFFFFF, w = sm.w[j];
    u64 s = 0;
    for (int u = 0; u <= ct2.jc; u++) s += (st[u] & 0xFFFFF) > lz ? sm.w[u] : 0ull;
    if (s <= tt && tt - s < w) {
      token[r] = int(0xFFFFF - lz);
      if (prob) prob[r] = float(double(w) / double(ct2.Wc));
    }
  }
}

// ---- spec_accept_sparse ------------------------------------------------------------------------------------------
constexpr int kAT = 512, kAW = kAT / 32;  // 16 warps: the rows of k ≤ 7 cut in one round
constexpr int kQ0 = kSparseMaxK + 1;  // draft row i at row[kQ0 + i]

// A row's kept set in index order.
struct SRow {
  u64 w[kK];
  int id[kK];
  u64 W;  // W'
  int n, top;
};
struct ASm {
  p016::Bv bv;
  SRow row[kQ0 + kSparseMaxK];
  u64 st[kAW][2 * kK];
};

// Warp: the row's records → its kept set (mt: min_p's weight threshold).
__device__ void cut_rec(const SparseCand* r0, const SparseCand* r1, float c, int kk, float pp, u64 mt, SRow& o,
                        u64* st) {
  rank_top(r0, r1, kk, st);
  const Cut2 ct2 = cut_top(st, c, kk, pp, mt);
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int j = 2 * lane + h;
    if (j > ct2.jc) continue;
    const u64 lz = st[j] & 0xFFFFF;
    int pos = 0;
    for (int u = 0; u <= ct2.jc; u++) pos += (st[u] & 0xFFFFF) > lz;
    o.id[pos] = int(0xFFFFF - lz), o.w[pos] = h ? ct2.w1 : ct2.w0;
  }
  if (lane == 0) o.n = ct2.jc + 1, o.W = ct2.Wc, o.top = int(0xFFFFF - (st[0] & 0xFFFFF));
  __syncwarp();
}

// Weight of token d in a kept set (0 if not kept).
__device__ __forceinline__ u64 lookup(const SRow& R, int d) {
  int lo = 0, hi = R.n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (R.id[mid] < d) lo = mid + 1;
    else hi = mid;
  }
  return lo < R.n && R.id[lo] == d ? R.w[lo] : 0ull;
}

// Warp: the entry e (lanes: entries 2·lane, 2·lane + 1) whose [excl, excl + w) holds t → its id (every lane).
template <class Wt>
__device__ __forceinline__ int find(const SRow& R, Wt t, Wt a0, Wt a1) {
  const int lane = threadIdx.x & 31;
  Wt inc = a0 + a1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const Wt y = shfl_up(inc, o);
    if (lane >= o) inc += y;
  }
  const Wt e0 = inc - a1 - a0, e1 = inc - a1;
  int hit = -1;
  if (e0 <= t && t - e0 < a0) hit = R.id[2 * lane];
  if (e1 <= t && t - e1 < a1) hit = R.id[2 * lane + 1];
  const unsigned bl = __ballot_sync(~0u, hit >= 0);
  return __shfl_sync(~0u, hit, __ffs(bl) - 1);
}

// Warp: a draw from a kept set as sample_rows draws a row (W' = 0: its argmax).
__device__ int draw_rec(const SRow& R, u64 U) {
  if (R.W == 0) return R.top;
  const int lane = threadIdx.x & 31, j0 = 2 * lane, j1 = j0 + 1;
  return find<u64>(R, __umul64hi(U, R.W), j0 < R.n ? R.w[j0] : 0ull, j1 < R.n ? R.w[j1] : 0ull);
}

// Warp: the base rule's residual draw at pair (P, Q): r_v = (wp_v·Wq − wq_v·Wp)⁺ over P's kept tokens (0 elsewhere);
// Σr = 0 → a draw from P.
__device__ int resid_draw(const SRow& P, const SRow& Q, u64 U) {
  const int lane = threadIdx.x & 31;
  u128 a[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int j = 2 * lane + h;
    a[h] = 0;
    if (j < P.n) {
      const u128 u = u128(P.w[j]) * Q.W, w = u128(lookup(Q, P.id[j])) * P.W;
      a[h] = u > w ? u - w : u128(0);
    }
  }
  u128 tot = a[0] + a[1];
#pragma unroll
  for (int o = 16; o; o >>= 1) tot += shfl_xor(tot, o);
  if (tot == 0) return draw_rec(P, U);
  return find<u128>(P, scale(U, tot), a[0], a[1]);
}

// P016 classification of P's kept token j (r = α·wp − β·wq ≥ 0 → taken), its wp / wq.
__device__ __forceinline__ bool bv_take(const SRow& P, const SRow& Q, int j, const p016::Bv& bv, u64& wp, u64& wq) {
  wp = P.w[j], wq = lookup(Q, P.id[j]);
  bool take, uns;
  p016::cls(wp, wq, bv.f, take, uns);
  if (uns) take = p016::big_take(bv.al, bv.be, wp, wq);
  return take;
}

// Warp: P016 row pass over pair (P, Q) — Σ wp, Σ wq of the taken tokens (row_setup done).
__device__ p016::S2 bv_sums(const SRow& P, const SRow& Q, const p016::Bv& bv) {
  const int lane = threadIdx.x & 31;
  u64 sp = 0, sq = 0;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int j = 2 * lane + h;
    u64 wp, wq;
    if (j < P.n && bv_take(P, Q, j, bv, wp, wq)) sp += wp, sq += wq;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) sp += shfl_xor(sp, o), sq += shfl_xor(sq, o);
  return {sp, sq};
}

// Warp: P016's residual draw at pair (P, Q) (bv.t = t*, bv.al / be = α, β): the first kept token v of P in index order
// with F(≤ v) = α·Σ_{taken ≤ v} wp − β·Σ_{taken ≤ v} wq > t* (F is non-decreasing; F(all) = num > t*).
__device__ int bv_pick(const SRow& P, const SRow& Q, const p016::Bv& bv) {
  const int lane = threadIdx.x & 31;
  u64 cp[2] = {0, 0}, cq[2] = {0, 0};
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int j = 2 * lane + h;
    u64 wp, wq;
    if (j < P.n && bv_take(P, Q, j, bv, wp, wq)) cp[h] = wp, cq[h] = wq;
  }
  u64 ip = cp[0] + cp[1], iq = cq[0] + cq[1];
  p016::scan2(ip, iq);
  const u64 ep = ip - cp[0] - cp[1], eq = iq - cq[0] - cq[1];
  const bool g0 = 2 * lane < P.n && p016::form_gt(bv.al, bv.be, ep + cp[0], eq + cq[0], bv.t);
  const bool g1 = 2 * lane + 1 < P.n && p016::form_gt(bv.al, bv.be, ip, iq, bv.t);
  const unsigned b0 = __ballot_sync(~0u, g0), b1 = __ballot_sync(~0u, g1);
  const int L = __ffs(b0 | b1) - 1;
  return P.id[2 * L + ((b0 >> L & 1) ? 0 : 1)];
}

// grid B, kAT threads: sequence b (spec_accept's arguments on records). BV: P016 for T > 0 and 1 ≤ k ≤ kBvMax.
template <bool BV>
__global__ void __launch_bounds__(kAT) k_accept_sparse(const SparseCand* pc, int pld, const int* p_start,
                                                       const SparseCand* qc, int qld, const int* q_start,
                                                       const int* drafts, int V, const float* temp, const int* top_k,
                                                       const float* top_p, const float* min_p, const float* q_min_p,
                                                       const u64* seed, const u64* ctr, int* n_acc, int* next,
                                                       int bvx) {
  extern __shared__ __align__(16) unsigned char raw[];
  ASm& sm = *reinterpret_cast<ASm*>(raw);
  const int b = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int p0 = p_start[b], q0 = q_start[b], nd = q_start[b + 1] - q0;
  if (nd < 0 || p_start[b + 1] - p0 != nd + 1) {  // as spec_accept
    if (t == 0) n_acc[b] = -1, next[b] = -1;
    return;
  }
  const float T = temp[b];
  const int tk = top_k[b];
  const float tp = top_p[b];
  const u64 sd = seed[b], ct = ctr[b];
  const bool greedy = !(T > 0.f);
  int kk = 0;
  float pp = 1.f;
  const float c = greedy ? 0.f : temp_scale(T);
  if (!greedy) trunc_params(tk, tp, V, kk, pp);
  const bool elig = greedy || (kk >= 1 && kk <= kK && isfinite(c) && nd <= kSparseMaxK);
  auto prec = [&](int i, int q) { return pc + (size_t(q) * pld + p0 + i) * kRec; };
  auto qrec = [&](int i, int q) { return qc + (size_t(q) * qld + q0 + i) * kRec; };
  int f = 0;
  if (elig) {  // flags of every row the rule may read (greedy: the target rows)
    const int nr = greedy ? nd + 1 : 2 * nd + 1;
    for (int j = t; j < 2 * nr; j += kAT) {
      const int row = j >> 1, q = j & 1;
      f |= (row <= nd ? prec(row, q) : qrec(row - nd - 1, q))[kK].id;
    }
  }
  if (__syncthreads_or(!elig || f != 0)) {
    if (t == 0) n_acc[b] = kSparseDense, next[b] = kSparseDense;
    return;
  }
  if (greedy) {  // accept while d_i == argmax p_i
    if (t == 0) {
      int i = 0, a;
      for (;; i++) {
        const SparseCand *r0 = prec(i, 0), *r1 = prec(i, 1);
        a = cand(max(ckey(r0[0]), ckey(r1[0]))).id;
        if (i == nd || drafts[q0 + i] != a) break;
      }
      n_acc[b] = i, next[b] = a;
    }
    return;
  }
  const u64 pm = mp_thr(min_p[b]), qm = mp_thr(q_min_p[b]);
  for (int j = warp; j < 2 * nd + 1; j += kAW) {  // every row's kept set
    if (j <= nd) cut_rec(prec(j, 0), prec(j, 1), c, kk, pp, pm, sm.row[j], sm.st[warp]);
    else cut_rec(qrec(j - nd - 1, 0), qrec(j - nd - 1, 1), c, kk, pp, qm, sm.row[kQ0 + j - nd - 1], sm.st[warp]);
  }
  __syncthreads();
  if (warp) return;
  const u64 U0 = philox(sd, ct, 0, 1);
  int tau = nd, x = -1;
  if (BV && nd >= 1 && nd <= p016::kBvMax) {  // P016: p016::accept_cut + rounds on the kept sets
    p016::Bv& bv = sm.bv;
    int kr = 0;
    bool dead = false;
    while (kr < nd) {
      const int i = kr++;
      const SRow &P = sm.row[i], &Q = sm.row[kQ0 + i];
      const int d = drafts[q0 + i];
      const bool in = d >= 0 && d < V;
      const u64 wpd = in ? lookup(P, d) : 0ull, wqd = in ? lookup(Q, d) : 0ull;
      if (lane == 0) bv.Wp[i] = P.W, bv.Wq[i] = Q.W, bv.ra[i] = u128(wpd) * Q.W, bv.rb[i] = u128(wqd) * P.W;
      if (wpd == 0 || Q.W == 0) {
        dead = true;
        break;
      }
    }
    __syncwarp();
    if (lane == 0) {
      p016::chain(bv, kr);
      bv.flag = !dead && p016::big_lt_scaled(bv.D[nd], philox(sd, ct, nd, 1), bv.N[nd], bv.tmp);
      uint32_t need = 0;
      for (int i = 1; i < kr && !bv.flag; i++) {
        const bool deg = bv.Wp[i] == 0 || bv.Wq[i] == 0;
        if (bv.N[i].n && (deg || p016::big_lt_scaled(bv.D[i], philox(sd, ct, i, 1), bv.N[i], bv.tmp))) need |= 1u << i;
      }
      bv.need = need;
    }
    __syncwarp();
    if (bv.flag) {
      x = draw_rec(sm.row[nd], U0);
    } else {
      tau = 0;
      const uint32_t need = bv.need;
      for (int i = kr - 1; i >= 1; i--) {
        if (!(need >> i & 1)) continue;
        if (lane == 0) p016::row_setup(bv, i, bvx != 0);
        __syncwarp();
        const p016::S2 s = bv_sums(sm.row[i], sm.row[kQ0 + i], bv);
        if (lane == 0) p016::row_num(bv, s.p, s.q), bv.flag = p016::row_test(bv, i, philox(sd, ct, i, 1));
        __syncwarp();
        if (bv.flag) {
          tau = i;
          break;
        }
      }
      if (tau == 0) {
        if (lane == 0) p016::row_setup(bv, 0, bvx != 0);
        __syncwarp();
        const p016::S2 s = bv_sums(sm.row[0], sm.row[kQ0], bv);
        if (lane == 0) p016::row_num(bv, s.p, s.q);
        __syncwarp();
      }
      if (bv.num.n == 0) {
        x = draw_rec(sm.row[tau], U0);
      } else {
        if (lane == 0) p016::big_scale(bv.t, bv.num, U0, bv.tmp);
        __syncwarp();
        x = bv_pick(sm.row[tau], sm.row[kQ0 + tau], bv);
      }
    }
  } else {  // the base rule: accept_trunc on the kept sets
    for (int i = 0; i < nd; i++) {
      const SRow &P = sm.row[i], &Q = sm.row[kQ0 + i];
      const int d = drafts[q0 + i];
      const bool in = d >= 0 && d < V;
      const u64 wpd = in ? lookup(P, d) : 0ull, wqd = in ? lookup(Q, d) : 0ull;
      if (scale(philox(sd, ct, 1 + i, 1), u128(wqd) * P.W) < u128(wpd) * Q.W) continue;  // min(1, p'(d) / q'(d))
      tau = i;
      x = resid_draw(P, Q, U0);
      break;
    }
    if (tau == nd) x = draw_rec(sm.row[nd], U0);
  }
  if (lane == 0) n_acc[b] = tau, next[b] = x;
}

// ---- host ----------------------------------------------------------------------------------------------------------
// k_cands' cluster sizes, once: cap = the largest it runs (non-portable sizes allowed); fit[c] = clusters of size c
// resident at once with one CTA per SM (the occupancy query with shared memory for one CTA per SM) — clusters live
// inside a GPC, so this is Σ over GPCs of ⌊SMs / c⌋ (GB10: 4·⌊12 / c⌋), not ⌊48 / c⌋.
struct CFit {
  int cap, fit[kCMaxC + 1];
};
const CFit& cands_fit() {
  static const CFit f = [] {
    CFit r{};
    CUDA_CHECK(cudaFuncSetAttribute(k_cands, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaFuncAttributes fa;
    CUDA_CHECK(cudaFuncGetAttributes(&fa, k_cands));
    int optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const int dyn = std::max(0, optin - int(fa.sharedSizeBytes));  // the most one CTA can take: one CTA per SM
    CUDA_CHECK(cudaFuncSetAttribute(k_cands, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    cfg.blockDim = dim3(kCT);
    cfg.attrs = at;
    cfg.numAttrs = 1;
    at[0].val.clusterDim.x = kCMaxC, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(kCMaxC, 1);
    int n = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&n, k_cands, &cfg));
    if (n < 1) qlog::fail("sample", "P019 cluster size %d", n);
    r.cap = std::min(n, kCMaxC);
    cfg.dynamicSmemBytes = size_t(dyn);
    for (int c = 1; c <= r.cap; c++) {
      at[0].val.clusterDim.x = c;
      cfg.gridDim = dim3(c, 1);
      CUDA_CHECK(cudaOccupancyMaxActiveClusters(&r.fit[c], k_cands, &cfg));
    }
    CUDA_CHECK(cudaFuncSetAttribute(k_cands, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));  // launches use none
    return r;
  }();
  return f;
}
// CTAs per row: the largest c ≤ 8 (measured best: larger clusters cost more in the merge than they save in the pass)
// whose R clusters run in one wave, at most one CTA per kCMinChunk4 float4; 1 if none fits.
int cands_cluster(int Vh, int R) {
  const CFit& f = cands_fit();
  if (g_force) return std::min(g_force, f.cap);
  for (int c = std::min({8, f.cap, std::max(1, cdiv(Vh / 4, kCMinChunk4))}); c > 1; c--)
    if (f.fit[c] >= R) return c;
  return 1;
}
// k_accept_sparse's dynamic shared memory, once per instantiation.
template <bool BV>
void accept_attr() {
  static const bool done = [] {
    CUDA_CHECK(cudaFuncSetAttribute(k_accept_sparse<BV>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(ASm))));
    return true;
  }();
  (void)done;
}

}  // namespace p019
