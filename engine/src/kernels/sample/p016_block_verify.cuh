// p016_block_verify.cuh — P016 block_verify: spec_accept by block verification (Sun et al. 2024, arXiv 2403.10444)
// for T > 0 and 1 ≤ k ≤ kBvMax drafts, in exact integers (rule, numerics and random numbers: sample.h). Included by
// sample.cu inside its anonymous namespace (Smem, Cl, walk, exchange, row_max, row_sum, pick_row, trunc_row, draw,
// weight, kept4, philox).
//
// A sequence's round (rows as in sample.h: target / draft row i = p_i / q_i, weights wp / wq, totals Wp / Wq):
//   pass 1   every pair's totals and the drafts' weights — the base kernel's all-accepted walk (pipelined, one pass per
//            pair) without its tests; a draft with wp_i(d_i) = 0 ends it (c = 0 from there: no later row can matter).
//   chain    c_i = N_i / D_i (big integers, thread 0 of every CTA; every CTA computes the same), then the test of
//            h_k = c_k. Passed → the bonus row's sum pass and draw, as the base kernel's all-accepted end.
//   rows     else, i = k−1 down to 1, only rows with η_i ≤ c_i (h_i ≤ c_i: the others cannot pass; tested with the
//            chain, so a round passes over ≈ Σ c_i rows, not k − 1): one pass over pair i — every token's
//            r_v = α·wp_v − β·wq_v (α = N_i·Wq_i, β = D_i·Wp_i) classified by sign, the taken tokens' wp and wq summed
//            (u64, exact) → R_i·D_i·Wp_i·Wq_i = num = α·S_p − β·S_q, h_i's test; the first passed i is τ. None → τ = 0
//            (row 0's pass, α = Wq_0, β = Wp_0: the base kernel's residual weights exactly).
//   draw     num = 0 → p_τ drawn as a row; else t* = ⌊U·num / 2^64⌋ and the token v with F(< v) ≤ t* < F(≤ v),
//            F(S) = α·Σ_S wp − β·Σ_S wq over the taken tokens (a linear form of two u64 prefix sums): ranks → warps →
//            32 parts of the owner warp's run (repeated while > 32 float4) → float4 → element, each level one
//            lane-parallel comparison of big integers.
// Classification: a token with r_v = 0 may be taken or not (it adds 0 to every F), so only the sign of r_v ≠ 0 has to
// be right. With s = max(0, bits(max(α, β)) − 64), a = ⌊α/2^s⌋, b = ⌊β/2^s⌋ (pad = 1 if s > 0, the floors' slack):
// a·wp ≥ (b + pad)·wq ⇒ r ≥ 0 (take); (a + pad)·wp ≤ b·wq ⇒ r ≤ 0 (leave); else unsure — then exactly (α·wp − β·wq
// over 32-bit limbs with a running signed carry). The walk only counts its unsure tokens (two u128 products per token,
// as the base residual pass); a thread with any reads its own float4s again and settles them. Real rows have none (it
// takes |r| < 2^-63 of α·wp); sample_force_exact makes every token with wp, wq > 0 unsure (tests).
// Sizes: wp ≤ 2^44, W < 2^64 (V < 2^20) → a_i, b_i < 2^108, N_i, D_i < 2^(108·i), α, β < 2^1684, num, den < 2^1748,
// U·den < 2^1812 ≤ 32·kL bits.
#pragma once

namespace p016 {

constexpr int kBvMax = 15;  // drafts at most (the engine's depth limit); more → the base rule
constexpr int kL = 60;      // 32-bit limbs per big integer (1920 bits)

struct Big {
  uint32_t d[kL];  // little-endian limbs
  int n;           // limbs in use: d[n − 1] ≠ 0, n = 0 for zero
};

// Classification constants of a row (see the top): a = ⌊α/2^s⌋, b = ⌊β/2^s⌋, pad = s > 0
struct Flt {
  u64 a, b, pad;
};

struct Bv {
  Arg ap[kBvMax + 1], aq[kBvMax];   // maxima (argmax) of target rows 0 … k and draft rows 0 … k−1
  u64 Wp[kBvMax + 1], Wq[kBvMax];   // totals (W' with truncation)
  u64 zp[kBvMax], zq[kBvMax];       // cuts (truncation)
  u128 ra[kBvMax], rb[kBvMax];      // a_i = wp_i(d_i)·Wq_i, b_i = wq_i(d_i)·Wp_i
  Big N[kBvMax + 1], D[kBvMax + 1];  // c_i = N_i / D_i
  Big al, be, num, den, t, tmp;     // α, β of the current row; h's numerator / denominator (num = Σr); t*; scratch
  Flt f;
  u64 off[2];                       // the pick's offset: Σ wp, Σ wq of the taken tokens before the current range
  int flag, ro, wo;                 // a test's outcome; owner rank / warp of the pick
  uint32_t need;                    // rows i whose h test can pass (η_i ≤ c_i)
};

// ---- big integers (one thread) ---------------------------------------------------------------------------------
__device__ __forceinline__ void big_set(Big& x, u128 v) {
  x.n = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    x.d[j] = uint32_t(v >> (32 * j));
    if (x.d[j]) x.n = j + 1;
  }
}
__device__ __forceinline__ void big_trim(Big& x) {
  while (x.n > 0 && x.d[x.n - 1] == 0) x.n--;
}
// o = x·m (m < 2^128); o ≠ x. Column by column: limb j = Σ_t x_{j−t}·m_t + carry in a register (the loads do not wait
// on the carry chain; one store per limb).
__device__ void big_mul(Big& o, const Big& x, u128 m) {
  const uint32_t ms[4] = {uint32_t(m), uint32_t(m >> 32), uint32_t(m >> 64), uint32_t(m >> 96)};
  const int mn = ms[3] ? 4 : ms[2] ? 3 : ms[1] ? 2 : ms[0] ? 1 : 0;
  if (x.n == 0 || mn == 0) {
    o.n = 0;
    return;
  }
  const int n = x.n + mn;
  if (n > kL) __trap();  // never (sizes at the top)
  u128 acc = 0;
  for (int j = 0; j < n; j++) {
#pragma unroll
    for (int t = 0; t < 4; t++) {
      const int i = j - t;
      if (t < mn && i >= 0 && i < x.n) acc += u64(x.d[i]) * ms[t];
    }
    o.d[j] = uint32_t(acc);
    acc >>= 32;
  }
  o.n = n;
  big_trim(o);
}
__device__ int big_cmp(const Big& a, const Big& b) {
  if (a.n != b.n) return a.n < b.n ? -1 : 1;
  for (int j = a.n - 1; j >= 0; j--)
    if (a.d[j] != b.d[j]) return a.d[j] < b.d[j] ? -1 : 1;
  return 0;
}
// o = a − b (a ≥ b); o may be a
__device__ void big_sub(Big& o, const Big& a, const Big& b) {
  const int n = a.n;
  int64_t br = 0;
  for (int j = 0; j < n; j++) {
    const int64_t t = int64_t(a.d[j]) - int64_t(j < b.n ? b.d[j] : 0u) + br;
    o.d[j] = uint32_t(t);
    br = t < 0 ? -1 : 0;
  }
  o.n = n;
  big_trim(o);
}
// o = a + b; o may be a or b
__device__ void big_add(Big& o, const Big& a, const Big& b) {
  const int n = max(a.n, b.n);
  if (n + 1 > kL) __trap();
  u64 cy = 0;
  for (int j = 0; j < n; j++) {
    cy += u64(j < a.n ? a.d[j] : 0u) + u64(j < b.n ? b.d[j] : 0u);
    o.d[j] = uint32_t(cy);
    cy >>= 32;
  }
  o.d[n] = uint32_t(cy);
  o.n = n + 1;
  big_trim(o);
}
// ⌊U·x / 2^64⌋ < y (tmp: scratch)
__device__ bool big_lt_scaled(const Big& x, u64 U, const Big& y, Big& tmp) {
  big_mul(tmp, x, U);
  const int n = tmp.n > 2 ? tmp.n - 2 : 0;
  if (n != y.n) return n < y.n;
  for (int j = n - 1; j >= 0; j--)
    if (tmp.d[j + 2] != y.d[j]) return tmp.d[j + 2] < y.d[j];
  return false;
}
// o = ⌊U·x / 2^64⌋ (tmp: scratch, ≠ o)
__device__ void big_scale(Big& o, const Big& x, u64 U, Big& tmp) {
  big_mul(tmp, x, U);
  o.n = tmp.n > 2 ? tmp.n - 2 : 0;
  for (int j = 0; j < o.n; j++) o.d[j] = tmp.d[j + 2];
}
__device__ __forceinline__ int big_bits(const Big& x) { return x.n ? 32 * x.n - __clz(int(x.d[x.n - 1])) : 0; }
// ⌊x / 2^s⌋ for x < 2^(s+64)
__device__ __forceinline__ u64 big_top(const Big& x, int s) {
  const int q = s >> 5, r = s & 31;
  u128 v = 0;
  for (int j = 2; j >= 0; j--) v = v << 32 | (q + j < x.n ? x.d[q + j] : 0u);
  return u64(v >> r);
}

// ---- token classification (r = α·wp − β·wq) -------------------------------------------------------------------------
// take: r ≥ 0 sure; unsure: neither r ≥ 0 nor r ≤ 0 sure (see the top)
__device__ __forceinline__ void cls(u64 wp, u64 wq, const Flt& f, bool& take, bool& unsure) {
  const u128 X = u128(f.a) * wp, Y = u128(f.b) * wq;
  take = X >= Y + (f.pad ? wq : 0ull);
  unsure = !take && X + (f.pad ? wp : 0ull) > Y;
}
// α·wp ≥ β·wq exactly: Σ_j (α_j·wp − β_j·wq)·2^(32j) with a running signed carry; the last carry's sign is the sum's
// (the limbs below it form a number in [0, 2^(32n))).
__device__ __noinline__ bool big_take(const Big& al, const Big& be, u64 wp, u64 wq) {
  const int n = max(al.n, be.n);
  __int128 c = 0;
  for (int j = 0; j < n; j++) {
    c += __int128(u128(j < al.n ? al.d[j] : 0u) * wp) - __int128(u128(j < be.n ? be.d[j] : 0u) * wq);
    c >>= 32;
  }
  return c >= 0;
}
// α·a − β·b > t (a, b < 2^64), the same way (|carry| < 2^97)
__device__ __noinline__ bool form_gt(const Big& al, const Big& be, u64 a, u64 b, const Big& t) {
  const int n = max(max(al.n, be.n), t.n);
  __int128 c = 0;
  uint32_t nz = 0;
  for (int j = 0; j < n; j++) {
    c += __int128(u128(j < al.n ? al.d[j] : 0u) * a) - __int128(u128(j < be.n ? be.d[j] : 0u) * b) -
         __int128(j < t.n ? t.d[j] : 0u);
    nz |= uint32_t(c);
    c >>= 32;
  }
  return c > 0 || (c == 0 && nz != 0);
}

// ---- the chain and the per-row tests (thread 0 of every CTA) --------------------------------------------------------
// c_0 = 1; c_{i+1} = 0 if c_i = 0 or a_i = 0; 1 if N_i·a_i ≥ D_i·b_i (b_i = 0 < a_i included); else N_i·a_i / D_i·b_i.
__device__ void chain(Bv& bv, int kr) {
  big_set(bv.N[0], 1), big_set(bv.D[0], 1);
  for (int i = 0; i < kr; i++) {
    Big &N = bv.N[i + 1], &D = bv.D[i + 1];
    if (bv.N[i].n == 0 || bv.ra[i] == 0) {
      big_set(N, 0), big_set(D, 1);
      continue;
    }
    big_mul(N, bv.N[i], bv.ra[i]);
    big_mul(D, bv.D[i], bv.rb[i]);
    if (big_cmp(N, D) >= 0) big_set(N, 1), big_set(D, 1);
  }
}
// α, β and the classification constants of row i (fx: every token with wp, wq > 0 unsure)
__device__ void row_setup(Bv& bv, int i, bool fx) {
  big_mul(bv.al, bv.N[i], bv.Wq[i]);
  big_mul(bv.be, bv.D[i], bv.Wp[i]);
  const int s = max(0, max(big_bits(bv.al), big_bits(bv.be)) - 64);
  bv.f = fx ? Flt{0, 0, 1} : Flt{big_top(bv.al, s), big_top(bv.be, s), u64(s > 0)};
}
// num = α·S_p − β·S_q (= Σ_v max(r_v, 0) ≥ 0)
__device__ void row_num(Bv& bv, u64 Sp, u64 Sq) {
  big_mul(bv.num, bv.al, Sp);
  big_mul(bv.tmp, bv.be, Sq);
  big_sub(bv.num, bv.num, bv.tmp);
}
// η_i ≤ h_i: den = num + (D_i − N_i)·Wp_i·Wq_i; den = 0 → h_i = 1; else ⌊U·den / 2^64⌋ < num
__device__ bool row_test(Bv& bv, int i, u64 U) {
  big_sub(bv.t, bv.D[i], bv.N[i]);
  big_mul(bv.den, bv.t, u128(bv.Wp[i]) * bv.Wq[i]);
  big_add(bv.den, bv.den, bv.num);
  return bv.den.n == 0 || big_lt_scaled(bv.den, U, bv.num, bv.tmp);
}

// ---- passes over pair i --------------------------------------------------------------------------------------------
struct S2 {
  u64 p, q;
};
// The taken tokens' weights among this thread's float4s of the walk (g ≡ w0 + lane mod 32), unsure ones settled.
template <bool CUT>
__device__ __noinline__ S2 settle(const float* p, const float* q, float mp, float mq, u64 zp, u64 zq, float c,
                                  const Bv& bv, int w0, int w1) {
  S2 s{0, 0};
  for (int g = w0 + int(threadIdx.x & 31); g < w1; g += 32) {
    float4 y = __ldg(reinterpret_cast<const float4*>(p) + g), z = __ldg(reinterpret_cast<const float4*>(q) + g);
    if (CUT) y = kept4(y, g, zp), z = kept4(z, g, zq);
    const float ys[4] = {y.x, y.y, y.z, y.w}, zs[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const u64 wp = weight(ys[e], mp, c), wq = weight(zs[e], mq, c);
      bool take, uns;
      cls(wp, wq, bv.f, take, uns);
      if (uns && big_take(bv.al, bv.be, wp, wq)) s.p += wp, s.q += wq;
    }
  }
  return s;
}

// Pair i's taken sums → sm.incl[0] / [1] (ranks' inclusive prefix of S_p / S_q), sm.ws[·][0 / 1] (warp sums).
template <bool CUT>
__device__ void row_pass(const float* p, const float* q, int i, float c, const Bv& bv, Cl& k) {
  const float mp = bv.ap[i].v, mq = bv.aq[i].v;
  const u64 zp = CUT ? bv.zp[i] : 0, zq = CUT ? bv.zq[i] : 0;
  const Flt f = bv.f;
  u64 s[2] = {0, 0};
  int nu = 0;
  const float* x[2] = {p, q};
  walk<2>(x, k, [&](int g, const float4* v) {
    const float4 y = CUT ? kept4(v[0], g, zp) : v[0], z = CUT ? kept4(v[1], g, zq) : v[1];
    const float ys[4] = {y.x, y.y, y.z, y.w}, zs[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const u64 wp = weight(ys[e], mp, c), wq = weight(zs[e], mq, c);
      bool take, uns;
      cls(wp, wq, f, take, uns);
      s[0] += take ? wp : 0ull;
      s[1] += take ? wq : 0ull;
      nu += uns;
    }
  });
  if (nu) {
    const S2 t = settle<CUT>(p, q, mp, mq, zp, zq, c, bv, k.w0, k.w1);
    s[0] += t.p, s[1] += t.q;
  }
  exchange<0, 2, u64>(nullptr, s, 0, k);
}

// The taken weights of float4 g (0 where not taken), unsure tokens settled.
template <bool CUT>
__device__ __forceinline__ void pair4(const float* p, const float* q, int g, float mp, float mq, u64 zp, u64 zq,
                                      float c, const Bv& bv, u64 (&cp)[4], u64 (&cq)[4]) {
  float4 y = __ldg(reinterpret_cast<const float4*>(p) + g), z = __ldg(reinterpret_cast<const float4*>(q) + g);
  if (CUT) y = kept4(y, g, zp), z = kept4(z, g, zq);
  const float ys[4] = {y.x, y.y, y.z, y.w}, zs[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const u64 wp = weight(ys[e], mp, c), wq = weight(zs[e], mq, c);
    bool take, uns;
    cls(wp, wq, bv.f, take, uns);
    if (uns) take = big_take(bv.al, bv.be, wp, wq);
    cp[e] = take ? wp : 0ull, cq[e] = take ? wq : 0ull;
  }
}

// Inclusive warp scan of a pair of u64
__device__ __forceinline__ void scan2(u64& a, u64& b) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const u64 ya = shfl_up(a, o), yb = shfl_up(b, o);
    if (lane >= o) a += ya, b += yb;
  }
}

// The residual draw at row i (its row_pass was the last exchange; bv.t = t*, bv.al / be = α, β) → *nxt (owner CTA)
template <bool CUT>
__device__ void pick(const float* p, const float* q, int i, float c, Bv& bv, Cl& k, int* nxt) {
  Smem& sm = k.sm;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp == 0) {  // owner rank: the first whose inclusive F exceeds t* (every CTA finds the same)
    const bool gt = lane < k.C && form_gt(bv.al, bv.be, u64(sm.incl[0][lane]), u64(sm.incl[1][lane]), bv.t);
    const int o = __ffs(__ballot_sync(~0u, gt)) - 1;
    if (lane == 0) {
      bv.ro = o;
      bv.off[0] = o ? u64(sm.incl[0][o - 1]) : 0ull, bv.off[1] = o ? u64(sm.incl[1][o - 1]) : 0ull;
    }
  }
  __syncthreads();
  if (bv.ro != k.c) return;
  if (warp == 0) {  // owner warp
    const u64 vp = lane < k.nw ? u64(sm.ws[lane][0]) : 0ull, vq = lane < k.nw ? u64(sm.ws[lane][1]) : 0ull;
    u64 ip = vp, iq = vq;
    scan2(ip, iq);
    const bool gt = lane < k.nw && form_gt(bv.al, bv.be, bv.off[0] + ip, bv.off[1] + iq, bv.t);
    const int w = __ffs(__ballot_sync(~0u, gt)) - 1;
    __syncwarp();
    if (lane == w) bv.wo = w, bv.off[0] += ip - vp, bv.off[1] += iq - vq;
  }
  __syncthreads();
  if (warp != bv.wo) return;
  const float mp = bv.ap[i].v, mq = bv.aq[i].v;
  const u64 zp = CUT ? bv.zp[i] : 0, zq = CUT ? bv.zq[i] : 0;
  u64 op = bv.off[0], oq = bv.off[1];
  int g0 = k.w0, g1 = k.w1;
  while (g1 - g0 > 32) {  // 32 contiguous parts, one per lane
    const int m = cdiv(g1 - g0, 32), a0 = min(g0 + lane * m, g1), a1 = min(a0 + m, g1);
    u64 sp = 0, sq = 0;
    for (int g = a0; g < a1; g++) {
      u64 cp[4], cq[4];
      pair4<CUT>(p, q, g, mp, mq, zp, zq, c, bv, cp, cq);
      sp += cp[0] + cp[1] + cp[2] + cp[3], sq += cq[0] + cq[1] + cq[2] + cq[3];
    }
    u64 ip = sp, iq = sq;
    scan2(ip, iq);
    const bool gt = form_gt(bv.al, bv.be, op + ip, oq + iq, bv.t);
    const int l = __ffs(__ballot_sync(~0u, gt)) - 1;
    op += __shfl_sync(~0u, ip - sp, l), oq += __shfl_sync(~0u, iq - sq, l);
    g0 = __shfl_sync(~0u, a0, l), g1 = __shfl_sync(~0u, a1, l);
  }
  const int g = g0 + lane;  // ≤ 32 float4: one per lane
  u64 cp[4] = {0, 0, 0, 0}, cq[4] = {0, 0, 0, 0};
  if (g < g1) pair4<CUT>(p, q, g, mp, mq, zp, zq, c, bv, cp, cq);
  const u64 sp = cp[0] + cp[1] + cp[2] + cp[3], sq = cq[0] + cq[1] + cq[2] + cq[3];
  u64 ip = sp, iq = sq;
  scan2(ip, iq);
  const bool gt = g < g1 && form_gt(bv.al, bv.be, op + ip, oq + iq, bv.t);
  const int l = __ffs(__ballot_sync(~0u, gt)) - 1;
  op += __shfl_sync(~0u, ip - sp, l), oq += __shfl_sync(~0u, iq - sq, l);
  u64 ep = op, eq = oq;  // lane e < 4: through element e of lane l's float4
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const u64 xp = __shfl_sync(~0u, cp[e], l), xq = __shfl_sync(~0u, cq[e], l);
    if (e <= lane) ep += xp, eq += xq;
  }
  const bool ge = lane < 4 && form_gt(bv.al, bv.be, ep, eq, bv.t);
  const int e = __ffs(__ballot_sync(~0u, ge)) - 1;
  if (lane == 0) *nxt = 4 * (g0 + l) + e;
}

// ---- the round -----------------------------------------------------------------------------------------------------
// After pass 1 (pairs 0 … kr−1 recorded; dead: a_{kr−1} = 0, so c_kr = 0 and τ < kr): the chain and h_k's test
// (thread 0), then — failed — the rows, τ and the residual draw. Returns τ (every thread); *nxt written unless τ = k
// (the caller draws the bonus row then).
template <bool CUT>
__device__ int rounds(const float* P, const float* Q, int V, int nd, int kr, bool dead, float c, u64 sd, u64 ct,
                      bool fx, int* nxt, Bv& bv, Cl& k) {
  const bool t0 = threadIdx.x == 0;
  if (t0) {
    chain(bv, kr);
    bv.flag = !dead && big_lt_scaled(bv.D[nd], philox(sd, ct, nd, 1), bv.N[nd], bv.tmp);
    // rows whose test can pass: h_i ≤ c_i (R_i ≤ c_i), and in the integer tests η_i ≤ h_i ⟺ U·den < num·2^64 ⇒
    // U·D_i < N_i·2^64 ⟺ ⌊U·D_i / 2^64⌋ < N_i (den = 0: c_i = 1 — or a degenerate row, Wp_i·Wq_i = 0, where num = den
    // = 0 and the convention h_i = 1 holds whatever c_i: always passed over). The other rows get no pass: the same τ.
    uint32_t need = 0;
    for (int i = 1; i < kr && !bv.flag; i++) {
      const bool deg = bv.Wp[i] == 0 || bv.Wq[i] == 0;
      if (bv.N[i].n && (deg || big_lt_scaled(bv.D[i], philox(sd, ct, i, 1), bv.N[i], bv.tmp))) need |= 1u << i;
    }
    bv.need = need;
  }
  __syncthreads();
  if (bv.flag) return nd;
  const uint32_t need = bv.need;
  auto prow = [&](int i) { return P + size_t(i) * V; };
  auto qrow = [&](int i) { return Q + size_t(i) * V; };
  int tau = 0;
  for (int i = kr - 1; i >= 1; i--) {
    if (!(need >> i & 1)) continue;  // η_i > c_i ≥ h_i (c_i = 0 included): no pass
    if (t0) row_setup(bv, i, fx);
    __syncthreads();
    row_pass<CUT>(prow(i), qrow(i), i, c, bv, k);
    if (t0) {
      row_num(bv, u64(k.sm.incl[0][k.C - 1]), u64(k.sm.incl[1][k.C - 1]));
      bv.flag = row_test(bv, i, philox(sd, ct, i, 1));
    }
    __syncthreads();
    if (bv.flag) {
      tau = i;
      break;
    }
  }
  if (tau == 0) {
    if (t0) row_setup(bv, 0, fx);
    __syncthreads();
    row_pass<CUT>(prow(0), qrow(0), 0, c, bv, k);
    if (t0) row_num(bv, u64(k.sm.incl[0][k.C - 1]), u64(k.sm.incl[1][k.C - 1]));
    __syncthreads();
  }
  const u64 U = philox(sd, ct, 0, 1);
  if (bv.num.n == 0) return -1 - tau;  // Σr = 0: the caller draws p_τ
  if (t0) big_scale(bv.t, bv.num, U, bv.tmp);
  __syncthreads();
  pick<CUT>(prow(tau), qrow(tau), tau, c, bv, k, nxt);
  return tau;
}

// Untruncated sequence: pass 1 as the base kernel's walk (pair i's sums with pair i+1's maxima), then rounds.
__device__ void accept_full(const float* P, const float* Q, const int* dr, int nd, int V, float c, u64 sd, u64 ct,
                            bool fx, int* n_acc, int* nxt, Bv& bv, Cl& k) {
  Smem& sm = k.sm;
  const bool lead = k.c == 0 && threadIdx.x == 0;
  auto prow = [&](int i) { return P + size_t(i) * V; };
  auto qrow = [&](int i) { return Q + size_t(i) * V; };
  Arg a[2];
  {
    const float* x[2] = {prow(0), qrow(0)};
    row_max<2>(x, a, k);
  }
  int kr = 0;
  bool dead = false;
  while (kr < nd) {
    const int i = kr++;
    const float* p = prow(i);
    const float* q = qrow(i);
    const int d = dr[i];
    const bool in = d >= 0 && d < V;
    const float xp = in ? __ldg(p + d) : 0.f, xq = in ? __ldg(q + d) : 0.f;
    Arg an[2];
    if (i + 1 < nd) {
      const float* x[4] = {p, q, prow(i + 1), qrow(i + 1)};
      row_sum<2, 2>(x, a, c, an, k);
    } else {
      const float* x[3] = {p, q, prow(nd)};
      row_sum<2, 1>(x, a, c, an, k);
    }
    const u64 Wp = u64(sm.incl[0][k.C - 1]), Wq = u64(sm.incl[1][k.C - 1]);
    const u64 wpd = in ? weight(xp, a[0].v, c) : 0ull, wqd = in ? weight(xq, a[1].v, c) : 0ull;
    if (threadIdx.x == 0) {
      bv.ap[i] = a[0], bv.aq[i] = a[1], bv.Wp[i] = Wp, bv.Wq[i] = Wq;
      bv.ra[i] = u128(wpd) * Wq, bv.rb[i] = u128(wqd) * Wp;
    }
    a[0] = an[0], a[1] = an[1];
    if (wpd == 0 || Wq == 0) {  // a_i = 0: c = 0 from here on
      dead = true;
      break;
    }
  }
  int tau = rounds<false>(P, Q, V, nd, kr, dead, c, sd, ct, fx, nxt, bv, k);
  if (tau == nd || tau < 0) {  // the bonus row (its maximum came with the last pass), or p_τ (Σr = 0)
    const int r = tau < 0 ? -1 - tau : nd;
    Arg b[1] = {tau < 0 ? bv.ap[r] : a[0]};
    const float* x[1] = {prow(r)};
    row_sum<1, 0>(x, b, c, nullptr, k);
    pick_row<false>(x[0], b[0], c, sm.incl[0], philox(sd, ct, 0, 1), nxt, nullptr, k);
    tau = r;
  }
  if (lead) *n_acc = tau;
}

// Truncated sequence: pass 1 through trunc_row per row (as accept_trunc; pm / qm: min_p's weight thresholds of the
// target / draft rows), then rounds; p_τ / the bonus row drawn as accept_trunc draws them.
__device__ __noinline__ void accept_cut(const float* P, const float* Q, const int* dr, int nd, int V, float c, int kk,
                                        float pp, u64 pm, u64 qm, int cap, u64 sd, u64 ct, bool fx, int* n_acc,
                                        int* nxt, Bv& bv, Cl k) {
  const bool lead = k.c == 0 && threadIdx.x == 0;
  int kr = 0;
  bool dead = false;
  while (kr < nd) {
    const int i = kr++;
    const float* p = P + size_t(i) * V;
    const float* q = Q + size_t(i) * V;
    const int d = dr[i];
    const bool in = d >= 0 && d < V;
    Arg ap, aq;
    const Cut cp = trunc_row(p, ap, c, kk, pp, pm, cap, k);
    const Cut cq = trunc_row(q, aq, c, kk, pp, qm, cap, k);
    const u64 wpd = in ? weight(kept(__ldg(p + d), d, cp.zc), ap.v, c) : 0ull;
    const u64 wqd = in ? weight(kept(__ldg(q + d), d, cq.zc), aq.v, c) : 0ull;
    if (threadIdx.x == 0) {
      bv.ap[i] = ap, bv.aq[i] = aq, bv.Wp[i] = cp.W, bv.Wq[i] = cq.W, bv.zp[i] = cp.zc, bv.zq[i] = cq.zc;
      bv.ra[i] = u128(wpd) * cq.W, bv.rb[i] = u128(wqd) * cp.W;
    }
    if (wpd == 0 || cq.W == 0) {
      dead = true;
      break;
    }
  }
  int tau = rounds<true>(P, Q, V, nd, kr, dead, c, sd, ct, fx, nxt, bv, k);
  if (tau == nd || tau < 0) {
    const int r = tau < 0 ? -1 - tau : nd;
    const float* p = P + size_t(r) * V;
    Arg a;
    const Cut cb = trunc_row(p, a, c, kk, pp, pm, cap, k);
    draw(p, a, c, cb, philox(sd, ct, 0, 1), nxt, nullptr, k);
    tau = r;
  }
  if (lead) *n_acc = tau;
}

}  // namespace p016
