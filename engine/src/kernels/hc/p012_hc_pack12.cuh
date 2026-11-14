// p012_hc_pack12.cuh — P012 hc_pack12 (patches.def): the decode-path HC kernels (M < kHcBigRows, 64-row passes) read a
// lossless 12-bit copy of the down (320 lora rows) and up weights instead of bf16 (−25 % of the bytes every decode step
// reads per module). Included by hc.cu inside its anonymous namespace after the host helpers; the prefill paths and the
// base decode kernels keep reading the bf16 copy (both copies stay on the device).
//
// Format (per weight row along K: 10240 values for down, 320 for up). Row base b = max(Emax − 14, 1) (Emax = largest
// exponent field of the row's nonzero values); a bf16 value s·E·m is stored as 12 bits s · c (4) · m (7):
//   c = E − b + 1 for b ≤ E ≤ b + 14, c = 0 for ±0 (sign kept); anything else (E below the window, subnormal) is an
//   exception: c = 0 in the stream and the exact bf16 in a side list (0.01–0.04 % of the values).
// Decode: mini = s<<15 | c<<7 | m is ±0 or a normal bf16 (exponent field c), and mini × 2^(b−1) (mul.rn.bf16x2 with the
// row scale, a power of two: exact) is the original value. b ≤ 128 is required (|w| < 2^16), else the module stays bf16.
// Stream (A fragments in pack_tiles order, words x y z w = rows g, g+8, g, g+8 of the tile): a k-step of a lane is 4 mini
// words M0..M3 whose bits 11..14 and 27..30 are zero; M3's 24 bits fill those gaps → 3 words
//   W0 = M0 | (M3 & 0x000F000F) << 11 · W1 = M1 | (M3 & 0x00F000F0) << 7 · W2 = M2 | (M3 & 0x07000700) << 3 | (M3 & 0x80008000) >> 1
// group = 4 k-steps = 3 uint4 per lane (uint4 v = W_v of the 4 k-steps), 512 B rows (32 lanes):
//   down: [kc 40][tile 20][group 4][v 3][32] (a K2 CTA's 120 KB contiguous) · up: [tile 640][group 5][v 3][32].
// Exceptions: per warp work unit a list of uint32 entries value << 16 | blk << 8 | lane << 3 | elem (sorted by the low
// 16 bits; blk = 4·k-step + tile slot for down, k-step for up). A warp holds 32 entries in registers (one per lane) and
// patches its decoded fragment when the next entry's blk is the current one (a warp-uniform test per fragment).
// The inject rows (4 × 10240) stay bf16 (compact tile of the base). Per element the MMA operands, k-step order, the 40
// K-chunks and every rounding point are those of the base kernels → bitwise equal outputs for every M.
#pragma once

namespace p12 {

constexpr int kDnG = kDnStep / 4, kUpG = kUpKs / 4;                    // groups per K2 CTA (4) / per up tile (5)
constexpr size_t kDnB = size_t(kDnChunks) * 20 * kDnG * 3 * 512;        // 4,915,200 B
constexpr size_t kUpB = size_t(kUpTiles) * kUpG * 3 * 512;              // 4,915,200 B
constexpr int kDnUnits = kDnChunks * kDnWarps, kUpUnits = kUpTiles;     // exception lists: (kc, warp) / up tile
constexpr int kUpPieces = 40;                                           // bulk L2 prefetch pieces of the up copy (K2b)
constexpr size_t al16(size_t x) { return (x + 15) / 16 * 16; }
// uint4 row (lane 0) of (kc, tile, group, v) in the down part / (tile, group, v) in the up part
__host__ __device__ constexpr size_t dn_row(int kc, int tl, int G, int v) {
  return ((size_t(kc) * 20 + tl) * kDnG + G) * 3 + v;
}
__host__ __device__ constexpr size_t up_row(int tl, int G, int v) { return (size_t(tl) * kUpG + G) * 3 + v; }
constexpr size_t kOScD = kDnB + kUpB, kOScU = kOScD + 20 * 8 * 4;       // row scales: per tile 8 × (s_g | s_g+8 << 16)
constexpr size_t kOEoD = al16(kOScU + size_t(kUpTiles) * 8 * 4);        // list offsets (absolute entry indices)
constexpr size_t kOEoU = al16(kOEoD + (kDnUnits + 1) * 4);
constexpr size_t kOEnt = al16(kOEoU + (kUpUnits + 1) * 4);              // entries: down lists, then up lists

// ---- device ------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// bulk prefetch of [p, p + bytes) into L2 (TMA engine; bytes multiple of 16)
__device__ __forceinline__ void l2_bulk(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint32_t comp(const uint4& v, int q) { return q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w; }

// one k-step of a lane: W0..W2 → A fragment (sg / sh: bf16x2 scales of rows g / g+8)
__device__ __forceinline__ uint4 dec(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t sg, uint32_t sh) {
  constexpr uint32_t K = 0x87FF87FFu;
  const uint32_t m3 = ((w0 >> 11) & 0x000F000Fu) | ((w1 >> 7) & 0x00F000F0u) | ((w2 >> 3) & 0x07000700u) |
                      ((w2 << 1) & 0x80008000u);
  return make_uint4(hmul2(w0 & K, sg), hmul2(w1 & K, sh), hmul2(w2 & K, sg), hmul2(m3, sh));
}
__device__ __forceinline__ uint4 dec_q(const uint4 (&p)[3], int q, uint32_t sg, uint32_t sh) {
  return dec(comp(p[0], q), comp(p[1], q), comp(p[2], q), sg, sh);
}
__device__ __forceinline__ void scales(uint32_t x, uint32_t& sg, uint32_t& sh) {
  sg = __byte_perm(x, 0, 0x1010), sh = __byte_perm(x, 0, 0x3232);
}

// A warp's exception list [p, end): entries c0 … c0+31 in registers (lane l holds c0 + l), nk = blk of entry p.
struct Exc {
  const uint32_t* ent;
  int p, end, c0;
  uint32_t r, nk;
  __device__ __forceinline__ void load(const uint32_t* e, int b, int n, int lane) {  // issue the load only
    ent = e, p = c0 = b, end = n;
    r = b + lane < n ? __ldg(e + b + lane) : 0u;
  }
  __device__ __forceinline__ void start() { nk = p < end ? (__shfl_sync(~0u, r, 0) >> 8) & 0xFF : 0x100u; }
  // patch fragment v (block blk) with every entry of blk
  __device__ __forceinline__ void apply(uint4& v, uint32_t blk, int lane) {
    if (nk != blk) return;
    do {
      const uint32_t e = __shfl_sync(~0u, r, p - c0);
      if (int((e >> 3) & 31) == lane) {
        const uint32_t el = e & 7, sh = (el & 1) * 16, km = ~(0xFFFFu << sh), nv = (e >> 16) << sh;
        if ((el >> 1) == 0) v.x = (v.x & km) | nv;
        else if ((el >> 1) == 1) v.y = (v.y & km) | nv;
        else if ((el >> 1) == 2) v.z = (v.z & km) | nv;
        else v.w = (v.w & km) | nv;
      }
      if (++p - c0 == 32 && p < end) c0 = p, r = c0 + lane < end ? __ldg(ent + c0 + lane) : 0u;  // > 32 entries: rare
      nk = p < end ? (__shfl_sync(~0u, r, p - c0) >> 8) & 0xFF : 0x100u;
    } while (nk == blk);
  }
};

// ---- K2: down projection from the packed copy (grid, warps, tiles, xs, MMA order and stores of hc.cu k_down) ---------
// Warp w: slots i = 0..2 ↔ tiles w + 7i; tile 20 (inject, bf16 compact) is slot 2 of warp 6. Two groups in flight.
// Order of the memory traffic of a module (measured, GB10): the DRAM serves open-row streams first, so a small DRAM miss
// issued during a stream waits for its end; the warps' loads are issue-limited (outstanding requests per SM). Hence:
// small tables first, then the bulk L2 prefetch of the slice, then the warps' own loads; dependents launched at half.
template <bool INJ>
__device__ __forceinline__ void dn_load(uint4 (&b)[3][3], const uint4* P, const uint2* Wi, int kc, int w, int lane,
                                        int G) {
#pragma unroll
  for (int i = 0; i < 3; i++) {
    const int rt = w + kDnWarps * i;
    if (rt < 20) {
#pragma unroll
      for (int v = 0; v < 3; v++) b[i][v] = __ldcs(P + dn_row(kc, rt, G, v) * 32 + lane);
    } else if (INJ && lane < 16) {
      const int ks = kc * kDnStep + 4 * G;
      const uint2 a0 = __ldcs(Wi + (ks + 0) * 16 + lane), a1 = __ldcs(Wi + (ks + 1) * 16 + lane);
      const uint2 a2 = __ldcs(Wi + (ks + 2) * 16 + lane), a3 = __ldcs(Wi + (ks + 3) * 16 + lane);
      b[i][0] = make_uint4(a0.x, a0.y, a1.x, a1.y), b[i][1] = make_uint4(a2.x, a2.y, a3.x, a3.y);
      b[i][2] = make_uint4(0u, 0u, 0u, 0u);
    } else {
#pragma unroll
      for (int v = 0; v < 3; v++) b[i][v] = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

template <int NT, bool INJ>
__global__ void __launch_bounds__(kDnWarps * 32) k_down(const uint4* __restrict__ P, const uint32_t* __restrict__ sc,
                                                         const uint32_t* __restrict__ eo, const uint32_t* __restrict__ ent,
                                                         const uint2* __restrict__ Wi, const bf16* X, const float* rrms,
                                                         const float* g, float* part, int M, int S) {
  extern __shared__ __align__(16) uint32_t xs[];  // xn chunk [NT·8][256 (+8)] bf16
  constexpr int N = INJ ? kHcDown : kHcRank, RT = INJ ? 21 : 20, PF = 2;  // groups in flight per warp
  const int kc = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
  uint32_t sw[3], sg[3], sh[3];  // small loads first (not queued behind the weight stream), used after it is issued
#pragma unroll
  for (int i = 0; i < 3; i++) sw[i] = w + kDnWarps * i < 20 ? __ldg(sc + (w + kDnWarps * i) * 8 + gq) : 0u;
  const int e0 = __ldg(eo + kc * kDnWarps + w), e1 = __ldg(eo + kc * kDnWarps + w + 1);
  // the CTA's 120 KB slice into L2 in one bulk request (TMA): a sequential stream per CTA, ahead of the warps' loads
  if (threadIdx.x == 0) l2_bulk(P + dn_row(kc, 0, 0, 0) * 32, 20 * kDnG * 3 * 512);
  uint4 bf[PF][3][3];
#pragma unroll
  for (int G = 0; G < PF; G++) dn_load<INJ>(bf[G], P, Wi, kc, w, lane, G);
  Exc ex;
  ex.load(ent, e0, e1, lane);
#pragma unroll
  for (int i = 0; i < 3; i++) scales(sw[i], sg[i], sh[i]);
  pdl_wait();
  {  // xn for columns [256·kc, +256) (one stream) of all rows; rows ≥ M are zero (as hc.cu k_down)
    constexpr int RW = kDnWarps, IT = (NT * 8 + RW - 1) / RW;
    const int s = kc / (H / 256), k0 = kc * 256, v = lane;
    float gw[8];
    load8f(g + k0 + 8 * v, gw);
    uint4 xv[IT];
    float rr[IT];
#pragma unroll
    for (int q = 0; q < IT; q++) {
      const int m = w + RW * q;
      xv[q] = make_uint4(0u, 0u, 0u, 0u), rr[q] = 0.f;
      if (m < M) xv[q] = *reinterpret_cast<const uint4*>(X + size_t(m) * D + k0 + 8 * v), rr[q] = rrms[m * kStreams + s];
    }
#pragma unroll
    for (int q = 0; q < IT; q++) {
      const int m = w + RW * q;
      if (m >= NT * 8) continue;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) {
        float x[8];
        unpack8(xv[q], x);
#pragma unroll
        for (int e = 0; e < 8; e++) x[e] = hc_norm1(x[e], rr[q], gw[e]);
        o = pack8(x);
      }
      *reinterpret_cast<uint4*>(xs + m * kXsW + 4 * v) = o;
    }
  }
  __syncthreads();
  ex.start();
  float c[3][NT][4] = {};
#pragma unroll
  for (int G = 0; G < kDnG; G++) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int j = 4 * G + q;
      uint4 cur[3];
#pragma unroll
      for (int i = 0; i < 3; i++) {
        const uint4 (&b)[3] = bf[G % PF][i];
        if (w + kDnWarps * i < 20) {
          cur[i] = dec_q(b, q, sg[i], sh[i]);
          ex.apply(cur[i], 4 * j + i, lane);
        } else {  // inject tile (bf16, lanes < 16) or nothing
          const uint4& u = b[q >> 1];
          cur[i] = (q & 1) ? make_uint4(u.z, 0u, u.w, 0u) : make_uint4(u.x, 0u, u.y, 0u);
        }
      }
      if (q == 3 && G + PF < kDnG) dn_load<INJ>(bf[G % PF], P, Wi, kc, w, lane, G + PF);
      // K2b / K3 launch when every CTA is half through its slice: K3's weight loads queue behind the rest of the down
      // stream (earlier they would delay it, later the DRAM would idle before them)
      if (q == 3 && G == 1) pdl_launch();
      uint32_t bb[NT][2];
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        bb[nt][0] = xs[(nt * 8 + gq) * kXsW + j * 8 + tq];
        bb[nt][1] = xs[(nt * 8 + gq) * kXsW + j * 8 + tq + 4];
      }
#pragma unroll
      for (int i = 0; i < 3; i++)
        if (w + kDnWarps * i < RT)
#pragma unroll
          for (int nt = 0; nt < NT; nt++) mma(c[i][nt], cur[i], bb[nt][0], bb[nt][1]);
    }
  }
  float* pk = part + size_t(kc) * S;  // K-chunk stride S ≥ N·M, a multiple of 32 (128-byte lines, see k_reduce)
#pragma unroll
  for (int i = 0; i < 3; i++) {
    const int rt = w + kDnWarps * i;
    if (rt >= RT) continue;
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int n = rt * 16 + gq + 8 * h;
        if (n >= N) continue;
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int m = nt * 8 + 2 * tq + e;
          if (m < M) pk[size_t(n) * M + m] = c[i][nt][2 * h + e];
        }
      }
  }
}

// ---- K2b: hc.cu k_down_reduce (same sum order and epilogue) over partials with K-chunk stride S (128-byte aligned) --
// Launched when K2 is half through its slices, CTA c < kUpPieces first sends piece c of the up copy into L2 (bulk, TMA):
// the up stream follows the down stream without waiting for K3's CTAs (they get SM room only as K2 CTAs exit), and
// this grid completes once the pieces are in L2, which is when K3 (waiting for it) needs them. The grid has at least
// kUpPieces CTAs (the extra ones only prefetch). Each warp then discards the 40 L2 lines of partials it read: they are
// dead, and dirty lines would be written to DRAM when evicted (M = 64: 3.3 MB per module). One warp reads a line.
__global__ void __launch_bounds__(256) k_reduce(const float* part, bf16* lora, bf16* inj, int M, int N, int S,
                                                const uint8_t* up) {
  if (threadIdx.x == 0 && blockIdx.x < kUpPieces) l2_bulk(up + size_t(blockIdx.x) * (kUpB / kUpPieces), kUpB / kUpPieces);
  pdl_launch();
  pdl_wait();
  const int idx = blockIdx.x * 256 + threadIdx.x, lane = threadIdx.x & 31;
  if (idx - lane >= N * M) return;
  float z = 0.f;
  if (idx < N * M) {
#pragma unroll
    for (int kc = 0; kc < kDnChunks; kc++) z += part[size_t(kc) * S + idx];
  }
  asm volatile("" ::"f"(z));  // every lane has its 40 values before the warp discards the lines
  __syncwarp();
  for (int kc = lane; kc < kDnChunks; kc += 32)
    asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(part + size_t(kc) * S + idx - lane) : "memory");
  if (idx >= N * M) return;
  const int n = idx / M, m = idx - n * M;
  const float zb = bfr(z);
  if (n < kHcRank) {
    const float x = zb * 0.25f;
    lora[m * kHcRank + n] = __float2bfloat16_rn(x * sigm(x));
  } else if (inj) {
    inj[m * kStreams + n - kHcRank] = __float2bfloat16_rn(zb);
  }
}

// ---- K3: up projection + gate mix from the packed copy (hc.cu k_up after the weight prologue) ------------------------
// The tile's 20 fragments (in L2 or on their way: K2b's prefetch) are decoded and patched before the dependency wait.
template <int NT, int WU = kUpWarps>
__global__ void __launch_bounds__(WU * 32) k_up(const uint4* __restrict__ P, const uint32_t* __restrict__ sc,
                                               const uint32_t* __restrict__ eo, const uint32_t* __restrict__ ent,
                                               const bf16* lora, const bf16* X, const float* rrms, const float* g,
                                               bf16* u, int M) {
  extern __shared__ __align__(16) uint32_t ls[];  // lora rows of one pass [NG·8][320 (+8)] bf16
  constexpr int NG = NT < 4 ? NT : 4;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
  const int j = blockIdx.x * WU + w, e = gq >> 2, h = 4 * j + (gq & 3);
  uint4 a[kUpKs];
  {
    const uint32_t sw = __ldg(sc + j * 8 + gq);
    const int e0 = __ldg(eo + j), e1 = __ldg(eo + j + 1);
    uint4 pw[kUpG][3];
#pragma unroll
    for (int G = 0; G < kUpG; G++)
#pragma unroll
      for (int v = 0; v < 3; v++) pw[G][v] = __ldcs(P + up_row(j, G, v) * 32 + lane);
    uint32_t sg, sh;
    scales(sw, sg, sh);
    Exc ex;
    ex.load(ent, e0, e1, lane);
    ex.start();
#pragma unroll
    for (int ks = 0; ks < kUpKs; ks++) {
      a[ks] = dec_q(pw[ks >> 2], ks & 3, sg, sh);
      ex.apply(a[ks], ks, lane);
    }
  }
  // xn of the first pass before the wait: X' and rrms are K1's, which completed before K2 passed its wait, and K3 is
  // launched only after K2 triggers (after that wait) → read from L2 (ld.cg); lora (K2b) only after the wait
  float gw[kStreams];
#pragma unroll
  for (int s = 0; s < kStreams; s++) gw[s] = g[s * H + h];
  float xn0[NG][kStreams];
#pragma unroll
  for (int nt = 0; nt < NG; nt++) {
    const int m = nt * 8 + 2 * tq + e;
#pragma unroll
    for (int s = 0; s < kStreams; s++)
      xn0[nt][s] = m < M ? hc_norm1(b2f(__ldcg(X + size_t(m) * D + s * H + h)), __ldcg(rrms + m * kStreams + s), gw[s])
                         : 0.f;
  }
  pdl_wait();
#pragma unroll
  for (int p0 = 0; p0 < NT; p0 += NG) {
    float xn[NG][kStreams];
#pragma unroll
    for (int nt = 0; nt < NG; nt++) {
      const int m = (p0 + nt) * 8 + 2 * tq + e;
#pragma unroll
      for (int s = 0; s < kStreams; s++)
        xn[nt][s] = p0 == 0 ? xn0[nt][s]
                            : m < M ? hc_norm1(b2f(X[size_t(m) * D + s * H + h]), rrms[m * kStreams + s], gw[s]) : 0.f;
    }
    if (p0) __syncthreads();
    for (int i = threadIdx.x; i < NG * 8 * 40; i += WU * 32) {
      const int r = i / 40, v = i - r * 40, m = p0 * 8 + r;
      uint4 o = make_uint4(0u, 0u, 0u, 0u);
      if (m < M) o = *reinterpret_cast<const uint4*>(lora + m * kHcRank + 8 * v);
      *reinterpret_cast<uint4*>(ls + r * kLsW + 4 * v) = o;
    }
    __syncthreads();
    float c[NG][4] = {};
#pragma unroll
    for (int ks = 0; ks < kUpKs; ks++)
#pragma unroll
      for (int nt = 0; nt < NG; nt++)
        if (p0 + nt < NT)
          mma(c[nt], a[ks], ls[(nt * 8 + gq) * kLsW + ks * 8 + tq], ls[(nt * 8 + gq) * kLsW + ks * 8 + tq + 4]);
#pragma unroll
    for (int nt = 0; nt < NG; nt++) {
      const float pa = __shfl_xor_sync(~0u, e ? c[nt][0] : c[nt][1], 16);
      const float pb = __shfl_xor_sync(~0u, e ? c[nt][2] : c[nt][3], 16);
      const float G[4] = {e ? pa : c[nt][0], e ? c[nt][1] : pa, e ? pb : c[nt][2], e ? c[nt][3] : pb};
      const int m = (p0 + nt) * 8 + 2 * tq + e;
      if (m >= M) continue;
      float acc = 0.f;
#pragma unroll
      for (int s = 0; s < kStreams; s++) acc = __fmaf_rn(sigm(bfr(G[s])), xn[nt][s], acc);
      u[size_t(m) * H + h] = __float2bfloat16_rn(acc * 0.25f);
    }
  }
}

// ---- upload check: decode every fragment as the kernels do and compare with the bf16 copy ----------------------------
// grid kDnChunks + kUpTiles / 4 CTAs × 128 threads: CTA < 40 = down chunk kc (warps 0..3 walk the 7 (kc, w) units),
// else 4 up tiles. bad[0] += differing fragments, bad[1] += lists not consumed to their end.
__global__ void __launch_bounds__(128) k_verify(const uint8_t* pk, const uint4* Wd, const uint4* Wu, unsigned* bad) {
  const uint4* P = reinterpret_cast<const uint4*>(pk);
  const uint32_t *scd = reinterpret_cast<const uint32_t*>(pk + kOScD), *scu = reinterpret_cast<const uint32_t*>(pk + kOScU);
  const uint32_t *eod = reinterpret_cast<const uint32_t*>(pk + kOEoD), *eou = reinterpret_cast<const uint32_t*>(pk + kOEoU);
  const uint32_t* ent = reinterpret_cast<const uint32_t*>(pk + kOEnt);
  const int w4 = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2;
  unsigned nb = 0, nl = 0;
  if (blockIdx.x < kDnChunks) {
    const int kc = blockIdx.x;
    for (int w = w4; w < kDnWarps; w += 4) {
      Exc ex;
      ex.load(ent, eod[kc * kDnWarps + w], eod[kc * kDnWarps + w + 1], lane);
      ex.start();
      uint32_t sg[3], sh[3];
      for (int i = 0; i < 3; i++) scales(w + kDnWarps * i < 20 ? scd[(w + kDnWarps * i) * 8 + gq] : 0u, sg[i], sh[i]);
      for (int G = 0; G < kDnG; G++) {
        uint4 b[3][3];
        dn_load<false>(b, P, nullptr, kc, w, lane, G);
        for (int q = 0; q < 4; q++)
          for (int i = 0; i < 3; i++) {
            const int rt = w + kDnWarps * i;
            if (rt >= 20) continue;
            uint4 v = dec_q(b[i], q, sg[i], sh[i]);
            ex.apply(v, 4 * (4 * G + q) + i, lane);
            const uint4 r = Wd[(size_t(rt) * kDnKs + kc * kDnStep + 4 * G + q) * 32 + lane];
            nb += v.x != r.x || v.y != r.y || v.z != r.z || v.w != r.w;
          }
      }
      nl += lane == 0 && ex.p != ex.end;
    }
  } else {
    const int j = (blockIdx.x - kDnChunks) * 4 + w4;
    Exc ex;
    ex.load(ent, eou[j], eou[j + 1], lane);
    ex.start();
    uint32_t sg, sh;
    scales(scu[j * 8 + gq], sg, sh);
    for (int G = 0; G < kUpG; G++) {
      uint4 b[3];
      for (int v = 0; v < 3; v++) b[v] = P[kDnB / 16 + up_row(j, G, v) * 32 + lane];
      for (int q = 0; q < 4; q++) {
        const int ks = 4 * G + q;
        uint4 v = dec_q(b, q, sg, sh);
        ex.apply(v, ks, lane);
        const uint4 r = Wu[(size_t(j) * kUpKs + ks) * 32 + lane];
        nb += v.x != r.x || v.y != r.y || v.z != r.z || v.w != r.w;
      }
    }
    nl += lane == 0 && ex.p != ex.end;
  }
  if (nb) atomicAdd(bad, nb);
  if (nl) atomicAdd(bad + 1, nl);
}

// ---- host: packing ---------------------------------------------------------------------------------------------------
struct Stats {
  int modules = 0, bf16_modules = 0;
  long long values = 0, exc = 0;
  int max_dn_list = 0, max_up_list = 0;
} g_st;

inline int row_base(const uint16_t* r, int K, int stride) {
  int mx = 0;
  for (int k = 0; k < K; k++) {
    const uint16_t v = r[size_t(k) * stride];
    if (v & 0x7FFF) mx = std::max(mx, (v >> 7) & 0xFF);
  }
  return std::max(mx - 14, 1);
}

// tiles of bf16 fragments t ([tile][ks][lane][8], pack_tiles), row bases base[tile·16 + r] → stream words + exceptions.
// at(tile, ks, v) → uint4 row of the k-step's group; unit(tile, ks) → list; blk(tile, ks) → block id.
template <class At, class Unit, class Blk>
void pack_part(const std::vector<uint16_t>& t, int tiles, int KS, const std::vector<int>& base, uint32_t* out, At at,
               Unit unit, Blk blk, std::vector<std::vector<uint32_t>>& lists, long long& nexc) {
  for (int tl = 0; tl < tiles; tl++)
    for (int ks = 0; ks < KS; ks++) {
      const int q = ks & 3;
      auto& L = lists[unit(tl, ks)];
      for (int lane = 0; lane < 32; lane++) {
        const uint16_t* o = t.data() + ((size_t(tl) * KS + ks) * 32 + lane) * 8;
        const int g = lane >> 2;
        uint32_t Mw[4];
        for (int wd = 0; wd < 4; wd++) {
          const int b = base[tl * 16 + g + 8 * (wd & 1)];
          uint32_t mm = 0;
          for (int hh = 0; hh < 2; hh++) {
            const uint16_t v = o[2 * wd + hh];
            const int E = (v >> 7) & 0xFF;
            uint32_t mini;
            if (E == 0 && (v & 0x7F) == 0) {
              mini = v & 0x8000u;
            } else if (E >= b && E <= b + 14) {
              mini = (v & 0x807Fu) | uint32_t(E - b + 1) << 7;
            } else {
              mini = v & 0x8000u;
              L.push_back(uint32_t(v) << 16 | uint32_t(blk(tl, ks)) << 8 | uint32_t(lane) << 3 | uint32_t(2 * wd + hh));
              nexc++;
            }
            mm |= mini << (16 * hh);
          }
          Mw[wd] = mm;
        }
        const uint32_t m3 = Mw[3];
        const uint32_t W[3] = {Mw[0] | (m3 & 0x000F000Fu) << 11, Mw[1] | (m3 & 0x00F000F0u) << 7,
                               Mw[2] | (m3 & 0x07000700u) << 3 | (m3 & 0x80008000u) >> 1};
        for (int v = 0; v < 3; v++) out[(at(tl, ks, v) * 32 + lane) * 4 + q] = W[v];
      }
    }
}

unsigned* g_bad = nullptr;  // device counters of k_verify

// 12-bit copy of a module (dn: [320, 10240] rows, up: [10240, 320] rows, row-major bf16; td / tu: their bf16 fragment
// tiles as uploaded; Wd / Wu: the device bf16 copies) → device block, checked bit-exact on the device; nullptr if a row
// does not fit (b > 128: the module stays on the bf16 kernels).
void* pack(const uint16_t* dn, const uint16_t* up, const std::vector<uint16_t>& td, const std::vector<uint16_t>& tu,
           const void* Wd, const void* Wu) {
  std::vector<int> bd(20 * 16), bu(size_t(kUpTiles) * 16);
  bool ok = true;
  for (int r = 0; r < kHcRank; r++) bd[r] = row_base(dn + size_t(r) * D, D, 1);
  for (int tl = 0; tl < kUpTiles; tl++)
    for (int r = 0; r < 16; r++) bu[tl * 16 + r] = row_base(up + size_t((r >> 2) * H + 4 * tl + (r & 3)) * kHcRank, kHcRank, 1);
  for (int b : bd) ok &= b <= 128;
  for (int b : bu) ok &= b <= 128;
  g_st.modules++;
  if (!ok) {
    g_st.bf16_modules++;
    qlog::warn("hc", "P012: row with exponent out of range → bf16 kernel for this module");
    return nullptr;
  }
  std::vector<uint32_t> head(kOEnt / 4, 0u);
  std::vector<std::vector<uint32_t>> ld(kDnUnits), lu(kUpUnits);
  long long nexc = 0;
  pack_part(td, 20, kDnKs, bd, head.data(),
            [](int tl, int ks, int v) { return dn_row(ks / kDnStep, tl, (ks % kDnStep) / 4, v); },
            [](int tl, int ks) { return (ks / kDnStep) * kDnWarps + tl % kDnWarps; },
            [](int tl, int ks) { return 4 * (ks % kDnStep) + tl / kDnWarps; }, ld, nexc);
  pack_part(tu, kUpTiles, kUpKs, bu, head.data() + kDnB / 4,
            [](int tl, int ks, int v) { return up_row(tl, ks / 4, v); }, [](int tl, int) { return tl; },
            [](int, int ks) { return ks; }, lu, nexc);
  auto sct = [&](const std::vector<int>& b, int tiles, uint32_t* o) {
    for (int tl = 0; tl < tiles; tl++)
      for (int g = 0; g < 8; g++)
        o[tl * 8 + g] = uint32_t(b[tl * 16 + g] + 126) << 7 | uint32_t(b[tl * 16 + g + 8] + 126) << 23;
  };
  sct(bd, 20, head.data() + kOScD / 4);
  sct(bu, kUpTiles, head.data() + kOScU / 4);
  std::vector<uint32_t> ent;
  auto lists = [&](std::vector<std::vector<uint32_t>>& L, uint32_t* off, int& mx) {
    for (size_t i = 0; i < L.size(); i++) {
      std::sort(L[i].begin(), L[i].end(), [](uint32_t a, uint32_t b) { return (a & 0xFFFF) < (b & 0xFFFF); });
      off[i] = uint32_t(ent.size());
      ent.insert(ent.end(), L[i].begin(), L[i].end());
      mx = std::max(mx, int(L[i].size()));
    }
    off[L.size()] = uint32_t(ent.size());
  };
  int mxd = 0, mxu = 0;
  lists(ld, head.data() + kOEoD / 4, mxd);
  lists(lu, head.data() + kOEoU / 4, mxu);
  uint8_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, kOEnt + ent.size() * 4 + 16));
  CUDA_CHECK(cudaMemcpy(d, head.data(), kOEnt, cudaMemcpyHostToDevice));
  if (!ent.empty()) CUDA_CHECK(cudaMemcpy(d + kOEnt, ent.data(), ent.size() * 4, cudaMemcpyHostToDevice));
  if (!g_bad) CUDA_CHECK(cudaMalloc(&g_bad, 8));
  CUDA_CHECK(cudaMemset(g_bad, 0, 8));
  k_verify<<<kDnChunks + kUpTiles / 4, 128>>>(d, static_cast<const uint4*>(Wd), static_cast<const uint4*>(Wu), g_bad);
  unsigned bad[2];
  CUDA_CHECK(cudaMemcpy(bad, g_bad, 8, cudaMemcpyDeviceToHost));
  if (bad[0] || bad[1]) qlog::fail("hc", "P012: 12-bit decode ≠ bf16 (fragments %u · lists %u)", bad[0], bad[1]);
  g_st.values += 2LL * kHcRank * D, g_st.exc += nexc;
  g_st.max_dn_list = std::max(g_st.max_dn_list, mxd), g_st.max_up_list = std::max(g_st.max_up_list, mxu);
  return d;
}

// ---- host: decode pass with the packed kernels (hc_combine_mix, M < kHcBigRows) --------------------------------------
void down(const HcW& w, const uint2* Wi, const bf16* x, float* part, int mc, int nt, int S, cudaStream_t s) {
  const uint8_t* b = static_cast<const uint8_t*>(w.pk);
  const uint4* P = reinterpret_cast<const uint4*>(b);
  const uint32_t *sc = reinterpret_cast<const uint32_t*>(b + kOScD), *eo = reinterpret_cast<const uint32_t*>(b + kOEoD),
                 *ent = reinterpret_cast<const uint32_t*>(b + kOEnt);
#define Q38_P12D(NT)                                                                                              \
  if (w.inject)                                                                                                   \
    launch(k_down<NT, true>, dim3(kDnChunks), dim3(kDnWarps * 32), size_t(NT) * 8 * kXsW * 4, s, true, P, sc, eo, \
           ent, Wi, x, (const float*)g_ws.rrms, w.norm, part, mc, S);                                                \
  else                                                                                                            \
    launch(k_down<NT, false>, dim3(kDnChunks), dim3(kDnWarps * 32), size_t(NT) * 8 * kXsW * 4, s, true, P, sc, eo, \
           ent, Wi, x, (const float*)g_ws.rrms, w.norm, part, mc, S);
  Q38_NT_SWITCH(nt, Q38_P12D)
#undef Q38_P12D
}

void up(const HcW& w, const bf16* lora, const bf16* x, bf16* u, int mc, int nt, cudaStream_t s) {
  const uint8_t* b = static_cast<const uint8_t*>(w.pk);
  const uint4* P = reinterpret_cast<const uint4*>(b + kDnB);
  const uint32_t *sc = reinterpret_cast<const uint32_t*>(b + kOScU), *eo = reinterpret_cast<const uint32_t*>(b + kOEoU),
                 *ent = reinterpret_cast<const uint32_t*>(b + kOEnt);
#define Q38_P12U(NT)                                                                                              \
  launch(k_up<NT>, dim3(kUpTiles / kUpWarps), dim3(kUpWarps * 32), size_t(NT < 4 ? NT : 4) * 8 * kLsW * 4, s, true, \
         P, sc, eo, ent, lora, x, (const float*)g_ws.rrms, w.norm, u, mc);
  Q38_NT_SWITCH(nt, Q38_P12U)
#undef Q38_P12U
}

}  // namespace p12
