// p010_moe_down_tune.cuh — P010 moe_down_tune (patches.def): the routed down (+ shared gate/up) and the shared down +
// combine launches of the large-M MoE path with better DRAM behaviour. Included by moe_large.cuh inside its anonymous
// namespace (LArgs, lup_item, swz, cp16, ldsm4, DS_SMEM, SD_SMEM, …); moe_large runs p010::down_sup / p010::sdown when
// patch::on(moe_down_tune), otherwise k_down_sup / k_sdown unchanged.
//
// Why (GB10, layer 0, M = 8192, base): k_down_sup 3.78 ms, k_sdown 2.18 ms, both DRAM-bound. k_down_sup writes y (419 MB)
// and reads the down weights (236 MB), h_r (52 MB) and the shared gate/up rows (42 MB); k_sdown reads y back (419 MB)
// and writes out (42 MB). Measured on this GPU: writes stream at ~194 GB/s, reads at ~236 GB/s (256-byte pieces at random
// rows: ~230 GB/s) — so y alone costs ≥ 2.16 ms to write and ≥ 1.8 ms to read back. Without its shared gate/up items
// the routed down already runs at 3.32 ms (≈ its DRAM floor), so the gains come from L2 behaviour and scheduling, not
// from the MMA loop (removing the MMAs does not change the time).
//
// What differs from the base (scheduling and memory movement only):
//  k_down_sup —
//   · y leaves the MMA fragments through a per-warp stmatrix.trans stage (2 × 512 B after the h tile): each lane stores
//     16 bytes (8 outputs of one pair), a warp store covers 8 pairs × 64 B, instead of 64 two-byte stores per block;
//   · y is stored with .cs (evict-first), so its 419 MB stream does not push the down weights, h_r tiles and shared
//     gate/up rows out of L2 — except the last 24 MB of y (the highest pairs, written last), stored normally so k_sdown
//     finds them in L2 (all of y .cs costs k_sdown 15–50 µs at M = 512–8192);
//   · the shared gate/up items are spread over the routed down items that exist (meta[0] token tiles, read on the
//     device) instead of over the launch's upper bound max_tiles(M), so they no longer run alone at the end of the grid.
//  k_sdown —
//   · out is stored with .cs (it is read next by the all-reduce, not from L2): the y lines kept by k_down_sup and the
//     shared down weights stay in L2;
//   · the next output block's shared down weights are loaded into registers right after the current block's MMAs (they
//     arrive during the combine), and each thread keeps two combine items (10 y rows each) in flight.
// Tried and dropped (layer 0, M = 8192): L2 bulk prefetch of an item's weights at its start (+200 µs), routed tiles in
// reverse order (+50 µs), 16-row CTAs with se of all 640 outputs first (+50 µs), y of the first combine item in flight
// during the shared down MMAs, 3 CTAs/SM or 8 warps (all ±10 µs).
//
// Bitwise: the same MMAs on the same operands in the same order; y = bf16(acc·(w·g2)) with one cvt.rn.bf16x2 per two
// values (= two cvt.rn.bf16); se and the combine are the base code. Every output equals the base path for every row and
// every M (tests: moe.large_p010 on == off, moe.large_bitwise with the patch on).
#pragma once

namespace p010 {

constexpr int kKeepPairs = (24 << 20) / (H * 2);  // y pairs stored with normal priority (the last 24 MB written)

__device__ __forceinline__ void stsm4t(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(a), "r"(b), "r"(c),
               "r"(d)
               : "memory");
}
__device__ __forceinline__ uint32_t pk(float a, float b) {  // bf16x2 {lo = bf16(a), hi = bf16(b)}
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ void st_cs(void* p, const uint4& v) {  // evict-first store
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

// ---- routed down, item = (token tile T, 640 outputs): base ldown_item with the y stage --------------------------------
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
  // y stage: per warp 2 × 512 B after the h tile. For n8 tile j the warp's 2 m16 tiles × 8 pairs go in as 4 matrices
  // (g = 2m + h: outputs 16·(8nb + 2warp) + 8g + 0..7) transposed: row k = pair 8j + k, 4 chunks of 16 B, chunk g at
  // g ^ ((k >> 1) & 3) (conflict-free stmatrix rows and 16-byte reads). Lane L then stores pair 8j + L/4, chunk L%4.
  uint8_t* stg = sm + LBT * I * 2 + warp * 1024;
  const int wk = lane & 7, wg = lane >> 3, rk = lane >> 2, rg = lane & 3;
  const uint32_t woff = smem_u32(stg) + wk * 64 + ((wg ^ ((wk >> 1) & 3)) << 4);
  const uint32_t roff = rk * 64 + ((rg ^ ((rk >> 1) & 3)) << 4);
  const bool cs = p0 + LBT <= kTopK * a.M - kKeepPairs;  // whole tile before the kept tail: evict-first
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
    bf16* yb = a.w.y + size_t(p0 + rk) * H + 16 * (8 * nb + 2 * warp) + 8 * rg;
    __syncwarp();  // previous block's stage reads done
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (j < nj) {
        const uint32_t buf = (j & 1) * 512;
        stsm4t(woff + buf, pk(acc[0][j][0] * wsc[j][0], acc[0][j][1] * wsc[j][1]),
               pk(acc[0][j][2] * wsc[j][0], acc[0][j][3] * wsc[j][1]), pk(acc[1][j][0] * wsc[j][0], acc[1][j][1] * wsc[j][1]),
               pk(acc[1][j][2] * wsc[j][0], acc[1][j][3] * wsc[j][1]));
        __syncwarp();
        const uint4 v = *reinterpret_cast<const uint4*>(stg + buf + roff);
        if (8 * j + rk < cnt) {
          if (cs) st_cs(yb + size_t(8 * j) * H, v);
          else *reinterpret_cast<uint4*>(yb + size_t(8 * j) * H) = v;
        }
      }
  }
}
static_assert(LBT * I * 2 + 4 * 1024 <= DS_SMEM, "y stage after the h tile");

// routed down + shared gate/up items: item b is a shared gate/up item iff ⌊(b+1)·ns/n⌋ > ⌊b·ns/n⌋ (index ⌊b·ns/n⌋),
// else routed down item b − ⌊b·ns/n⌋, with n = 4·meta[0] + ns (the launch covers 4·max_tiles(M) + ns; the rest exits)
__global__ void __launch_bounds__(128, 2) k_down_sup(LArgs a, int ns) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ int stok[LBT];
  const long long n = (long long)__ldg(a.w.meta) * DN_NS + ns, b = blockIdx.x;
  if (b >= n) return;
  const int c0 = int(b * ns / n), c1 = int((b + 1) * ns / n);
  if (c1 > c0) lup_item<false>(a, c0 / 5, c0 % 5, sm, stok);
  else p010::ldown_item(a, int(b - c0) / DN_NS, int(b - c0) % DN_NS, sm);  // qualified: ADL also sees the base
}

// ---- shared down + combine, item = (32 rows, 640 outputs): base k_sdown, pipelined -------------------------------------
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
  uint4 w[2][10];
  uint32_t sw[2][5];
  auto ldw = [&](int nb) {
#pragma unroll
    for (int m = 0; m < 2; m++) {
      const int n = 8 * nb + 2 * warp + m;
#pragma unroll
      for (int c = 0; c < 10; c++) w[m][c] = ld_stream(reinterpret_cast<const uint4*>(a.sw2) + (size_t(n) * 10 + c) * 32 + lane);
#pragma unroll
      for (int j = 0; j < 5; j++) sw[m][j] = ld_stream1(a.ss2 + size_t(n) * 8 * 20 + r * 20 + 4 * j);
    }
  };
  ldw(part * NB);
  cp_wait<0>();
  __syncthreads();
  for (int nb = part * NB; nb < part * NB + NB; nb++) {
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
    if (nb + 1 < part * NB + NB) ldw(nb + 1);  // in flight during the combine below
    __syncthreads();
    // out = bf16(bf16(Σ_k y[pair_k]) + se), pairs in expert-id order; item = (row, 8 outputs), two items in flight
    const int nit = cnt * 16;
    auto ldy = [&](int it, uint4 (&yv)[kTopK]) {
      const int tok = t0 + (it >> 4), col = 128 * nb + 8 * (it & 15);
#pragma unroll
      for (int k = 0; k < kTopK; k++)
        yv[k] = __ldcs(reinterpret_cast<const uint4*>(a.w.y + size_t(__ldg(a.w.tpair + tok * kTopK + k)) * H + col));
    };
    auto fin = [&](int it, const uint4 (&yv)[kTopK]) {
      const int t = it >> 4, ch = it & 15, tok = t0 + t, col = 128 * nb + 8 * ch;
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
      st_cs(a.out + size_t(tok) * H + col, make_uint4(o[0], o[1], o[2], o[3]));
      if (a.dbg_routed) *reinterpret_cast<uint4*>(a.dbg_routed + size_t(tok) * H + col) = make_uint4(ro[0], ro[1], ro[2], ro[3]);
      if (a.dbg_shared) *reinterpret_cast<uint4*>(a.dbg_shared + size_t(tok) * H + col) = sev;
    };
    uint4 ya[kTopK], yb[kTopK];
    if (tid < nit) ldy(tid, ya);
    for (int it = tid; it < nit; it += 256) {
      if (it + 128 < nit) ldy(it + 128, yb);
      fin(it, ya);
      if (it + 256 < nit) ldy(it + 256, ya);
      if (it + 128 < nit) fin(it + 128, yb);
    }
    __syncthreads();
  }
}

// ---- host ---------------------------------------------------------------------------------------------------------------
void setup() {
  CUDA_CHECK(cudaFuncSetAttribute(p010::k_down_sup, cudaFuncAttributeMaxDynamicSharedMemorySize, DS_SMEM));
  CUDA_CHECK(cudaFuncSetAttribute(p010::k_sdown, cudaFuncAttributeMaxDynamicSharedMemorySize, SD_SMEM));
}

void down_sup(const LArgs& a, int nd, int ns, cudaStream_t s) { p010::k_down_sup<<<nd + ns, 128, DS_SMEM, s>>>(a, ns); }

void sdown(const LArgs& a, int M, cudaStream_t s) { p010::k_sdown<<<cdiv(M, SD_BT) * SD_NS, 128, SD_SMEM, s>>>(a); }

}  // namespace p010
