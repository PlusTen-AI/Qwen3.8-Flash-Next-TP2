// p004_gemm_tune.cuh — P004 gemm_tune: faster large-M path of lin_forward (M > 64, split-1 shapes), selected by
// patch::on(gemm_tune); the base path (linear_prefill.cuh / decode passes) stays and runs when the patch is off or for
// K-split shapes. Included by linear.cu inside its anonymous namespace (Fmt, mma, dq4, cp_commit/cp_wait, big::).
//
// Bitwise contract (same as linear_prefill.cuh): every MMA is the decode kernel's instruction with identical operands —
// swap-AB (weight rows on the MMA M side, activation rows on N), the same k → slot permutation (lane t of a quad holds k
// 8t..8t+7 of a k32 block: k16 step 0 = k 8t..8t+3, step 1 = 8t+4..8t+7), the same chain (k32 blocks ascending, step 0
// then step 1, into one fp32 accumulator from 0), the same exact MXFP8 dequant and the same bf16/fp32 output rounding.
// No split, no atomics. Only scheduling differs from the base 128×128 / 8-warp kernel:
//   · 4 warps per CTA, each 32 weight rows × all BM activation rows (2 × 16 MMA tiles at BM = 128): every weight tile is
//     loaded and dequantized by one warp only (the base dequantizes each tile in 2 warps), half the A work per MMA;
//   · 2–3 CTAs per SM (≤ 49 KB shared memory each, ≤ 255 registers): one CTA's barrier waits, pipeline fill and
//     epilogue run under the other CTAs' MMAs;
//   · one k32 block per stage, all stages in flight; register software pipeline: the next block's raw weights are read
//     at the start of the current block, its step-0 A half is dequantized after the current step 0 and its step-1 half
//     after step 1, and B fragment j is reloaded right after its last use — the barrier sits behind queued MMAs;
//   · tile 128 × 128 or 128 × 64 activation rows, chosen per (N, M) by a wave model (pick() below).
#pragma once

namespace p004 {

using big::Args;
using big::cpz;

template <int FMT_, int BN_, int BM_, int WN_, int WM_, int ST_, int MINB_>
struct Cfg {
  static constexpr int FMT = FMT_, BN = BN_, BM = BM_, WN = WN_, WM = WM_, ST = ST_, MINB = MINB_;
  static constexpr int TB = Fmt<FMT>::TB, SB = Fmt<FMT>::SB;
  static constexpr int MI = BN / WN / 16, NJ = BM / WM / 8;  // warp tile: MI m16 weight tiles × NJ n8 activation tiles
  static constexpr int NTH = WN * WM * 32, TILES = BN / 16;
  static constexpr int WST = TILES * TB, SST = TILES * SB, XST = BM * 64;  // one k32 block per stage
  static constexpr int STAGE = WST + SST + XST;
  static constexpr int PIPE = ST * STAGE;
  template <bool F32> static constexpr int pitch() { return F32 ? BN + 4 : BN + 8; }  // epilogue row (elements)
  // epilogue passes: the tile's activation rows are staged in NP slices when they do not fit the pipeline buffers
  template <bool F32> static constexpr int passes() {
    int np = 1;
    while (BM / np > 8 && (BM / np) * pitch<F32>() * (F32 ? 4 : 2) > PIPE) np *= 2;
    return np;
  }
  template <bool F32> static constexpr int smem() {
    return std::max(PIPE, (BM / passes<F32>()) * pitch<F32>() * (F32 ? 4 : 2));
  }
};

template <class C, bool F32>
__global__ void __launch_bounds__(C::NTH, C::MINB) kernel(Args a) {
  constexpr int FMT = C::FMT, TB = C::TB, MI = C::MI, NJ = C::NJ, ST = C::ST;
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp % C::WN, wm = warp / C::WN;
  // grouped raster: GM activation tiles share the sweep over all weight tiles (their activations stay in L2)
  constexpr int GM = 8;
  const int per = GM * a.NT, grp = blockIdx.x / per, fm = grp * GM, gm = min(a.MT - fm, GM);
  const int rem = blockIdx.x - grp * per, mt = fm + rem % gm, nt = rem / gm;
  const int T0 = nt * C::TILES, m0 = mt * C::BM, KB = a.KB;

  auto load = [&](int kb, int buf) {  // k32 block kb → stage buffer buf: weights [tile][TB], scales [tile][16], x [row][64 B]
    uint8_t* st = sm + buf * C::STAGE;
    constexpr int PT = TB / 16;
#pragma unroll
    for (int p = tid; p < C::WST / 16; p += C::NTH) {
      const int T = T0 + p / PT;
      const bool ok = T < a.tiles;
      cpz(st + p * 16, a.w + (size_t(ok ? T : 0) * KB + kb) * TB + (p % PT) * 16, ok);
    }
    if constexpr (C::SST > 0) {
      if (tid < C::TILES) {
        const int T = T0 + tid;
        const bool ok = T < a.tiles;
        cpz(st + C::WST + tid * 16, a.s + (size_t(ok ? T : 0) * KB + kb) * 16, ok);
      }
    }
#pragma unroll
    for (int p = tid; p < C::BM * 4; p += C::NTH) {
      const int r = p >> 2, ch = p & 3, m = m0 + r;
      const bf16* src = a.x + size_t(min(m, a.M - 1)) * a.K + kb * 32 + ch * 8;
      cpz(st + C::WST + C::SST + p * 16, src, m < a.M);
    }
  };

  // fragment offsets inside a stage (all conflict-free: a warp reads 512 contiguous bytes per 16-B load)
  const int aoff = (wn * MI) * TB + lane * 16;                          // + i·TB: weight tile wn·MI + i (row g ‖ g+8)
  const int soff = C::WST + (wn * MI) * 16 + 2 * g;                     // + i·16: scale pair (row g, row g+8)
  const int boff = C::WST + C::SST + ((wm * NJ * 8 + g) * 4 + t) * 16;  // + j·512: activation row 8j+g, k 8t..8t+7

  float acc[MI][NJ][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NJ; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  uint32_t A0[MI][4], A1[MI][4];  // A fragments of k16 steps 0 and 1 (same slots as lin_kernel / big::kernel)
  uint4 B[NJ];                    // B fragments: .x .y step 0, .z .w step 1
  uint4 RW[MI];                   // MXFP8: raw weights of the next block
  uint32_t RS[MI];                // MXFP8: scale pair of the next block

  auto lds_raw = [&](const uint8_t* st) {
#pragma unroll
    for (int i = 0; i < MI; i++) {
      RW[i] = *reinterpret_cast<const uint4*>(st + aoff + i * TB);
      RS[i] = *reinterpret_cast<const uint16_t*>(st + soff + i * 16);
    }
  };
  auto deq0 = [&](const uint8_t* st) {  // step-0 half (bf16 weights: straight from shared memory)
#pragma unroll
    for (int i = 0; i < MI; i++) {
      if constexpr (FMT == WMat::MXFP8) {
        const float s0 = ue8m0_to_float(RS[i] & 0xff), s1 = ue8m0_to_float(RS[i] >> 8);
        dq4(RW[i].x, s0, A0[i][0], A0[i][2]);
        dq4(RW[i].z, s1, A0[i][1], A0[i][3]);
      } else {
        const uint2 u = *reinterpret_cast<const uint2*>(st + aoff + i * TB);
        const uint2 v = *reinterpret_cast<const uint2*>(st + aoff + i * TB + 512);
        A0[i][0] = u.x, A0[i][1] = v.x, A0[i][2] = u.y, A0[i][3] = v.y;
      }
    }
  };
  auto deq1 = [&](const uint8_t* st) {  // step-1 half
#pragma unroll
    for (int i = 0; i < MI; i++) {
      if constexpr (FMT == WMat::MXFP8) {
        const float s0 = ue8m0_to_float(RS[i] & 0xff), s1 = ue8m0_to_float(RS[i] >> 8);
        dq4(RW[i].y, s0, A1[i][0], A1[i][2]);
        dq4(RW[i].w, s1, A1[i][1], A1[i][3]);
      } else {
        const uint2 u = *reinterpret_cast<const uint2*>(st + aoff + i * TB + 8);
        const uint2 v = *reinterpret_cast<const uint2*>(st + aoff + i * TB + 512 + 8);
        A1[i][0] = u.x, A1[i][1] = v.x, A1[i][2] = u.y, A1[i][3] = v.y;
      }
    }
  };

  // prologue: all ST stages in flight, then the fragments of block 0
#pragma unroll
  for (int s = 0; s < ST; s++) {
    if (s < KB) load(s, s);
    cp_commit();
  }
  cp_wait<ST - 1>();
  __syncthreads();
  if constexpr (FMT == WMat::MXFP8) lds_raw(sm);
  deq0(sm);
  deq1(sm);
#pragma unroll
  for (int j = 0; j < NJ; j++) B[j] = *reinterpret_cast<const uint4*>(sm + boff + j * 512);

  int buf = 0;  // stage buffer of block kb
  for (int kb = 0; kb < KB; kb++) {
    // block kb's fragments are in registers: once every warp is here its buffer is free for block kb + ST
    cp_wait<ST - 2>();  // block kb + 1 has landed
    __syncthreads();
    if (kb + ST < KB) load(kb + ST, buf);
    cp_commit();
    buf = buf + 1 == ST ? 0 : buf + 1;
    const uint8_t* nx = sm + buf * C::STAGE;  // block kb + 1 (after the last block: stale data, never used)
    if constexpr (FMT == WMat::MXFP8) lds_raw(nx);
#pragma unroll
    for (int j = 0; j < NJ; j++)
#pragma unroll
      for (int i = 0; i < MI; i++) mma(acc[i][j], A0[i], B[j].x, B[j].y);  // k16 step 0
    deq0(nx);
#pragma unroll
    for (int j = 0; j < NJ; j++) {
#pragma unroll
      for (int i = 0; i < MI; i++) mma(acc[i][j], A1[i], B[j].z, B[j].w);  // then step 1
      B[j] = *reinterpret_cast<const uint4*>(nx + boff + j * 512);
    }
    deq1(nx);
  }
  cp_wait<0>();
  __syncthreads();

  // epilogue: C fragment c0 (w-row g, act-row 2t) c1 (g, 2t+1) c2 (g+8, 2t) c3 (g+8, 2t+1) → smem [act row][w row]
  // (NP slices of RP activation rows), then 16-B coalesced stores
  using E = typename std::conditional<F32, float, bf16>::type;
  constexpr int P = C::template pitch<F32>(), NP = C::template passes<F32>(), RP = C::BM / NP;
  constexpr int V = 16 / sizeof(E);
  E* Ps = reinterpret_cast<E*>(sm);
  E* y = static_cast<E*>(a.y);
  const int n0 = nt * C::BN;
  const bool vec = a.N % V == 0;
#pragma unroll
  for (int pass = 0; pass < NP; pass++) {
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      const int r8 = wm * NJ * 8 + j * 8;  // first activation row of n8 tile j in the CTA tile
      if (r8 / RP != pass) continue;
#pragma unroll
      for (int i = 0; i < MI; i++) {
        const int wr = (wn * MI + i) * 16 + g, ar = r8 - pass * RP + 2 * t;
        if constexpr (F32) {
          Ps[ar * P + wr] = acc[i][j][0], Ps[(ar + 1) * P + wr] = acc[i][j][1];
          Ps[ar * P + wr + 8] = acc[i][j][2], Ps[(ar + 1) * P + wr + 8] = acc[i][j][3];
        } else {
          Ps[ar * P + wr] = __float2bfloat16_rn(acc[i][j][0]);
          Ps[(ar + 1) * P + wr] = __float2bfloat16_rn(acc[i][j][1]);
          Ps[ar * P + wr + 8] = __float2bfloat16_rn(acc[i][j][2]);
          Ps[(ar + 1) * P + wr + 8] = __float2bfloat16_rn(acc[i][j][3]);
        }
      }
    }
    __syncthreads();
    for (int p = tid; p < RP * (C::BN / V); p += C::NTH) {
      const int r = p / (C::BN / V), q = p % (C::BN / V), m = m0 + pass * RP + r, n = n0 + q * V;
      if (m >= a.M || n >= a.N) continue;
      const E* src = Ps + r * P + q * V;
      E* dst = y + size_t(m) * a.N + n;
      if (vec) {
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      } else {
        for (int e = 0; e < V && n + e < a.N; e++) dst[e] = src[e];
      }
    }
    if (pass + 1 < NP) __syncthreads();
  }
}

// Tiles (4 warps of 32 weight rows): Row 128 × 128 activation rows, 2 CTAs/SM; Small 128 × 64, 3 CTAs/SM (MXFP8) or 2
// (bf16, whose small tile needs 4 stages to keep up with its 2× weight bytes).
template <int FMT> using Row = Cfg<FMT, 128, 128, 4, 1, FMT == WMat::MXFP8 ? 4 : 3, 2>;
template <int FMT> using Small = Cfg<FMT, 128, 64, 4, 1, FMT == WMat::MXFP8 ? 3 : 4, FMT == WMat::MXFP8 ? 3 : 2>;

template <class C, bool F32>
void launch(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  Args a{static_cast<const uint8_t*>(L.w), L.s, x, y, M, L.n, L.k, L.k / 32, L.k / 32, L.npad / 16,
         cdiv(M, C::BM), cdiv(L.npad, C::BN)};
  kernel<C, F32><<<a.MT * a.NT, C::NTH, C::template smem<F32>(), s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

template <class C, bool F32>
int set_smem() {  // → resident CTAs per SM
  CUDA_CHECK(cudaFuncSetAttribute(kernel<C, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::template smem<F32>()));
  int n = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel<C, F32>, C::NTH, C::template smem<F32>()));
  return std::max(1, n);
}

int g_sms = 48;                // SMs of the device (set in init)
int g_occ[2][2][2] = {};       // [fmt][F32][tile 0 = Row, 1 = Small] resident CTAs per SM
int g_force = -1;              // tests: -1 = pick(), 0 = Row, 1 = Small

template <int FMT, bool F32>
void set_smem_fmt() {
  g_occ[FMT][F32][0] = set_smem<Row<FMT>, F32>();
  g_occ[FMT][F32][1] = set_smem<Small<FMT>, F32>();
}
void init() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, dev));
  set_smem_fmt<WMat::MXFP8, false>(), set_smem_fmt<WMat::MXFP8, true>();
  set_smem_fmt<WMat::BF16, false>(), set_smem_fmt<WMat::BF16, true>();
}

// split-1 shapes only (K-split shapes keep the base kernels and their split sum)
inline bool takes(const Lin& L, int M) { return L.split == 1 && M > kDecodeRows; }

// Wave model (scheduling only — the tile never changes a value): `tiles` CTAs over g_sms SMs with `per` resident each;
// full waves cost `wave`, a last partial wave whose busiest SM holds j < per CTAs costs part[j-1]·wave.
inline double cost(long tiles, int per, double wave, const double* part) {
  const long slots = long(g_sms) * per, full = tiles / slots, r = tiles % slots;
  const int j = int((r + g_sms - 1) / g_sms);
  return wave * (double(full) + (r == 0 ? 0.0 : j < per ? part[j - 1] : 1.0));
}

// Row or Small for (N, M). M ≤ 256 is weight-bandwidth bound: Small (more CTAs in flight). Above, bf16 weights take Row
// (the bf16 Small tile is bandwidth bound: ~72 TF vs 88–90); MXFP8 compares wave costs measured on GB10 (K = 2560/3072,
// relative to a full Row wave): Row partial wave with 1 CTA/SM 0.64; Small (half the rows per tile, 3 CTAs/SM) full
// wave 0.76, partial 0.49 (1/SM) · 0.80 (2/SM). The model's misses measured ≤ 1.3 %.
template <int FMT, bool F32>
int pick(const Lin& L, int M) {
  if (g_force >= 0) return g_force;
  if (M <= 256) return 1;
  if (FMT != WMat::MXFP8) return 0;
  static constexpr double row_part[] = {0.64, 1.0}, small_part[] = {0.49, 0.80, 1.0};
  const long nt = cdiv(L.npad, 128);
  const double r = cost(long(cdiv(M, 128)) * nt, std::min(2, g_occ[FMT][F32][0]), 1.0, row_part);
  const double s = cost(long(cdiv(M, 64)) * nt, std::min(3, g_occ[FMT][F32][1]), 0.76, small_part);
  return s < r ? 1 : 0;
}

template <int FMT, bool F32>
void forward(const Lin& L, const bf16* x, void* y, int M, cudaStream_t s) {
  if (pick<FMT, F32>(L, M) == 0) launch<Row<FMT>, F32>(L, x, y, M, s);
  else launch<Small<FMT>, F32>(L, x, y, M, s);
}

}  // namespace p004
