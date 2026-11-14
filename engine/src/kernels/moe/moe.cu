// moe.cu — MoE block for one layer on this rank (spec §2.6), two launches per call (graph-capturable, PDL):
//   k_route   : 96 router blocks (16 experts × one K third, bf16; the last third to finish adds the three partials in
//               order → bf16 logits), shared-expert gate per row, v → B-fragment order. The block completing the last
//               router tile runs softmax/top-10/renormalise per row and the plan: union of selected experts sorted by id,
//               per expert its rows ascending, per row its (expert, row) pairs in expert-id order, routed 8-row tiles.
//   k_experts : persistent warps on one work queue; launched as a programmatic dependent it runs alongside k_route
//               (integer flags with release/acquire order the data). Items, one 8-row token tile each:
//               shared gate/up (16-row tile × K/5) → shared down (16 output rows) → routed up (16-row gate/up tile × K/5)
//               → routed down (16 output rows: y = bf16(acc·w·g2), waits for the expert's 40 gate/up tiles).
//               The last K part of a gate/up tile adds the five fp32 partials in order and writes
//               h = bf16(bf16(silu(g))·u); the last contribution to a 16-row output tile writes
//               out = bf16(bf16(Σ_e y_e) + shared), Σ in fp32 in expert-id order.
// Every selected expert's weights are read exactly once per call whatever the number of rows using it.
// Math: tensor-core mma.m16n8k16 bf16 → fp32. Weights are dequantised exactly to bf16 (fp4·e4m3 ≤ 6 significant bits,
// e4m3·2^e exact); the NVFP4 global scale is applied to the fp32 sum (as Marlin). A row's result depends only on its own
// data: fixed K splits, partial sums added in a fixed order, no float atomics (integer counters only order and schedule
// work) — deterministic and batch-invariant.
// Rounding follows the reference (Marlin W4A16, vLLM 0.32): router bf16 logits, GEMM-1 bf16 out,
// silu_and_mul bf16(bf16(silu(g))·u), GEMM-2 bf16(acc·w·g) per expert, moe_sum in fp32 → bf16,
// shared bf16(bf16(sigmoid(bf16 logit))·bf16(down)), bf16 add.
// (The reference quantises the shared down_proj input to MXFP8 even at decode; we keep it bf16 — lossless.)
// M > 64 (prefill): moe_large.cuh, included below — same weights in place, same per-row fold points, bitwise equal rows.
// P014 moe_scale4 (p014_moe_scale4.cuh): with the patch on, k_experts reads the routed experts' scales from a lossless
// 4-bit copy (88,320 B per expert, built at upload) instead of the block's expanded e4m3 bytes (153,600 B) — the same
// scale words reach sc_fp4, so the same bits; the expanded bytes stay for M > 64 and the patch off.
// P017 router_pack12 (p017_router_pack12.cuh): with the patch on, k_route reads a lossless 12-bit copy of the bf16 router
// (1,991,168 B per layer, built at upload) instead of its bf16 fragments (2,621,440 B) — the same A operands, the same
// MMAs in the same order, so the same logits; the bf16 fragments stay for M > 64, the patch off and escaped spans.
// P022 route_fast (p022_route_fast.cuh): with the patch on, k_route22 replaces k_route — the same router GEMV as 48
// CTAs of 16 warps, one barrier stage instead of two (the last CTA forms the logits from the K-third partials itself)
// and a faster routing tail on an SM of its own (top-3 lists for the top-10 rounds) — and k_experts22 replaces
// k_experts (routed gate/up items queued ahead of the shared down items); same logits, top-k, plan, sig, xf, out.
#include <algorithm>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "kernels/moe/moe.h"
#include "core/log.h"
#include "model/patches.h"

namespace q38 {

namespace {
constexpr int H = kHidden;            // 2560
constexpr int I = kExpInter / 2;      // 320 rank-local intermediate (TP=2)
constexpr int KS1 = H / 16;           // 160 k-steps over K=2560
constexpr int KS2 = I / 16;           // 20 k-steps over K=320
constexpr int T13 = I / 8;            // 40 gate/up tiles per expert (8 gate + 8 up rows)
constexpr int T2 = H / 16;            // 160 output tiles of 16 rows
constexpr int TR = kExperts / 16;     // 32 router tiles
constexpr int KP = 5;                 // K parts of an up tile (32 k-steps each)
constexpr int kMaxM = 64;
constexpr int kMaxPairs = kMaxM * kTopK;
constexpr int kMaxTT = kExperts + kMaxPairs / 8;  // routed token tiles (≥1 per selected expert)
// routed expert block (bytes): A fragments + scales, gate/up then down
constexpr size_t kW13 = size_t(T13) * KS1 * 32 * 4;  // 819,200
constexpr size_t kS13 = size_t(T13) * 8 * KS1 * 2;   // 102,400
constexpr size_t kW2 = size_t(T2) * KS2 * 32 * 4;    // 409,600
constexpr size_t kS2 = size_t(T2) * 8 * KS2 * 2;     //  51,200
constexpr size_t kEB = kW13 + kS13 + kW2 + kS2;      // 1,382,400
constexpr int kRouteBlocks = 3 * TR;                  // 96 router tile thirds

struct Plan {
  int U;                                   // distinct experts
  int uexp[kExperts], ucnt[kExperts], uoff[kExperts];
  int utok[kMaxPairs];                     // rows of expert slot u at pairs [uoff[u], +ucnt[u])
  float uw[kMaxPairs];                     // their renormalised top-k weights
  int tpair[kMaxPairs];                    // [M,10] pairs of row t in expert-id order
  int NT;                                  // routed token tiles: expert slot tt_u[T], its rows 8·tt_t[T] ..
  int tt_u[kMaxTT], tt_t[kMaxTT];
};
struct Sync {                              // integer counters/flags; all back to 0 when a call ends
  unsigned rtile[TR];                      // router K thirds done per tile
  unsigned rdone;                          // router tiles done
  unsigned sig_done;                       // router blocks past the shared-gate step
  unsigned h_done;                         // shared gate/up tiles done (h_s ready)
  unsigned scnt[T13];                      // K parts done per shared gate/up tile
  unsigned plan_ready;
  unsigned queue, exits;                   // k_experts work queue head, exited warps
  unsigned uready[kExperts];               // gate/up tiles done per expert slot
  unsigned ncnt[T2];                       // token tiles done per output tile (NT routed + nt shared)
  unsigned tcnt[kExperts * T13];           // K parts × token tiles done per (expert slot, gate/up tile)
};

struct Args {
  const bf16* v;
  bf16* out;
  int M;
  const uint8_t* ex;
  const float* gs;
  const uint4* rt;
  const uint8_t *sw13, *ss13, *sw2, *ss2;
  const bf16* sg;
  bf16 *xf, *hs, *hr, *logits, *y, *se;
  float *sig, *rpart, *upart, *spart;
  Plan* plan;
  Sync* sy;
  int* topk_out;
  float* topw_out;
  bf16 *dbg_routed, *dbg_shared;
  const uint8_t* sc4;  // P014: 4-bit scale records [512][p014::kSC] (k_experts<true> only)
  const uint8_t *rp, *rpb;  // P017: router 12-bit blocks, row bases (k_route<NT, true> only)
  const uint32_t* rpx;      // P017: exception slots per warp span
};
}  // namespace

struct MoeW {
  int layer = 0;
  uint8_t* ex = nullptr;      // [512][kEB]
  float* gs = nullptr;        // [512][3] global scale ×256: gate, up, down
  uint4* rt = nullptr;        // router A fragments [32][160][32]
  uint8_t* sw13 = nullptr;    // shared gate/up e4m3 fragments [40][80][32][16 B]
  uint8_t* ss13 = nullptr;    // [40][8][80][2] (ue8m0 + 120)
  uint8_t* sw2 = nullptr;     // shared down [160][10][32][16 B]
  uint8_t* ss2 = nullptr;     // [160][8][10][2]
  bf16* sg = nullptr;         // shared_expert_gate [2560]
  uint8_t* sc4 = nullptr;     // P014: routed scales, 4-bit records [512][88,320] (built when the patch is on at upload)
  long long p14[5] = {};      // its build: scales, exceptions, items, items over 8 exceptions, escaped rows
  uint8_t *rp = nullptr, *rpb = nullptr;  // P017: router 12-bit copy (blocks, bases; built when the patch is on at upload)
  uint32_t* rpx = nullptr;                // P017: its exception slots
  long long p17[5] = {};      // its build: values, exceptions, spans with exceptions, escaped spans, most in one span
  // per-layer scratch
  bf16 *xf = nullptr, *hs = nullptr, *hr = nullptr, *logits = nullptr, *y = nullptr, *se = nullptr;
  float *sig = nullptr, *rpart = nullptr, *upart = nullptr, *spart = nullptr;
  Plan* plan = nullptr;
  Sync* sy = nullptr;
};

namespace {

// ---- device helpers ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_stream(const void* p) {  // weights: read once, keep out of L1
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ld_stream2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ld_stream1(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void wait_ge(const unsigned* p, unsigned v, int lane) {  // lane 0 spins (acquire)
  if (lane == 0)
    while (ld_acquire(p) < v) __nanosleep(32);
  __syncwarp();
}
__device__ __forceinline__ void mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {  // bf16x2, subnormals kept (no .ftz for bf16)
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
// vLLM silu_and_mul: bf16(silu(g)) · u rounded to bf16; g, u already bf16 values
__device__ __forceinline__ bf16 silu_mul(float g, float u) {
  const float s = bf16r(g / (1.0f + expf(-g)));
  return __float2bfloat16_rn(s * u);
}
// element offset of k inside a row stored in B-fragment order (32-k blocks: lane q holds k=2q,2q+1 | +8 | +16 | +24)
__host__ __device__ __forceinline__ int frag_off(int k) {
  return (k & ~31) + ((k & 7) >> 1) * 8 + ((k >> 4) & 1) * 4 + ((k >> 3) & 1) * 2 + (k & 1);
}
// NVFP4 scales of rows r / r+8 (e4m3 bytes in the low 16 bits) → bf16x2 broadcasts of s·2^118
__device__ __forceinline__ void sc_fp4(uint32_t two, uint32_t& lo, uint32_t& hi) {
  const uint32_t bits = (__byte_perm(two, 0, 0x4140) << 4) + 0x77007700u;
  lo = __byte_perm(bits, 0, 0x1010);
  hi = __byte_perm(bits, 0, 0x3232);
}
// MXFP8 scales of rows r / r+8 (stored ue8m0+120) → bf16x2 broadcasts of 2^(s-7)
__device__ __forceinline__ void sc_fp8(uint32_t two, uint32_t& lo, uint32_t& hi) {
  const uint32_t bits = __byte_perm(two, 0, 0x4140) << 7;
  lo = __byte_perm(bits, 0, 0x1010);
  hi = __byte_perm(bits, 0, 0x3232);
}
// 8 fp4 codes [a6,a4,a2,a0,a7,a5,a3,a1] → A fragment (bf16x2 ×4) = fp4·s·2^-8 exactly
__device__ __forceinline__ void dq_fp4(uint32_t q, uint32_t lo, uint32_t hi, uint32_t a[4]) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const uint32_t w = q << (4 * j);
    a[j] = hmul2((w & 0x80008000u) | ((w & 0x70007000u) >> 6), (j & 1) ? hi : lo);  // fp4·2^-126 × s·2^118
  }
}
// 8 e4m3 bytes (w0 = a0..a3, w1 = a4..a7) → A fragment = e4m3·2^(s-127) exactly
__device__ __forceinline__ uint32_t e4m3x2(uint32_t w, uint32_t sel) {
  const uint32_t t = __byte_perm(w, 0, sel);
  return (t & 0x80008000u) | ((t & 0x7F007F00u) >> 4);  // e4m3·2^-120
}
__device__ __forceinline__ void dq_fp8(uint32_t w0, uint32_t w1, uint32_t lo, uint32_t hi, uint32_t a[4]) {
  a[0] = hmul2(e4m3x2(w0, 0x1404), lo);
  a[1] = hmul2(e4m3x2(w0, 0x3424), hi);
  a[2] = hmul2(e4m3x2(w1, 0x1404), lo);
  a[3] = hmul2(e4m3x2(w1, 0x3424), hi);
}


#include "kernels/moe/p017_router_pack12.cuh"  // P017 router_pack12: 12-bit router copy, decode, host builder

// ---- k_route ---------------------------------------------------------------------------------------------------------
// Routing tail: softmax over 512 (fp32), top-10 (ties → lower id), renormalise; plan. One CTA (256 threads).
__device__ void route_tail(const Args& a) {
  __shared__ int tk_i[kMaxM][kTopK];
  __shared__ float tk_w[kMaxM][kTopK];
  __shared__ unsigned long long msk[kExperts];
  __shared__ int off_e[kExperts];
  __shared__ int wsum[8][3];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, M = a.M;
  Plan* P = a.plan;
  for (int e = tid; e < kExperts; e += 256) msk[e] = 0;
  float nx[16];  // next row's logits (prefetched)
  if (warp < M)
#pragma unroll
    for (int i = 0; i < 16; i++) nx[i] = __bfloat162float(__ldcg(a.logits + size_t(warp) * kExperts + lane + 32 * i));
  for (int tok = warp; tok < M; tok += 8) {
    float p[16];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      p[i] = nx[i];
      mx = fmaxf(mx, p[i]);
    }
    if (tok + 8 < M)
#pragma unroll
      for (int i = 0; i < 16; i++) nx[i] = __bfloat162float(__ldcg(a.logits + size_t(tok + 8) * kExperts + lane + 32 * i));
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
    auto lane_best = [&]() {  // max p among this lane's free entries, lowest index on ties
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
      tk_i[tok][lane] = ik;
      tk_w[tok][lane] = wk / tot;
    }
  }
  __syncthreads();
  for (int i = tid; i < M * kTopK; i += 256) atomicOr(&msk[tk_i[i / kTopK][i % kTopK]], 1ull << (i / kTopK));
  __syncthreads();
  // expert slots in id order: thread t owns experts 2t, 2t+1 — exclusive scan of (used, rows, token tiles)
  const unsigned long long m0 = msk[2 * tid], m1 = msk[2 * tid + 1];
  const int f0 = m0 != 0, f1 = m1 != 0, c0 = __popcll(m0), c1 = __popcll(m1), t0 = (c0 + 7) >> 3, t1 = (c1 + 7) >> 3;
  int sf = f0 + f1, sc = c0 + c1, st = t0 + t1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int xf = __shfl_up_sync(~0u, sf, o), xc = __shfl_up_sync(~0u, sc, o), xt = __shfl_up_sync(~0u, st, o);
    if (lane >= o) { sf += xf; sc += xc; st += xt; }
  }
  if (lane == 31) { wsum[warp][0] = sf; wsum[warp][1] = sc; wsum[warp][2] = st; }
  __syncthreads();
  int slot = sf - (f0 + f1), off = sc - (c0 + c1), tt = st - (t0 + t1);
  for (int w = 0; w < warp; w++) { slot += wsum[w][0]; off += wsum[w][1]; tt += wsum[w][2]; }
  const int fs[2] = {f0, f1}, cs[2] = {c0, c1}, ts[2] = {t0, t1};
#pragma unroll
  for (int h = 0; h < 2; h++) {
    off_e[2 * tid + h] = off;
    if (fs[h]) {
      P->uexp[slot] = 2 * tid + h;
      P->ucnt[slot] = cs[h];
      P->uoff[slot] = off;
      for (int j = 0; j < ts[h]; j++) {
        P->tt_u[tt + j] = slot;
        P->tt_t[tt + j] = j;
      }
    }
    slot += fs[h];
    off += cs[h];
    tt += ts[h];
  }
  if (tid == 255) {
    P->U = slot;
    P->NT = tt;
  }
  __syncthreads();
  // pairs: (row, k) → slot position of the row inside its expert; row's pairs sorted by expert id
  for (int i = tid; i < M * kTopK; i += 256) {
    const int tok = i / kTopK, k = i % kTopK, e = tk_i[tok][k];
    const int pair = off_e[e] + __popcll(msk[e] & ((1ull << tok) - 1));
    int pos = 0;
#pragma unroll
    for (int j = 0; j < kTopK; j++) pos += tk_i[tok][j] < e;
    const float w = tk_w[tok][k];
    P->utok[pair] = tok;
    P->uw[pair] = w;
    P->tpair[tok * kTopK + pos] = pair;
    if (a.topk_out) a.topk_out[i] = e;
    if (a.topw_out) a.topw_out[i] = w;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    a.sy->rdone = 0;
    atomicExch(&a.sy->plan_ready, 1u);
  }
}

// 96 blocks: router tile (16 experts) × K third (56/56/48 k-steps, 8 warps × 7/7/6); the third part to finish adds the
// three fp32 partials in order → bf16 logits; the block finishing the last tile runs the routing tail.
// P17: the warp's fragments come from the 12-bit copy (loads at the same point, decoded before the first MMA).
template <int NT, bool P17>
__global__ void __launch_bounds__(256, 2) k_route(Args a) {
  __shared__ float red[8][NT][32][4];
  __shared__ int role;  // 0 part done, 1 last part of a tile (adds parts), 2 + last router tile (tail)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
  const int M = a.M, nt = (M + 7) >> 3, tile = blockIdx.x / 3, part = blockIdx.x % 3;
  const int nks = part < 2 ? 7 : 6, ks0 = part * 56 + warp * nks;
  const uint4* A = a.rt + (size_t(tile) * KS1 + ks0) * 32 + lane;
  uint4 w[7];
  [[maybe_unused]] p017::Span sp;
  if constexpr (P17) p017::load(a.rp, a.rpb, a.rpx, tile, blockIdx.x * 8 + warp, ks0, nks, lane, sp);  // before the wait
  else
#pragma unroll
    for (int s = 0; s < 7; s++) w[s] = s < nks ? ld_stream(A + s * 32) : make_uint4(0, 0, 0, 0);  // before the wait
  pdl_wait();
  pdl_trigger();
  if (warp == 0) {  // shared expert gate of row blockIdx.x: bf16(x·wg) → bf16(sigmoid)
    const int tok = blockIdx.x;
    if (tok < M) {
      float d = 0.f;
      const uint4* xv = reinterpret_cast<const uint4*>(a.v + size_t(tok) * H);
      const uint4* gv = reinterpret_cast<const uint4*>(a.sg);
#pragma unroll
      for (int i = 0; i < H / 256; i++) {
        const uint4 x = __ldg(xv + lane + 32 * i), g = __ldg(gv + lane + 32 * i);
        const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, gw[4] = {g.x, g.y, g.z, g.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xs[j]));
          const float2 gf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[j]));
          d = fmaf(xf.x, gf.x, d);
          d = fmaf(xf.y, gf.y, d);
        }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(~0u, d, o);
      const float gl = bf16r(d);
      if (lane == 0) a.sig[tok] = bf16r(1.0f / (1.0f + expf(-gl)));
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) atomicAdd(&a.sy->sig_done, 1u);
  }
  // v → B-fragment order for the routed up items
  for (int i = blockIdx.x * 256 + threadIdx.x; i < M * (H / 8); i += kRouteBlocks * 256) {
    const int tok = i / (H / 8), rem = i % (H / 8), kc = rem >> 2, qq = rem & 3;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(a.v + size_t(tok) * H + kc * 32 + 2 * qq);
    reinterpret_cast<uint4*>(a.xf)[i] = make_uint4(src[0], src[4], src[8], src[12]);
  }
  const bf16* vrow[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) vrow[t] = (8 * t + r < M) ? a.v + size_t(8 * t + r) * H + 2 * q : nullptr;
  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
  if constexpr (P17) p017::decode(sp, A, nks, lane, w);
#pragma unroll
  for (int s = 0; s < 7; s++)
    if (s < nks) {
      const int k = (ks0 + s) * 16;
#pragma unroll
      for (int t = 0; t < NT; t++)
        if (t < nt) {
          uint32_t b0 = 0, b1 = 0;
          if (vrow[t]) {
            b0 = __ldg(reinterpret_cast<const uint32_t*>(vrow[t] + k));
            b1 = __ldg(reinterpret_cast<const uint32_t*>(vrow[t] + k + 8));
          }
          mma(acc[t], w[s].x, w[s].y, w[s].z, w[s].w, b0, b1);
        }
    }
#pragma unroll
  for (int t = 0; t < NT; t++)
    if (t < nt)
#pragma unroll
      for (int j = 0; j < 4; j++) red[warp][t][lane][j] = acc[t][j];
  __syncthreads();
  if (warp < nt) {  // warp t adds the 8 K-slices of token tile t in warp order → fp32 partial [part][M][512]
    const int t = warp;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w8 = 0; w8 < 8; w8++)
#pragma unroll
      for (int j = 0; j < 4; j++) c[j] += red[w8][t][lane][j];
    float* rp = a.rpart + size_t(part) * kMaxM * kExperts;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int tok = 8 * t + 2 * q + (j & 1), e = 16 * tile + r + 8 * (j >> 1);
      if (tok < M) rp[size_t(tok) * kExperts + e] = c[j];
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) role = atomicAdd(&a.sy->rtile[tile], 1u) == 2 ? 1 : 0;
  __syncthreads();
  if (!role) return;
  // last part: logits = bf16((p0 + p1) + p2)
  __threadfence();
  for (int i = threadIdx.x; i < 16 * M; i += 256) {
    const int tok = i >> 4, e = 16 * tile + (i & 15);
    float s = 0.f;
#pragma unroll
    for (int p = 0; p < 3; p++) s += __ldcg(a.rpart + (size_t(p) * kMaxM + tok) * kExperts + e);
    a.logits[size_t(tok) * kExperts + e] = __float2bfloat16_rn(s);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    a.sy->rtile[tile] = 0;
    role = atomicAdd(&a.sy->rdone, 1u) == TR - 1 ? 2 : 1;
  }
  __syncthreads();
  if (role != 2) return;
  __threadfence();
  route_tail(a);
}

// ---- k_experts: work items (one warp, one 8-row token tile each) -----------------------------------------------------
struct Chunk {
  uint4 w;       // 4 k-steps of A
  uint2 s;       // 4 k-steps × (row r, row r+8) scales (P14: unused, scales from the 4-bit record)
  uint4 b0, b1;  // B fragments (2 k-steps each)
};

#include "kernels/moe/p014_moe_scale4.cuh"  // P014 moe_scale4: 4-bit scale records, decode, host builder

// routed token tile T (expert slot u, rows 8t..8t+7 of it), gate/up tile i, K part kp → fp32 partial; the last of the
// KP × (token tiles of u) parts to finish adds the KP partials in order and writes h for all rows of u (B-fragment
// order), then counts the tile in uready[u]. P14: the scales come from the item's 4-bit record (loaded first, decoded
// to the same scale words before the first MMA) instead of the expanded bytes.
template <bool P14>
__device__ void up_item(const Args& a, const Plan* P, int T, int i, int kp, int lane) {
  const int r = lane >> 2, q = lane & 3;
  const int u = __ldcg(&P->tt_u[T]), t = __ldcg(&P->tt_t[T]);
  const int e = __ldcg(&P->uexp[u]), cnt = __ldcg(&P->ucnt[u]), off = __ldcg(&P->uoff[u]), ntu = (cnt + 7) >> 3;
  const uint8_t* eb = a.ex + size_t(e) * kEB;
  constexpr int NC = KS1 / 4 / KP;  // 8 chunks of 4 k-steps
  const uint4* A = reinterpret_cast<const uint4*>(eb) + (size_t(i) * (KS1 / 4) + kp * NC) * 32 + lane;
  const uint2* S = reinterpret_cast<const uint2*>(eb + kW13 + size_t(i) * 8 * KS1 * 2 + r * KS1 * 2) + kp * NC;
  p014::UpRec rec;
  uint32_t sw[P14 ? 2 * NC : 1];
  if constexpr (P14) rec = p014::up_load(a.sc4 + size_t(e) * p014::kSC + size_t(i * KP + kp) * p014::kRecU, r);
  const int sl0 = 8 * t + r;
  const uint4* B = sl0 < cnt ? reinterpret_cast<const uint4*>(a.xf) + size_t(__ldcg(&P->utok[off + sl0])) * (H / 8) +
                                   kp * NC * 8 + q
                             : nullptr;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  constexpr int G = 4, NG = NC / G;  // 2 groups of 4 chunks, double-buffered
  Chunk cur[G], nxt[G];
  auto load = [&](Chunk* d, int g) {
#pragma unroll
    for (int j = 0; j < G; j++) {
      const int c = g * G + j;
      d[j].w = ld_stream(A + c * 32);
      if constexpr (!P14) d[j].s = ld_stream2(S + c);
      d[j].b0 = B ? __ldcg(B + 8 * c) : make_uint4(0, 0, 0, 0);
      d[j].b1 = B ? __ldcg(B + 8 * c + 4) : make_uint4(0, 0, 0, 0);
    }
  };
  load(cur, 0);
#pragma unroll
  for (int g = 0; g < NG; g++) {
    if (g + 1 < NG) load(nxt, g + 1);
    if constexpr (P14)
      if (g == 0) p014::up_decode(rec, S, r, sw);
#pragma unroll
    for (int j = 0; j < G; j++) {
      const uint32_t wq[4] = {cur[j].w.x, cur[j].w.y, cur[j].w.z, cur[j].w.w};
      uint32_t s0, s1;
      if constexpr (P14) s0 = sw[2 * (g * G + j)], s1 = sw[2 * (g * G + j) + 1];
      else s0 = cur[j].s.x, s1 = cur[j].s.y;
#pragma unroll
      for (int s4 = 0; s4 < 4; s4++) {
        uint32_t lo, hi, af[4];
        sc_fp4((s4 < 2 ? s0 : s1) >> (16 * (s4 & 1)), lo, hi);
        dq_fp4(wq[s4], lo, hi, af);
        const uint4& bb = (s4 < 2) ? cur[j].b0 : cur[j].b1;
        mma(acc, af[0], af[1], af[2], af[3], (s4 & 1) ? bb.z : bb.x, (s4 & 1) ? bb.w : bb.y);
      }
    }
    if (g + 1 < NG)
#pragma unroll
      for (int j = 0; j < G; j++) cur[j] = nxt[j];
  }
  // partial [pair][tile][kp][16 rows]: rows r (gate) / r+8 (up) of slots 8t+2q+{0,1}
  auto pidx = [&](int sl, int row, int k) { return ((size_t(off + sl) * T13 + i) * KP + k) * 16 + row; };
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int sl = 8 * t + 2 * q + (j & 1);
    if (sl < cnt) a.upart[pidx(sl, r + 8 * (j >> 1), kp)] = acc[j];
  }
  __threadfence();
  __syncwarp();
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(&a.sy->tcnt[u * T13 + i], 1u);
  old = __shfl_sync(~0u, old, 0);
  if (old != unsigned(KP * ntu - 1)) return;
  __threadfence();
  const float gg = a.gs[3 * e], gu = a.gs[3 * e + 1];
  bf16* hrow = a.hr + size_t(off) * I + frag_off(8 * i + r);
  for (int sl = q; sl < cnt; sl += 4) {
    float g = 0.f, v = 0.f;
#pragma unroll
    for (int k = 0; k < KP; k++) {
      g += __ldcg(a.upart + pidx(sl, r, k));
      v += __ldcg(a.upart + pidx(sl, r + 8, k));
    }
    hrow[size_t(sl) * I] = silu_mul(bf16r(g * gg), bf16r(v * gu));
  }
  __threadfence();
  __syncwarp();
  if (lane == 0) {
    a.sy->tcnt[u * T13 + i] = 0;
    atomicAdd(&a.sy->uready[u], 1u);
  }
}

// shared expert gate/up tile i (8 gate + 8 up rows), K part kp, token tile t; the last part writes h_s for all rows
__device__ void sgu_item(const Args& a, int i, int kp, int t, int lane) {
  const int r = lane >> 2, q = lane & 3, M = a.M, nt = (M + 7) >> 3;
  constexpr int NC = KS1 / 2 / KP;  // 16 chunks of 2 k-steps (one 32-k scale group each)
  const uint4* A = reinterpret_cast<const uint4*>(a.sw13) + (size_t(i) * (KS1 / 2) + NC * kp) * 32 + lane;
  const uint4* S = reinterpret_cast<const uint4*>(a.ss13 + size_t(i) * 8 * (KS1 / 2) * 2 + r * (KS1 / 2) * 2 + 2 * NC * kp);
  uint4 w[NC];
#pragma unroll
  for (int c = 0; c < NC; c++) w[c] = ld_stream(A + c * 32);
  const uint4 s0 = ld_stream(S), s1 = ld_stream(S + 1);
  const uint32_t sw[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
  const int tok0 = 8 * t + r;
  const bf16* vrow = tok0 < M ? a.v + size_t(tok0) * H + 2 * q : nullptr;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < NC; c++) {
    uint32_t lo, hi;
    sc_fp8(sw[c >> 1] >> (16 * (c & 1)), lo, hi);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint32_t af[4];
      dq_fp8(h ? w[c].z : w[c].x, h ? w[c].w : w[c].y, lo, hi, af);
      const int k = (2 * (NC * kp + c) + h) * 16;
      uint32_t b0 = 0, b1 = 0;
      if (vrow) {  // L2 reads: this grid does not wait on the producer of v (see k_experts)
        b0 = __ldcg(reinterpret_cast<const uint32_t*>(vrow + k));
        b1 = __ldcg(reinterpret_cast<const uint32_t*>(vrow + k + 8));
      }
      mma(acc, af[0], af[1], af[2], af[3], b0, b1);
    }
  }
  auto pidx = [&](int tok, int row, int k) { return ((size_t(tok) * T13 + i) * KP + k) * 16 + row; };
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int tok = 8 * t + 2 * q + (j & 1);
    if (tok < M) a.spart[pidx(tok, r + 8 * (j >> 1), kp)] = acc[j];
  }
  __threadfence();
  __syncwarp();
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(&a.sy->scnt[i], 1u);
  old = __shfl_sync(~0u, old, 0);
  if (old != unsigned(KP * nt - 1)) return;
  __threadfence();
  for (int tok = q; tok < M; tok += 4) {
    float g = 0.f, v = 0.f;
#pragma unroll
    for (int k = 0; k < KP; k++) {
      g += __ldcg(a.spart + pidx(tok, r, k));
      v += __ldcg(a.spart + pidx(tok, r + 8, k));
    }
    a.hs[size_t(tok) * I + frag_off(8 * i + r)] = silu_mul(bf16r(g), bf16r(v));
  }
  __threadfence();
  __syncwarp();
  if (lane == 0) {
    a.sy->scnt[i] = 0;
    atomicAdd(&a.sy->h_done, 1u);
  }
}

// last contribution to output tile n: out rows 16n.. = bf16(bf16(Σ_e y_e, expert-id order) + shared)
__device__ void final_sum(const Args& a, const Plan* P, int n, int lane) {
  __threadfence();
  const int M = a.M;
  for (int idx = lane; idx < 16 * M; idx += 32) {
    const int row = idx & 15, tok = idx >> 4;
    const size_t col = size_t(16 * n + row);
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; k++)
      s += __bfloat162float(__ldcg(a.y + size_t(__ldcg(&P->tpair[tok * kTopK + k])) * H + col));
    const float rt = bf16r(s), se = __bfloat162float(__ldcg(a.se + size_t(tok) * H + col));
    a.out[size_t(tok) * H + col] = __float2bfloat16_rn(rt + se);
    if (a.dbg_routed) a.dbg_routed[size_t(tok) * H + col] = __float2bfloat16_rn(rt);
    if (a.dbg_shared) a.dbg_shared[size_t(tok) * H + col] = __float2bfloat16_rn(se);
  }
  if (lane == 0) a.sy->ncnt[n] = 0;
}

// contributions to tile n: NT routed token tiles + nt shared token tiles
__device__ __forceinline__ void signal_tile(const Args& a, const Plan* P, int n, int lane) {
  __threadfence();
  __syncwarp();
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(&a.sy->ncnt[n], 1u);
  old = __shfl_sync(~0u, old, 0);
  if (old == unsigned(__ldcg(&P->NT) + ((a.M + 7) >> 3) - 1)) final_sum(a, P, n, lane);
}

// shared expert down, output tile n, token tile t: se = bf16(sig · bf16(h_s · w_dᵀ))
__device__ void sdown_item(const Args& a, const Plan* P, int n, int t, int lane) {
  const int r = lane >> 2, q = lane & 3, M = a.M;
  const uint4* A = reinterpret_cast<const uint4*>(a.sw2) + size_t(n) * 10 * 32 + lane;
  const uint8_t* S = a.ss2 + size_t(n) * 8 * 20 + r * 20;
  uint4 w[10];
  uint32_t sw[5];
#pragma unroll
  for (int c = 0; c < 10; c++) w[c] = ld_stream(A + c * 32);
#pragma unroll
  for (int j = 0; j < 5; j++) sw[j] = ld_stream1(S + 4 * j);
  wait_ge(&a.sy->h_done, T13, lane);
  wait_ge(&a.sy->sig_done, kRouteBlocks, lane);
  const int tok0 = 8 * t + r;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < 10; c++) {
    uint32_t lo, hi;
    sc_fp8(sw[c >> 1] >> (16 * (c & 1)), lo, hi);
    const uint4 b = tok0 < M ? __ldcg(reinterpret_cast<const uint4*>(a.hs) + size_t(tok0) * (I / 8) + 4 * c + q)
                             : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint32_t af[4];
      dq_fp8(h ? w[c].z : w[c].x, h ? w[c].w : w[c].y, lo, hi, af);
      mma(acc, af[0], af[1], af[2], af[3], h ? b.z : b.x, h ? b.w : b.y);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int tok = 8 * t + 2 * q + (j & 1);
    if (tok < M)
      a.se[size_t(tok) * H + 16 * n + r + 8 * (j >> 1)] = __float2bfloat16_rn(__ldcg(a.sig + tok) * bf16r(acc[j]));
  }
  wait_ge(&a.sy->plan_ready, 1u, lane);  // the tile's contribution count needs the plan
  signal_tile(a, P, n, lane);
}

// routed down, token tile T (expert slot u, rows 8t..), output tile n: y[pair][16n..] = bf16(acc · w · g2)
template <bool P14>
__device__ void down_item(const Args& a, const Plan* P, int T, int n, int lane) {
  const int r = lane >> 2, q = lane & 3;
  const int u = __ldcg(&P->tt_u[T]), t = __ldcg(&P->tt_t[T]);
  const int e = __ldcg(&P->uexp[u]), cnt = __ldcg(&P->ucnt[u]), off = __ldcg(&P->uoff[u]);
  const uint8_t* eb = a.ex + size_t(e) * kEB + kW13 + kS13;
  const uint4* A = reinterpret_cast<const uint4*>(eb) + size_t(n) * (KS2 / 4) * 32 + lane;
  const uint2* S = reinterpret_cast<const uint2*>(eb + kW2 + size_t(n) * 8 * KS2 * 2 + r * KS2 * 2);
  uint4 w[5];
  uint2 s[5];
  p014::DnRec rec;
#pragma unroll
  for (int c = 0; c < 5; c++) {
    w[c] = ld_stream(A + c * 32);
    if constexpr (!P14) s[c] = ld_stream2(S + c);
  }
  if constexpr (P14) rec = p014::dn_load(a.sc4 + size_t(e) * p014::kSC + p014::kUpB + size_t(n) * p014::kRecD, r);
  wait_ge(&a.sy->uready[u], T13, lane);
  const int sl0 = 8 * t + r;
  const uint4* B = reinterpret_cast<const uint4*>(a.hr) + size_t(off + sl0) * (I / 8) + q;
  uint4 b[10];
#pragma unroll
  for (int c = 0; c < 10; c++) b[c] = sl0 < cnt ? __ldcg(B + 4 * c) : make_uint4(0, 0, 0, 0);
  if constexpr (P14) {  // after the h loads are issued (escaped rows read S here)
    uint32_t sw[10];
    p014::dn_decode(rec, S, r, sw);
#pragma unroll
    for (int c = 0; c < 5; c++) s[c] = make_uint2(sw[2 * c], sw[2 * c + 1]);
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < 5; c++) {
    const uint32_t wq[4] = {w[c].x, w[c].y, w[c].z, w[c].w};
#pragma unroll
    for (int s4 = 0; s4 < 4; s4++) {
      const int ks = 4 * c + s4;
      uint32_t lo, hi, af[4];
      sc_fp4((s4 < 2 ? s[c].x : s[c].y) >> (16 * (s4 & 1)), lo, hi);
      dq_fp4(wq[s4], lo, hi, af);
      const uint4& bb = b[ks >> 1];
      mma(acc, af[0], af[1], af[2], af[3], (ks & 1) ? bb.z : bb.x, (ks & 1) ? bb.w : bb.y);
    }
  }
  const float g2 = a.gs[3 * e + 2];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int sl = 8 * t + 2 * q + (j & 1);
    if (sl < cnt)
      a.y[size_t(off + sl) * H + 16 * n + r + 8 * (j >> 1)] = __float2bfloat16_rn(acc[j] * (__ldcg(&P->uw[off + sl]) * g2));
  }
  signal_tile(a, P, n, lane);
}

// queue: shared gate/up (nt × 200) | shared down (nt × 160) | routed up (NT × 200) | routed down (NT × 160);
// token tiles of one tile/part are adjacent so the weights they share come from L2. Launched as a programmatic dependent
// of k_route without griddepcontrol.wait: everything k_route (or its predecessor) wrote is read through L2 (ld.cg) after
// the matching flag (sig_done, h_done, plan_ready, uready) was observed with acquire; an item only waits on flags of
// k_route or of earlier queue items, so the queue cannot deadlock.
template <bool P14>
__global__ void __launch_bounds__(128) k_experts(Args a) {
  const Plan* P = a.plan;
  const int lane = threadIdx.x & 31, nt = (a.M + 7) >> 3;
  constexpr int UPI = T13 * KP;
  const int SGU = nt * UPI, SH = SGU + nt * T2;
  int NT = -1, total = 0x7fffffff;
  for (;;) {
    int item = 0;
    if (lane == 0) item = int(atomicAdd(&a.sy->queue, 1u));
    item = __shfl_sync(~0u, item, 0);
    if (item >= SH && NT < 0) {
      wait_ge(&a.sy->plan_ready, 1u, lane);
      NT = __ldcg(&P->NT);
      total = SH + NT * (UPI + T2);
    }
    if (item >= total) break;
    if (item < SGU) {
      const int x = item / nt;
      sgu_item(a, x / KP, x % KP, item % nt, lane);
    } else if (item < SH) {
      const int x = item - SGU;
      sdown_item(a, P, x / nt, x % nt, lane);
    } else if (item < SH + NT * UPI) {
      const int x = item - SH, T = x / UPI, y = x % UPI;
      up_item<P14>(a, P, T, y / KP, y % KP, lane);
    } else {
      const int x = item - SH - NT * UPI;
      down_item<P14>(a, P, x / T2, x % T2, lane);
    }
  }
  // last warp out resets the per-call flags and counters
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(&a.sy->exits, 1u);
  old = __shfl_sync(~0u, old, 0);
  if (old != gridDim.x * (blockDim.x / 32) - 1) return;
  __threadfence();
  const int U = __ldcg(&P->U);
  for (int u = lane; u < U; u += 32) a.sy->uready[u] = 0;
  if (lane == 0) {
    a.sy->queue = 0;
    a.sy->sig_done = 0;
    a.sy->h_done = 0;
    a.sy->plan_ready = 0;
    a.sy->exits = 0;
  }
}

template <bool P14>
int expert_blocks() {
  static int nb = 0;
  if (!nb) {
    int dev, sms, per;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_experts<P14>, 128, 0));
    nb = sms * (per > 0 ? per : 1);
  }
  return nb;
}

template <class... P>
void launch_pdl(void (*k)(P...), int grid, int block, cudaStream_t s, const Args& a) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid, cfg.blockDim = block, cfg.dynamicSmemBytes = 0, cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at, cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k, a));
}

#include "kernels/moe/p022_route_fast.cuh"  // P022 route_fast: k_route22 + k_experts22 (earlier plan, routed work first)

template <int NT>
void launch(const Args& a, cudaStream_t s) {
  if (patch::on(patch::route_fast)) return p022::launch<NT>(a, s);  // P022 (read per call; a graph keeps the choice)
  if (a.rp) launch_pdl(k_route<NT, true>, kRouteBlocks, 256, s, a);  // P017 (rp set only when on)
  else launch_pdl(k_route<NT, false>, kRouteBlocks, 256, s, a);
  if (a.sc4) launch_pdl(k_experts<true>, expert_blocks<true>(), 128, s, a);  // P014 (sc4 set only when on)
  else launch_pdl(k_experts<false>, expert_blocks<false>(), 128, s, a);
}

// ---- upload helpers ----------------------------------------------------------------------------------------------
template <class T>
T* dupload(const void* src, size_t bytes) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  CUDA_CHECK(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice));
  return p;
}
std::vector<uint8_t> need(const Weights& W, const std::string& name, std::vector<int64_t> shape) {
  std::vector<int64_t> got;
  auto b = W.slice(name, &got);
  if (got != shape) {
    std::string g, s;
    for (auto x : got) g += std::to_string(x) + "x";
    for (auto x : shape) s += std::to_string(x) + "x";
    qlog::fail("moe", "%s: shape %s ≠ %s", name.c_str(), g.c_str(), s.c_str());
  }
  return b;
}
inline uint8_t nib(const uint8_t* row, int k) { return (row[k >> 1] >> ((k & 1) * 4)) & 15; }
// NVFP4 A fragment word: rows lo (a0,a1,a4,a5) / hi (a2,a3,a6,a7), k-step ks, lane q → codes [a6,a4,a2,a0,a7,a5,a3,a1]
inline uint32_t fp4_word(const uint8_t* lo, const uint8_t* hi, int ks, int q) {
  const int k = 16 * ks + 2 * q;
  const uint8_t a[8] = {nib(lo, k), nib(lo, k + 1), nib(hi, k), nib(hi, k + 1),
                        nib(lo, k + 8), nib(lo, k + 9), nib(hi, k + 8), nib(hi, k + 9)};
  const int order[8] = {6, 4, 2, 0, 7, 5, 3, 1};
  uint32_t w = 0;
  for (int i = 0; i < 8; i++) w |= uint32_t(a[order[i]]) << (4 * i);
  return w;
}
// router bf16 [512, 2560] (bf16 pairs) → k_route fragments [32][160][32] × {W[r][k..], W[r+8][k..], W[r][k+8..], W[r+8][k+8..]}
std::vector<uint4> router_frags(const uint32_t* src) {
  std::vector<uint4> rt(size_t(TR) * KS1 * 32);
  for (int i = 0; i < TR; i++)
    for (int ks = 0; ks < KS1; ks++)
      for (int lane = 0; lane < 32; lane++) {
        const int r = lane >> 2, q = lane & 3, k2 = (16 * ks + 2 * q) / 2;
        const size_t lo = size_t(16 * i + r) * (H / 2), hi = size_t(16 * i + r + 8) * (H / 2);
        rt[(size_t(i) * KS1 + ks) * 32 + lane] = make_uint4(src[lo + k2], src[hi + k2], src[lo + k2 + 4], src[hi + k2 + 4]);
      }
  return rt;
}
// P017: the router's 12-bit copy from its bf16 bits [512][2560] (every value checked on the host in p017::build)
void p017_upload(MoeW* m, const uint16_t* w) {
  p017::Stats st;
  const p017::Copy c = p017::build(w, st);
  for (void* p : {(void*)m->rp, (void*)m->rpb, (void*)m->rpx})
    if (p) CUDA_CHECK(cudaFree(p));
  m->rp = dupload<uint8_t>(c.blk.data(), c.blk.size());
  m->rpb = dupload<uint8_t>(c.base.data(), c.base.size());
  m->rpx = dupload<uint32_t>(c.slots.data(), c.slots.size() * sizeof(uint32_t));
  const long long v[5] = {st.values, st.exc, st.spans_exc, st.esc_spans, st.max_span};
  std::memcpy(m->p17, v, sizeof v);
}
// zero-scale groups → scale 1.0 and zero codes (exact: the group is zero either way); NaN scales are refused
void fix_scales(uint8_t* w, uint8_t* s, int rows, int kbytes, const std::string& name) {
  const int groups = kbytes / 8;
  for (int r = 0; r < rows; r++)
    for (int g = 0; g < groups; g++) {
      uint8_t& b = s[r * groups + g];
      if ((b & 0x7f) == 0x7f || (b & 0x80)) qlog::fail("moe", "%s: scale 0x%02x (row %d, group %d)", name.c_str(), b, r, g);
      if (b == 0) {
        b = 0x38;
        std::memset(w + size_t(r) * kbytes + 8 * g, 0, 8);
      }
    }
}
}  // namespace

#include "kernels/moe/moe_large.cuh"  // large-M (prefill) path, M > 64

MoeW* moe_upload(const Weights& W, int layer) {
  MoeW* m = moe_upload(W, "model.language_model.layers." + std::to_string(layer) + ".mlp.", false);
  m->layer = layer;
  return m;
}

MoeW* moe_upload(const Weights& W, const std::string& p, bool mtp) {
  auto* m = new MoeW;
  const std::string sc = mtp ? ".weight_scale" : ".weight_scale.nvfp4_csf_fixed";  // MTP experts: plain E4M3 scales
  // routed experts
  CUDA_CHECK(cudaMalloc(&m->ex, kEB * kExperts));
  std::vector<float> gs(3 * kExperts);
  std::vector<uint8_t> blk(kEB);
  // P014: 4-bit scale records built from the same expanded bytes (every row checked on the host), when the patch is on
  const bool p14 = patch::on(patch::moe_scale4);
  constexpr size_t SX = kS13 + kS2;
  std::vector<uint8_t> scx(p14 ? SX * kExperts : 0);  // the experts' expanded scales, as in the block
  for (int e = 0; e < kExperts; e++) {
    const std::string pe = p + "experts." + std::to_string(e) + ".";
    auto wg = need(W, pe + "gate_proj.weight", {I, H / 2});
    auto wu = need(W, pe + "up_proj.weight", {I, H / 2});
    auto wd = need(W, pe + "down_proj.weight", {H, I / 2});
    auto sgt = need(W, pe + "gate_proj" + sc, {I, H / 16});
    auto sup = need(W, pe + "up_proj" + sc, {I, H / 16});
    auto sdn = need(W, pe + "down_proj" + sc, {H, I / 16});
    fix_scales(wg.data(), sgt.data(), I, H / 2, pe + "gate_proj");
    fix_scales(wu.data(), sup.data(), I, H / 2, pe + "up_proj");
    fix_scales(wd.data(), sdn.data(), H, I / 2, pe + "down_proj");
    const char* gn[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int j = 0; j < 3; j++) {
      auto g = W.raw(pe + gn[j] + ".weight_scale_2");
      float f;
      std::memcpy(&f, g.data(), 4);
      gs[3 * e + j] = f * 256.f;  // fp4·2^-126 × e4m3·2^118 = w/g·2^-8
    }
    uint32_t* w13 = reinterpret_cast<uint32_t*>(blk.data());
    uint8_t* s13 = blk.data() + kW13;
    uint32_t* w2 = reinterpret_cast<uint32_t*>(blk.data() + kW13 + kS13);
    uint8_t* s2 = blk.data() + kW13 + kS13 + kW2;
    for (int i = 0; i < T13; i++)
      for (int r = 0; r < 8; r++) {
        const uint8_t* lo = wg.data() + size_t(8 * i + r) * (H / 2);
        const uint8_t* hi = wu.data() + size_t(8 * i + r) * (H / 2);
        for (int ks = 0; ks < KS1; ks++) {
          for (int q = 0; q < 4; q++) w13[((size_t(i) * (KS1 / 4) + ks / 4) * 32 + 4 * r + q) * 4 + ks % 4] = fp4_word(lo, hi, ks, q);
          s13[((size_t(i) * 8 + r) * KS1 + ks) * 2 + 0] = sgt[size_t(8 * i + r) * KS1 + ks];
          s13[((size_t(i) * 8 + r) * KS1 + ks) * 2 + 1] = sup[size_t(8 * i + r) * KS1 + ks];
        }
      }
    for (int n = 0; n < T2; n++)
      for (int r = 0; r < 8; r++) {
        const uint8_t* lo = wd.data() + size_t(16 * n + r) * (I / 2);
        const uint8_t* hi = wd.data() + size_t(16 * n + r + 8) * (I / 2);
        for (int ks = 0; ks < KS2; ks++) {
          for (int q = 0; q < 4; q++) w2[((size_t(n) * (KS2 / 4) + ks / 4) * 32 + 4 * r + q) * 4 + ks % 4] = fp4_word(lo, hi, ks, q);
          s2[((size_t(n) * 8 + r) * KS2 + ks) * 2 + 0] = sdn[size_t(16 * n + r) * KS2 + ks];
          s2[((size_t(n) * 8 + r) * KS2 + ks) * 2 + 1] = sdn[size_t(16 * n + r + 8) * KS2 + ks];
        }
      }
    CUDA_CHECK(cudaMemcpy(m->ex + kEB * e, blk.data(), kEB, cudaMemcpyHostToDevice));
    if (p14) {
      std::memcpy(scx.data() + SX * e, s13, kS13);
      std::memcpy(scx.data() + SX * e + kS13, s2, kS2);
    }
  }
  m->gs = dupload<float>(gs.data(), gs.size() * 4);
  if (p14) {
    std::vector<uint8_t> sc4(p014::kSC * kExperts);
    std::vector<p014::Stats> st(kExperts);
    const int nt = std::max(1, std::min<int>(16, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++)
      th.emplace_back([&, t] {
        for (int e = t; e < kExperts; e += nt)
          p014::build(scx.data() + SX * e, scx.data() + SX * e + kS13, sc4.data() + p014::kSC * e, st[e]);
      });
    for (auto& x : th) x.join();
    m->sc4 = dupload<uint8_t>(sc4.data(), sc4.size());
    for (const auto& x : st) {
      const long long v[5] = {x.scales, x.exc, x.items, x.esc_items, x.esc_rows};
      for (int j = 0; j < 5; j++) m->p14[j] += v[j];
    }
  }

  // router bf16 [512, 2560] → k_route fragments (router_frags); P017: + its 12-bit copy when the patch is on
  {
    auto rw = need(W, p + "gate.weight", {kExperts, H});
    const std::vector<uint4> rt = router_frags(reinterpret_cast<const uint32_t*>(rw.data()));
    m->rt = dupload<uint4>(rt.data(), rt.size() * sizeof(uint4));
    if (patch::on(patch::router_pack12)) p017_upload(m, reinterpret_cast<const uint16_t*>(rw.data()));  // P017
    auto g = need(W, p + "shared_expert_gate.weight", {1, H});
    m->sg = dupload<bf16>(g.data(), g.size());
  }
  // shared expert MXFP8: gate/up tiles (8 gate + 8 up rows), down tiles (16 rows); scales stored as ue8m0 + 120.
  // MTP: the checkpoint's shared expert is bf16 (not representable in e4m3 without loss) — it is uploaded as zeros
  // here, so the kernel's shared term is bf16(σ·0) = 0 and out = bf16(routed + 0) = the routed sum exactly; the caller
  // adds the bf16 shared expert itself (model/mtp.cc, kernels/mtp).
  {
    const std::string ps = p + "shared_expert.";
    auto zeros = [](size_t n) { return std::vector<uint8_t>(n, 0); };
    auto wg = mtp ? zeros(size_t(I) * H) : need(W, ps + "gate_proj.weight", {I, H});
    auto wu = mtp ? zeros(size_t(I) * H) : need(W, ps + "up_proj.weight", {I, H});
    auto sgt = mtp ? zeros(size_t(I) * H / 32) : need(W, ps + "gate_proj.weight_scale", {I, H / 32});
    auto sup = mtp ? zeros(size_t(I) * H / 32) : need(W, ps + "up_proj.weight_scale", {I, H / 32});
    auto wd = mtp ? zeros(size_t(H) * I) : need(W, ps + "down_proj.weight", {H, I});
    auto sdn = mtp ? zeros(size_t(H) * I / 32) : need(W, ps + "down_proj.weight_scale", {H, I / 32});
    for (auto* v : {&sgt, &sup, &sdn})
      for (auto& b : *v) {
        if (b > 134) qlog::fail("moe", "%s: ue8m0 scale %d > 134 (bf16 dequant range)", ps.c_str(), b);
        b = uint8_t(b + 120);
      }
    auto frag8 = [](const uint8_t* lo, const uint8_t* hi, int ks, int q, uint8_t* o) {
      const int k = 16 * ks + 2 * q;
      const uint8_t a[8] = {lo[k], lo[k + 1], hi[k], hi[k + 1], lo[k + 8], lo[k + 9], hi[k + 8], hi[k + 9]};
      std::memcpy(o, a, 8);
    };
    std::vector<uint8_t> w13(size_t(T13) * KS1 * 32 * 8), s13(size_t(T13) * 8 * (KS1 / 2) * 2);
    for (int i = 0; i < T13; i++)
      for (int r = 0; r < 8; r++) {
        const uint8_t* lo = wg.data() + size_t(8 * i + r) * H;
        const uint8_t* hi = wu.data() + size_t(8 * i + r) * H;
        for (int ks = 0; ks < KS1; ks++)
          for (int q = 0; q < 4; q++)
            frag8(lo, hi, ks, q, &w13[((size_t(i) * (KS1 / 2) + ks / 2) * 32 + 4 * r + q) * 16 + (ks % 2) * 8]);
        for (int g = 0; g < KS1 / 2; g++) {
          s13[((size_t(i) * 8 + r) * (KS1 / 2) + g) * 2 + 0] = sgt[size_t(8 * i + r) * (H / 32) + g];
          s13[((size_t(i) * 8 + r) * (KS1 / 2) + g) * 2 + 1] = sup[size_t(8 * i + r) * (H / 32) + g];
        }
      }
    std::vector<uint8_t> w2(size_t(T2) * KS2 * 32 * 8), s2(size_t(T2) * 8 * (KS2 / 2) * 2);
    for (int n = 0; n < T2; n++)
      for (int r = 0; r < 8; r++) {
        const uint8_t* lo = wd.data() + size_t(16 * n + r) * I;
        const uint8_t* hi = wd.data() + size_t(16 * n + r + 8) * I;
        for (int ks = 0; ks < KS2; ks++)
          for (int q = 0; q < 4; q++)
            frag8(lo, hi, ks, q, &w2[((size_t(n) * (KS2 / 2) + ks / 2) * 32 + 4 * r + q) * 16 + (ks % 2) * 8]);
        for (int g = 0; g < KS2 / 2; g++) {
          s2[((size_t(n) * 8 + r) * (KS2 / 2) + g) * 2 + 0] = sdn[size_t(16 * n + r) * (I / 32) + g];
          s2[((size_t(n) * 8 + r) * (KS2 / 2) + g) * 2 + 1] = sdn[size_t(16 * n + r + 8) * (I / 32) + g];
        }
      }
    m->sw13 = dupload<uint8_t>(w13.data(), w13.size());
    m->ss13 = dupload<uint8_t>(s13.data(), s13.size());
    m->sw2 = dupload<uint8_t>(w2.data(), w2.size());
    m->ss2 = dupload<uint8_t>(s2.data(), s2.size());
  }
  // scratch
  CUDA_CHECK(cudaMalloc(&m->xf, sizeof(bf16) * kMaxM * H));
  CUDA_CHECK(cudaMalloc(&m->hs, sizeof(bf16) * kMaxM * I));
  CUDA_CHECK(cudaMalloc(&m->hr, sizeof(bf16) * kMaxPairs * I));
  CUDA_CHECK(cudaMalloc(&m->logits, sizeof(bf16) * kMaxM * kExperts));
  CUDA_CHECK(cudaMalloc(&m->y, sizeof(bf16) * kMaxPairs * H));
  CUDA_CHECK(cudaMalloc(&m->se, sizeof(bf16) * kMaxM * H));
  CUDA_CHECK(cudaMalloc(&m->sig, sizeof(float) * kMaxM));
  CUDA_CHECK(cudaMalloc(&m->rpart, sizeof(float) * 3 * kMaxM * kExperts));
  CUDA_CHECK(cudaMalloc(&m->upart, sizeof(float) * kMaxPairs * T13 * KP * 16));
  CUDA_CHECK(cudaMalloc(&m->spart, sizeof(float) * kMaxM * T13 * KP * 16));
  CUDA_CHECK(cudaMalloc(&m->plan, sizeof(Plan)));
  CUDA_CHECK(cudaMalloc(&m->sy, sizeof(Sync)));
  CUDA_CHECK(cudaMemset(m->sy, 0, sizeof(Sync)));
  CUDA_CHECK(cudaMemset(m->plan, 0, sizeof(Plan)));
  lw_acquire();
  return m;
}


void moe_free(MoeW* m) {
  if (!m) return;
  for (void* p : {(void*)m->ex, (void*)m->sc4, (void*)m->rp, (void*)m->rpb, (void*)m->rpx, (void*)m->gs, (void*)m->rt,
                  (void*)m->sw13, (void*)m->ss13, (void*)m->sw2, (void*)m->ss2, (void*)m->sg, (void*)m->xf, (void*)m->hs, (void*)m->hr, (void*)m->logits, (void*)m->y,
                  (void*)m->se, (void*)m->sig, (void*)m->rpart, (void*)m->upart, (void*)m->spart, (void*)m->plan,
                  (void*)m->sy})
    if (p) CUDA_CHECK(cudaFree(p));
  delete m;
  lw_release();
}

namespace {
Args make_args(const MoeW* w, const bf16* v, bf16* out, int M, int* topk_out, float* topw_out, const MoeDbg& dbg) {
  const bool p17 = patch::on(patch::router_pack12) && w->rp;
  return Args{v, out, M, w->ex, w->gs, w->rt, w->sw13, w->ss13, w->sw2, w->ss2, w->sg, w->xf, w->hs, w->hr,
              w->logits, w->y, w->se, w->sig, w->rpart, w->upart, w->spart, w->plan, w->sy, topk_out, topw_out,
              dbg.routed, dbg.shared, patch::on(patch::moe_scale4) ? w->sc4 : nullptr,
              p17 ? w->rp : nullptr, p17 ? w->rpb : nullptr, p17 ? w->rpx : nullptr};
}
}  // namespace

void moe_forward_dbg(const MoeW* w, const bf16* v, bf16* out, int M, int* topk_out, float* topw_out, const MoeDbg& dbg,
                     cudaStream_t s) {
  if (M <= 0) return;
  if (M > kMaxM) {
    if (M > kMaxML) qlog::fail("moe", "M %d > %d", M, kMaxML);
    moe_large(w, v, out, M, topk_out, topw_out, dbg, s);
    return;
  }
  const Args a = make_args(w, v, out, M, topk_out, topw_out, dbg);
  if (M <= 8) launch<1>(a, s);
  else if (M <= 16) launch<2>(a, s);
  else if (M <= 32) launch<4>(a, s);
  else launch<8>(a, s);
  CUDA_CHECK(cudaGetLastError());
  if (dbg.logits) CUDA_CHECK(cudaMemcpyAsync(dbg.logits, w->logits, sizeof(bf16) * M * kExperts, cudaMemcpyDeviceToDevice, s));
  if (dbg.h_shared) CUDA_CHECK(cudaMemcpyAsync(dbg.h_shared, w->hs, sizeof(bf16) * M * I, cudaMemcpyDeviceToDevice, s));
  if (dbg.n_experts) CUDA_CHECK(cudaMemcpyAsync(dbg.n_experts, &w->plan->U, sizeof(int), cudaMemcpyDeviceToDevice, s));
}

void moe_forward(const MoeW* w, const bf16* v, bf16* out, int M, int* topk_out, float* topw_out, cudaStream_t s) {
  moe_forward_dbg(w, v, out, M, topk_out, topw_out, MoeDbg{}, s);
}

namespace {
__global__ void k_silu_mul(const bf16* gu, bf16* h, int M, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M * n; i += gridDim.x * blockDim.x) {
    const int m = i / n, j = i % n;
    h[i] = silu_mul(__bfloat162float(gu[size_t(m) * 2 * n + j]), __bfloat162float(gu[size_t(m) * 2 * n + n + j]));
  }
}
}  // namespace

void moe_silu_mul(const bf16* gu, bf16* h, int M, int n, cudaStream_t s) {
  k_silu_mul<<<cdiv(M * n, 256), 256, 0, s>>>(gu, h, M, n);
  CUDA_CHECK(cudaGetLastError());
}

int moe_frag_off(int k) { return frag_off(k); }

// ---- P014 tests ---------------------------------------------------------------------------------------------------
bool moe_p014_built(const MoeW* w) { return w->sc4 != nullptr; }

void moe_p014_info(const MoeW* w, long long* v) { std::memcpy(v, w->p14, sizeof w->p14); }

void moe_p014_unpack(const MoeW* w, uint8_t* dec, uint8_t* ref, cudaStream_t s) {
  if (!w->sc4) qlog::fail("moe", "p014: no 4-bit scales (patch off at upload)");
  constexpr int items = kExperts * (T13 * KP + T2);
  p014::unpack_kernel<<<(items + 3) / 4, 128, 0, s>>>(w->sc4, w->ex, dec);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy2DAsync(ref, kS13 + kS2, w->ex + kW13, kEB, kS13, kExperts, cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpy2DAsync(ref + kS13, kS13 + kS2, w->ex + kW13 + kS13 + kW2, kEB, kS2, kExperts,
                               cudaMemcpyDeviceToDevice, s));
}

void moe_p014_get_scales(const MoeW* w, int e, uint8_t* s13s2) {
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(s13s2, w->ex + kEB * e + kW13, kS13, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(s13s2 + kS13, w->ex + kEB * e + kW13 + kS13 + kW2, kS2, cudaMemcpyDeviceToHost));
}

void moe_p014_set_scales(MoeW* w, int e, const uint8_t* s13s2, long long* v) {
  if (!w->sc4) qlog::fail("moe", "p014: no 4-bit scales (patch off at upload)");
  std::vector<uint8_t> rec(p014::kSC);
  p014::Stats st;
  p014::build(s13s2, s13s2 + kS13, rec.data(), st);  // checks every row
  const long long x[5] = {st.scales, st.exc, st.items, st.esc_items, st.esc_rows};
  std::memcpy(v, x, sizeof x);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(w->ex + kEB * e + kW13, s13s2, kS13, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(w->ex + kEB * e + kW13 + kS13 + kW2, s13s2 + kS13, kS2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(w->sc4 + p014::kSC * e, rec.data(), p014::kSC, cudaMemcpyHostToDevice));
}

// ---- P017 tests ---------------------------------------------------------------------------------------------------
bool moe_p017_built(const MoeW* w) { return w->rp != nullptr; }

void moe_p017_info(const MoeW* w, long long* v) { std::memcpy(v, w->p17, sizeof w->p17); }

void moe_p017_unpack(const MoeW* w, bf16* out, cudaStream_t s) {
  if (!w->rp) qlog::fail("moe", "p017: no 12-bit router (patch off at upload)");
  p017::unpack_kernel<<<p017::kSpans / 4, 128, 0, s>>>(w->rp, w->rpb, w->rpx, w->rt, reinterpret_cast<uint32_t*>(out));
  CUDA_CHECK(cudaGetLastError());
}

void moe_p017_set_router(MoeW* w, const uint16_t* bits, long long* v) {
  CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<uint4> rt = router_frags(reinterpret_cast<const uint32_t*>(bits));
  CUDA_CHECK(cudaMemcpy(w->rt, rt.data(), rt.size() * sizeof(uint4), cudaMemcpyHostToDevice));
  if (w->rp) p017_upload(w, bits);
  std::memcpy(v, w->p17, sizeof w->p17);
}

void moe_p017_roundtrip(const uint16_t* bits, bf16* out, long long* v, cudaStream_t s) {
  MoeW t;
  const std::vector<uint4> rt = router_frags(reinterpret_cast<const uint32_t*>(bits));
  t.rt = dupload<uint4>(rt.data(), rt.size() * sizeof(uint4));
  p017_upload(&t, bits);
  moe_p017_unpack(&t, out, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::memcpy(v, t.p17, sizeof t.p17);
  for (void* p : {(void*)t.rt, (void*)t.rp, (void*)t.rpb, (void*)t.rpx}) CUDA_CHECK(cudaFree(p));
}

// ---- P022 tests ---------------------------------------------------------------------------------------------------
void moe_p022_state(const MoeW* w, int M, int* plan, float* sig, bf16* xf, cudaStream_t s) {
  if (M <= 0 || M > kMaxM) qlog::fail("moe", "p022 state: M %d outside [1, %d]", M, kMaxM);
  CUDA_CHECK(cudaMemcpyAsync(plan, w->plan, sizeof(Plan), cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(sig, w->sig, sizeof(float) * M, cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(xf, w->xf, sizeof(bf16) * M * H, cudaMemcpyDeviceToDevice, s));
}

int moe_p022_plan_ints() { return int(sizeof(Plan) / sizeof(int)); }

void moe_forward_prof(const MoeW* w, const bf16* v, bf16* out, int M, float* ms, cudaStream_t s) {
  if (M <= kMaxM || M > kMaxML) qlog::fail("moe", "prof: M %d outside (%d, %d]", M, kMaxM, kMaxML);
  cudaEvent_t ev[8];
  for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
  moe_large(w, v, out, M, nullptr, nullptr, MoeDbg{}, s, ev);
  CUDA_CHECK(cudaEventSynchronize(ev[7]));
  for (int i = 0; i < 7; i++) CUDA_CHECK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
  for (auto& e : ev) CUDA_CHECK(cudaEventDestroy(e));
}

}  // namespace q38
