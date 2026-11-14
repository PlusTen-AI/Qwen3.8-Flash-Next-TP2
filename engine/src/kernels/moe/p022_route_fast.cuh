// p022_route_fast.cuh — P022 route_fast: a shorter path from the decode router's last MMA to plan_ready (k_route22,
// M ≤ 64) and the routed expert stream started as soon as the plan is there (k_experts22), when
// patch::on(route_fast). Included by moe.cu inside its anonymous namespace, after k_experts and launch_pdl (Args,
// Plan, Sync, kRouteBlocks, p017, the k_experts items…).
//
// Where k_route's time to plan_ready went (stamped scratch build; results/runs/20261113-p022/README.txt), M = 20, µs after
// the first k_route block starts: last MMA 14.6 · counted 16.0 · tile logits 19.4 · tail start 19.7 · softmax + top-10
// 30.6 · masks 31.4 · scan 32.2 · pairs 32.5 · plan_ready 33.1, and k_experts' first routed item ends at 38.3.
//   · route_tail's top-10 costs ≈ 2 µs per row (lane_best: a 16-step dependent select chain redone in every round) and
//     M rows share 8 warps (M = 20: three rows per warp);
//   · the tail CTA shares its SM with k_experts warps streaming the shared expert: its loads stall behind theirs;
//   · the K reduction takes two barrier stages (per-tile count → last third adds → router-wide count);
//   · k_experts parks the warps done with shared gate/up items on shared down items (h_done) before routed items.
//
// k_route22: the same 96 router blocks of 8 warps, launched as 48 CTAs of 16 warps (warps 8h … 8h+7 = k_route's block
// 2·CTA + h: the same loads, MMAs in the same order and 8-warp partial sums), then
//   · one barrier stage: each CTA stores its two K-third partials (fp32) and counts itself; the last of the 48 runs
//     tail() — no per-tile stage. A CTA of 16 warps holds the whole register file, so no k_experts block shares the
//     tail's SM (k_route's 8-warp blocks leave room for one).
//   · tail(): warp w takes rows w, w+16, …; per row each lane loads its 16 entries' three partials (four 16-byte loads
//     per third; the next row's issued before this row's softmax) and forms logit = bf16((0 + p0 + p1) + p2) exactly as
//     k_route's last third does (stored to Args::logits); softmax exactly as route_tail (same lane ↔ entry map, same
//     reduction trees); top-10 with route_tail's rounds (warp max of the lane bests, the lowest id among equal bits,
//     tot summed in pick order), the lane best taken from a sorted top-3 of the lane (one pass), route_tail's lane_best
//     only for a lane that gave all three; masks of the selected experts set right after the rounds; the plan as
//     route_tail (one expert per thread in the scan).
//   · the shared gate moved behind the CTA's count (the last CTA: behind the tail); only k_experts' shared down items
//     read sig, after sig_done (still 96 counts).
//   · fences: fence.acq_rel.gpu by the thread that counts, around a CTA barrier, instead of every thread's
//     __threadfence (fence.sc).
// k_experts22: k_experts with the routed gate/up items ahead of the shared down items (below).
// Exactness: every float operation that reaches the logits, top-k ids/weights, the plan and sig is k_route's /
// route_tail's operation on the same operands in the same order (the top-3 list yields exactly route_tail's lane best
// in each round; masks, scan and pairs are integer); xf is the same copy; k_experts22 runs k_experts' items. So logits,
// ids, weights, plan, sig, xf and out are bitwise the base's for every M ≤ 64 and row. A row's result depends only on
// its own row (batch invariant); integer counters only.
#pragma once

namespace p022 {

constexpr int kCtas = kRouteBlocks / 2;  // 48 CTAs × 16 warps = k_route's 96 blocks × 8 warps
constexpr int kThreads = 512;
static_assert(kThreads == kExperts, "p022: one expert per tail thread in the scan");

// release / acquire fence at gpu scope, by the one thread that does the counting atomic, behind / before a CTA barrier
// (the pattern of cooperative groups' grid sync; PTX release / acquire patterns are cumulative through the barrier)
__device__ __forceinline__ void fence() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }

// a lane's three best entries by (p bits desc, index asc): one scan in index order, strict > (an equal value stays
// behind the earlier, lower index) — the first three picks route_tail's lane_best makes in this lane
__device__ __forceinline__ void top3(const float (&p)[16], unsigned& u0, int& i0, unsigned& u1, int& i1, unsigned& u2,
                                     int& i2) {
  u0 = __float_as_uint(p[0]), i0 = 0;
  {
    const unsigned u = __float_as_uint(p[1]);
    const bool c0 = u > u0;
    u1 = c0 ? u0 : u, i1 = c0 ? 0 : 1;
    u0 = c0 ? u : u0, i0 = c0 ? 1 : 0;
  }
  {
    const unsigned u = __float_as_uint(p[2]);
    const bool c0 = u > u0, c1 = u > u1;
    u2 = c1 ? u1 : u, i2 = c1 ? i1 : 2;
    u1 = c0 ? u0 : (c1 ? u : u1), i1 = c0 ? i0 : (c1 ? 2 : i1);
    u0 = c0 ? u : u0, i0 = c0 ? 2 : i0;
  }
#pragma unroll
  for (int i = 3; i < 16; i++) {
    const unsigned u = __float_as_uint(p[i]);
    const bool c0 = u > u0, c1 = u > u1, c2 = u > u2;
    u2 = c1 ? u1 : (c2 ? u : u2), i2 = c1 ? i1 : (c2 ? i : i2);
    u1 = c0 ? u0 : (c1 ? u : u1), i1 = c0 ? i0 : (c1 ? i : i1);
    u0 = c0 ? u : u0, i0 = c0 ? i : i0;
  }
}

// route_tail's lane_best: the lane's best untaken entry (largest p bits, lowest index on equal bits)
__device__ __forceinline__ void lane_best(const float (&p)[16], unsigned taken, unsigned& bp, int& bi) {
  bp = 0;
  bi = 16;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const unsigned u = __float_as_uint(p[i]);
    if (!((taken >> i) & 1) && (bi == 16 || u > bp)) { bp = u; bi = i; }
  }
}

// K-third partial layout in Args::rpart: entry e = lane + 32·i of row tok, third q at ((tok·3 + q)·4 + i/4)·128 +
// lane·4 + i%4 — the tail's lane reads its 16 entries of a third with four 16-byte loads
__host__ __device__ __forceinline__ int part_at(int tok, int q, int i, int lane) {
  return ((tok * 3 + q) * 4 + (i >> 2)) * 128 + lane * 4 + (i & 3);
}

// the routing tail (the last CTA, 16 warps): logits from the K-third partials, softmax, top-10, renormalise; plan
__device__ void tail(const Args& a) {
  __shared__ int tk_i[kMaxM][kTopK];
  __shared__ float tk_w[kMaxM][kTopK];
  __shared__ unsigned mlo[kExperts], mhi[kExperts];  // bit t: row t (mlo) / row 32 + t (mhi) selected expert e
  __shared__ int off_e[kExperts];
  __shared__ int wsum[kThreads / 32][3];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, M = a.M;
  constexpr int NW = kThreads / 32;
  Plan* P = a.plan;
  mlo[tid] = 0, mhi[tid] = 0;
  float4 nx[3][4];  // the warp's next row: K-third partials of its 16 entries (entries i = 4j … 4j+3 in nx[·][j])
  auto fetch = [&](int tok) {
#pragma unroll
    for (int q = 0; q < 3; q++)
#pragma unroll
      for (int j = 0; j < 4; j++) nx[q][j] = __ldcg(reinterpret_cast<const float4*>(a.rpart) + part_at(tok, q, 4 * j, lane) / 4);
  };
  if (warp < M) fetch(warp);
  __syncthreads();  // masks cleared
  for (int tok = warp; tok < M; tok += NW) {
    float p[16];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; i++) {  // logits = bf16((p0 + p1) + p2), as k_route's last K third
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < 3; q++) s += reinterpret_cast<const float*>(&nx[q][i >> 2])[i & 3];
      const bf16 lg = __float2bfloat16_rn(s);
      a.logits[size_t(tok) * kExperts + lane + 32 * i] = lg;
      p[i] = __bfloat162float(lg);
      mx = fmaxf(mx, p[i]);
    }
    if (tok + NW < M) fetch(tok + NW);
    // softmax: route_tail's operations in route_tail's order
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
    // top-10: route_tail's rounds (largest lane best, then the lowest id among equal bits); the lane best (bp, bi) is
    // the head of the lane's top-3 list; a lane that gave its three heads takes route_tail's lane_best over its
    // untaken entries (rare: a lane holds 16 of the 512)
    unsigned bp, u1, u2, taken = 0;
    int bi, i1, i2, nv = 3;
    top3(p, bp, bi, u1, i1, u2, i2);
    float tot = 0.f, wk = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; k++) {
      const unsigned best = __reduce_max_sync(~0u, bp);
      bool w = bp == best;
      const unsigned eq = __ballot_sync(~0u, w);
      if (eq & (eq - 1))  // several lanes hold the largest bits: the lowest id, as route_tail
        w = lane == int(__reduce_min_sync(~0u, w ? unsigned(lane + 32 * bi) : 0xffffffffu) & 31);
      if (w) {
        tk_i[tok][k] = lane + 32 * bi;
        taken |= 1u << bi;
        bp = u1, bi = i1, u1 = u2, i1 = i2;
        --nv;
      }
      if (k >= 2 && k + 1 < kTopK && __any_sync(~0u, nv == 0))
        if (nv == 0) {
          lane_best(p, taken, bp, bi);
          nv = 1;
        }
      if (lane == k) wk = __uint_as_float(best);
      tot += __uint_as_float(best);
    }
    __syncwarp();
    if (lane < kTopK) {
      const int e = tk_i[tok][lane];
      tk_w[tok][lane] = wk / tot;
      if (tok < 32) atomicOr(&mlo[e], 1u << tok);
      else atomicOr(&mhi[e], 1u << (tok - 32));
    }
  }
  __syncthreads();
  // route_tail's plan: expert slots in id order (thread t owns expert t), exclusive scan of (used, rows, token tiles)
  const unsigned long long m = mlo[tid] | (unsigned long long)mhi[tid] << 32;
  const int f = m != 0, c = __popcll(m), t8 = (c + 7) >> 3;
  int sf = f, sc = c, st = t8;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int xf = __shfl_up_sync(~0u, sf, o), xc = __shfl_up_sync(~0u, sc, o), xt = __shfl_up_sync(~0u, st, o);
    if (lane >= o) { sf += xf; sc += xc; st += xt; }
  }
  if (lane == 31) { wsum[warp][0] = sf; wsum[warp][1] = sc; wsum[warp][2] = st; }
  __syncthreads();
  int slot = sf - f, off = sc - c, tt = st - t8;
  for (int w = 0; w < warp; w++) { slot += wsum[w][0]; off += wsum[w][1]; tt += wsum[w][2]; }
  off_e[tid] = off;
  if (f) {
    P->uexp[slot] = tid;
    P->ucnt[slot] = c;
    P->uoff[slot] = off;
    for (int j = 0; j < t8; j++) {
      P->tt_u[tt + j] = slot;
      P->tt_t[tt + j] = j;
    }
  }
  if (tid == kThreads - 1) {
    P->U = slot + f;
    P->NT = tt + t8;
  }
  __syncthreads();
  // pairs: (row, k) → slot position of the row inside its expert; row's pairs sorted by expert id
  for (int i = tid; i < M * kTopK; i += kThreads) {
    const int tok = i / kTopK, k = i % kTopK, e = tk_i[tok][k];
    const unsigned long long me = mlo[e] | (unsigned long long)mhi[e] << 32;
    const int pair = off_e[e] + __popcll(me & ((1ull << tok) - 1));
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
  __syncthreads();
  if (tid == 0) {  // release: the plan (cumulative through the barrier) before plan_ready
    fence();
    a.sy->rdone = 0;
    atomicExch(&a.sy->plan_ready, 1u);
  }
}

// k_route's shared expert gate of row tok (one warp): bf16(x·wg) → bf16(sigmoid), counted in sig_done
__device__ __forceinline__ void shared_gate(const Args& a, int tok, int lane) {
  if (tok < a.M) {
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
  __syncwarp();
  if (lane == 0) {  // release: sig (lane 0's store) before the count
    fence();
    atomicAdd(&a.sy->sig_done, 1u);
  }
}

}  // namespace p022

// k_route22: 48 CTAs of 16 warps; warps 8h … 8h+7 do the work of k_route's block vb = 2·blockIdx.x + h (same K third,
// same k-steps per warp, same exception span vb·8 + warp). Dynamic shared memory: the 8-warp partial sums, [2][8][NT][32].
template <int NT, bool P17>
__global__ void __launch_bounds__(p022::kThreads, 1) k_route22(Args a) {
  extern __shared__ float4 red22[];
  __shared__ int last;
  const int half = threadIdx.x >> 8, warp = (threadIdx.x >> 5) & 7, lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
  const int vb = 2 * blockIdx.x + half;  // k_route's block
  const int M = a.M, nt = (M + 7) >> 3, tile = vb / 3, part = vb % 3;
  const int nks = part < 2 ? 7 : 6, ks0 = part * 56 + warp * nks;
  const uint4* A = a.rt + (size_t(tile) * KS1 + ks0) * 32 + lane;
  uint4 w[7];
  [[maybe_unused]] p017::Span sp;
  if constexpr (P17) p017::load(a.rp, a.rpb, a.rpx, tile, vb * 8 + warp, ks0, nks, lane, sp);  // before the wait
  else
#pragma unroll
    for (int s = 0; s < 7; s++) w[s] = s < nks ? ld_stream(A + s * 32) : make_uint4(0, 0, 0, 0);  // before the wait
  pdl_wait();
  pdl_trigger();
  // v → B-fragment order for the routed up items
  for (int i = blockIdx.x * p022::kThreads + threadIdx.x; i < M * (H / 8); i += p022::kCtas * p022::kThreads) {
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
  float4* red = red22 + (half * 8) * NT * 32;  // this half's [8][NT][32]
#pragma unroll
  for (int t = 0; t < NT; t++)
    if (t < nt) red[(warp * NT + t) * 32 + lane] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
  __syncthreads();
  if (warp < nt) {  // warp t adds the 8 K-slices of token tile t in warp order → fp32 partial (P022 layout)
    const int t = warp;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w8 = 0; w8 < 8; w8++) {
      const float4 x = red[(w8 * NT + t) * 32 + lane];
      c[0] += x.x, c[1] += x.y, c[2] += x.z, c[3] += x.w;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int tok = 8 * t + 2 * q + (j & 1), e = 16 * tile + r + 8 * (j >> 1);
      if (tok < M) a.rpart[p022::part_at(tok, part, e >> 5, e & 31)] = c[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {  // release the partials (cumulative through the barrier), count the CTA; the last acquires
    p022::fence();
    last = atomicAdd(&a.sy->rdone, 1u) == unsigned(p022::kCtas - 1);  // one stage, 48 CTAs
    if (last) p022::fence();
  }
  __syncthreads();
  if (last) p022::tail(a);
  if (warp == 0) p022::shared_gate(a, vb, lane);  // warps 0 and 8: rows vb (k_route: warp 0 of block vb)
}

// k_experts with the routed gate/up items ahead of the shared down items: queue = shared gate/up (nt × 200) | routed up
// (NT × 200) | shared down (nt × 160) | routed down (NT × 160). k_experts' order parks most warps on shared down items
// until every shared gate/up tile is done (h_done), so an early plan_ready finds few warps to start the routed stream;
// here they wait for plan_ready on routed up items instead. Items, their math and every cross-item sum order (K parts,
// expert-id order, integer contribution counts) are k_experts' — only the order warps take them in differs. An item
// still waits only on k_route22's flags or on earlier items (shared down: h_done of the shared gate/up items, routed
// down: uready of the routed up items), so the queue cannot deadlock.
template <bool P14>
__global__ void __launch_bounds__(128) k_experts22(Args a) {
  const Plan* P = a.plan;
  const int lane = threadIdx.x & 31, nt = (a.M + 7) >> 3;
  constexpr int UPI = T13 * KP;
  const int SGU = nt * UPI;
  int NT = -1, RU = 0, SD = 0, total = 0x7fffffff;
  for (;;) {
    int item = 0;
    if (lane == 0) item = int(atomicAdd(&a.sy->queue, 1u));
    item = __shfl_sync(~0u, item, 0);
    if (item >= SGU && NT < 0) {
      wait_ge(&a.sy->plan_ready, 1u, lane);
      NT = __ldcg(&P->NT);
      RU = SGU + NT * UPI, SD = RU + nt * T2, total = SD + NT * T2;
    }
    if (item >= total) break;
    if (item < SGU) {
      const int x = item / nt;
      sgu_item(a, x / KP, x % KP, item % nt, lane);
    } else if (item < RU) {
      const int x = item - SGU, T = x / UPI, y = x % UPI;
      up_item<P14>(a, P, T, y / KP, y % KP, lane);
    } else if (item < SD) {
      const int x = item - RU;
      sdown_item(a, P, x / nt, x % nt, lane);
    } else {
      const int x = item - SD;
      down_item<P14>(a, P, x / T2, x % T2, lane);
    }
  }
  // last warp out resets the per-call flags and counters (as k_experts)
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

namespace p022 {
template <bool P14>
int expert_blocks22() {
  static int nb = 0;
  if (!nb) {
    int dev, sms, per;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_experts22<P14>, 128, 0));
    nb = sms * (per > 0 ? per : 1);
  }
  return nb;
}

// k_route22<NT, P17> as a programmatic dependent (as k_route), 48 × 512, the partial sums in dynamic smem
template <int NT, bool P17>
void launch_route(const Args& a, cudaStream_t s) {
  constexpr int smem = 2 * 8 * NT * 32 * int(sizeof(float4));
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_route22<NT, P17>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = kCtas, cfg.blockDim = kThreads, cfg.dynamicSmemBytes = smem, cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at, cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_route22<NT, P17>, a));
}

// the decode MoE call with the patch on: k_route22, then k_experts22 as its programmatic dependent (P017 / P014 as
// set in Args, as moe.cu's launch)
template <int NT>
void launch(const Args& a, cudaStream_t s) {
  if (a.rp) launch_route<NT, true>(a, s);
  else launch_route<NT, false>(a, s);
  if (a.sc4) launch_pdl(k_experts22<true>, expert_blocks22<true>(), 128, s, a);
  else launch_pdl(k_experts22<false>, expert_blocks22<false>(), 128, s, a);
}
}  // namespace p022
