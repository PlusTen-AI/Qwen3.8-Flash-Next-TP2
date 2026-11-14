// p005_gdn_fast_prefill.cuh — P005 gdn_fast_prefill (patches.def): warp-specialised variant of gdn_core_kernel for
// long sequences. Included by gdn.cu inside its anonymous namespace (uses its constants and helpers).
//
// Same grid (8 k-heads × 3·NC, B), cluster (3 v-heads × NC = 2 V-halves of 64 rows), same per-element arithmetic and
// reduction orders as gdn_core_kernel → core, o, S and conv state are bit-identical. Only the schedule changes:
//   • Recurrence warps hold S [64,128] fp32 in registers and run the delta rule token after token; they meet the
//     producer warps only at chunk boundaries (named barriers), never the cluster.
//   • Producer warps run one chunk ahead and one behind: cp.async of the raw rows two chunks ahead, conv1d +
//     silu, g/β and l2norm of the next chunk into a double-buffered stage; for the previous chunk the Σo² partial of
//     this V-half goes to the peer CTA by st.async (DSMEM, mbarrier tx count — no cluster barrier per chunk), then
//     the gated RMSNorm and the core stores.
//   • One cluster barrier at the start: mbarrier inits visible, and every CTA has read the old q|k conv state before
//     cluster rank 0 overwrites it at the end.
// Cheaper but bit-identical forms of the base operations (checked on GB10, see p5_div_b1 / p5_div_pos / p5_sqrt):
// div.rn / sqrt.rn fast paths without their slow-path call (rare operands fall back to `/` / sqrtf), and paired
// cvt.rn.bf16x2.f32 conversions. Smem ~96 KB → one CTA per SM; 8 clusters of 6 fit on the 48 SMs (B = 1 in one wave).

constexpr int kP5RW = 4, kP5PW = 8;                         // recurrence | producer warps (one | two per SMSP)
constexpr int kP5Rec = 32 * kP5RW, kP5Prod = 32 * kP5PW, kP5Threads = kP5Rec + kP5Prod;
constexpr int kP5R = 128 / kNC, kP5CH = 256 + kP5R, kP5RM = kP5R / (8 * kP5RW);  // rows per lane
constexpr int kP5G = kTC * kP5CH / (5 * kP5Prod);           // conv: tokens per item (5 items per producer thread)
static_assert(kNC == 2 && kP5RM * 8 * kP5RW == kP5R && kTC % kP5G == 0 && kTC % kP5PW == 0, "P005 split");
// named barriers (0 = __syncthreads, unused after the role split)
constexpr int kBarFull = 1;   // +stage: producer → recurrence (stage inputs ready, os[stage] free)
constexpr int kBarDone = 3;   // +stage: recurrence → producer (chunk done: os[stage] filled, inputs consumed)
constexpr int kBarProd = 5;   // producer warps only

struct P5Smem {
  bf16 xs[3][kTC + 3][kP5CH];  // raw x rows t0−3 … t0+n−1 of the CTA's channels (q | k | v rows), ring of 3
  bf16 bas[3][kTC][48];        // b | a rows, ring of 3
  bf16 zs[4][kTC][kP5R];       // z rows, ring of 4 (the epilogue reads them one chunk after the conv)
  float qs[2][kTC][128], ks[2][kTC][128], vs[2][kTC][kP5R];  // stage: conv+silu (+l2norm) outputs
  float eg[2][kTC], beta[2][kTC];
  bf16 os[2][kTC][kP5R];       // stage: bf16 o of the recurrence
  bf16 cws[kP5CH][4], nws[kP5R];
  float adt[2][24];            // A, dt_bias
  float part[kTC], rstd[kTC];  // this V-half's Σo², 1/rms
  float pin[2][kTC];           // the peer V-half's Σo² (st.async from the peer), ring of 2
  unsigned long long pbar[2];  // pin[s] landed (tx bytes)
};

__device__ __forceinline__ uint32_t p5_sa(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void p5_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void p5_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ uint32_t p5_pk(float a, float b) {  // two bf16 (same RN conversion as base), a low
  return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
         (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
}
__device__ __forceinline__ uint32_t p5_mapa(uint32_t a, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void p5_expect(unsigned long long* b, uint32_t bytes) {  // arrive + expect tx bytes
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(p5_sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void p5_wait(const unsigned long long* b, uint32_t parity) {
  uint32_t done;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
        " selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(p5_sa(b)), "r"(parity)
        : "memory");
  } while (!done);
}
__device__ __forceinline__ void p5_st_async(uint32_t raddr, float v, uint32_t rbar) {  // remote store + tx on its bar
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(raddr),
               "r"(__float_as_uint(v)), "r"(rbar)
               : "memory");
}

// a / b (IEEE RN: the same bits as the `/` operator) for b ≥ 1 or b = +inf (b = 1 + e^x), without div.rn.f32's
// slow-path call, so independent divisions schedule together: div.rn's own fast path (MUFU.RCP + 5 FFMA, as ptxas
// emits it) on operands scaled by 2^-64 when b > 2^64 (exact; keeps 1/b normal), special operands selected.
// Checked bit-exact against `/` for all 2^32 fp32 a with b = 1 + __expf(-a) (conv silu) and for all 65536 bf16 z with
// a = 1, b = 1 + expf(-z) (sigmoid), NaN payloads included.
__device__ __forceinline__ float p5_div_b1(float a, float b) {
  const float inf = __int_as_float(0x7f800000);
  const float sc = b > 0x1p64f ? 0x1p-64f : 1.f, bs = b * sc, as = a * sc;
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(bs));
  const float e = fmaf(-bs, r, 1.f), r1 = fmaf(r, e, r), q = as * r1, rem = fmaf(-bs, q, as);
  float y = fmaf(r1, rem, q);
  const bool ainf = fabsf(a) == inf;
  y = (a == 0.f || ainf) ? a : y;                                              // ±0 / b, ±inf / b (b finite)
  y = b == inf ? (ainf ? __int_as_float(0x7fffffff) : copysignf(0.f, a)) : y;  // a / inf
  return y;
}
// a / b for b > 0: div.rn's fast path; ok = false outside 2^-64 ≤ b ≤ 2^64, |a| ≤ 2^64, |a| ≥ 2^-64 or a = 0
// (caller then uses `/`). Inside that range the fast path is exact (Markstein; no under/overflow) — also checked
// against `/` on 1.3·10^10 random operand pairs.
__device__ __forceinline__ float p5_div_pos(float a, float b, bool& ok) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(b));
  const float e = fmaf(-b, r, 1.f), r1 = fmaf(r, e, r), q = a * r1, rem = fmaf(-b, q, a);
  const float y = fmaf(r1, rem, q), aa = fabsf(a);
  ok &= b >= 0x1p-64f && b <= 0x1p64f && aa <= 0x1p64f && (aa >= 0x1p-64f || a == 0.f);
  return a == 0.f ? a : y;
}
// sqrtf(x): sqrt.rn's fast path as ptxas emits it (MUFU.RSQ, FMUL.FTZ ×2, 2 FFMA); ok = false where sqrt.rn takes
// its slow path (x < 2^-101, negative, inf/NaN). Checked equal to sqrtf for all 2^32 inputs in range.
__device__ __forceinline__ float p5_sqrt(float x, bool& ok) {
  float r, y, hh;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(r));
  asm("mul.ftz.f32 %0, %1, 0f3F000000;" : "=f"(hh) : "f"(r));
  ok &= __float_as_uint(x) - 0x0d000000u <= 0x727fffffu;
  return fmaf(fmaf(-y, y, x), hh, y);
}
// bf16 rounding of two values at once (cvt.rn.bf16x2.f32: same bits as two cvt.rn.bf16.f32, checked for all 2^32
// inputs; half the conversion-pipe cost)
__device__ __forceinline__ void bfr2(float& x, float& y) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(x, y);
  x = __low2float(v), y = __high2float(v);
}

// conv1d (k=4) + silu of chunk j (n tokens) into stage s: items = (channel, group of kP5G tokens), 5 per thread.
// Base arithmetic: products rounded to bf16, ((p0 + p1) + p2) + p3 in fp32, y = bf16(acc / (1 + e^-acc)).
template <bool Full>
__device__ __forceinline__ void p5_conv(P5Smem& sm, int pt, int bx, int s, int n) {
#pragma unroll
  for (int u = 0; u < 5; u++) {
    const int item = pt + kP5Prod * u, c = item % kP5CH, i0 = (item / kP5CH) * kP5G;  // warp-uniform group, segment
    if (!Full && i0 >= n) continue;
    const int nt = Full ? kP5G : min(n - i0, kP5G);
    float* dst = c < 128 ? &sm.qs[s][i0][c] : c < 256 ? &sm.ks[s][i0][c - 128] : &sm.vs[s][i0][c - 256];
    const int ds = c < 256 ? 128 : kP5R;
    const uint2 wv = *reinterpret_cast<const uint2*>(sm.cws[c]);  // the 4 taps (bf16) in one load
    const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
    const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
    float x[kP5G + 3], y[kP5G];
#pragma unroll
    for (int r = 0; r < kP5G + 3; r++) x[r] = (Full || r < nt + 3) ? bf(sm.xs[bx][i0 + r][c]) : 0.f;
#pragma unroll
    for (int t = 0; t < kP5G; t++) {
      float p0 = w0 * x[t], p1 = w1 * x[t + 1], p2 = w2 * x[t + 2], p3 = w3 * x[t + 3];
      bfr2(p0, p1), bfr2(p2, p3);
      float acc = p0;
      acc += p1;
      acc += p2;
      acc += p3;
      y[t] = p5_div_b1(acc, 1.f + __expf(-acc));
    }
#pragma unroll
    for (int t = 0; t < kP5G; t += 2) bfr2(y[t], y[t + 1]);
#pragma unroll
    for (int t = 0; t < kP5G; t++)
      if (Full || t < nt) dst[t * ds] = y[t];
  }
}

__device__ __forceinline__ float row_sum4(float v) {  // over the 4 lanes of a row (steps 2, 3 of the base's row_sum8)
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  return v + __shfl_xor_sync(0xffffffffu, v, 16);
}

__global__ void __launch_bounds__(kP5Threads, 1)
    gdn_core_p005_kernel(const bf16* __restrict__ qkvz, const bf16* __restrict__ ba, bf16* __restrict__ core,
                         bf16* __restrict__ o_out, GdnState* const* __restrict__ states,
                         const int* __restrict__ seq_start, GdnW w) {
  constexpr int R = kP5R, CH = kP5CH;
  extern __shared__ __align__(16) unsigned char p5_raw[];
  P5Smem& sm = *reinterpret_cast<P5Smem*>(p5_raw);

  cg::cluster_group cl = cg::this_cluster();
  const int cr = int(cl.block_rank()), hk = blockIdx.x / (3 * kNC), hl = cr / kNC, half = cr % kNC, v0 = half * R;
  const int h = hk * 3 + hl, b = blockIdx.y, tid = threadIdx.x;
  const int row0 = seq_start[b], T = seq_start[b + 1] - row0;
  if (T <= 0) return;  // uniform over the cluster: no barrier is reached
  const int nch = (T + kTC - 1) / kTC;
  auto nrows = [&](int j) { return min(kTC, T - j * kTC); };
  const GdnState st = *states[b];
  auto chan = [&](int c) {  // local conv channel → channel of the rank's q|k|v (as in the base kernel)
    return c < 128 ? hk * 128 + c : c < 256 ? 1024 + hk * 128 + (c - 128) : kV0 + h * 128 + v0 + (c - 256);
  };
  const int pt = tid - kP5Rec, pw = pt >> 5;  // producer thread / warp (valid if tid ≥ kP5Rec)
  auto fetch = [&](int j) {  // all producer threads: cp.async of chunk j's raw rows (slots j % 3, z: j % 4)
    constexpr int XC = CH / 8, ZC = R / 8;
    const int t0 = j * kTC, n = nrows(j), bx = j % 3, bz = j % 4, nx = (n + 3) * XC, nz = n * ZC;
    for (int i = pt; i < nx + nz + n * 6; i += kP5Prod) {
      if (i < nx) {
        const int r = i / XC, c = (i % XC) * 8, t = t0 - 3 + r;
        cp16(&sm.xs[bx][r][c], (t < 0 ? st.conv + (3 + t) * kQKV : qkvz + size_t(row0 + t) * kQKVZ) + chan(c));
      } else if (i < nx + nz) {
        const int r = (i - nx) / ZC, c = ((i - nx) % ZC) * 8;
        cp16(&sm.zs[bz][r][c], qkvz + size_t(row0 + t0 + r) * kQKVZ + kZ0 + h * 128 + v0 + c);
      } else {
        const int r = (i - nx - nz) / 6, c = ((i - nx - nz) % 6) * 8;
        cp16(&sm.bas[bx][r][c], ba + size_t(row0 + t0 + r) * 48 + c);
      }
    }
  };

  if (tid == 0) {
    for (int s = 0; s < 2; s++)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(p5_sa(&sm.pbar[s])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int s = 0; s < 2 && s < nch; s++) p5_expect(&sm.pbar[s], uint32_t(nrows(s) * 4));  // peer partials of 0, 1
  }
  __syncthreads();
  if (tid >= kP5Rec) {
    for (int i = pt; i < CH / 2 + R / 8 + 12; i += kP5Prod) {  // weights: conv | norm | A | dt_bias
      if (i < CH / 2) cp16(&sm.cws[2 * i][0], w.conv_w + size_t(chan(2 * i)) * 4);
      else if (i < CH / 2 + R / 8) cp16(&sm.nws[(i - CH / 2) * 8], w.norm_w + v0 + (i - CH / 2) * 8);
      else if (i < CH / 2 + R / 8 + 6) cp16(&sm.adt[0][(i - CH / 2 - R / 8) * 4], w.A + (i - CH / 2 - R / 8) * 4);
      else cp16(&sm.adt[1][(i - CH / 2 - R / 8 - 6) * 4], w.dt_bias + (i - CH / 2 - R / 8 - 6) * 4);
    }
    fetch(0);
    cp_commit();
    if (nch > 1) fetch(1);
    cp_commit();
    cp_wait();  // chunk 0 (incl. the old conv state rows) has landed before the cluster barrier
  }
  __syncthreads();
  cl_arrive(), cl_wait();  // mbarrier inits visible cluster-wide; all old q|k conv state reads done

  if (tid < kP5Rec) {
    // ---------------- recurrence warps: S [64,128] in registers, delta rule per token ----------------
    // lane = (a, g) = (lane / 8, lane % 8): rows wp·8·RM + g + 8m (m < RM); a row's 128 columns over 4 lanes, lane a
    // holding the base kernel's column groups p = 2a, 2a+1 (float4 columns p + 8j, j < 4). Each group's partial v_p is
    // formed exactly as in the base kernel; v_2a + v_2a+1 is the base's first butterfly step (in-thread), row_sum4
    // (xor 8, xor 16 = a^1, a^2) its other two. a in the high lane bits: k/q float4 broadcasts cost half (2 vs 4 clk).
    constexpr int RM = kP5RM;
    const int wp = tid >> 5, lane = tid & 31, g = lane & 7, a = lane >> 3, rw = wp * 8 * RM + g;
    float* Sg = st.S + (size_t(h) * 128 + v0 + rw) * 128 + a * 8;
    float4 S[RM][2][4];  // [m][pp][j] = row rw + 8m, columns 4(2a+pp) + 32j … +3
#pragma unroll
    for (int m = 0; m < RM; m++)
#pragma unroll
      for (int pp = 0; pp < 2; pp++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          S[m][pp][j] = __ldcs(reinterpret_cast<const float4*>(Sg + m * 1024 + pp * 4 + j * 32));
    for (int c = 0; c < nch; c++) {
      const int s = c & 1, n = nrows(c);
      p5_sync(kBarFull + s, kP5Threads);
#pragma unroll 2
      for (int i = 0; i < n; i++) {  // S ← S·e^g; δ = β(v − S k); S ← S + δ kᵀ; o = S q   (base arithmetic)
        float4 k[2][4], q[2][4];
#pragma unroll
        for (int pp = 0; pp < 2; pp++)
#pragma unroll
          for (int j = 0; j < 4; j++) {
            k[pp][j] = reinterpret_cast<const float4*>(sm.ks[s][i])[2 * a + pp + 8 * j];
            q[pp][j] = reinterpret_cast<const float4*>(sm.qs[s][i])[2 * a + pp + 8 * j];
          }
        const float eg = sm.eg[s][i], beta = sm.beta[s][i];
        float d[RM], vp[2];
#pragma unroll
        for (int m = 0; m < RM; m++) {
#pragma unroll
          for (int pp = 0; pp < 2; pp++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
              float4& x = S[m][pp][j];
              x.x *= eg, x.y *= eg, x.z *= eg, x.w *= eg;
            }
            vp[pp] = (dot4(S[m][pp][0], k[pp][0]) + dot4(S[m][pp][1], k[pp][1])) +
                     (dot4(S[m][pp][2], k[pp][2]) + dot4(S[m][pp][3], k[pp][3]));
          }
          d[m] = row_sum4(vp[0] + vp[1]);
        }
#pragma unroll
        for (int m = 0; m < RM; m++) {
          const float dl = (sm.vs[s][i][rw + 8 * m] - d[m]) * beta;
#pragma unroll
          for (int pp = 0; pp < 2; pp++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
              float4& x = S[m][pp][j];
              x.x = fmaf(dl, k[pp][j].x, x.x), x.y = fmaf(dl, k[pp][j].y, x.y);
              x.z = fmaf(dl, k[pp][j].z, x.z), x.w = fmaf(dl, k[pp][j].w, x.w);
            }
            vp[pp] = (dot4(S[m][pp][0], q[pp][0]) + dot4(S[m][pp][1], q[pp][1])) +
                     (dot4(S[m][pp][2], q[pp][2]) + dot4(S[m][pp][3], q[pp][3]));
          }
          d[m] = row_sum4(vp[0] + vp[1]);
        }
        if (a == 0) {
#pragma unroll
          for (int m = 0; m < RM; m++) sm.os[s][i][rw + 8 * m] = __float2bfloat16_rn(d[m]);
        }
      }
      p5_arrive(kBarDone + s, kP5Threads);
    }
#pragma unroll
    for (int m = 0; m < RM; m++)
#pragma unroll
      for (int pp = 0; pp < 2; pp++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          __stcs(reinterpret_cast<float4*>(Sg + m * 1024 + pp * 4 + j * 32), S[m][pp][j]);
    return;
  }

  // ---------------- producer warps ----------------
  const int lane = pt & 31;
  const uint32_t peer_pin = p5_mapa(p5_sa(&sm.pin[0][0]), cr ^ 1), peer_bar = p5_mapa(p5_sa(&sm.pbar[0]), cr ^ 1);
  auto epilogue = [&](int e) {  // chunk e: Σo² exchange, gated RMSNorm, core stores
    const int s = e & 1, n = nrows(e), t0 = e * kTC;
    p5_sync(kBarDone + s, kP5Threads);
#pragma unroll
    for (int u = 0; u < kTC / kP5PW; u++) {  // Σ bf16(o)² over this CTA's rows (base order: lane-strided, warp_sum)
      const int i = pw + u * kP5PW;
      if (i < n) {
        float a = 0.f;
#pragma unroll
        for (int r = lane; r < R; r += 32) {
          const float o = bf(sm.os[s][i][r]);
          a = fmaf(o, o, a);
        }
        a = warp_sum(a);
        if (lane == 0) {
          sm.part[i] = a;
          p5_st_async(peer_pin + uint32_t((s * kTC + i) * 4), a, peer_bar + uint32_t(s * 8));
        }
      }
    }
    p5_sync(kBarProd, kP5Prod);
    if (pt < n) {
      p5_wait(&sm.pbar[s], uint32_t(e >> 1) & 1u);
      const float mine = sm.part[pt], other = sm.pin[s][pt];
      float tot = 0.f;  // V-half 0 first, as the base kernel's loop over cluster ranks
      tot += half == 0 ? mine : other;
      tot += half == 0 ? other : mine;
      sm.rstd[pt] = rsqrtf(tot / 128.f + kEps);
    }
    p5_sync(kBarProd, kP5Prod);  // pin[s] read: re-arm it for chunk e + 2 (the peer sends that after our e + 1)
    if (pt == 0 && e + 2 < nch) p5_expect(&sm.pbar[s], uint32_t(nrows(e + 2) * 4));
    // gated RMSNorm: ((o·rstd)·w)·sigmoid(z), 4 elements per step → one 8-B store
    for (int e4 = pt; e4 < n * (R / 4); e4 += kP5Prod) {
      const int i = e4 / (R / 4), r0 = (e4 % (R / 4)) * 4;
      const size_t at = size_t(row0 + t0 + i) * kCore + h * 128 + v0 + r0;
      float y[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const float o = bf(sm.os[s][i][r0 + u]), z = bf(sm.zs[e % 4][i][r0 + u]);
        y[u] = o * sm.rstd[i] * bf(sm.nws[r0 + u]) * p5_div_b1(1.f, 1.f + expf(-z));  // = 1 / (1 + e^-z)
      }
      *reinterpret_cast<uint2*>(core + at) = make_uint2(p5_pk(y[0], y[1]), p5_pk(y[2], y[3]));
      if (o_out) *reinterpret_cast<uint2*>(o_out + at) = *reinterpret_cast<const uint2*>(&sm.os[s][i][r0]);
    }
  };

  for (int j = 0; j < nch; j++) {
    const int s = j & 1, bx = j % 3, n = nrows(j);
    p5_sync(kBarProd, kP5Prod);  // slots of chunk j−1 (x, b|a) / j−2 (z) no longer read by any producer thread
    if (j + 2 < nch) fetch(j + 2);
    cp_commit();  // one group per iteration (maybe empty): chunk j's group is the third newest
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    p5_sync(kBarProd, kP5Prod);  // chunk j's raw rows visible to every producer thread
    if (n == kTC) p5_conv<true>(sm, pt, bx, s, n);
    else p5_conv<false>(sm, pt, bx, s, n);
    if (pt < n) {  // e^g with g = A·softplus(a + dt_bias), A = −exp(A_log); β = sigmoid(b)
      const float xa = bf(sm.bas[bx][pt][24 + h]) + sm.adt[1][h];
      const float sp = xa <= 20.f ? logf(1.f + expf(xa)) : xa;
      sm.eg[s][pt] = expf(sm.adt[0][h] * sp);
      sm.beta[s][pt] = 1.f / (1.f + expf(-bf(sm.bas[bx][pt][h])));
    }
    p5_sync(kBarProd, kP5Prod);
    {  // l2norm (eps inside sqrt), q·128^-½ — base expressions; fast div/sqrt paths, `/` and sqrtf if out of range
      constexpr int U = kTC / kP5PW;
      float4 qv[U], kv[U], qn[U], kn[U];
      bool ok = true;
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int i = min(pw + u * kP5PW, n - 1);  // (a duplicate row past n is computed, not stored)
        qv[u] = reinterpret_cast<const float4*>(sm.qs[s][i])[lane];
        kv[u] = reinterpret_cast<const float4*>(sm.ks[s][i])[lane];
        const float nq = p5_sqrt(warp_sum(dot4(qv[u], qv[u])) + 1e-6f, ok);
        const float nk = p5_sqrt(warp_sum(dot4(kv[u], kv[u])) + 1e-6f, ok);
        qn[u].x = p5_div_pos(qv[u].x, nq, ok) * kQScale, qn[u].y = p5_div_pos(qv[u].y, nq, ok) * kQScale;
        qn[u].z = p5_div_pos(qv[u].z, nq, ok) * kQScale, qn[u].w = p5_div_pos(qv[u].w, nq, ok) * kQScale;
        kn[u].x = p5_div_pos(kv[u].x, nk, ok), kn[u].y = p5_div_pos(kv[u].y, nk, ok);
        kn[u].z = p5_div_pos(kv[u].z, nk, ok), kn[u].w = p5_div_pos(kv[u].w, nk, ok);
      }
      if (__any_sync(0xffffffffu, !ok)) {  // rare operands: the base expressions verbatim
#pragma unroll
        for (int u = 0; u < U; u++) {
          float4 q = qv[u], k = kv[u];
          const float nq = sqrtf(warp_sum(dot4(q, q)) + 1e-6f), nk = sqrtf(warp_sum(dot4(k, k)) + 1e-6f);
          q.x = q.x / nq * kQScale, q.y = q.y / nq * kQScale, q.z = q.z / nq * kQScale, q.w = q.w / nq * kQScale;
          k.x /= nk, k.y /= nk, k.z /= nk, k.w /= nk;
          qn[u] = q, kn[u] = k;
        }
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int i = pw + u * kP5PW;
        if (i < n) {
          reinterpret_cast<float4*>(sm.qs[s][i])[lane] = qn[u];
          reinterpret_cast<float4*>(sm.ks[s][i])[lane] = kn[u];
        }
      }
    }
    p5_arrive(kBarFull + s, kP5Threads);
    if (j) epilogue(j - 1);
  }
  epilogue(nch - 1);
  // new conv state = x(T−3 … T−1) = rows n … n+2 of the last chunk (q|k by cluster rank 0, v rows by their owner)
  {
    const int bx = (nch - 1) % 3, n = nrows(nch - 1);
    for (int i = pt; i < 3 * (CH / 8); i += kP5Prod) {
      const int s = i / (CH / 8), c = (i % (CH / 8)) * 8;
      if (c >= 256 || cr == 0)
        *reinterpret_cast<uint4*>(st.conv + s * kQKV + chan(c)) = *reinterpret_cast<const uint4*>(&sm.xs[bx][n + s][c]);
    }
  }
}

void gdn_p005_prepare() {  // > 48 KB dynamic smem opt-in; from gdn_upload (outside any graph capture), once
  static bool done = false;
  if (!done) {
    CUDA_CHECK(cudaFuncSetAttribute(gdn_core_p005_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(sizeof(P5Smem))));
    done = true;
  }
}

void gdn_core_p005(const GdnW& w, const bf16* qkvz, const bf16* ba, bf16* core, bf16* o, GdnState* const* states,
                   const int* seq_start, int B, cudaStream_t s) {
  gdn_p005_prepare();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(8 * 3 * kNC, B);
  cfg.blockDim = dim3(kP5Threads);
  cfg.dynamicSmemBytes = sizeof(P5Smem);
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 3 * kNC, at[0].val.clusterDim.y = 1, at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, gdn_core_p005_kernel, qkvz, ba, core, o, states, seq_start, w));
}
