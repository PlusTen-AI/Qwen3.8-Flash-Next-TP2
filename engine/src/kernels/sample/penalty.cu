// penalty.cu — vLLM's repetition / frequency / presence penalties on fp32 logits rows, and the per-sequence seen-token
// state they read (contract: sample.h, "penalties").
//
// penalize_rows: grid (G, B), one sequence per grid row, its seen list split over G CTAs. A thread reads a listed id
// (cnt ≠ 0) and its cnt once and applies them to every row of the sequence (row i: plus the extras d_1 … d_i equal to
// it), the rows' logits loaded four at a time (independent loads in flight; the work is a gather of scattered logits,
// so latency, not bandwidth, decides). CTA 0 also takes the extra tokens that are not listed (cnt = 0), each at its first
// occurrence, in the rows after it. Each logit is written by exactly one thread, and its value depends only on itself,
// its count and prompt bit — never on the list order, G or the other sequences.
// The state: ids grows by one entry the first time a token's cnt becomes nonzero (the atomic that sees the old value 0
// appends it), so ids holds exactly the tokens with cnt ≠ 0, each once.
#include <algorithm>

#include "core/log.h"
#include "kernels/sample/sample.h"

namespace q38 {
namespace {

constexpr int kPT = 256;                    // threads per CTA
constexpr uint32_t kPrompt = 0x80000000u;   // cnt bit 31

// One logit of a row of sequence q: n generated occurrences (with the row's extras), pr = in the prompt.
__device__ __forceinline__ float pen(float l, uint32_t n, bool pr, const PenSeq& q) {
  // vLLM 0.32 (v1/worker/gpu/sample/penalties.py): logits *= where(logits > 0, 1.0 / scale, scale) — a product with the
  // correctly rounded reciprocal, not a division
  if (pr || n) l = l > 0.f ? __fmul_rn(l, __frcp_rn(q.rep)) : __fmul_rn(l, q.rep);
  if (n) l = __fsub_rn(__fsub_rn(l, __fmul_rn(q.freq, __uint2float_rn(n))), q.pres);
  return l;
}

// Token v (at rows[i·ld + v]) in rows i0 … nr − 1 of a sequence: c = its cnt; row i adds the extras xs[u] == v, u < i
// (u < nx). Loads four rows at a time before any store.
__device__ __forceinline__ void pen_rows(float* rows, int ld, int v, int i0, int nr, uint32_t c, const int* xs, int nx,
                                         const PenSeq& q) {
  uint32_t ex = 0;
  for (int u = 0; u < i0 && u < nx; u++) ex += xs[u] == v;
  for (int i = i0; i < nr; i += 4) {
    float l[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (i + j < nr) l[j] = rows[size_t(i + j) * ld + v];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (i + j >= nr) break;
      if (i + j > i0 && i + j <= nx) ex += xs[i + j - 1] == v;
      l[j] = pen(l[j], (c & ~kPrompt) + ex, c >> 31, q);
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (i + j < nr) rows[size_t(i + j) * ld + v] = l[j];
  }
}

__global__ void __launch_bounds__(kPT) k_penalize(float* x, int ld, int off, int Vb, int V, int R, const int* p_start,
                                                  const int* q_start, const int* drafts, const PenSeq* seqs) {
  __shared__ int xs[kPenMaxX];
  const int b = blockIdx.y;
  const PenSeq q = seqs[b];
  if (!q.seen.cnt || (q.rep == 1.f && q.freq == 0.f && q.pres == 0.f)) return;
  const int r0 = p_start[b], nr = min(p_start[b + 1], R) - r0;
  if (nr <= 0) return;
  // row i's extras: xs[0 … min(i, k_b)) — at most nx = min(k_b, kPenMaxX, nr − 1) of them are ever read
  const int nx = q_start ? max(0, min(min(q_start[b + 1] - q_start[b], kPenMaxX), nr - 1)) : 0;
  if (int(threadIdx.x) < nx) xs[threadIdx.x] = drafts[q_start[b] + threadIdx.x];
  __syncthreads();
  float* rows = x + size_t(r0) * ld - off;  // rows[i·ld + v] for v ∈ [off, off + Vb)
  const uint32_t* cnt = q.seen.cnt;
  const int n = *q.seen.n;
  for (int e = blockIdx.x * kPT + threadIdx.x; e < n; e += gridDim.x * kPT) {  // listed ids
    const int v = q.seen.ids[e];
    if (v < off || v >= off + Vb) continue;
    pen_rows(rows, ld, v, 0, nr, cnt[v], xs, nx, q);
  }
  if (blockIdx.x == 0 && int(threadIdx.x) < nx) {  // extras not listed, from the row after their first occurrence
    const int j = threadIdx.x, v = xs[j];
    if (v < 0 || v >= V || v < off || v >= off + Vb || cnt[v] != 0) return;
    for (int u = 0; u < j; u++)
      if (xs[u] == v) return;
    pen_rows(rows, ld, v, j + 1, nr, 0, xs, nx, q);
  }
}

// token v of a sequence's state: cnt[v] |= or_bits, += add; listed the first time it becomes nonzero
__device__ __forceinline__ void seen_put(const PenSeen& s, int v, int V, uint32_t add) {
  if (v < 0 || v >= V) return;
  const uint32_t old = add ? atomicAdd(s.cnt + v, add) : atomicOr(s.cnt + v, kPrompt);
  if (old == 0) s.ids[atomicAdd(s.n, 1)] = v;
}

__global__ void k_pen_prompt(PenSeen s, const int* tok, int L, int V) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < L; i += gridDim.x * blockDim.x) seen_put(s, tok[i], V, 0);
}

// grid B, 32 threads: sequence b's tokens tok[t_start[b] + j] (j < its count), then last[b]
__global__ void k_pen_add(const PenSeq* seqs, const int* tok, const int* t_start, const int* n_tok, const int* last,
                          int V) {
  const int b = blockIdx.x;
  const PenSeen s = seqs[b].seen;
  if (!s.cnt) return;
  int nb = 0;
  if (tok) {
    const int span = t_start[b + 1] - t_start[b];
    nb = n_tok ? min(n_tok[b], span) : span;
    if (nb < 0) return;  // (spec_accept's −1 / kSparseDense: no tokens, not even last)
  }
  for (int j = threadIdx.x; j < nb; j += blockDim.x) seen_put(s, tok[t_start[b] + j], V, 1);
  if (last && threadIdx.x == 0) seen_put(s, last[b], V, 1);
}

// CTAs per sequence: ~6 resident threads' worth per SM over the batch (a list entry per thread or a few: the gather is
// latency-bound), at most 64.
int pen_ctas(int B) {
  static const int sms = [] {
    int n = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0));
    return n;
  }();
  return std::max(1, std::min(64, cdiv(6 * sms, B)));
}

}  // namespace

void penalize_rows(float* x, int ld, int off, int Vb, int V, int R, const int* p_start, const int* q_start,
                   const int* drafts, int B, const PenSeq* seqs, cudaStream_t s) {
  if (R <= 0 || B <= 0) return;
  if (Vb <= 0 || off < 0 || off + Vb > V || ld < Vb || B > 65535 || (q_start && !drafts))
    qlog::fail("penalty", "penalize_rows: off %d + Vb %d ≤ V %d, ld %d ≥ Vb, B %d ≤ 65535, drafts with q_start", off, Vb,
               V, ld, B);
  k_penalize<<<dim3(pen_ctas(B), B), kPT, 0, s>>>(x, ld, off, Vb, V, R, p_start, q_start, drafts, seqs);
  CUDA_CHECK(cudaGetLastError());
}

void penalty_reset(const PenSeen& st, int V, cudaStream_t s) {
  CUDA_CHECK(cudaMemsetAsync(st.cnt, 0, size_t(V) * sizeof(uint32_t), s));
  CUDA_CHECK(cudaMemsetAsync(st.n, 0, sizeof(int), s));
}

void penalty_prompt(const PenSeen& st, const int* tok, int L, int V, cudaStream_t s) {
  if (L <= 0) return;
  k_pen_prompt<<<std::min(cdiv(L, 256), 1024), 256, 0, s>>>(st, tok, L, V);
  CUDA_CHECK(cudaGetLastError());
}

void penalty_add(const PenSeq* seqs, int B, const int* tok, const int* t_start, const int* n_tok, const int* last, int V,
                 cudaStream_t s) {
  if (B <= 0 || (!tok && !last)) return;
  if (tok && !t_start) qlog::fail("penalty", "penalty_add: tok without t_start");
  k_pen_add<<<B, 32, 0, s>>>(seqs, tok, t_start, n_tok, last, V);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace q38
