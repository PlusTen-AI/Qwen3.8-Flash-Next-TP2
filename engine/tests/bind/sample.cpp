// Test binding for the sample family (engine/src/kernels/sample/sample.cu): sample_rows, spec_accept, the truncation
// hooks (cut, weights, cap), cluster-size hook, P016 switches (patch, exact classification), timing; P019 sparse_logits
// (cands, sample_sparse, accept_sparse, slow-path counter, sparse_ok, bench_sparse); min_p (optional min_p / q_min_p
// arguments, the ex2 probe weight_monotone); penalties (penalty.cu: pen_reset, pen_prompt, pen_add, penalize,
// pen_bench).
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

#include "core/log.h"
#include "kernels/sample/sample.h"
#include "model/patches.h"

namespace {

using OptT = c10::optional<torch::Tensor>;

cudaStream_t st() { return at::cuda::getCurrentCUDAStream(); }
const uint64_t* u64p(const torch::Tensor& t) { return reinterpret_cast<const uint64_t*>(t.data_ptr<int64_t>()); }

void dev(const torch::Tensor& t, torch::ScalarType ty, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == ty, name, ": contiguous CUDA tensor of the right type");
}

// per-row truncation parameters: top_k [n] i32, top_p [n] f32 (None: 0 / 1 — no truncation)
torch::Tensor topk_t(const OptT& t, int64_t n) {
  if (!t) return torch::zeros({n}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  dev(*t, torch::kInt32, "top_k");
  TORCH_CHECK(t->numel() == n, "top_k: one per row");
  return *t;
}
torch::Tensor topp_t(const OptT& t, int64_t n) {
  if (!t) return torch::ones({n}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  dev(*t, torch::kFloat32, "top_p");
  TORCH_CHECK(t->numel() == n, "top_p: one per row");
  return *t;
}
// min_p [n] f32 or None → the pointer the kernels get (None: null — the calls without min_p)
const float* minp_p(const OptT& t, int64_t n, const char* name) {
  if (!t) return nullptr;
  dev(*t, torch::kFloat32, name);
  TORCH_CHECK(t->numel() == n, name, ": one per row");
  return t->data_ptr<float>();
}

// logits [R, V] f32, temp [R] f32, seed [R] i64, ctr [R] or [N, R] i64 (+ top_k [R] i32, top_p [R] f32) → token, prob
// [N, R]: N calls on the same rows, call n with counters ctr[n].
std::vector<torch::Tensor> sample(torch::Tensor logits, torch::Tensor temp, torch::Tensor seed, torch::Tensor ctr,
                                  OptT top_k, OptT top_p, OptT min_p) {
  dev(logits, torch::kFloat32, "logits"), dev(temp, torch::kFloat32, "temp"), dev(seed, torch::kInt64, "seed");
  dev(ctr, torch::kInt64, "ctr");
  const int R = int(logits.size(0)), V = int(logits.size(1));
  const int N = ctr.dim() == 1 ? 1 : int(ctr.size(0));
  TORCH_CHECK(temp.numel() == R && seed.numel() == R && ctr.numel() == int64_t(N) * R, "row counts");
  auto tk = topk_t(top_k, R), tp = topp_t(top_p, R);
  const float* mp = minp_p(min_p, R, "min_p");
  auto o = logits.options();
  auto tok = torch::empty({N, R}, o.dtype(torch::kInt32)), prob = torch::empty({N, R}, o);
  for (int n = 0; n < N; n++)
    q38::sample_rows(logits.data_ptr<float>(), R, V, temp.data_ptr<float>(), tk.data_ptr<int>(), tp.data_ptr<float>(), mp,
                     u64p(seed),
                     u64p(ctr) + size_t(n) * R, tok.data_ptr<int>() + size_t(n) * R, prob.data_ptr<float>() + size_t(n) * R,
                     st());
  return {tok, prob};
}

// p_logits [P, V], p_start [B+1] i32, q_logits [Q, V], q_start [B+1] i32, drafts [Q] or [N, Q] i32, temp [B] f32,
// seed [B] i64, ctr [B] or [N, B] i64 (+ top_k [B] i32, top_p [B] f32) → n_acc, next [N, B]: N calls on the same rows
// (call n: drafts[n], ctr[n]).
std::vector<torch::Tensor> accept(torch::Tensor pl, torch::Tensor p_start, torch::Tensor ql, torch::Tensor q_start,
                                  torch::Tensor drafts, torch::Tensor temp, torch::Tensor seed, torch::Tensor ctr,
                                  OptT top_k, OptT top_p, OptT min_p, OptT q_min_p) {
  dev(pl, torch::kFloat32, "p_logits"), dev(ql, torch::kFloat32, "q_logits"), dev(p_start, torch::kInt32, "p_start");
  dev(q_start, torch::kInt32, "q_start"), dev(drafts, torch::kInt32, "drafts"), dev(temp, torch::kFloat32, "temp");
  dev(seed, torch::kInt64, "seed"), dev(ctr, torch::kInt64, "ctr");
  const int B = int(temp.numel()), V = int(pl.size(1));
  const int N = ctr.dim() == 1 ? 1 : int(ctr.size(0));
  const int64_t Q = drafts.numel() / N;
  TORCH_CHECK(p_start.numel() == B + 1 && q_start.numel() == B + 1 && ctr.numel() == int64_t(N) * B, "counts");
  TORCH_CHECK(ql.size(1) == V && drafts.numel() == N * Q, "q / drafts");
  auto tk = topk_t(top_k, B), tp = topp_t(top_p, B);
  const float *mp = minp_p(min_p, B, "min_p"), *qmp = minp_p(q_min_p, B, "q_min_p");
  auto o = pl.options().dtype(torch::kInt32);
  auto n_acc = torch::empty({N, B}, o), next = torch::empty({N, B}, o);
  for (int n = 0; n < N; n++)
    q38::spec_accept(pl.data_ptr<float>(), p_start.data_ptr<int>(), ql.data_ptr<float>(), q_start.data_ptr<int>(),
                     drafts.data_ptr<int>() + n * Q, B, V, temp.data_ptr<float>(), tk.data_ptr<int>(),
                     tp.data_ptr<float>(), mp, qmp, u64p(seed),
                     u64p(ctr) + size_t(n) * B, n_acc.data_ptr<int>() + size_t(n) * B, next.data_ptr<int>() + size_t(n) * B,
                     st());
  return {n_acc, next};
}

// sample_cut: logits [R, V], temp [R] (+ top_k, top_p, min_p) → [R, 2] i64 (cut z, W')
torch::Tensor cut(torch::Tensor logits, torch::Tensor temp, OptT top_k, OptT top_p, OptT min_p) {
  dev(logits, torch::kFloat32, "logits"), dev(temp, torch::kFloat32, "temp");
  const int R = int(logits.size(0)), V = int(logits.size(1));
  auto out = torch::empty({R, 2}, logits.options().dtype(torch::kInt64));
  auto tk = topk_t(top_k, R), tp = topp_t(top_p, R);
  q38::sample_cut(logits.data_ptr<float>(), R, V, temp.data_ptr<float>(), tk.data_ptr<int>(), tp.data_ptr<float>(),
                  minp_p(min_p, R, "min_p"), reinterpret_cast<uint64_t*>(out.data_ptr<int64_t>()), st());
  return out;
}

// ---- penalties (kernels/sample/penalty.cu) -------------------------------------------------------------------------
// State of S sequences: cnt [S, V] i32 (uint32 bits), ids [S, V] i32, n [S] i32 (CUDA). Sequence b of a call uses state
// slots[b] (−1: none) with penalties rep[b], freq[b], pres[b].
q38::PenSeen seen(torch::Tensor& cnt, torch::Tensor& ids, torch::Tensor& n, int64_t s) {
  dev(cnt, torch::kInt32, "cnt"), dev(ids, torch::kInt32, "ids"), dev(n, torch::kInt32, "n");
  TORCH_CHECK(cnt.dim() == 2 && ids.sizes() == cnt.sizes() && n.numel() == cnt.size(0) && s >= 0 && s < cnt.size(0),
              "state shapes / slot");
  const int64_t V = cnt.size(1);
  return {reinterpret_cast<uint32_t*>(cnt.data_ptr<int>()) + s * V, ids.data_ptr<int>() + s * V, n.data_ptr<int>() + s};
}
torch::Tensor pen_seqs(torch::Tensor& cnt, torch::Tensor& ids, torch::Tensor& n, const std::vector<int64_t>& slots,
                       const std::vector<double>& rep, const std::vector<double>& freq, const std::vector<double>& pres) {
  const size_t B = slots.size();
  TORCH_CHECK(rep.size() == B && freq.size() == B && pres.size() == B, "one penalty per sequence");
  std::vector<q38::PenSeq> h(B);
  for (size_t b = 0; b < B; b++)
    h[b] = {slots[b] < 0 ? q38::PenSeen{nullptr, nullptr, nullptr} : seen(cnt, ids, n, slots[b]), float(rep[b]),
            float(freq[b]), float(pres[b])};
  auto d = torch::empty({int64_t(std::max<size_t>(B, 1) * sizeof(q38::PenSeq))}, cnt.options().dtype(torch::kUInt8));
  CUDA_CHECK(cudaMemcpyAsync(d.data_ptr(), h.data(), B * sizeof(q38::PenSeq), cudaMemcpyHostToDevice, st()));
  CUDA_CHECK(cudaStreamSynchronize(st()));  // (h is pageable and goes out of scope)
  return d;
}
const q38::PenSeq* psq(const torch::Tensor& d) { return reinterpret_cast<const q38::PenSeq*>(d.data_ptr()); }
const int* opt_i32(const OptT& t, const char* name) {
  if (!t) return nullptr;
  dev(*t, torch::kInt32, name);
  return t->data_ptr<int>();
}

void pen_reset(torch::Tensor cnt, torch::Tensor ids, torch::Tensor n, int64_t slot) {
  q38::penalty_reset(seen(cnt, ids, n, slot), int(cnt.size(1)), st());
}
void pen_prompt(torch::Tensor cnt, torch::Tensor ids, torch::Tensor n, int64_t slot, torch::Tensor tok) {
  dev(tok, torch::kInt32, "tok");
  q38::penalty_prompt(seen(cnt, ids, n, slot), tok.data_ptr<int>(), int(tok.numel()), int(cnt.size(1)), st());
}
void pen_add(torch::Tensor cnt, torch::Tensor ids, torch::Tensor n, std::vector<int64_t> slots, OptT tok, OptT t_start,
             OptT n_tok, OptT last) {
  const std::vector<double> z(slots.size(), 0.0);
  auto d = pen_seqs(cnt, ids, n, slots, z, z, z);
  q38::penalty_add(psq(d), int(slots.size()), opt_i32(tok, "tok"), opt_i32(t_start, "t_start"), opt_i32(n_tok, "n_tok"),
                   opt_i32(last, "last"), int(cnt.size(1)), st());
}
// penalize_rows in place on x [R, ld] (row = tokens off … off + Vb − 1, Vb = ld), p_start [B+1], q_start [B+1] and
// drafts (or None)
void penalize(torch::Tensor x, int64_t off, torch::Tensor p_start, OptT q_start, OptT drafts, torch::Tensor cnt,
              torch::Tensor ids, torch::Tensor n, std::vector<int64_t> slots, std::vector<double> rep,
              std::vector<double> freq, std::vector<double> pres) {
  dev(x, torch::kFloat32, "x"), dev(p_start, torch::kInt32, "p_start");
  const int B = int(slots.size());
  TORCH_CHECK(p_start.numel() == B + 1, "p_start [B+1]");
  auto d = pen_seqs(cnt, ids, n, slots, rep, freq, pres);
  q38::penalize_rows(x.data_ptr<float>(), int(x.size(1)), int(off), int(x.size(1)), int(cnt.size(1)), int(x.size(0)),
                     p_start.data_ptr<int>(), opt_i32(q_start, "q_start"), opt_i32(drafts, "drafts"), B, psq(d), st());
}
// µs per penalize_rows call (a graph of `iters` calls on the same rows, best of reps) — arguments as penalize
double pen_bench(torch::Tensor x, int64_t off, torch::Tensor p_start, OptT q_start, OptT drafts, torch::Tensor cnt,
                 torch::Tensor ids, torch::Tensor n, std::vector<int64_t> slots, std::vector<double> rep,
                 std::vector<double> freq, std::vector<double> pres, int iters, int reps) {
  const int B = int(slots.size());
  auto d = pen_seqs(cnt, ids, n, slots, rep, freq, pres);
  const int *qs = opt_i32(q_start, "q_start"), *dr = opt_i32(drafts, "drafts");
  auto call = [&](cudaStream_t s) {
    q38::penalize_rows(x.data_ptr<float>(), int(x.size(1)), int(off), int(x.size(1)), int(cnt.size(1)), int(x.size(0)),
                       p_start.data_ptr<int>(), qs, dr, B, psq(d), s);
  };
  cudaStream_t s = st();
  call(s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaStream_t cs;
  CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  cudaGraph_t gr;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < iters; i++) call(cs);
  CUDA_CHECK(cudaStreamEndCapture(cs, &gr));
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  cudaGraphDestroy(gr);
  cudaStreamDestroy(cs);
  CUDA_CHECK(cudaGraphLaunch(ge, s));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    cudaEventRecord(e0, s);
    CUDA_CHECK(cudaGraphLaunch(ge, s));
    cudaEventRecord(e1, s);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    best = std::min(best, 1000.0 * ms / iters);
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  cudaGraphExecDestroy(ge);
  return best;
}
// the ex2 probe → (violations, total descent: 2^44 when the walk covered every argument)
std::vector<int64_t> weight_monotone() {
  uint64_t descent = 0;
  const uint64_t bad = q38::sample_weight_monotone(&descent);
  return {int64_t(bad), int64_t(descent)};
}

// sample_weights: logits [R, V], temp [R] → w [R, V] i64
torch::Tensor weights(torch::Tensor logits, torch::Tensor temp) {
  dev(logits, torch::kFloat32, "logits"), dev(temp, torch::kFloat32, "temp");
  auto out = torch::empty(logits.sizes(), logits.options().dtype(torch::kInt64));
  q38::sample_weights(logits.data_ptr<float>(), int(logits.size(0)), int(logits.size(1)), temp.data_ptr<float>(),
                      reinterpret_cast<uint64_t*>(out.data_ptr<int64_t>()), st());
  return out;
}

int force_cluster(int c) { return q38::sample_force_cluster(c); }
int force_cap(int c) { return q38::sample_force_cap(c); }
int force_exact(int on) { return q38::sample_force_exact(on); }
// P016 block_verify on / off → the previous state
bool block(bool on) {
  const bool was = q38::patch::on(q38::patch::block_verify);
  q38::patch::set(q38::patch::block_verify, on);
  return was;
}

// µs per call at vocabulary V, cycling `sets` disjoint copies of the inputs (sets × rows ≫ L2: cold rows from DRAM, as
// after a step), best of `reps`. iters > 1: `iters` back-to-back calls replayed as one CUDA graph (kernel time +
// inter-kernel gap). iters == 1: one call alone — a one-call graph per input set, each launch timed by itself (graph
// launch included), sets in turn.
// what 0: sample_rows of R rows at temperature temp (0 = argmax). what 1: spec_accept of R sequences × k drafts at T = 1
// with every draft accepted (q rows = p rows bitwise, drafts = argmax: 2k + 1 rows read, then the bonus draw). what 2:
// q rows = the p rows shifted by one token, drafts = argmax of q (the drafts' p is small: rejections from the first
// draft). what 3: q rows = p rows + N(0, 0.7²), drafts drawn from q (sample_rows, T = 1, the same truncation): mixed.
// what ≥ 2 (iters == 1) returns the mean over the input sets of each set's best (the sets take different paths). top_k /
// top_p: every row's truncation (0 / 1: none). src [n, V] (optional): the logits rows, tiled (else N(0, 3²)).
double bench(int what, int V, int R, int k, float temp, int sets, int iters, int reps, int top_k, float top_p, OptT src) {
  auto o = torch::TensorOptions().device(torch::kCUDA);
  const int P = what == 0 ? R : R * (k + 1), Q = what == 0 ? 0 : R * k;
  torch::Tensor pl;
  if (src) {
    dev(*src, torch::kFloat32, "src");
    TORCH_CHECK(src->size(1) == V, "src: V");
    const int64_t n = int64_t(sets) * P;
    pl = src->repeat({(n + src->size(0) - 1) / src->size(0), 1}).narrow(0, 0, n).contiguous();
  } else {
    pl = torch::randn({int64_t(sets) * P, V}, o) * 3.f;
  }
  auto ql = torch::empty({what >= 1 ? int64_t(sets) * Q : 1, V}, o);
  auto ps = torch::arange(0, R + 1, o.dtype(torch::kInt32)) * (k + 1), qs = torch::arange(0, R + 1, o.dtype(torch::kInt32)) * k;
  auto drafts = torch::empty({what >= 1 ? int64_t(sets) * Q : 1}, o.dtype(torch::kInt32));
  auto tp = torch::full({R}, what == 0 ? temp : 1.f, o);
  auto tk = torch::full({R}, top_k, o.dtype(torch::kInt32)), pp = torch::full({R}, top_p, o);
  if (what >= 1) {
    auto gen = at::detail::createCPUGenerator(1234);
    auto qt = torch::ones({k}, o), qk = torch::full({k}, top_k, o.dtype(torch::kInt32)), qp = torch::full({k}, top_p, o);
    auto dsd = torch::arange(1, k + 1, o.dtype(torch::kInt64)) * 11, dct = torch::zeros({k}, o.dtype(torch::kInt64));
    auto dprob = torch::empty({k}, o);
    for (int j = 0; j < sets; j++)
      for (int b = 0; b < R; b++) {
        auto rows = pl.narrow(0, int64_t(j) * P + b * (k + 1), k);
        auto qr = ql.narrow(0, int64_t(j) * Q + b * k, k);
        auto dr = drafts.narrow(0, int64_t(j) * Q + b * k, k);
        if (what == 1) qr.copy_(rows), dr.copy_(rows.argmax(1).to(torch::kInt32));
        if (what == 2) qr.copy_(rows.roll(1, 1)), dr.copy_(qr.argmax(1).to(torch::kInt32));
        if (what == 3) {
          qr.copy_(rows + (torch::randn(rows.sizes(), gen) * 0.7f).to(rows.device()));
          dct.fill_(int64_t(j) * R + b);
          q38::sample_rows(qr.data_ptr<float>(), k, V, qt.data_ptr<float>(), qk.data_ptr<int>(), qp.data_ptr<float>(),
                           u64p(dsd), u64p(dct), dr.data_ptr<int>(), dprob.data_ptr<float>(), st());
        }
      }
  }
  auto seed = torch::arange(1, R + 1, o.dtype(torch::kInt64)), ctr = seed * 7;
  auto out0 = torch::empty({R}, o.dtype(torch::kInt32)), out1 = torch::empty({R}, o.dtype(torch::kInt32));
  auto prob = torch::empty({R}, o);
  cudaStream_t s = st();
  auto call = [&](int i) {
    const int j = i % sets;
    if (what == 0)
      q38::sample_rows(pl.data_ptr<float>() + size_t(j) * P * V, R, V, tp.data_ptr<float>(), tk.data_ptr<int>(),
                       pp.data_ptr<float>(), u64p(seed), u64p(ctr), out0.data_ptr<int>(), prob.data_ptr<float>(), s);
    else
      q38::spec_accept(pl.data_ptr<float>() + size_t(j) * P * V, ps.data_ptr<int>(), ql.data_ptr<float>() + size_t(j) * Q * V,
                       qs.data_ptr<int>(), drafts.data_ptr<int>() + size_t(j) * Q, R, V, tp.data_ptr<float>(),
                       tk.data_ptr<int>(), pp.data_ptr<float>(), u64p(seed), u64p(ctr), out0.data_ptr<int>(),
                       out1.data_ptr<int>(), s);
  };
  for (int i = 0; i < sets; i++) call(i);  // first call outside the capture (one-time kernel attributes)
  if (what == 1) TORCH_CHECK(torch::equal(out0, torch::full({R}, k, out0.options())), "bench: not every draft accepted");
  // graphs: one of `iters` calls, or (iters == 1) one single call per input set
  std::vector<cudaGraphExec_t> ge(iters > 1 ? 1 : sets);
  cudaStream_t cs;
  CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  std::swap(s, cs);
  for (size_t g = 0; g < ge.size(); g++) {
    cudaGraph_t gr;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; i++) call(int(g) + i);
    CUDA_CHECK(cudaStreamEndCapture(s, &gr));
    CUDA_CHECK(cudaGraphInstantiate(&ge[g], gr, 0));
    cudaGraphDestroy(gr);
  }
  std::swap(s, cs);
  cudaStreamDestroy(cs);
  CUDA_CHECK(cudaGraphLaunch(ge[0], s));  // warm
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  std::vector<double> best(ge.size(), 1e30);
  const int n = iters > 1 ? reps : reps * sets;
  for (int r = 0; r < n; r++) {
    const size_t g = (r + 1) % ge.size();
    cudaEventRecord(e0, s);
    CUDA_CHECK(cudaGraphLaunch(ge[g], s));
    cudaEventRecord(e1, s);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    best[g] = std::min(best[g], 1000.0 * ms / iters);
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  for (auto g : ge) cudaGraphExecDestroy(g);
  if (what >= 2 && iters == 1) {
    double m = 0;
    for (double b : best) m += b;
    return m / double(best.size());
  }
  return *std::min_element(best.begin(), best.end());
}

// ---- P019 sparse_logits --------------------------------------------------------------------------------------------
q38::SparseCand* cp(const torch::Tensor& t) { return reinterpret_cast<q38::SparseCand*>(t.data_ptr<int>()); }
void recs(const torch::Tensor& t, const char* name) {
  dev(t, torch::kInt32, name);
  TORCH_CHECK(t.dim() == 4 && t.size(0) == 2 && t.size(2) == q38::kSparseRec && t.size(3) == 2, name,
              ": records [2, ld, kSparseRec, 2] i32");
}

// sample_cands: half [R, Vh] f32, id offset → records [R, kSparseRec, 2] i32 (id, logit bits; entry kSparseK = flag)
torch::Tensor cands(torch::Tensor half, int64_t off) {
  dev(half, torch::kFloat32, "half");
  const int R = int(half.size(0)), Vh = int(half.size(1));
  auto out = torch::empty({R, q38::kSparseRec, 2}, half.options().dtype(torch::kInt32));
  q38::sample_cands(half.data_ptr<float>(), R, Vh, int(off), cp(out), st());
  return out;
}

// sample_rows_sparse: records [2, ld, kSparseRec, 2], V, temp [R], seed [R], ctr [R] or [N, R] (+ top_k, top_p) →
// token, prob [N, R]
std::vector<torch::Tensor> sample_sparse(torch::Tensor c, int64_t V, torch::Tensor temp, torch::Tensor seed,
                                         torch::Tensor ctr, OptT top_k, OptT top_p, OptT min_p) {
  recs(c, "cands"), dev(temp, torch::kFloat32, "temp"), dev(seed, torch::kInt64, "seed"), dev(ctr, torch::kInt64, "ctr");
  const int R = int(temp.numel()), ld = int(c.size(1));
  const int N = ctr.dim() == 1 ? 1 : int(ctr.size(0));
  TORCH_CHECK(seed.numel() == R && ctr.numel() == int64_t(N) * R && ld >= R, "row counts");
  auto tk = topk_t(top_k, R), tp = topp_t(top_p, R);
  const float* mp = minp_p(min_p, R, "min_p");
  auto o = temp.options();
  auto tok = torch::empty({N, R}, o.dtype(torch::kInt32)), prob = torch::empty({N, R}, o);
  for (int n = 0; n < N; n++)
    q38::sample_rows_sparse(cp(c), ld, R, int(V), temp.data_ptr<float>(), tk.data_ptr<int>(), tp.data_ptr<float>(), mp,
                            u64p(seed), u64p(ctr) + size_t(n) * R, tok.data_ptr<int>() + size_t(n) * R,
                            prob.data_ptr<float>() + size_t(n) * R, st());
  return {tok, prob};
}

// spec_accept_sparse: target records pc [2, P, kSparseRec, 2], p_start [B+1], draft records qc [2, Q', …], q_start,
// drafts [Q] or [N, Q], V, temp [B], seed [B], ctr [B] or [N, B] (+ top_k, top_p) → n_acc, next [N, B]
std::vector<torch::Tensor> accept_sparse(torch::Tensor pc, torch::Tensor p_start, torch::Tensor qc, torch::Tensor q_start,
                                         torch::Tensor drafts, int64_t V, torch::Tensor temp, torch::Tensor seed,
                                         torch::Tensor ctr, OptT top_k, OptT top_p, OptT min_p, OptT q_min_p) {
  recs(pc, "p_cands"), recs(qc, "q_cands"), dev(p_start, torch::kInt32, "p_start"), dev(q_start, torch::kInt32, "q_start");
  dev(drafts, torch::kInt32, "drafts"), dev(temp, torch::kFloat32, "temp"), dev(seed, torch::kInt64, "seed");
  dev(ctr, torch::kInt64, "ctr");
  const int B = int(temp.numel());
  const int N = ctr.dim() == 1 ? 1 : int(ctr.size(0));
  const int64_t Q = drafts.numel() / N;
  TORCH_CHECK(p_start.numel() == B + 1 && q_start.numel() == B + 1 && ctr.numel() == int64_t(N) * B, "counts");
  auto tk = topk_t(top_k, B), tp = topp_t(top_p, B);
  const float *mp = minp_p(min_p, B, "min_p"), *qmp = minp_p(q_min_p, B, "q_min_p");
  auto o = temp.options().dtype(torch::kInt32);
  auto n_acc = torch::empty({N, B}, o), next = torch::empty({N, B}, o);
  for (int n = 0; n < N; n++)
    q38::spec_accept_sparse(cp(pc), int(pc.size(1)), p_start.data_ptr<int>(), cp(qc), int(qc.size(1)),
                            q_start.data_ptr<int>(), drafts.data_ptr<int>() + n * Q, B, int(V), temp.data_ptr<float>(),
                            tk.data_ptr<int>(), tp.data_ptr<float>(), mp, qmp, u64p(seed), u64p(ctr) + size_t(n) * B,
                            n_acc.data_ptr<int>() + size_t(n) * B, next.data_ptr<int>() + size_t(n) * B, st());
  return {n_acc, next};
}

unsigned cands_slow(bool reset) { return q38::sample_cands_slow(reset); }
int cands_cluster(int Vh, int R) { return q38::sample_cands_cluster(Vh, R); }
bool sparse_ok(double temp, int top_k, double top_p, int V) { return q38::sparse_ok(float(temp), top_k, float(top_p), V); }

// µs per call of `call(i)` (input set i % sets), as bench: iters > 1 — a graph of `iters` back-to-back calls, best of
// reps; iters == 1 — a one-call graph per set, best of reps per set, the minimum (mean_sets: the mean of the sets' bests).
template <class F>
double time_calls(F call, int sets, int iters, int reps, bool mean_sets) {
  cudaStream_t s = st();
  for (int i = 0; i < sets; i++) call(i, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<cudaGraphExec_t> ge(iters > 1 ? 1 : sets);
  cudaStream_t cs;
  CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  for (size_t g = 0; g < ge.size(); g++) {
    cudaGraph_t gr;
    CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; i++) call(int(g) + i, cs);
    CUDA_CHECK(cudaStreamEndCapture(cs, &gr));
    CUDA_CHECK(cudaGraphInstantiate(&ge[g], gr, 0));
    cudaGraphDestroy(gr);
  }
  cudaStreamDestroy(cs);
  CUDA_CHECK(cudaGraphLaunch(ge[0], s));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  std::vector<double> best(ge.size(), 1e30);
  const int n = iters > 1 ? reps : reps * sets;
  for (int r = 0; r < n; r++) {
    const size_t g = (r + 1) % ge.size();
    cudaEventRecord(e0, s);
    CUDA_CHECK(cudaGraphLaunch(ge[g], s));
    cudaEventRecord(e1, s);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    best[g] = std::min(best[g], 1000.0 * ms / iters);
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  for (auto g : ge) cudaGraphExecDestroy(g);
  if (mean_sets) {
    double m = 0;
    for (double b : best) m += b;
    return m / double(best.size());
  }
  return *std::min_element(best.begin(), best.end());
}

// P019 timing at V = src's width (rows of src tiled; src [n, V] f32): what 0 — sample_cands of R half rows (sets disjoint
// copies: sets × R × Vh·4 B ≫ L2 for rows from DRAM, sets = 1: the same rows, L2-resident); what 1 — sample_rows_sparse
// of R rows (T = 1, top_k / top_p); what 2 / 3 — spec_accept_sparse of R sequences × k drafts at T = 1: 2 mixed (q = p +
// N(0, 0.7²), drafts drawn from q by sample_rows), 3 every draft accepted (q = p, drafts = argmax). iters, reps, mean as
// time_calls (what 2: the mean of the sets).
double bench_sparse(int what, int R, int k, int sets, int iters, int reps, int top_k, float top_p, torch::Tensor src) {
  dev(src, torch::kFloat32, "src");
  const int V = int(src.size(1)), Vh = V / 2;
  auto o = src.options();
  auto tile = [&](int64_t n) { return src.repeat({(n + src.size(0) - 1) / src.size(0), 1}).narrow(0, 0, n).contiguous(); };
  auto halves = [&](const torch::Tensor& rows) {  // full rows [n, V] → records [2, n, rec, 2]
    auto a = cands(rows.narrow(1, 0, Vh).contiguous(), 0), b = cands(rows.narrow(1, Vh, Vh).contiguous(), Vh);
    return torch::stack({a, b}).contiguous();
  };
  auto i32 = o.dtype(torch::kInt32);
  auto seed = torch::arange(1, R + 1, o.dtype(torch::kInt64)), ctr = seed * 7;
  auto temp = torch::ones({R}, o), tk = torch::full({R}, top_k, i32), tp = torch::full({R}, top_p, o);
  auto out0 = torch::empty({R}, i32), out1 = torch::empty({R}, i32);
  auto prob = torch::empty({R}, o);
  if (what == 0) {
    auto x = tile(int64_t(sets) * R).narrow(1, 0, Vh).contiguous();
    auto rec = torch::empty({R, q38::kSparseRec, 2}, i32);
    return time_calls([&](int i, cudaStream_t s) {
      q38::sample_cands(x.data_ptr<float>() + size_t(i % sets) * R * Vh, R, Vh, 0, cp(rec), s);
    }, sets, iters, reps, false);
  }
  if (what == 1) {
    std::vector<torch::Tensor> rs;
    auto all = tile(int64_t(sets) * R);
    for (int j = 0; j < sets; j++) rs.push_back(halves(all.narrow(0, int64_t(j) * R, R)));
    return time_calls([&](int i, cudaStream_t s) {
      q38::sample_rows_sparse(cp(rs[i % sets]), R, R, V, temp.data_ptr<float>(), tk.data_ptr<int>(), tp.data_ptr<float>(),
                              u64p(seed), u64p(ctr), out0.data_ptr<int>(), prob.data_ptr<float>(), s);
    }, sets, iters, reps, false);
  }
  const int P = R * (k + 1), Q = R * k;
  auto ps = torch::arange(0, R + 1, i32) * (k + 1), qs = torch::arange(0, R + 1, i32) * k;
  auto gen = at::detail::createCPUGenerator(1234);
  std::vector<torch::Tensor> pr, qr, dr;
  auto all = tile(int64_t(sets) * P);
  for (int j = 0; j < sets; j++) {
    auto pl = all.narrow(0, int64_t(j) * P, P);
    auto ql = torch::empty({Q, V}, o);
    auto d = torch::empty({Q}, i32);
    for (int b = 0; b < R; b++) {
      auto rows = pl.narrow(0, b * (k + 1), k), qb = ql.narrow(0, b * k, k), db = d.narrow(0, b * k, k);
      if (what == 3) {
        qb.copy_(rows), db.copy_(rows.argmax(1).to(torch::kInt32));
      } else {
        qb.copy_(rows + (torch::randn(rows.sizes(), gen) * 0.7f).to(rows.device()));
        auto qt = torch::ones({k}, o), qk = torch::full({k}, top_k, i32), qp = torch::full({k}, top_p, o);
        auto dsd = torch::arange(1, k + 1, o.dtype(torch::kInt64)) * 11, dct = torch::full({k}, int64_t(j) * R + b, o.dtype(torch::kInt64));
        auto dprob = torch::empty({k}, o);
        q38::sample_rows(qb.data_ptr<float>(), k, V, qt.data_ptr<float>(), qk.data_ptr<int>(), qp.data_ptr<float>(),
                         u64p(dsd), u64p(dct), db.data_ptr<int>(), dprob.data_ptr<float>(), st());
      }
    }
    pr.push_back(halves(pl)), qr.push_back(halves(ql)), dr.push_back(d);
  }
  return time_calls([&](int i, cudaStream_t s) {
    const int j = i % sets;
    q38::spec_accept_sparse(cp(pr[j]), P, ps.data_ptr<int>(), cp(qr[j]), Q, qs.data_ptr<int>(), dr[j].data_ptr<int>(), R, V,
                            temp.data_ptr<float>(), tk.data_ptr<int>(), tp.data_ptr<float>(), u64p(seed), u64p(ctr),
                            out0.data_ptr<int>(), out1.data_ptr<int>(), s);
  }, sets, iters, reps, what == 2 && iters == 1);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cands", &cands);
  m.def("sample_sparse", &sample_sparse, py::arg("cands"), py::arg("V"), py::arg("temp"), py::arg("seed"), py::arg("ctr"),
        py::arg("top_k") = py::none(), py::arg("top_p") = py::none(), py::arg("min_p") = py::none());
  m.def("accept_sparse", &accept_sparse, py::arg("pc"), py::arg("p_start"), py::arg("qc"), py::arg("q_start"),
        py::arg("drafts"), py::arg("V"), py::arg("temp"), py::arg("seed"), py::arg("ctr"), py::arg("top_k") = py::none(),
        py::arg("top_p") = py::none(), py::arg("min_p") = py::none(), py::arg("q_min_p") = py::none());
  m.def("pen_reset", &pen_reset);
  m.def("pen_prompt", &pen_prompt);
  m.def("pen_add", &pen_add, py::arg("cnt"), py::arg("ids"), py::arg("n"), py::arg("slots"), py::arg("tok") = py::none(),
        py::arg("t_start") = py::none(), py::arg("n_tok") = py::none(), py::arg("last") = py::none());
  m.def("penalize", &penalize, py::arg("x"), py::arg("off"), py::arg("p_start"), py::arg("q_start"), py::arg("drafts"),
        py::arg("cnt"), py::arg("ids"), py::arg("n"), py::arg("slots"), py::arg("rep"), py::arg("freq"), py::arg("pres"));
  m.def("pen_bench", &pen_bench, py::arg("x"), py::arg("off"), py::arg("p_start"), py::arg("q_start"), py::arg("drafts"),
        py::arg("cnt"), py::arg("ids"), py::arg("n"), py::arg("slots"), py::arg("rep"), py::arg("freq"), py::arg("pres"),
        py::arg("iters"), py::arg("reps"));
  m.def("weight_monotone", &weight_monotone);
  m.def("cands_slow", &cands_slow);
  m.def("cands_cluster", &cands_cluster);
  m.def("sparse_ok", &sparse_ok);
  m.def("bench_sparse", &bench_sparse, py::arg("what"), py::arg("R"), py::arg("k"), py::arg("sets"), py::arg("iters"),
        py::arg("reps"), py::arg("top_k"), py::arg("top_p"), py::arg("src"));
  m.def("sample", &sample, py::arg("logits"), py::arg("temp"), py::arg("seed"), py::arg("ctr"), py::arg("top_k") = py::none(),
        py::arg("top_p") = py::none(), py::arg("min_p") = py::none());
  m.def("accept", &accept, py::arg("pl"), py::arg("p_start"), py::arg("ql"), py::arg("q_start"), py::arg("drafts"),
        py::arg("temp"), py::arg("seed"), py::arg("ctr"), py::arg("top_k") = py::none(), py::arg("top_p") = py::none(),
        py::arg("min_p") = py::none(), py::arg("q_min_p") = py::none());
  m.def("cut", &cut, py::arg("logits"), py::arg("temp"), py::arg("top_k") = py::none(), py::arg("top_p") = py::none(),
        py::arg("min_p") = py::none());
  m.def("weights", &weights);
  m.def("force_cluster", &force_cluster);
  m.def("force_cap", &force_cap);
  m.def("force_exact", &force_exact);
  m.def("block", &block);
  m.def("bench", &bench, py::arg("what"), py::arg("V"), py::arg("R"), py::arg("k"), py::arg("temp"), py::arg("sets"),
        py::arg("iters"), py::arg("reps"), py::arg("top_k") = 0, py::arg("top_p") = 1.f, py::arg("src") = py::none());
}
