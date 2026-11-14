// Test binding for the GDN core (engine/src/kernels/gdn.cu): upload, run on torch tensors, timing; speculative rows
// (gdn_core_spec + gdn_commit) with records kept in a uint8 tensor [slots, gdn_rec_bytes()]; P020 (gdn_core_spec_pending
// + gdn_flush) with records addressed by their own row lists.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <map>
#include <memory>
#include <tuple>

#include "kernels/gdn/gdn.h"
#include "core/log.h"
#include "model/patches.h"

namespace {
std::unique_ptr<q38::Weights> W;
std::map<int, q38::GdnW> L;
q38::GdnState* d_st = nullptr;   // [cap] structs
q38::GdnState** d_ptr = nullptr; // [cap] pointers
q38::GdnRec* d_rec = nullptr;    // [cap] structs
q38::GdnRec** d_rp = nullptr;    // [cap] pointers
int* d_seq = nullptr;            // [cap + 1] (commit: n per sequence)
q38::GdnRec* d_prec = nullptr;   // P020: [cap] pending record structs
q38::GdnRec** d_pp = nullptr;    // [cap] pointers
int* d_np = nullptr;             // [cap] pending rows per sequence
int cap = 0;

void ensure(int n) {
  if (n <= cap) return;
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d_st), cudaFree(d_ptr), cudaFree(d_rec), cudaFree(d_rp), cudaFree(d_seq);
  cudaFree(d_prec), cudaFree(d_pp), cudaFree(d_np);
  cap = std::max(n, 64);
  CUDA_CHECK(cudaMalloc(&d_st, cap * sizeof(q38::GdnState)));
  CUDA_CHECK(cudaMalloc(&d_ptr, cap * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&d_rec, cap * sizeof(q38::GdnRec)));
  CUDA_CHECK(cudaMalloc(&d_rp, cap * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&d_seq, (cap + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_prec, cap * sizeof(q38::GdnRec)));
  CUDA_CHECK(cudaMalloc(&d_pp, cap * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&d_np, cap * sizeof(int)));
}

// Device state (and record) pointers for slots idx[p] of S [N,24,128,128] f32, conv [N,3,5120] bf16, rec [N, bytes].
void put_ptrs(torch::Tensor S, torch::Tensor conv, const torch::Tensor* rec, const std::vector<int>& idx,
              cudaStream_t s) {
  const int P = int(idx.size());
  ensure(P);
  std::vector<q38::GdnState> st(P);
  std::vector<q38::GdnState*> pt(P);
  std::vector<q38::GdnRec> rc(P);
  std::vector<q38::GdnRec*> rp(P);
  for (int p = 0; p < P; p++) {
    TORCH_CHECK(idx[p] >= 0 && idx[p] < S.size(0) && idx[p] < conv.size(0) && (!rec || idx[p] < rec->size(0)), "slot");
    st[p].S = S.data_ptr<float>() + size_t(idx[p]) * 24 * 128 * 128;
    st[p].conv = reinterpret_cast<q38::bf16*>(conv.data_ptr()) + size_t(idx[p]) * 3 * 5120;
    pt[p] = d_st + p;
    if (rec) rc[p] = q38::gdn_rec_at(rec->data_ptr<uint8_t>() + size_t(idx[p]) * q38::gdn_rec_bytes());
    rp[p] = d_rec + p;
  }
  CUDA_CHECK(cudaMemcpyAsync(d_st, st.data(), P * sizeof(q38::GdnState), cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_ptr, pt.data(), P * sizeof(void*), cudaMemcpyHostToDevice, s));
  if (rec) {
    CUDA_CHECK(cudaMemcpyAsync(d_rec, rc.data(), P * sizeof(q38::GdnRec), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_rp, rp.data(), P * sizeof(void*), cudaMemcpyHostToDevice, s));
  }
}

void open(const std::string& model_dir, const std::string& table, int rank) {
  W = std::make_unique<q38::Weights>(model_dir, table, rank);
}

// Uploads layer `i` (core fields); returns copies: conv_w [5120,4] bf16, A [24] f32, dt_bias [24] f32, norm_w [128].
std::vector<torch::Tensor> upload(int i) {
  if (!L.count(i)) L[i] = q38::gdn_upload(*W, i);
  const auto& g = L[i];
  auto o = torch::TensorOptions().device(torch::kCUDA);
  auto cw = torch::empty({5120, 4}, o.dtype(torch::kBFloat16)), nw = torch::empty({128}, o.dtype(torch::kBFloat16));
  auto A = torch::empty({24}, o.dtype(torch::kFloat32)), dt = torch::empty({24}, o.dtype(torch::kFloat32));
  CUDA_CHECK(cudaMemcpy(cw.data_ptr(), g.conv_w, 5120 * 4 * 2, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(nw.data_ptr(), g.norm_w, 128 * 2, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(A.data_ptr(), g.A, 24 * 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(dt.data_ptr(), g.dt_bias, 24 * 4, cudaMemcpyDeviceToDevice));
  return {cw, A, dt, nw};
}

// qkvz [M,8192] bf16, ba [M,48] bf16, S [N,24,128,128] f32, conv [N,3,5120] bf16 (state slots), slots [B] (slot of
// sequence b), seq_start [B+1] (host list), core [M,3072] bf16, o [M,3072] bf16 or None.
void run(int i, torch::Tensor qkvz, torch::Tensor ba, torch::Tensor S, torch::Tensor conv, std::vector<int> slots,
         std::vector<int> seq_start, torch::Tensor core, c10::optional<torch::Tensor> o) {
  const int B = int(slots.size()), M = int(qkvz.size(0));
  TORCH_CHECK(seq_start.size() == size_t(B + 1) && seq_start[B] == M, "seq_start");
  TORCH_CHECK(qkvz.is_contiguous() && ba.is_contiguous() && S.is_contiguous() && conv.is_contiguous() &&
              core.is_contiguous(), "contiguous");
  auto s = at::cuda::getCurrentCUDAStream().stream();
  put_ptrs(S, conv, nullptr, slots, s);
  CUDA_CHECK(cudaMemcpyAsync(d_seq, seq_start.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice, s));
  auto P = [](torch::Tensor t) { return reinterpret_cast<q38::bf16*>(t.data_ptr()); };
  q38::gdn_core_tap(L.at(i), P(qkvz), P(ba), P(core), o ? P(*o) : nullptr, d_ptr, d_seq, B, M, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

int64_t rec_bytes() { return int64_t(q38::gdn_rec_bytes()); }

// gdn_core_spec: as run (no o), plus rec [N, rec_bytes] uint8 — slot slots[b] holds sequence b's state and record.
void run_spec(int i, torch::Tensor qkvz, torch::Tensor ba, torch::Tensor S, torch::Tensor conv, torch::Tensor rec,
              std::vector<int> slots, std::vector<int> seq_start, torch::Tensor core) {
  const int B = int(slots.size()), M = int(qkvz.size(0));
  TORCH_CHECK(seq_start.size() == size_t(B + 1) && seq_start[B] == M, "seq_start");
  TORCH_CHECK(qkvz.is_contiguous() && ba.is_contiguous() && S.is_contiguous() && conv.is_contiguous() &&
              core.is_contiguous() && rec.is_contiguous() && rec.size(1) == rec_bytes(), "contiguous / rec");
  auto s = at::cuda::getCurrentCUDAStream().stream();
  put_ptrs(S, conv, &rec, slots, s);
  CUDA_CHECK(cudaMemcpyAsync(d_seq, seq_start.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice, s));
  auto P = [](torch::Tensor t) { return reinterpret_cast<q38::bf16*>(t.data_ptr()); };
  q38::gdn_core_spec(L.at(i), P(qkvz), P(ba), P(core), d_ptr, d_rp, d_seq, B, M, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

// gdn_commit over L·B pairs: pair p = l·B + b uses slot idx[p] of S / conv / rec and keeps n[b] rows.
void commit(torch::Tensor S, torch::Tensor conv, torch::Tensor rec, std::vector<int> idx, std::vector<int> n) {
  const int B = int(n.size()), P = int(idx.size());
  TORCH_CHECK(B > 0 && P % B == 0 && S.is_contiguous() && conv.is_contiguous() && rec.is_contiguous() &&
              rec.size(1) == rec_bytes(), "commit");
  auto s = at::cuda::getCurrentCUDAStream().stream();
  put_ptrs(S, conv, &rec, idx, s);
  CUDA_CHECK(cudaMemcpyAsync(d_seq, n.data(), B * sizeof(int), cudaMemcpyHostToDevice, s));
  q38::gdn_commit(d_ptr, d_rp, d_seq, P / B, B, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

// ---- P020 gdn_fused_commit: records addressed separately from states (double-buffered: rec rows ≠ state slots) ------
// Record structs / pointers of rows idx[p] of rec [N, rec_bytes] into the given device arrays.
void put_recs(const torch::Tensor& rec, const std::vector<int>& idx, q38::GdnRec* drec, q38::GdnRec** dptr,
              cudaStream_t s) {
  const int P = int(idx.size());
  std::vector<q38::GdnRec> rc(P);
  std::vector<q38::GdnRec*> rp(P);
  for (int p = 0; p < P; p++) {
    TORCH_CHECK(idx[p] >= 0 && idx[p] < rec.size(0), "record row");
    rc[p] = q38::gdn_rec_at(rec.data_ptr<uint8_t>() + size_t(idx[p]) * q38::gdn_rec_bytes());
    rp[p] = drec + p;
  }
  CUDA_CHECK(cudaMemcpyAsync(drec, rc.data(), P * sizeof(q38::GdnRec), cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(dptr, rp.data(), P * sizeof(void*), cudaMemcpyHostToDevice, s));
}

// gdn_core_spec_pending: sequence b uses state slot slots[b], writes record row new_idx[b] of rec, applies the first
// npend[b] rows of record row pend_idx[b] first. new_idx == pend_idx → in place, one pointer array for both.
void run_spec_pending(int i, torch::Tensor qkvz, torch::Tensor ba, torch::Tensor S, torch::Tensor conv,
                      torch::Tensor rec, std::vector<int> slots, std::vector<int> new_idx, std::vector<int> pend_idx,
                      std::vector<int> npend, std::vector<int> seq_start, torch::Tensor core) {
  const int B = int(slots.size()), M = int(qkvz.size(0));
  TORCH_CHECK(seq_start.size() == size_t(B + 1) && seq_start[B] == M && new_idx.size() == size_t(B) &&
              pend_idx.size() == size_t(B) && npend.size() == size_t(B), "seq_start / idx");
  TORCH_CHECK(qkvz.is_contiguous() && ba.is_contiguous() && S.is_contiguous() && conv.is_contiguous() &&
              core.is_contiguous() && rec.is_contiguous() && rec.size(1) == rec_bytes(), "contiguous / rec");
  auto s = at::cuda::getCurrentCUDAStream().stream();
  put_ptrs(S, conv, nullptr, slots, s);
  put_recs(rec, new_idx, d_rec, d_rp, s);
  const bool inplace = new_idx == pend_idx;
  if (!inplace) put_recs(rec, pend_idx, d_prec, d_pp, s);
  CUDA_CHECK(cudaMemcpyAsync(d_seq, seq_start.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_np, npend.data(), B * sizeof(int), cudaMemcpyHostToDevice, s));
  auto P = [](torch::Tensor t) { return reinterpret_cast<q38::bf16*>(t.data_ptr()); };
  q38::gdn_core_spec_pending(L.at(i), P(qkvz), P(ba), P(core), d_ptr, d_rp, inplace ? d_rp : d_pp, d_np, d_seq, B, M,
                             s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

// gdn_flush over L·B pairs: pair p = l·B + b uses state slot st_idx[p], record row rec_idx[p], n[b] rows.
void flush(torch::Tensor S, torch::Tensor conv, torch::Tensor rec, std::vector<int> st_idx, std::vector<int> rec_idx,
           std::vector<int> n) {
  const int B = int(n.size()), P = int(st_idx.size());
  TORCH_CHECK(B > 0 && P % B == 0 && rec_idx.size() == size_t(P) && S.is_contiguous() && conv.is_contiguous() &&
              rec.is_contiguous() && rec.size(1) == rec_bytes(), "flush");
  auto s = at::cuda::getCurrentCUDAStream().stream();
  put_ptrs(S, conv, nullptr, st_idx, s);
  put_recs(rec, rec_idx, d_rec, d_rp, s);
  CUDA_CHECK(cudaMemcpyAsync(d_seq, n.data(), B * sizeof(int), cudaMemcpyHostToDevice, s));
  q38::gdn_flush(d_ptr, d_rp, d_seq, P / B, B, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

// µs per call of layer i with B sequences × T rows, cycling `sets` distinct state sets (cold state like in the real
// step); inputs random-ish bf16. mode 0: mean over `iters` back-to-back eager launches; 1: the same launches replayed
// as one CUDA graph (kernel time + the GPU's inter-kernel gap); 2: median of `iters` single launches, each between two
// events on an idle stream (includes the launch latency). spec: gdn_core_spec with one record per state; pend ≥ 0
// (with spec): gdn_core_spec_pending with pend pending rows per sequence from a second record per state (small floats).
double bench_impl(int i, int B, int T, int iters, int sets, int mode, bool spec, int pend = -1) {
  const int M = B * T, NR = spec ? (pend >= 0 ? 2 : 1) * sets * B : 0;
  auto o = torch::TensorOptions().device(torch::kCUDA);
  auto qkvz = (torch::randn({M, 8192}, o) * 0.5).to(torch::kBFloat16);
  auto ba = torch::randn({M, 48}, o).to(torch::kBFloat16);
  auto core = torch::empty({M, 3072}, o.dtype(torch::kBFloat16));
  auto S = torch::zeros({int64_t(sets) * B, 24, 128, 128}, o.dtype(torch::kFloat32));
  auto conv = torch::zeros({int64_t(sets) * B, 3, 5120}, o.dtype(torch::kBFloat16));
  auto rec = torch::empty({NR, rec_bytes()}, o.dtype(torch::kUInt8));
  if (pend >= 0) rec.view(torch::kFloat32).normal_(0.f, 0.01f);
  q38::GdnState *st = nullptr, **pt = nullptr;
  q38::GdnRec *rc = nullptr, **rp = nullptr;
  int *seq = nullptr, *npd = nullptr;
  CUDA_CHECK(cudaMalloc(&st, size_t(sets) * B * sizeof(q38::GdnState)));
  CUDA_CHECK(cudaMalloc(&pt, size_t(sets) * B * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&rc, std::max(NR, sets * B) * sizeof(q38::GdnRec)));
  CUDA_CHECK(cudaMalloc(&rp, std::max(NR, sets * B) * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&seq, (B + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&npd, B * sizeof(int)));
  std::vector<q38::GdnState> hs(size_t(sets) * B);
  std::vector<q38::GdnState*> hp(size_t(sets) * B);
  std::vector<q38::GdnRec> hr(std::max(NR, sets * B));
  std::vector<q38::GdnRec*> hrp(hr.size());
  std::vector<int> hq(B + 1), hn(B, pend);
  for (int b = 0; b <= B; b++) hq[b] = b * T;
  for (size_t k = 0; k < hs.size(); k++) {
    hs[k].S = S.data_ptr<float>() + k * 24 * 128 * 128;
    hs[k].conv = reinterpret_cast<q38::bf16*>(conv.data_ptr()) + k * 3 * 5120;
    hp[k] = st + k;
  }
  for (size_t k = 0; k < hr.size(); k++) {
    if (k < size_t(NR)) hr[k] = q38::gdn_rec_at(rec.data_ptr<uint8_t>() + k * q38::gdn_rec_bytes());
    hrp[k] = rc + k;
  }
  CUDA_CHECK(cudaMemcpy(st, hs.data(), hs.size() * sizeof(q38::GdnState), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pt, hp.data(), hp.size() * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rc, hr.data(), hr.size() * sizeof(q38::GdnRec), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rp, hrp.data(), hrp.size() * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(seq, hq.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(npd, hn.data(), B * sizeof(int), cudaMemcpyHostToDevice));
  auto s = at::cuda::getCurrentCUDAStream().stream();
  auto P = [](torch::Tensor t) { return reinterpret_cast<q38::bf16*>(t.data_ptr()); };
  const auto& g = L.at(i);
  auto call = [&](int k) {
    const size_t at = size_t(k % sets) * B;
    if (spec && pend >= 0)
      q38::gdn_core_spec_pending(g, P(qkvz), P(ba), P(core), pt + at, rp + at, rp + size_t(sets) * B + at, npd, seq, B,
                                 M, s);
    else if (spec) q38::gdn_core_spec(g, P(qkvz), P(ba), P(core), pt + at, rp + at, seq, B, M, s);
    else q38::gdn_core(g, P(qkvz), P(ba), P(core), pt + at, seq, B, M, s);
  };
  for (int k = 0; k < sets; k++) call(k);
  auto launches = [&] {
    for (int k = 0; k < iters; k++) call(k);
  };
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  double us = 0;
  if (mode == 2) {
    std::vector<float> t(iters);
    for (int k = 0; k < iters; k++) {
      cudaEventRecord(e0, s);
      call(k);
      cudaEventRecord(e1, s);
      CUDA_CHECK(cudaEventSynchronize(e1));
      cudaEventElapsedTime(&t[k], e0, e1);
    }
    std::nth_element(t.begin(), t.begin() + iters / 2, t.end());
    us = 1000.0 * t[iters / 2];
  } else {
    cudaGraphExec_t ge = nullptr;
    if (mode == 1) {
      cudaStream_t cs;
      CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
      cudaGraph_t gr;
      std::swap(s, cs);
      CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      launches();
      CUDA_CHECK(cudaStreamEndCapture(s, &gr));
      std::swap(s, cs);
      cudaStreamDestroy(cs);
      CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
      cudaGraphDestroy(gr);
      CUDA_CHECK(cudaGraphLaunch(ge, s));  // warm
    }
    cudaEventRecord(e0, s);
    if (ge) CUDA_CHECK(cudaGraphLaunch(ge, s));
    else launches();
    cudaEventRecord(e1, s);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    us = 1000.0 * ms / iters;
    if (ge) cudaGraphExecDestroy(ge);
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  cudaFree(st), cudaFree(pt), cudaFree(rc), cudaFree(rp), cudaFree(seq), cudaFree(npd);
  return us;
}
double bench(int i, int B, int T, int iters, int sets, int mode) {
  return bench_impl(i, B, T, iters, sets, mode, false);
}
double bench_spec(int i, int B, int T, int iters, int sets, int mode) {
  return bench_impl(i, B, T, iters, sets, mode, true);
}
double bench_spec_pending(int i, int B, int T, int pend, int iters, int sets, int mode) {
  return bench_impl(i, B, T, iters, sets, mode, true, pend);
}

// µs per verify round of the GDN part: `layers` layers (layer i's weights) × B sequences of T rows, keeping n rows per
// round. fused = false: today — gdn_core_spec per layer, then one gdn_commit of all layers; true: P020 —
// gdn_core_spec_pending per layer (records double-buffered, the parity alternating per round, n pending). One CUDA
// graph of `rounds` rounds (even) after a warm replay; states random small floats, 36 · B · 1.57 MB ≫ L2.
double bench_round(int i, int layers, int B, int T, int n, int rounds, bool fused) {
  const int M = B * T, P = layers * B;
  auto o = torch::TensorOptions().device(torch::kCUDA);
  auto qkvz = (torch::randn({M, 8192}, o) * 0.5).to(torch::kBFloat16);
  auto ba = torch::randn({M, 48}, o).to(torch::kBFloat16);
  auto core = torch::empty({M, 3072}, o.dtype(torch::kBFloat16));
  auto S = torch::randn({int64_t(P), 24, 128, 128}, o) * 0.01;
  auto conv = torch::zeros({int64_t(P), 3, 5120}, o.dtype(torch::kBFloat16));
  auto rec = torch::empty({2 * int64_t(P), rec_bytes()}, o.dtype(torch::kUInt8));
  rec.view(torch::kFloat32).normal_(0.f, 0.01f);
  q38::GdnState *st = nullptr, **pt = nullptr;
  q38::GdnRec *rc = nullptr, **rp = nullptr;
  int *seq = nullptr, *nn = nullptr;
  CUDA_CHECK(cudaMalloc(&st, P * sizeof(q38::GdnState)));
  CUDA_CHECK(cudaMalloc(&pt, P * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&rc, 2 * P * sizeof(q38::GdnRec)));
  CUDA_CHECK(cudaMalloc(&rp, 2 * P * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&seq, (B + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&nn, B * sizeof(int)));
  std::vector<q38::GdnState> hs(P);
  std::vector<q38::GdnState*> hp(P);
  std::vector<q38::GdnRec> hr(2 * P);
  std::vector<q38::GdnRec*> hrp(2 * P);
  std::vector<int> hq(B + 1), hn(B, n);
  for (int b = 0; b <= B; b++) hq[b] = b * T;
  for (int k = 0; k < P; k++) {
    hs[k].S = S.data_ptr<float>() + size_t(k) * 24 * 128 * 128;
    hs[k].conv = reinterpret_cast<q38::bf16*>(conv.data_ptr()) + size_t(k) * 3 * 5120;
    hp[k] = st + k;
  }
  for (int k = 0; k < 2 * P; k++) {  // record (parity k / P, pair k % P)
    hr[k] = q38::gdn_rec_at(rec.data_ptr<uint8_t>() + size_t(k) * q38::gdn_rec_bytes());
    hrp[k] = rc + k;
  }
  CUDA_CHECK(cudaMemcpy(st, hs.data(), hs.size() * sizeof(q38::GdnState), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pt, hp.data(), hp.size() * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rc, hr.data(), hr.size() * sizeof(q38::GdnRec), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(rp, hrp.data(), hrp.size() * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(seq, hq.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(nn, hn.data(), B * sizeof(int), cudaMemcpyHostToDevice));
  auto Pb = [](torch::Tensor t) { return reinterpret_cast<q38::bf16*>(t.data_ptr()); };
  const auto& g = L.at(i);
  auto round = [&](int r, cudaStream_t s) {
    const int par = r & 1;
    for (int l = 0; l < layers; l++) {
      if (fused)
        q38::gdn_core_spec_pending(g, Pb(qkvz), Pb(ba), Pb(core), pt + l * B, rp + (par ^ 1) * P + l * B,
                                   rp + par * P + l * B, nn, seq, B, M, s);
      else q38::gdn_core_spec(g, Pb(qkvz), Pb(ba), Pb(core), pt + l * B, rp + l * B, seq, B, M, s);
    }
    if (!fused) q38::gdn_commit(pt, rp, nn, layers, B, s);
  };
  auto s = at::cuda::getCurrentCUDAStream().stream();
  cudaStream_t cs;
  CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  cudaGraph_t gr;
  cudaGraphExec_t ge = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
  for (int r = 0; r < rounds; r++) round(r, cs);
  CUDA_CHECK(cudaStreamEndCapture(cs, &gr));
  cudaStreamDestroy(cs);
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  cudaGraphDestroy(gr);
  CUDA_CHECK(cudaGraphLaunch(ge, s));  // warm
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  cudaEventRecord(e0, s);
  CUDA_CHECK(cudaGraphLaunch(ge, s));
  cudaEventRecord(e1, s);
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  cudaGraphExecDestroy(ge);
  cudaFree(st), cudaFree(pt), cudaFree(rc), cudaFree(rp), cudaFree(seq), cudaFree(nn);
  return 1000.0 * ms / rounds;
}

// µs per gdn_commit of `layers` layers × B sequences keeping n rows each: mean of `iters` back-to-back launches over 2
// state sets cycled (a set is ≫ L2, so S comes from DRAM as in the real step); S and records random small floats.
double bench_commit(int layers, int B, int n, int iters) {
  const int P = layers * B, sets = 2;
  auto o = torch::TensorOptions().device(torch::kCUDA);
  auto S = torch::randn({int64_t(sets) * P, 24, 128, 128}, o) * 0.01;
  auto conv = torch::zeros({int64_t(sets) * P, 3, 5120}, o.dtype(torch::kBFloat16));
  auto rec = torch::empty({int64_t(sets) * P, rec_bytes()}, o.dtype(torch::kUInt8));
  rec.view(torch::kFloat32).normal_(0.f, 0.01f);  // e^g, δ, k: small finite floats (x is only copied)
  std::vector<int> nn(B, n), idx(size_t(sets) * P);
  for (size_t k = 0; k < idx.size(); k++) idx[k] = int(k);
  auto s = at::cuda::getCurrentCUDAStream().stream();
  put_ptrs(S, conv, &rec, idx, s);
  CUDA_CHECK(cudaMemcpyAsync(d_seq, nn.data(), B * sizeof(int), cudaMemcpyHostToDevice, s));
  auto call = [&](int k) {
    q38::gdn_commit(d_ptr + size_t(k % sets) * P, d_rp + size_t(k % sets) * P, d_seq, layers, B, s);
  };
  for (int k = 0; k < sets; k++) call(k);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  cudaEventRecord(e0, s);
  for (int k = 0; k < iters; k++) call(k);
  cudaEventRecord(e1, s);
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.f;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  return 1000.0 * ms / iters;
}
// P005 gdn_fast_prefill on/off inside this process (base vs patch); min_rows: rows per sequence from which the patch
// kernel is taken (engine default 32; 1 = every call). Returns the previous (on, min_rows).
std::tuple<bool, int> set_patch(bool on, int min_rows) {
  const bool was = q38::patch::on(q38::patch::gdn_fast_prefill);
  q38::patch::set(q38::patch::gdn_fast_prefill, on);
  return {was, q38::gdn_set_fast_min_rows(min_rows)};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("upload", &upload);
  m.def("run", &run);
  m.def("bench", &bench);
  m.def("set_patch", &set_patch);
  m.def("rec_bytes", &rec_bytes);
  m.def("run_spec", &run_spec);
  m.def("commit", &commit);
  m.def("bench_spec", &bench_spec);
  m.def("bench_commit", &bench_commit);
  m.def("run_spec_pending", &run_spec_pending);
  m.def("flush", &flush);
  m.def("bench_spec_pending", &bench_spec_pending);
  m.def("bench_round", &bench_round);
}
