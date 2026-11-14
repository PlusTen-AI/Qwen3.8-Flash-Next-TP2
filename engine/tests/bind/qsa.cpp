// Test binding for the QSA core (engine/src/kernels/qsa.cu): upload, caches on torch tensors, run, test hooks.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <map>
#include <memory>

#include "kernels/qsa/qsa.h"
#include "core/log.h"
#include "model/patches.h"

namespace {
std::unique_ptr<q38::Weights> W;
std::map<int, q38::QsaW> L;

q38::bf16* P(const torch::Tensor& t) { return reinterpret_cast<q38::bf16*>(t.data_ptr()); }

void open(const std::string& model_dir, const std::string& table, int rank) {
  W = std::make_unique<q38::Weights>(model_dir, table, rank);
}

// Uploads layer i (core fields); returns copies (q_norm [256], k_norm [256], iq_norm [128], ik_norm [128]).
std::vector<torch::Tensor> upload(int i) {
  if (!L.count(i)) L[i] = q38::qsa_upload(*W, i);
  const auto& w = L[i];
  auto o = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  std::vector<torch::Tensor> r;
  const std::vector<std::pair<const q38::bf16*, int>> ws = {{w.q_norm, 256}, {w.k_norm, 256}, {w.iq_norm, 128},
                                                            {w.ik_norm, 128}};
  for (auto [p, n] : ws) {
    auto t = torch::empty({n}, o);
    CUDA_CHECK(cudaMemcpy(t.data_ptr(), p, n * 2, cudaMemcpyDeviceToDevice));
    r.push_back(t);
  }
  return r;
}

// RoPE table rows [0, n) as a bf16 [n, 64] copy (needs one upload first).
torch::Tensor rope(int n) {
  TORCH_CHECK(!L.empty() && n <= q38::qsa_rope().rows, "rope");
  auto t = torch::empty({n, 64}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
  CUDA_CHECK(cudaMemcpy(t.data_ptr(), L.begin()->second.rope, size_t(n) * 64 * 2, cudaMemcpyDeviceToDevice));
  return t;
}

// qsa_rope_set (shared table + workspace for `rows`); the uploaded layers take the new table.
void rope_set(double factor, int original, int ramp_len, int rows) {
  const q38::bf16* t = q38::qsa_rope_set(q38::QsaRope{factor, original, ramp_len, rows});
  for (auto& kv : L) kv.second.rope = t;
}

// The table for these parameters (qsa_rope_fill, the same kernel) as a new bf16 [rows, 64] tensor.
torch::Tensor rope_fill(double factor, int original, int ramp_len, int rows) {
  auto t = torch::empty({rows, 64}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
  q38::qsa_rope_fill(q38::QsaRope{factor, original, ramp_len, rows}, reinterpret_cast<q38::bf16*>(t.data_ptr()));
  return t;
}

// Device argument block for a set of caches: caches given as (kv [cap,512], ik_raw [cap,128], ik_comp [cap/4,128])
// bf16 tensors per sequence. Returns (structs u8, pointers i64, seq_start i32) on the GPU; keep them alive while used.
std::vector<torch::Tensor> args(std::vector<std::vector<torch::Tensor>> caches, std::vector<int> seq_start) {
  const int B = int(caches.size());
  TORCH_CHECK(seq_start.size() == size_t(B + 1), "seq_start");
  auto o = torch::TensorOptions().device(torch::kCUDA);
  auto st = torch::empty({int64_t(B * sizeof(q38::QsaCache))}, o.dtype(torch::kUInt8));
  auto pt = torch::empty({B}, o.dtype(torch::kInt64));
  auto sq = torch::tensor(std::vector<int32_t>(seq_start.begin(), seq_start.end()), torch::kInt32).to(torch::kCUDA);
  std::vector<q38::QsaCache> h(B);
  std::vector<q38::QsaCache*> hp(B);
  for (int b = 0; b < B; b++) {
    auto& c = caches[b];
    TORCH_CHECK(c.size() == 3 && c[0].is_contiguous() && c[1].is_contiguous() && c[2].is_contiguous(), "cache");
    h[b] = q38::QsaCache{P(c[0]), P(c[1]), P(c[2]), int(c[0].size(0))};
    hp[b] = reinterpret_cast<q38::QsaCache*>(st.data_ptr<uint8_t>()) + b;
  }
  CUDA_CHECK(cudaMemcpy(st.data_ptr(), h.data(), B * sizeof(q38::QsaCache), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pt.data_ptr(), hp.data(), B * sizeof(void*), cudaMemcpyHostToDevice));
  return {st, pt, sq};
}

// qkv [M,7296] bf16, pos [M] i32, attn [M,3072] bf16 (out), ptrs/seq from args(), sel [M,2052] i32 or None,
// max_pos: qsa_core's host hint (largest row position, −1 = unknown). No sync (timing loops); call
// torch.cuda.synchronize() before reading.
void run(int i, torch::Tensor qkv, torch::Tensor pos, torch::Tensor attn, torch::Tensor ptrs, torch::Tensor seq,
         c10::optional<torch::Tensor> sel, int max_pos) {
  const int M = int(qkv.size(0)), B = int(seq.size(0)) - 1;
  TORCH_CHECK(qkv.is_contiguous() && pos.is_contiguous() && attn.is_contiguous() && qkv.size(1) == 7296, "args");
  TORCH_CHECK(pos.scalar_type() == torch::kInt32 && pos.size(0) == M, "pos");
  q38::qsa_core(L.at(i), P(qkv), pos.data_ptr<int>(), P(attn),
                reinterpret_cast<q38::QsaCache* const*>(ptrs.data_ptr<int64_t>()), seq.data_ptr<int>(), B, M,
                sel ? sel->data_ptr<int>() : nullptr, at::cuda::getCurrentCUDAStream().stream(), max_pos);
}

// Copies of the last call's Q [M,12,256] and indexer q [M,4,128].
std::vector<torch::Tensor> last_q(int M) {
  const q38::bf16 *q, *qi;
  q38::qsa_last_q(&q, &qi);
  auto o = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  auto a = torch::empty({M, 12, 256}, o), b = torch::empty({M, 4, 128}, o);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(a.data_ptr(), q, size_t(M) * 12 * 256 * 2, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr(), qi, size_t(M) * 4 * 128 * 2, cudaMemcpyDeviceToDevice));
  return {a, b};
}

// P021: the decode attention stage alone — variant 0 base (k_attn + k_merge), 1 k_pattn, 2 P021 as qsa_core runs it
// (with the max_pos hint), 3 the P021 cluster kernel for any call. q [M,12,256] bf16, qkv [M,7296] bf16 (gates), sel
// [M,2052] i32, attn [M,3072] bf16 (out). No sync.
void attn_stage(int variant, torch::Tensor q, torch::Tensor qkv, torch::Tensor sel, torch::Tensor ptrs, torch::Tensor seq,
                torch::Tensor attn, int max_pos) {
  const int M = int(q.size(0)), B = int(seq.size(0)) - 1;
  TORCH_CHECK(q.is_contiguous() && qkv.is_contiguous() && sel.is_contiguous() && attn.is_contiguous(), "contiguous");
  TORCH_CHECK(q.numel() == int64_t(M) * 12 * 256 && qkv.size(0) == M && qkv.size(1) == 7296 && sel.size(0) == M &&
                  sel.size(1) == 2052 && attn.numel() == int64_t(M) * 3072 && sel.scalar_type() == torch::kInt32,
              "shapes");
  q38::qsa_decode_attn_stage(variant, P(q), P(qkv), sel.data_ptr<int>(),
                             reinterpret_cast<q38::QsaCache* const*>(ptrs.data_ptr<int64_t>()), seq.data_ptr<int>(), B, M,
                             P(attn), at::cuda::getCurrentCUDAStream().stream(), max_pos);
}

// numbered patches (model/patches.def): switch / read inside this process (base vs patch comparisons)
void patch_set(const std::string& name, bool on) { q38::patch::set(name, on); }
bool patch_on(int id) { return q38::patch::on(q38::patch::Id(id)); }

// P009: candidates per (row, key split) of the last round's first `rows` rows [rows, splits] and the fallback row count
// of the last call.
std::tuple<torch::Tensor, int> p009_last(int rows) {
  const int *cnt, *fb;
  int ks;
  q38::qsa_p009_last(&cnt, &fb, &ks);
  TORCH_CHECK(cnt && fb, "P009 not run");
  auto t = torch::empty({rows, ks}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
  int n = 0;
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(t.data_ptr(), cnt, size_t(rows) * ks * 4, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(&n, fb, 4, cudaMemcpyDeviceToHost));
  return {t, n};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("upload", &upload);
  m.def("rope", &rope);
  m.def("rope_set", &rope_set, py::arg("factor"), py::arg("original"), py::arg("ramp_len"), py::arg("rows"));
  m.def("rope_fill", &rope_fill, py::arg("factor"), py::arg("original"), py::arg("ramp_len"), py::arg("rows"));
  m.def("rope_rows", []() { return q38::qsa_rope().rows; });
  m.def("args", &args);
  m.def("run", &run, py::arg("i"), py::arg("qkv"), py::arg("pos"), py::arg("attn"), py::arg("ptrs"), py::arg("seq"),
        py::arg("sel") = py::none(), py::arg("max_pos") = -1);
  m.def("last_q", &last_q);
  m.def("reserve", &q38::qsa_reserve);
  m.def("patch_set", &patch_set);
  m.def("patch_on", &patch_on);
  m.def("p009_last", &p009_last);
  m.def("attn_stage", &attn_stage, py::arg("variant"), py::arg("q"), py::arg("qkv"), py::arg("sel"), py::arg("ptrs"),
        py::arg("seq"), py::arg("attn"), py::arg("max_pos") = -1);
}
