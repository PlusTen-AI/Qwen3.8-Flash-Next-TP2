// Test binding for the hc family (engine/src/kernels/hc.cu): HC modules, embedding, PLE, timing.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <chrono>
#include <map>
#include <memory>

#include "kernels/hc/hc.h"
#include "core/log.h"
#include "model/patches.h"

namespace {
using q38::bf16;
std::unique_ptr<q38::Weights> W;
std::map<std::string, q38::HcW> HC;  // key: prefix, or prefix + "#bf" (norm rounded to bf16 like the reference's param)
q38::PleW* P = nullptr;
q38::PleW* Pbf = nullptr;  // PLE norms rounded to bf16 (as the reference deployment does), test variant
bf16* embed = nullptr;
int rank_ = 0;
void* dbuf = nullptr;  // small device scratch for pointer / index arrays
size_t dcap = 0;

cudaStream_t st() { return at::cuda::getCurrentCUDAStream(); }
bf16* bp(const torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr()); }
const bf16* bpo(const c10::optional<torch::Tensor>& t) { return t ? reinterpret_cast<const bf16*>(t->data_ptr()) : nullptr; }
void* scratch(size_t n) {
  if (n > dcap) {
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dbuf);
    dcap = std::max(n, size_t(1) << 16);
    CUDA_CHECK(cudaMalloc(&dbuf, dcap));
  }
  return dbuf;
}

void open(const std::string& model_dir, const std::string& table, int rank) {
  W = std::make_unique<q38::Weights>(model_dir, table, rank);
  rank_ = rank;
}

// prefix e.g. "model.language_model.layers.3.attn_hyper_connection"; round_norm: fp32 norm rounded to bf16
std::string hc_load(const std::string& prefix, bool inject, bool round_norm) {
  const std::string key = prefix + (round_norm ? "#bf" : "");
  if (HC.count(key)) return key;
  if (!HC.count(prefix)) HC[prefix] = q38::hc_upload(*W, prefix, inject);
  if (round_norm) {
    q38::HcW w = HC[prefix];
    std::vector<float> g(10240);
    CUDA_CHECK(cudaMemcpy(g.data(), w.norm, g.size() * 4, cudaMemcpyDeviceToHost));
    for (auto& v : g) v = __bfloat162float(__float2bfloat16_rn(v));
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, g.size() * 4));
    CUDA_CHECK(cudaMemcpy(d, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
    w.norm = d;
    HC[key] = w;
  }
  return key;
}

torch::Tensor hc_norm(const std::string& key) {
  auto t = torch::empty({10240}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
  CUDA_CHECK(cudaMemcpy(t.data_ptr(), HC.at(key).norm, 10240 * 4, cudaMemcpyDeviceToDevice));
  return t;
}

// X [M,10240] in place; bo [M,2560] | None; inj [M,4] | None; u [M,2560]; inj_out [M,4] | None. Returns lora [M,320].
torch::Tensor combine_mix(const std::string& key, torch::Tensor X, c10::optional<torch::Tensor> bo,
                          c10::optional<torch::Tensor> inj, torch::Tensor u, c10::optional<torch::Tensor> inj_out) {
  const int M = int(X.size(0));
  q38::hc_combine_mix(HC.at(key), bp(X), bpo(bo), bpo(inj), bp(u), inj_out ? bp(*inj_out) : nullptr, M, st());
  auto l = torch::empty({M, 320}, X.options());
  CUDA_CHECK(cudaMemcpyAsync(l.data_ptr(), q38::hc_last_lora(), l.numel() * 2, cudaMemcpyDeviceToDevice, st()));
  return l;
}

// P003 hc_fused_prefill on/off inside this process (base vs patch comparisons); returns the previous state
bool p003(bool on) {
  const bool was = q38::patch::on(q38::patch::hc_fused_prefill);
  q38::patch::set(q38::patch::hc_fused_prefill, on);
  return was;
}

// P012 hc_pack12 on/off (decode kernels on the packed copy; modules uploaded while it was on keep their copy)
bool p012(bool on) {
  const bool was = q38::patch::on(q38::patch::hc_pack12);
  q38::patch::set(q38::patch::hc_pack12, on);
  return was;
}

// P012: the module uploaded with the patch on (key prefix + "#p12": always has the packed copy, whatever the switch)
std::string hc_load_packed(const std::string& prefix, bool inject) {
  const std::string key = prefix + "#p12";
  if (!HC.count(key)) {
    const bool was = p012(true);
    HC[key] = q38::hc_upload(*W, prefix, inject);
    p012(was);
  }
  return key;
}

// P012: copy of the module's packed device block (uint8, empty if the module has none)
torch::Tensor pack_blob(const std::string& key) {
  const auto& w = HC.at(key);
  const size_t n = q38::hc_pack_bytes(w);
  auto t = torch::empty({int64_t(n)}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));
  if (n) CUDA_CHECK(cudaMemcpy(t.data_ptr(), w.pk, n, cudaMemcpyDeviceToDevice));
  return t;
}

void combine(torch::Tensor X, torch::Tensor bo, c10::optional<torch::Tensor> inj) {
  q38::hc_combine(bp(X), bp(bo), bpo(inj), int(X.size(0)), st());
}

void expand(torch::Tensor x0, torch::Tensor X) { q38::hc_expand(bp(x0), bp(X), int(x0.size(0)), st()); }

// ids int32 [M] (device) → partial [M,2560] (this rank's vocab half)
void embed_lookup(torch::Tensor ids, torch::Tensor out) {
  if (!embed) {
    auto b = W->slice("model.language_model.embed_tokens.weight");
    CUDA_CHECK(cudaMalloc(&embed, b.size()));
    CUDA_CHECK(cudaMemcpy(embed, b.data(), b.size(), cudaMemcpyHostToDevice));
  }
  q38::embed_lookup(embed, rank_, ids.data_ptr<int>(), bp(out), int(ids.size(0)), st());
}

void ple_load() {
  if (!P) P = q38::ple_upload(*W, rank_);
}

// ids int32 [M], prev int32 [M,2] (device) → E partial [M,2560]; returns n-gram ids int64 [M,16]
torch::Tensor ple_lookup(torch::Tensor ids, torch::Tensor prev, torch::Tensor E) {
  const int M = int(ids.size(0));
  auto out = torch::empty({M, 16}, ids.options().dtype(torch::kInt64));
  q38::ple_ngram_ids(P, ids.data_ptr<int>(), prev.data_ptr<int>(), reinterpret_cast<long long*>(out.data_ptr()), M, st());
  q38::ple_lookup(P, ids.data_ptr<int>(), prev.data_ptr<int>(), bp(E), M, st());
  return out;
}

// X [M,10240] in place, E [M,2560] (all-reduced), state [S,9,10240] bf16 slots, slot [B], seq_start [B+1] (host).
// Returns kv [M,12800].
torch::Tensor ple_apply(torch::Tensor X, torch::Tensor E, torch::Tensor state, std::vector<int> slot,
                        std::vector<int> seq_start, bool bf_norms) {
  if (bf_norms && !Pbf) Pbf = q38::ple_debug_clone(P, true);
  q38::PleW* w = bf_norms ? Pbf : P;
  const int M = int(X.size(0)), B = int(slot.size());
  TORCH_CHECK(int(seq_start.size()) == B + 1 && seq_start[B] == M, "seq_start");
  std::vector<bf16*> ptr(B);
  for (int b = 0; b < B; b++) ptr[b] = bp(state) + size_t(slot[b]) * 9 * 10240;
  char* d = static_cast<char*>(scratch(B * sizeof(void*) + (B + 1) * sizeof(int) + 64));
  bf16** dptr = reinterpret_cast<bf16**>(d);
  int* dseq = reinterpret_cast<int*>(d + ((B * sizeof(void*) + 15) / 16) * 16);
  CUDA_CHECK(cudaMemcpyAsync(dptr, ptr.data(), B * sizeof(void*), cudaMemcpyHostToDevice, st()));
  CUDA_CHECK(cudaMemcpyAsync(dseq, seq_start.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice, st()));
  q38::ple_apply(w, bp(X), bp(E), dptr, dseq, B, M, st());
  auto kv = torch::empty({M <= 1024 ? M : 0, 12800}, X.options());  // kv workspace holds ≤ 1024 rows
  if (kv.numel()) CUDA_CHECK(cudaMemcpyAsync(kv.data_ptr(), q38::ple_last_kv(w), kv.numel() * 2, cudaMemcpyDeviceToDevice, st()));
  CUDA_CHECK(cudaStreamSynchronize(st()));
  return kv;
}

// Mean µs per hc_combine_mix over `iters` calls, cycling through `keys` (≥ 4 modules > L2), best of `reps`.
double bench_mix(std::vector<std::string> keys, int M, int iters, int reps) {
  auto o = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  auto X = torch::randn({M, 10240}, o), bo = torch::randn({M, 2560}, o), inj = torch::randn({M, 4}, o);
  auto u = torch::empty({M, 2560}, o), io = torch::empty({M, 4}, o);
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  double best = 1e30;
  const auto t0 = std::chrono::steady_clock::now();  // ~0.4 s warm-up (GPU clocks ramp up after idle)
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.4) {
    for (int i = 0; i < 64; i++) q38::hc_combine_mix(HC.at(keys[i % keys.size()]), bp(X), bp(bo), bp(inj), bp(u), bp(io), M, st());
    CUDA_CHECK(cudaStreamSynchronize(st()));
  }
  for (int r = 0; r < reps; r++) {
    cudaEventRecord(a, st());
    for (int i = 0; i < iters; i++) {
      const auto& w = HC.at(keys[i % keys.size()]);
      q38::hc_combine_mix(w, bp(X), bp(bo), bp(inj), bp(u), w.inject ? bp(io) : nullptr, M, st());
    }
    cudaEventRecord(b, st());
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    best = std::min(best, 1e3 * ms / iters);
  }
  cudaEventDestroy(a), cudaEventDestroy(b);
  return best;
}

// µs of one ple_lookup + ple_apply (fresh slot each run), best of `reps`
double bench_ple(int M, int reps) {
  auto o = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
  auto X = torch::randn({M, 10240}, o), E = torch::randn({M, 2560}, o) * 0.05;
  auto state = torch::zeros({1, 9, 10240}, o);
  auto ids = torch::randint(0, 248320, {M}, o.dtype(torch::kInt32)), prev = torch::randint(0, 248320, {M, 2}, o.dtype(torch::kInt32));
  std::vector<bf16*> ptr(1, bp(state));
  std::vector<int> seq = {0, M};
  char* d = static_cast<char*>(scratch(64));
  CUDA_CHECK(cudaMemcpy(d, ptr.data(), sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d + 16, seq.data(), 8, cudaMemcpyHostToDevice));
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    cudaEventRecord(a, st());
    q38::ple_lookup(P, ids.data_ptr<int>(), prev.data_ptr<int>(), bp(E), M, st());
    q38::ple_apply(P, bp(X), bp(E), reinterpret_cast<bf16**>(d), reinterpret_cast<int*>(d + 16), 1, M, st());
    cudaEventRecord(b, st());
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    best = std::min(best, 1e3 * double(ms));
  }
  cudaEventDestroy(a), cudaEventDestroy(b);
  return best;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("hc_load", &hc_load);
  m.def("hc_norm", &hc_norm);
  m.def("combine_mix", &combine_mix);
  m.def("combine", &combine);
  m.def("p003", &p003);
  m.def("p012", &p012);
  m.def("hc_load_packed", &hc_load_packed);
  m.def("pack_blob", &pack_blob);
  m.def("pack_stats", &q38::hc_pack_stats);
  m.def("expand", &expand);
  m.def("embed_lookup", &embed_lookup);
  m.def("ple_load", &ple_load);
  m.def("ple_lookup", &ple_lookup);
  m.def("ple_apply", &ple_apply);
  m.def("bench_mix", &bench_mix);
  m.def("bench_ple", &bench_ple);
}
