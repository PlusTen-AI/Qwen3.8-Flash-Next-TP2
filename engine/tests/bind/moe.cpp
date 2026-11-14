// Test binding for the MoE family (engine/src/kernels/moe.cu): upload per (rank, layer), forward with debug outputs,
// silu_and_mul check, CUDA-graph timing, P014 / P017 copies, P022 routing state.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <map>
#include <memory>

#include "kernels/moe/moe.h"
#include "core/log.h"
#include "model/patches.h"

namespace {
std::map<int, std::unique_ptr<q38::Weights>> Ws;
std::map<int, q38::MoeW*> Hs;
int next_h = 1;

void open(const std::string& model_dir, const std::string& table, int rank) {
  if (!Ws.count(rank)) Ws[rank] = std::make_unique<q38::Weights>(model_dir, table, rank);
}

int upload(int rank, int layer) {
  Hs[next_h] = q38::moe_upload(*Ws.at(rank), layer);
  CUDA_CHECK(cudaDeviceSynchronize());
  return next_h++;
}

void free_(int h) {
  CUDA_CHECK(cudaDeviceSynchronize());
  q38::moe_free(Hs.at(h));
  Hs.erase(h);
}

// v [M,2560] bf16 cuda → (out, topk ids [M,10], topk w [M,10], logits [M,512], routed, shared [M,2560], h_shared [M,320]
// fragment order, distinct experts)
std::vector<torch::Tensor> forward(int h, torch::Tensor v) {
  TORCH_CHECK(v.is_cuda() && v.dtype() == torch::kBFloat16 && v.dim() == 2 && v.size(1) == q38::kHidden && v.is_contiguous());
  const int M = v.size(0);
  auto o = v.options();
  auto out = torch::empty({M, q38::kHidden}, o), routed = torch::empty_like(out), shared = torch::empty_like(out);
  auto logits = torch::empty({M, q38::kExperts}, o), hsh = torch::empty({M, 320}, o);
  auto ids = torch::empty({M, q38::kTopK}, o.dtype(torch::kInt32));
  auto tw = torch::empty({M, q38::kTopK}, o.dtype(torch::kFloat32));
  auto nu = torch::empty({1}, o.dtype(torch::kInt32));
  q38::MoeDbg d;
  d.logits = reinterpret_cast<q38::bf16*>(logits.data_ptr());
  d.routed = reinterpret_cast<q38::bf16*>(routed.data_ptr());
  d.shared = reinterpret_cast<q38::bf16*>(shared.data_ptr());
  d.h_shared = reinterpret_cast<q38::bf16*>(hsh.data_ptr());
  d.n_experts = nu.data_ptr<int>();
  q38::moe_forward_dbg(Hs.at(h), reinterpret_cast<const q38::bf16*>(v.data_ptr()), reinterpret_cast<q38::bf16*>(out.data_ptr()),
                       M, ids.data_ptr<int>(), tw.data_ptr<float>(), d, at::cuda::getCurrentCUDAStream());
  return {out, ids, tw, logits, routed, shared, hsh, nu};
}

// production entry point only (no debug outputs)
torch::Tensor forward_plain(int h, torch::Tensor v) {
  const int M = v.size(0);
  auto out = torch::empty({M, q38::kHidden}, v.options());
  q38::moe_forward(Hs.at(h), reinterpret_cast<const q38::bf16*>(v.data_ptr()), reinterpret_cast<q38::bf16*>(out.data_ptr()), M,
                   nullptr, nullptr, at::cuda::getCurrentCUDAStream());
  return out;
}

torch::Tensor silu_mul(torch::Tensor gu) {
  const int M = gu.size(0), n = gu.size(1) / 2;
  auto h = torch::empty({M, n}, gu.options());
  q38::moe_silu_mul(reinterpret_cast<const q38::bf16*>(gu.data_ptr()), reinterpret_cast<q38::bf16*>(h.data_ptr()), M, n,
                    at::cuda::getCurrentCUDAStream());
  return h;
}

// One CUDA graph = one moe_forward per handle in `hs` (inputs vs[i]); returns the best of `reps` graph replays
// (each `iters` launches) in µs per graph.
double bench(std::vector<int> hs, std::vector<torch::Tensor> vs, int iters, int reps) {
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  std::vector<torch::Tensor> outs;
  for (auto& v : vs) outs.push_back(torch::empty_like(v));
  auto run = [&] {
    for (size_t i = 0; i < hs.size(); i++)
      q38::moe_forward(Hs.at(hs[i]), reinterpret_cast<const q38::bf16*>(vs[i].data_ptr()),
                       reinterpret_cast<q38::bf16*>(outs[i].data_ptr()), vs[i].size(0), nullptr, nullptr, s);
  };
  run();
  CUDA_CHECK(cudaStreamSynchronize(s));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  run();
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  for (int i = 0; i < 3; i++) CUDA_CHECK(cudaGraphLaunch(ge, s));
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    CUDA_CHECK(cudaEventRecord(e0, s));
    for (int i = 0; i < iters; i++) CUDA_CHECK(cudaGraphLaunch(ge, s));
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    best = std::min(best, 1e3 * ms / iters);
  }
  CUDA_CHECK(cudaGraphExecDestroy(ge));
  CUDA_CHECK(cudaGraphDestroy(g));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaStreamDestroy(s));
  return best;
}

// large-M call timed per launch (ms): router, top-k, scan, place, routed up, routed down + shared up, shared down + combine
std::vector<double> prof(int h, torch::Tensor v) {
  const int M = v.size(0);
  auto out = torch::empty({M, q38::kHidden}, v.options());
  float ms[7];
  q38::moe_forward_prof(Hs.at(h), reinterpret_cast<const q38::bf16*>(v.data_ptr()), reinterpret_cast<q38::bf16*>(out.data_ptr()),
                        M, ms, at::cuda::getCurrentCUDAStream());
  return std::vector<double>(ms, ms + 7);
}

// numbered patches (model/patches.def): switch / read inside this process (base vs patch comparisons)
void patch_set(int id, bool on) { q38::patch::set(q38::patch::Id(id), on); }
bool patch_on(int id) { return q38::patch::on(q38::patch::Id(id)); }

std::vector<int> frag_perm(int K) {  // perm[k] = position of element k in a fragment-ordered row
  std::vector<int> p(K);
  for (int k = 0; k < K; k++) p[k] = q38::moe_frag_off(k);
  return p;
}

// P014 moe_scale4
int upload_mtp(int rank) {
  Hs[next_h] = q38::moe_upload(*Ws.at(rank), "mtp.layers.0.mlp.", true);
  CUDA_CHECK(cudaDeviceSynchronize());
  return next_h++;
}
constexpr int64_t kScaleBytes = 102400 + 51200;  // one expert's expanded scales (gate/up, down)
std::vector<long long> p014_info(int h) {
  std::vector<long long> v(5);
  q38::moe_p014_info(Hs.at(h), v.data());
  return v;
}
// (decoded as k_experts does, the block's expanded bytes), uint8 cuda [512, 153600] each
std::vector<torch::Tensor> p014_unpack(int h) {
  auto o = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto dec = torch::empty({q38::kExperts, kScaleBytes}, o), ref = torch::empty_like(dec);
  q38::moe_p014_unpack(Hs.at(h), dec.data_ptr<uint8_t>(), ref.data_ptr<uint8_t>(), at::cuda::getCurrentCUDAStream());
  return {dec, ref};
}
torch::Tensor p014_get(int h, int e) {
  auto t = torch::empty({kScaleBytes}, torch::kUInt8);
  q38::moe_p014_get_scales(Hs.at(h), e, t.data_ptr<uint8_t>());
  return t;
}
std::vector<long long> p014_set(int h, int e, torch::Tensor t) {
  TORCH_CHECK(t.device().is_cpu() && t.dtype() == torch::kUInt8 && t.numel() == kScaleBytes && t.is_contiguous());
  std::vector<long long> v(5);
  q38::moe_p014_set_scales(Hs.at(h), e, t.data_ptr<uint8_t>(), v.data());
  return v;
}

// P017 router_pack12
constexpr int64_t kRouterVals = int64_t(q38::kExperts) * q38::kHidden;
std::vector<long long> p017_info(int h) {
  std::vector<long long> v(5);
  q38::moe_p017_info(Hs.at(h), v.data());
  return v;
}
torch::Tensor p017_unpack(int h) {  // bf16 cuda [512, 2560]
  auto out = torch::empty({q38::kExperts, q38::kHidden}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  q38::moe_p017_unpack(Hs.at(h), reinterpret_cast<q38::bf16*>(out.data_ptr()), at::cuda::getCurrentCUDAStream());
  return out;
}
torch::Tensor bits_of(torch::Tensor w) {  // bf16 / int16 / uint8 CPU router → contiguous int16 [512, 2560]
  w = w.contiguous();
  TORCH_CHECK(w.device().is_cpu() && w.numel() * w.element_size() == kRouterVals * 2);
  return w.view(torch::kInt16).view({q38::kExperts, q38::kHidden});
}
std::vector<long long> p017_set(int h, torch::Tensor w) {
  auto b = bits_of(w);
  std::vector<long long> v(5);
  q38::moe_p017_set_router(Hs.at(h), reinterpret_cast<const uint16_t*>(b.data_ptr()), v.data());
  return v;
}
std::tuple<torch::Tensor, std::vector<long long>> p017_roundtrip(torch::Tensor w) {
  auto b = bits_of(w);
  auto out = torch::empty({q38::kExperts, q38::kHidden}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  std::vector<long long> v(5);
  q38::moe_p017_roundtrip(reinterpret_cast<const uint16_t*>(b.data_ptr()), reinterpret_cast<q38::bf16*>(out.data_ptr()),
                          v.data(), at::cuda::getCurrentCUDAStream());
  return {out, v};
}

// P022 route_fast: the routing state the last decode call on h left (plan int32 [moe_p022_plan_ints], sig fp32 [M],
// xf bf16 [M, 2560] in B-fragment order); cuda tensors
std::vector<torch::Tensor> p022_state(int h, int M) {
  auto o = torch::TensorOptions().device(torch::kCUDA);
  auto plan = torch::empty({q38::moe_p022_plan_ints()}, o.dtype(torch::kInt32));
  auto sig = torch::empty({M}, o.dtype(torch::kFloat32));
  auto xf = torch::empty({M, q38::kHidden}, o.dtype(torch::kBFloat16));
  q38::moe_p022_state(Hs.at(h), M, plan.data_ptr<int>(), sig.data_ptr<float>(), reinterpret_cast<q38::bf16*>(xf.data_ptr()),
                      at::cuda::getCurrentCUDAStream());
  return {plan, sig, xf};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("upload", &upload);
  m.def("free", &free_);
  m.def("forward", &forward);
  m.def("forward_plain", &forward_plain);
  m.def("silu_mul", &silu_mul);
  m.def("bench", &bench);
  m.def("frag_perm", &frag_perm);
  m.def("prof", &prof);
  m.def("patch_set", &patch_set);
  m.def("patch_on", &patch_on);
  m.def("upload_mtp", &upload_mtp);
  m.def("p014_built", [](int h) { return q38::moe_p014_built(Hs.at(h)); });
  m.def("p014_info", &p014_info);
  m.def("p014_unpack", &p014_unpack);
  m.def("p014_get", &p014_get);
  m.def("p014_set", &p014_set);
  m.def("p017_built", [](int h) { return q38::moe_p017_built(Hs.at(h)); });
  m.def("p017_info", &p017_info);
  m.def("p017_unpack", &p017_unpack);
  m.def("p017_set", &p017_set);
  m.def("p017_roundtrip", &p017_roundtrip);
  m.def("p022_state", &p022_state);
}
