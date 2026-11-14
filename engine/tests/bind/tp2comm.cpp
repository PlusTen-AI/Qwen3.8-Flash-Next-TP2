// Test binding for the TP2 communicator (engine/src/comm.cc): init + timed small all-reduces / all-gathers.
#include <torch/extension.h>

#include <memory>

#include "core/comm.h"
#include "core/log.h"

namespace {
std::unique_ptr<q38::Comm> C;
cudaStream_t S;

void init(int rank, const std::string& ip, int port) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&S, cudaStreamNonBlocking));
  C = std::make_unique<q38::Comm>(rank, ip, port);
}

// all-reduce of a bf16 tensor in place (sum)
void all_reduce(torch::Tensor t) {
  C->all_reduce_bf16(reinterpret_cast<q38::bf16*>(t.data_ptr()), t.numel(), S);
  CUDA_CHECK(cudaStreamSynchronize(S));
}

// best / median µs of `iters` back-to-back bf16 all-reduces of n elements, captured in a CUDA graph (decode-like)
std::vector<double> time_ar(int64_t n, int iters, bool graph) {
  auto t = torch::ones({n}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  auto* p = reinterpret_cast<q38::bf16*>(t.data_ptr());
  cudaGraphExec_t ge = nullptr;
  if (graph) {
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iters; i++) C->all_reduce_bf16(p, n, S);
    CUDA_CHECK(cudaStreamEndCapture(S, &g));
    CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  }
  std::vector<double> ts;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int rep = 0; rep < 7; rep++) {
    C->barrier(S);
    cudaEventRecord(a, S);
    if (graph) CUDA_CHECK(cudaGraphLaunch(ge, S));
    else for (int i = 0; i < iters; i++) C->all_reduce_bf16(p, n, S);
    cudaEventRecord(b, S);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    ts.push_back(ms * 1000.0 / iters);
  }
  std::sort(ts.begin(), ts.end());
  return {ts.front(), ts[ts.size() / 2]};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init);
  m.def("all_reduce", &all_reduce);
  m.def("time_ar", &time_ar);
}
