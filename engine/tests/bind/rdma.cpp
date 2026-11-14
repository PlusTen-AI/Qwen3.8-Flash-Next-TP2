// Test binding for P001 rdma_ar (engine/src/kernels/comm/rdma_ar.cu) next to the NCCL communicator (core/comm.cc):
// init both, in-place all-reduce on torch's current stream, timed back-to-back loops (eager / CUDA graph), and an
// all-reduce loop that runs in the background (a graph on its own stream) for the overlap test.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <map>
#include <memory>
#include <tuple>

#include "core/comm.h"
#include "core/log.h"
#include "kernels/comm/rdma_ar.h"

namespace {
std::unique_ptr<q38::Comm> C;
std::unique_ptr<q38::RdmaAR> R;
cudaStream_t S;
cudaEvent_t E0, E1;

// which: 0 = NCCL, 1 = RdmaAR
void ar_raw(q38::bf16* p, int64_t n, int which, cudaStream_t s) {
  if (which) R->all_reduce_bf16(p, size_t(n), s);
  else C->all_reduce_bf16(p, size_t(n), s);
}

// prio: the all-reduce stream S gets the highest priority (its short copy/add kernels go ahead of queued compute CTAs)
void init(int rank, const std::string& ip, int port, int64_t max_elems, bool prio) {
  int lo = 0, hi = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
  CUDA_CHECK(cudaStreamCreateWithPriority(&S, cudaStreamNonBlocking, prio ? hi : lo));
  CUDA_CHECK(cudaEventCreate(&E0));
  CUDA_CHECK(cudaEventCreate(&E1));
  C = std::make_unique<q38::Comm>(rank, ip, port);
  R = std::make_unique<q38::RdmaAR>(rank, ip, port + 1, size_t(max_elems));
}

void barrier() { C->barrier(S); }
int rails() { return R->rails(); }

// in place on torch's current stream (capturable from torch.cuda.graph), no sync
void ar(torch::Tensor t, int which) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16 && t.is_contiguous());
  ar_raw(reinterpret_cast<q38::bf16*>(t.data_ptr()), t.numel(), which, at::cuda::getCurrentCUDAStream().stream());
}

cudaGraphExec_t capture(q38::bf16* p, int64_t n, int iters, int which) {
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < iters; i++) ar_raw(p, n, which, S);
  CUDA_CHECK(cudaStreamEndCapture(S, &g));
  CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
  CUDA_CHECK(cudaGraphDestroy(g));
  return ge;
}

// best / median µs per call of `iters` back-to-back all-reduces of n elements (reps after a barrier each)
std::vector<double> time_ar(int64_t n, int iters, bool graph, int which, int reps) {
  auto t = torch::zeros({n}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));  // zeros stay zeros
  auto* p = reinterpret_cast<q38::bf16*>(t.data_ptr());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaGraphExec_t ge = graph ? capture(p, n, iters, which) : nullptr;
  std::vector<double> ts;
  for (int rep = 0; rep < reps + 1; rep++) {  // first rep = warm-up
    C->barrier(S);
    CUDA_CHECK(cudaEventRecord(E0, S));
    if (graph) CUDA_CHECK(cudaGraphLaunch(ge, S));
    else for (int i = 0; i < iters; i++) ar_raw(p, n, which, S);
    CUDA_CHECK(cudaEventRecord(E1, S));
    CUDA_CHECK(cudaEventSynchronize(E1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, E0, E1));
    if (rep) ts.push_back(ms * 1000.0 / iters);
  }
  if (ge) CUDA_CHECK(cudaGraphExecDestroy(ge));
  std::sort(ts.begin(), ts.end());
  return {ts.front(), ts[ts.size() / 2]};
}

// background loop for the overlap test: `count` all-reduces of t (a graph, captured once per key) on S
std::map<std::tuple<void*, int64_t, int, int>, cudaGraphExec_t> loops;
void loop_start(torch::Tensor t, int count, int which) {
  auto key = std::make_tuple(t.data_ptr(), t.numel(), count, which);
  if (!loops.count(key)) loops[key] = capture(reinterpret_cast<q38::bf16*>(t.data_ptr()), t.numel(), count, which);
  CUDA_CHECK(cudaEventRecord(E0, S));
  CUDA_CHECK(cudaGraphLaunch(loops[key], S));
  CUDA_CHECK(cudaEventRecord(E1, S));
}
double loop_wait() {  // ms of the loop
  CUDA_CHECK(cudaEventSynchronize(E1));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, E0, E1));
  return ms;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init);
  m.def("barrier", &barrier);
  m.def("rails", &rails);
  m.def("ar", &ar);
  m.def("time_ar", &time_ar);
  m.def("loop_start", &loop_start);
  m.def("loop_wait", &loop_wait);
}
