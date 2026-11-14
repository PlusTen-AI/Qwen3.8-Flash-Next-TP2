// Test binding for the linear family (engine/src/kernels/linear.cu): upload from the checkpoint (either rank), build
// from host bytes, forward, CUDA-graph timing, patch switches (base vs patch comparisons in one process); P013 12-bit
// copies (pack_all: every bf16 Lin made or uploaded from then on also gets one).
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>

#include "kernels/linear/linear.h"
#include "core/log.h"
#include "model/patches.h"

namespace {
std::string g_dir, g_table;
std::unique_ptr<q38::Weights> g_w[2];
std::vector<q38::Lin> g_lin;
bool g_pack_all = false;

void open(const std::string& model_dir, const std::string& table) { g_dir = model_dir, g_table = table; }

int add(const q38::Lin& L) {
  g_lin.push_back(L);
  return int(g_lin.size()) - 1;
}

int upload(int rank, const std::vector<std::string>& names) {
  TORCH_CHECK(rank == 0 || rank == 1);
  if (!g_w[rank]) g_w[rank] = std::make_unique<q38::Weights>(g_dir, g_table, rank);
  return add(q38::lin_upload(*g_w[rank], names, g_pack_all && g_w[rank]->info(names.at(0)).dtype == "BF16"));
}

// w: uint8 CPU [n, k] (e4m3) or [n, 2k] (bf16 bytes); s: uint8 CPU [n, k/32] or empty
int make(int fmt, torch::Tensor w, torch::Tensor s, int n, int k, int split) {
  w = w.contiguous(), s = s.contiguous();
  TORCH_CHECK(w.dtype() == torch::kUInt8 && !w.is_cuda());
  return add(q38::lin_make(fmt, n, k, w.data_ptr<uint8_t>(), s.numel() ? s.data_ptr<uint8_t>() : nullptr, split,
                           g_pack_all && fmt == q38::WMat::BF16));
}

// P013: bf16 bytes [n, 2k] CPU → Lin with the bf16 copy and the 12-bit copy (packing checked bit-exact on the host)
int make12(torch::Tensor w, int n, int k, int split) {
  w = w.contiguous();
  TORCH_CHECK(w.dtype() == torch::kUInt8 && !w.is_cuda() && w.numel() == int64_t(n) * k * 2);
  return add(q38::lin_make(q38::WMat::BF16, n, k, w.data_ptr<uint8_t>(), nullptr, split, true));
}
void pack_all(bool on) { g_pack_all = on; }

// the 12-bit copy unpacked on the device: bf16 [n, k]
torch::Tensor unpack12(int h) {
  const auto& L = g_lin.at(h);
  auto y = torch::empty({L.n, L.k}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  q38::lin_pack12_unpack(L, reinterpret_cast<q38::bf16*>(y.data_ptr()), c10::cuda::getCurrentCUDAStream().stream());
  return y;
}

// [exceptions, most in one 16-row tile, device bytes of the copy (blocks + bases + offsets + exceptions)]
std::vector<int64_t> pack_info(int h) {
  const auto& L = g_lin.at(h);
  if (!L.pk) return {-1, -1, 0};
  const int tiles = L.npad / 16;
  std::vector<uint32_t> xo(tiles + 1);
  CUDA_CHECK(cudaMemcpy(xo.data(), L.pk_xo, xo.size() * 4, cudaMemcpyDeviceToHost));
  int64_t mx = 0;
  for (int t = 0; t < tiles; t++) mx = std::max<int64_t>(mx, xo[t + 1] - xo[t]);
  const int64_t bytes = int64_t(tiles) * (L.k / 32) * 768 + L.npad + 4 * (tiles + 1) + 4 * int64_t(L.pk_nx);
  return {L.pk_nx, mx, bytes};
}

// w: bf16 CPU [n, k] → NVFP4 copy (lin_make_nvfp4, draft-only format)
int make_nvfp4(torch::Tensor w, int n, int k) {
  w = w.contiguous();
  TORCH_CHECK(w.dtype() == torch::kBFloat16 && !w.is_cuda() && w.numel() == int64_t(n) * k);
  return add(q38::lin_make_nvfp4(n, k, reinterpret_cast<const q38::bf16*>(w.data_ptr())));
}

std::vector<int64_t> info(int h) {
  const auto& L = g_lin.at(h);
  return {L.fmt, L.n, L.k, L.npad, L.split};
}

void free_(int h) { q38::lin_free(g_lin.at(h)); }

torch::Tensor forward(int h, torch::Tensor x, bool f32) {
  const auto& L = g_lin.at(h);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.size(1) == L.k && x.is_contiguous());
  const int M = int(x.size(0));
  auto y = torch::empty({M, L.n}, x.options().dtype(f32 ? torch::kFloat32 : torch::kBFloat16));
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  const auto* xp = reinterpret_cast<const q38::bf16*>(x.data_ptr());
  if (f32) q38::lin_forward_f32(L, xp, y.data_ptr<float>(), M, st);
  else q38::lin_forward(L, xp, reinterpret_cast<q38::bf16*>(y.data_ptr()), M, st);
  return y;
}

// µs per call: `reps` rounds over handles hs (same shape; cycling defeats L2 reuse), captured in one CUDA graph,
// best of `trials` replays.
double bench(const std::vector<int64_t>& hs, torch::Tensor x, bool f32, int reps, int trials) {
  const auto& L0 = g_lin.at(hs.at(0));
  const int M = int(x.size(0));
  auto y = torch::empty({M, L0.n}, x.options().dtype(f32 ? torch::kFloat32 : torch::kBFloat16));
  const auto* xp = reinterpret_cast<const q38::bf16*>(x.data_ptr());
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto run = [&] {
    for (int r = 0; r < reps; r++)
      for (auto h : hs) {
        if (f32) q38::lin_forward_f32(g_lin.at(h), xp, y.data_ptr<float>(), M, st);
        else q38::lin_forward(g_lin.at(h), xp, reinterpret_cast<q38::bf16*>(y.data_ptr()), M, st);
      }
  };
  run();  // warm-up (also sets kernel attributes outside the capture)
  CUDA_CHECK(cudaStreamSynchronize(st));
  cudaGraph_t gr;
  cudaGraphExec_t ge;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  run();
  CUDA_CHECK(cudaStreamEndCapture(st, &gr));
  CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  float best = 1e30f;
  for (int i = 0; i < trials; i++) {
    CUDA_CHECK(cudaEventRecord(e0, st));
    CUDA_CHECK(cudaGraphLaunch(ge, st));
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    best = std::min(best, ms);
  }
  cudaEventDestroy(e0), cudaEventDestroy(e1);
  cudaGraphExecDestroy(ge), cudaGraphDestroy(gr), cudaStreamDestroy(st);
  return best * 1e3 / (double(reps) * hs.size());
}
// numbered patches (model/patches.def): switch / read inside this process; P004 tile override (-1 = automatic)
void patch_set(int id, bool on) { q38::patch::set(q38::patch::Id(id), on); }
bool patch_on(int id) { return q38::patch::on(q38::patch::Id(id)); }
void p004_tile(int t) { q38::lin_p004_tile(t); }
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("upload", &upload);
  m.def("make", &make);
  m.def("make_nvfp4", &make_nvfp4);
  m.def("make12", &make12);
  m.def("pack_all", &pack_all);
  m.def("unpack12", &unpack12);
  m.def("pack_info", &pack_info);
  m.def("info", &info);
  m.def("free", &free_);
  m.def("forward", &forward);
  m.def("bench", &bench);
  m.def("patch_set", &patch_set);
  m.def("patch_on", &patch_on);
  m.def("p004_tile", &p004_tile);
}
