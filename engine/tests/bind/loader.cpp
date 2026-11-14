// Test binding for the production weight loader (engine/src/weights.cc).
#include <torch/extension.h>

#include <memory>

#include "core/weights.h"

namespace {
std::unique_ptr<q38::Weights> W;

void open(const std::string& model_dir, const std::string& table, int rank) {
  W = std::make_unique<q38::Weights>(model_dir, table, rank);
}

// (uint8 tensor of this rank's bytes, local shape, dtype string, tp rule)
std::tuple<torch::Tensor, std::vector<int64_t>, std::string, std::string> slice(const std::string& name) {
  std::vector<int64_t> shape;
  auto b = W->slice(name, &shape);
  auto t = torch::empty({int64_t(b.size())}, torch::kUInt8);
  std::memcpy(t.data_ptr(), b.data(), b.size());
  const auto& ti = W->info(name);
  return {t, shape, ti.dtype, ti.tp};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("open", &open);
  m.def("slice", &slice);
  m.def("has", [](const std::string& n) { return W->has(n); });
}
