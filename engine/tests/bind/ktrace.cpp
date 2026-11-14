// Kernel timeline for tests (CUPTI activity API, no kernel changes): start() records every kernel this process runs
// from then on (CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: GPU start/end timestamps, so overlapping kernels — PDL — keep
// their real start/end, unlike event marks between launches); stop() flushes and returns the records.
#include <cupti.h>
#include <torch/extension.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {
using Rec = std::tuple<std::string, uint64_t, uint64_t, uint32_t, int, int, int>;  // name, start, end, stream, grid x·y·z
std::vector<Rec> g_recs;
std::mutex g_mu;
bool g_on = false;

void chk(CUptiResult r, const char* what) {
  if (r != CUPTI_SUCCESS) {
    const char* s = nullptr;
    cuptiGetResultString(r, &s);
    throw std::runtime_error(std::string("CUPTI ") + what + ": " + (s ? s : "?"));
  }
}

void CUPTIAPI buf_req(uint8_t** buf, size_t* size, size_t* max_records) {
  *size = size_t(8) << 20;
  *buf = static_cast<uint8_t*>(aligned_alloc(8, *size));
  *max_records = 0;
}

void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  CUpti_Activity* rec = nullptr;
  std::lock_guard<std::mutex> lk(g_mu);
  while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = reinterpret_cast<CUpti_ActivityKernel12*>(rec);
      const uint64_t t0 = k->start, t1 = k->end;  // packed fields: copy before binding
      const uint32_t st = k->streamId;
      const int gx = k->gridX, gy = k->gridY, gz = k->gridZ;
      g_recs.emplace_back(k->name ? k->name : "?", t0, t1, st, gx, gy, gz);
    }
  }
  free(buf);
}

void start() {
  if (g_on) return;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_recs.clear();
  }
  chk(cuptiActivityRegisterCallbacks(buf_req, buf_done), "register");
  chk(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable");
  g_on = true;
}

std::vector<Rec> stop() {
  if (g_on) {
    chk(cuptiActivityFlushAll(1), "flush");
    chk(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "disable");
    g_on = false;
  }
  std::lock_guard<std::mutex> lk(g_mu);
  return g_recs;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("start", &start);
  m.def("stop", &stop);
}
