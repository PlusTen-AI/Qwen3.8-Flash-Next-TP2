// kv_pool.cc — see kv_pool.h.
#include "core/kv_pool.h"

#include "core/log.h"

namespace q38 {
namespace {
void ck(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  qlog::fail("kv pool", "%s: %s", what, s ? s : "?");
}
}  // namespace

KvPool::KvPool(size_t chunks) {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaFree(nullptr));  // the runtime's primary context is current: the driver calls below use it
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  size_t g = 0;
  ck(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "granularity");
  if (kChunk % g) qlog::fail("kv pool", "chunk %zu B is not a multiple of the granularity %zu B", kChunk, g);
  acc_.location = prop.location;
  acc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  free_.reserve(chunks);
  for (size_t i = 0; i < chunks; i++) {
    CUmemGenericAllocationHandle h;
    const CUresult r = cuMemCreate(&h, kChunk, &prop, 0);
    if (r != CUDA_SUCCESS) {
      for (auto x : free_) cuMemRelease(x);
      ck(r, "creating the pool's chunks (less memory than planned?)");
    }
    free_.push_back(h);
  }
  total_ = chunks;
}

KvPool::~KvPool() {
  cudaDeviceSynchronize();
  for (auto& a : arr_) {
    for (size_t i = 0; i < a.h.size(); i++) cuMemUnmap(a.base + i * kChunk, kChunk), free_.push_back(a.h[i]);
    if (a.base) cuMemAddressFree(a.base, a.va);
  }
  for (auto h : free_) cuMemRelease(h);
}

int KvPool::add(size_t bytes) {
  Arr a;
  a.va = chunks_for(bytes) * kChunk;
  ck(cuMemAddressReserve(&a.base, a.va, kChunk, 0, 0), "reserving a cache's address range");
  arr_.push_back(std::move(a));
  return int(arr_.size()) - 1;
}

bool KvPool::grow(int id, size_t bytes, cudaStream_t s) {
  auto& a = arr_[id];
  if (bytes > a.va) qlog::fail("kv pool", "array %d: %zu B past its range of %zu B", id, bytes, a.va);
  const size_t want = chunks_for(bytes), have = a.h.size();
  if (want <= have) return true;
  if (want - have > free_.size()) return false;
  for (size_t i = have; i < want; i++) {
    const CUmemGenericAllocationHandle h = free_.back();
    free_.pop_back();
    ck(cuMemMap(a.base + i * kChunk, kChunk, 0, h, 0), "mapping a chunk");
    a.h.push_back(h);
  }
  ck(cuMemSetAccess(a.base + have * kChunk, (want - have) * kChunk, &acc_, 1), "chunk access");
  CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(a.base + have * kChunk), 0, (want - have) * kChunk, s));
  return true;
}

void KvPool::shrink(int id, size_t bytes) {
  auto& a = arr_[id];
  const size_t keep = chunks_for(bytes);
  while (a.h.size() > keep) {
    const size_t i = a.h.size() - 1;
    ck(cuMemUnmap(a.base + i * kChunk, kChunk), "unmapping a chunk");
    free_.push_back(a.h[i]);
    a.h.pop_back();
  }
}

}  // namespace q38
