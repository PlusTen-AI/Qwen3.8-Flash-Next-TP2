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
    for (size_t i = 0; i < a.h.size(); i++) cuMemUnmap(a.base + i * kChunk, kChunk), unref(a.h[i]);
    if (a.base) cuMemAddressFree(a.base, a.va);
  }
  if (tmp_) cuMemAddressFree(tmp_, kChunk);
  for (auto h : free_) cuMemRelease(h);
}

CUmemGenericAllocationHandle KvPool::take() {
  const CUmemGenericAllocationHandle h = free_.back();
  free_.pop_back();
  ref_[h] = 1;
  return h;
}

void KvPool::unref(CUmemGenericAllocationHandle h) {
  auto it = ref_.find(h);
  if (it == ref_.end()) qlog::fail("kv pool", "unmapping a chunk the pool does not count");
  if (--it->second == 0) {
    ref_.erase(it);
    free_.push_back(h);
  }
}

void KvPool::map(CUdeviceptr at, CUmemGenericAllocationHandle h, bool access) {
  ck(cuMemMap(at, kChunk, 0, h, 0), "mapping a chunk");
  if (access) ck(cuMemSetAccess(at, kChunk, &acc_, 1), "chunk access");
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
    const CUmemGenericAllocationHandle h = take();
    map(a.base + i * kChunk, h, false);
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
    unref(a.h[i]);
    a.h.pop_back();
  }
}

void KvPool::share(int dst, int src, size_t n) {
  auto& d = arr_[dst];
  const auto& a = arr_[src];
  if (dst == src || !d.h.empty() || n > a.h.size() || n * kChunk > d.va)
    qlog::fail("kv pool", "share: array %d (%zu mapped) ← %zu chunks of array %d (%zu mapped)", dst, d.h.size(), n, src,
               a.h.size());
  for (size_t i = 0; i < n; i++) {
    map(d.base + i * kChunk, a.h[i], false);
    d.h.push_back(a.h[i]);
    ref_[a.h[i]]++;
  }
  if (n) ck(cuMemSetAccess(d.base, n * kChunk, &acc_, 1), "chunk access");
}

bool KvPool::grow_copy(int dst, int src, cudaStream_t s) {
  auto& d = arr_[dst];
  const auto& a = arr_[src];
  const size_t i = d.h.size();
  if (dst == src || i >= a.h.size() || (i + 1) * kChunk > d.va)
    qlog::fail("kv pool", "grow_copy: array %d chunk %zu from array %d (%zu mapped)", dst, i, src, a.h.size());
  if (free_.empty()) return false;
  const CUmemGenericAllocationHandle h = take();
  map(d.base + i * kChunk, h, true);
  d.h.push_back(h);
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d.base + i * kChunk), reinterpret_cast<const void*>(a.base + i * kChunk),
                             kChunk, cudaMemcpyDeviceToDevice, s));
  return true;
}

bool KvPool::shared(int id, size_t i) const {
  const auto& a = arr_[id];
  return i < a.h.size() && ref_.at(a.h[i]) > 1;
}

bool KvPool::own(int id, size_t i, cudaStream_t s) {
  auto& a = arr_[id];
  if (!shared(id, i)) return true;
  if (free_.empty()) return false;
  if (!tmp_) ck(cuMemAddressReserve(&tmp_, kChunk, kChunk, 0, 0), "reserving the copy range");
  const CUdeviceptr at = a.base + i * kChunk;
  const CUmemGenericAllocationHandle h = take();
  map(tmp_, h, true);
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(tmp_), reinterpret_cast<const void*>(at), kChunk,
                             cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  ck(cuMemUnmap(tmp_, kChunk), "unmapping the copy range");
  ck(cuMemUnmap(at, kChunk), "unmapping a chunk");
  unref(a.h[i]);
  map(at, h, true);
  a.h[i] = h;
  return true;
}

size_t KvPool::sole(int id) const {
  size_t n = 0;
  for (auto h : arr_[id].h) n += ref_.at(h) == 1;
  return n;
}

}  // namespace q38
