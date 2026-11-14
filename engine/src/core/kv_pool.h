// kv_pool.h — physical memory for the sequence caches, shared by all slots (CUDA VMM). A fixed pool of 2 MiB chunks is
// created at start (the memory is taken then, so later growth never meets an empty machine); each cache array reserves
// its whole virtual range up front and gets chunks mapped — zeroed, as cudaMalloc'd caches were — as its sequence
// grows, and unmapped (back to the pool) when the caller shrinks it. Kernels see an ordinary contiguous array.
// Measured on GB10 (results/runs/20261113-1430-step2-memory): map 1.7 µs + access ~44 µs per call, memory taken at
// cuMemCreate, given back at release; remapped addresses and chunk boundaries read back exactly.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace q38 {

class KvPool {
 public:
  static constexpr size_t kChunk = size_t(2) << 20;
  explicit KvPool(size_t chunks);  // creates the chunks now on the current device
  ~KvPool();
  KvPool(const KvPool&) = delete;
  KvPool& operator=(const KvPool&) = delete;
  int add(size_t bytes);  // reserve an array's virtual range (rounded up to chunks) → its id
  void* ptr(int a) const { return reinterpret_cast<void*>(arr_[a].base); }
  size_t mapped(int a) const { return arr_[a].h.size() * kChunk; }
  // map (and zero on stream s) chunks until [0, bytes) is covered; false (nothing mapped) if the pool has too few
  bool grow(int a, size_t bytes, cudaStream_t s);
  // unmap the chunks wholly past `bytes`; the caller makes sure no queued kernel still uses them (synchronise first)
  void shrink(int a, size_t bytes);
  size_t free_chunks() const { return free_.size(); }
  size_t total_chunks() const { return total_; }
  static size_t chunks_for(size_t bytes) { return (bytes + kChunk - 1) / kChunk; }

 private:
  struct Arr {
    CUdeviceptr base = 0;
    size_t va = 0;
    std::vector<CUmemGenericAllocationHandle> h;
  };
  std::vector<CUmemGenericAllocationHandle> free_;
  std::vector<Arr> arr_;
  size_t total_ = 0;
  CUmemAccessDesc acc_{};
};

}  // namespace q38
