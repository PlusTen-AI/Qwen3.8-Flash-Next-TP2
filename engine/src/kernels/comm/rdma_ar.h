// rdma_ar.h — P001 rdma_ar: two-rank in-place bf16 all-reduce (sum) over the CX7 RoCE rails with RDMA WRITE.
//
// Data path (no host callbacks, CUDA-graph capturable; both ranks must issue the same calls in the same order, and the
// calls of one RdmaAR must be ordered on the GPU — one in flight at a time, like the engine's s_/cs_ event chains):
//   1. GPU: copy buf → pinned host staging (parity p = call & 1), then ring a doorbell per chunk {seq, src, dst, len}
//      in host memory (release, system scope)
//   2. CPU proxy thread (busy-polls the doorbells): per chunk one RDMA WRITE to the peer's staging (chunks alternate
//      the rails), plus — except for LL — an inline 8-byte WRITE of seq to the peer's flag slot on the same QP
//   3. GPU: wait until the peer's data is there, buf = bf16(float(buf) + float(peer)) — the same bits as NCCL's 2-rank
//      bf16 sum (commutative; one rounding of the exact fp32 sum = the correctly rounded bf16 add)
// Three paths by size (data bytes): ≤ 16 KiB LL — 16-byte lines {d0, flag, d1, flag} (NCCL's LL format; relies on
// the NIC writing 8-byte words atomically), one kernel, no flag WRITE, the GPU polls the lines; ≤ 1 MiB fused — one
// kernel, one chunk per block (~16 KiB), flag after each chunk; larger — 1 MiB chunks, a 2-block copy kernel per pair
// of chunks running one pair ahead, a 1-warp wait kernel (nanosleep) and a 4-block add kernel per pair: while the NIC
// moves data no SMs are held, and the short copy/add bursts leave DRAM bandwidth to the NIC's DMA.
// Parity double buffering makes reuse safe without credits: a rank can only start call i+2 after the peer finished
// call i+1, which it started after finishing call i (so peer recv[p] is consumed and our send[p] has been read).
// The NIC only touches the pinned host region (cudaHostAlloc, GPU-mapped): no GPUDirect / dma-buf needed
// (GB10: cudaMalloc memory cannot be exported as dma-buf).
//
// Rails: env Q38_RDMA_HCA, else NCCL_IB_HCA (comma list, same rail order on both ranks, e.g. C "roceP2p1s0f0,
// rocep1s0f0" ↔ D "roceP2p1s0f1,rocep1s0f1"); port 1; RoCE v2 IPv4 GID found per HCA (Q38_RDMA_GID_INDEX overrides).
// Proxy CPU: the highest-capacity CPU allowed (Q38_RDMA_CPU=n to choose, -1 = unpinned). Q38_RDMA_TIMEOUT_S=s makes a
// GPU wait longer than s seconds trap instead of hanging (default 0 = wait forever, like NCCL). Tuning (defaults are
// the measured best): Q38_RDMA_LL_MAX (bytes, 16384), Q38_RDMA_CHUNK_KB (1024), Q38_RDMA_COPY_BLOCKS (2),
// Q38_RDMA_ADD_BLOCKS (4).
// Memory: pinned host 4 × max_elems × 2 B + 4 MiB (LL lines, flags, doorbells), registered on every rail; calls with
// n > max_elems are split into several calls of ≤ max_elems. Device: 1 KiB of counters.
#pragma once
#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "core/common.h"

namespace q38 {

class RdmaAR {
 public:
  // TCP bootstrap only for the QP exchange: rank 0 listens on head_ip:port, rank 1 connects (retries for 300 s).
  RdmaAR(int rank, const std::string& head_ip, int port, size_t max_elems);
  ~RdmaAR();  // call after the GPU work that uses it has finished
  RdmaAR(const RdmaAR&) = delete;
  RdmaAR& operator=(const RdmaAR&) = delete;

  void all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s);  // in place, sum, stream-ordered
  int rails() const;
  size_t max_elems() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

}  // namespace q38
