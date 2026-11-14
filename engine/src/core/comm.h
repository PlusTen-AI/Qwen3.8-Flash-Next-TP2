// comm.h — TP=2 collectives between C (rank 0) and D (rank 1) over the CX7 link.
// Bootstrap: rank 0 listens on <head_ip>:<port>, rank 1 connects and receives the NCCL unique id. NCCL then runs over
// RoCE (NCCL_IB_HCA / NCCL_IB_GID_INDEX / NCCL_SOCKET_IFNAME from the environment, as in tools/dump/run_dump.sh).
// First version: NCCL for everything. The decode path's small all-reduces get a one-shot RDMA path later (latency).
#pragma once
#include <cuda_runtime.h>
#include <nccl.h>

#include <string>

#include "core/common.h"

namespace q38 {

class Comm {
 public:
  Comm(int rank, const std::string& head_ip, int port);
  ~Comm();
  int rank() const { return rank_; }
  void all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s);                     // in place, sum
  void all_gather_bf16(const bf16* send, bf16* recv, size_t n_per_rank, cudaStream_t s);
  void all_reduce_f32(float* buf, size_t n, cudaStream_t s);
  void all_gather_f32(const float* send, float* recv, size_t n_per_rank, cudaStream_t s);  // recv = [rank0 | rank1]
  void barrier(cudaStream_t s);

 private:
  int rank_;
  ncclComm_t nc_ = nullptr;
};

}  // namespace q38
