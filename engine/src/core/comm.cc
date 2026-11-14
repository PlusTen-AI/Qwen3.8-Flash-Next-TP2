#include "core/comm.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <thread>

#include "core/log.h"

#define NCCL_CHECK(x)                                                                       \
  do {                                                                                      \
    ncclResult_t r_ = (x);                                                                  \
    if (r_ != ncclSuccess) qlog::fail("nccl", "%s · %s:%d", ncclGetErrorString(r_), __FILE__, __LINE__); \
  } while (0)

namespace q38 {

static void send_all(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    ssize_t k = ::send(fd, c, n, 0);
    if (k <= 0) qlog::fail("comm", "bootstrap send failed");
    c += k, n -= k;
  }
}
static void recv_all(int fd, void* p, size_t n) {
  char* c = static_cast<char*>(p);
  while (n) {
    ssize_t k = ::recv(fd, c, n, 0);
    if (k <= 0) qlog::fail("comm", "bootstrap receive failed");
    c += k, n -= k;
  }
}

Comm::Comm(int rank, const std::string& head_ip, int port) : rank_(rank) {
  if (head_ip.empty()) qlog::fail("comm", "no head address (EngineCfg.head_ip; tests: Q38_HEAD_IP)");
  ncclUniqueId id;
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, head_ip.c_str(), &a.sin_addr) != 1) qlog::fail("comm", "bad address: %s", head_ip.c_str());
  auto t0 = std::chrono::steady_clock::now();
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 1)) qlog::fail("comm", "cannot listen on %s:%d", head_ip.c_str(), port);
    int fd = accept(ls, nullptr, nullptr);
    send_all(fd, &id, sizeof id);
    close(fd);
    close(ls);
  } else {
    int fd = -1;
    for (int i = 0; i < 3000; i++) {  // the head may start later (e.g. still compiling): retry for 300 s
      fd = socket(AF_INET, SOCK_STREAM, 0);
      if (connect(fd, (sockaddr*)&a, sizeof a) == 0) break;
      close(fd), fd = -1;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (fd < 0) qlog::fail("comm", "cannot reach the head %s:%d (300 s)", head_ip.c_str(), port);
    recv_all(fd, &id, sizeof id);
    close(fd);
  }
  NCCL_CHECK(ncclCommInitRank(&nc_, 2, id, rank));
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  qlog::ok("comm", "rank%d · NCCL TP2 · %s:%d · %.0f ms", rank, head_ip.c_str(), port, ms);
}

Comm::~Comm() {
  if (nc_) ncclCommDestroy(nc_);
}

void Comm::all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s) {
  NCCL_CHECK(ncclAllReduce(buf, buf, n, ncclBfloat16, ncclSum, nc_, s));
}
void Comm::all_reduce_f32(float* buf, size_t n, cudaStream_t s) {
  NCCL_CHECK(ncclAllReduce(buf, buf, n, ncclFloat32, ncclSum, nc_, s));
}
void Comm::all_gather_bf16(const bf16* send, bf16* recv, size_t n, cudaStream_t s) {
  NCCL_CHECK(ncclAllGather(send, recv, n, ncclBfloat16, nc_, s));
}
void Comm::all_gather_f32(const float* send, float* recv, size_t n, cudaStream_t s) {
  NCCL_CHECK(ncclAllGather(send, recv, n, ncclFloat32, nc_, s));
}
void Comm::barrier(cudaStream_t s) {
  static float* d = nullptr;
  if (!d) CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
  all_reduce_f32(d, 1, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

}  // namespace q38
