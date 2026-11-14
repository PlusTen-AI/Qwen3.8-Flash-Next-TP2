// rdma_ar.cu — P001 rdma_ar (contract in rdma_ar.h): GPU kernels, the CPU proxy thread and the ibverbs setup.
#include "kernels/comm/rdma_ar.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "core/log.h"

namespace q38 {
namespace {

constexpr int kMaxRails = 4;
constexpr int kMaxChunks = 64;
constexpr size_t kLLMaxCap = size_t(512) << 10;  // largest LL call (data bytes); LL lines take twice that
constexpr size_t kLLRegion = 2 * kLLMaxCap;     // per parity, send and recv
constexpr size_t kFusedMax = size_t(1) << 20;   // data bytes ≤: one fused kernel (copy · signal · spin · add)
constexpr size_t kBigChunk = size_t(1) << 20;
constexpr size_t kHdrBytes = size_t(64) << 10;  // flags + doorbells
constexpr size_t kBulk = kHdrBytes + 4 * kLLRegion;  // bulk staging send[2], recv[2] from here (page aligned)
constexpr int kThreads = 512;
constexpr int kSqDepth = 512;
constexpr uint32_t kMagic = 0x51385242;          // "Q8RB"

struct alignas(64) Flag {  // written by the peer's NIC (inline RDMA WRITE after the chunk's data, same QP)
  unsigned long long seq;
  unsigned long long pad[7];
};
struct alignas(64) Db {    // written by our GPU (seq last, release), read by our proxy
  unsigned long long seq;
  unsigned long long src, dst, bytes;  // byte offsets in the (same-layout) regions of both ranks
  long long flag;                      // peer flag slot to write after the data, -1 = none (LL)
  unsigned long long last;
  unsigned long long pad[2];
};
struct Hdr {
  Flag flag[2][kMaxChunks];
  Db db[2][kMaxChunks];
};
static_assert(sizeof(Hdr) <= kHdrBytes, "header");

__host__ __device__ constexpr size_t ll_send_off(int p) { return kHdrBytes + size_t(p) * kLLRegion; }
__host__ __device__ constexpr size_t ll_recv_off(int p) { return kHdrBytes + size_t(2 + p) * kLLRegion; }

struct Ctl {                      // device memory, zeroed once
  unsigned long long seq;         // last call issued on the GPU (1, 2, …); parity = seq & 1
  unsigned int cnt[kMaxChunks];   // blocks that finished copying their part of a chunk
  unsigned int done;              // blocks that finished the kernel
  unsigned long long ll_swept[2]; // last LL call per parity that cleared stale lines
};

struct Args {
  bf16* buf;
  size_t n;          // elements of this call
  size_t chunk;      // elements per chunk (bulk) or LL lines per chunk
  int C;             // chunks
  char* base;        // host-pinned region, GPU view
  size_t stride;     // bulk bytes per parity buffer
  Ctl* ctl;
  long long timeout_ns;
  __device__ size_t send_off(int p) const { return kBulk + size_t(p) * stride; }
  __device__ size_t recv_off(int p) const { return kBulk + size_t(2 + p) * stride; }
  __device__ Hdr* hdr() const { return reinterpret_cast<Hdr*>(base); }
};

// ---- device ------------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long ld_relaxed_sys(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ void wait_flag(const Args& a, int p, int k, unsigned long long seq, bool nap) {
  const unsigned long long* f = &a.hdr()->flag[p][k].seq;
  if ((long long)(ld_relaxed_sys(f) - seq) < 0) {
    const unsigned long long t0 = gtimer();
    unsigned it = 0;
    while ((long long)(ld_relaxed_sys(f) - seq) < 0) {
      if (nap) __nanosleep(200);
      if (a.timeout_ns && (++it & 1023) == 0 && (long long)(gtimer() - t0) > a.timeout_ns) {
        printf("[rdma_ar] GPU wait timeout · call %llu chunk %d · flag %llu\n", seq, k, ld_relaxed_sys(f));
        __trap();
      }
    }
  }
  __threadfence_system();  // acquire: the data written before the flag is read after this
}

// rows of chunk k handled by this block: [e0, e1), multiples of 64 elements except at n
__device__ __forceinline__ void block_part(const Args& a, size_t c0, size_t c1, size_t& e0, size_t& e1) {
  const size_t per = ((c1 - c0 + gridDim.x - 1) / gridDim.x + 63) & ~size_t(63);
  e0 = min(c1, c0 + size_t(blockIdx.x) * per);
  e1 = min(c1, e0 + per);
}

__device__ __forceinline__ void copy_part(char* dst, const bf16* __restrict__ src, size_t e0, size_t e1, bool vec) {
  if (vec && e1 > e0) {
    const uint4* __restrict__ s = reinterpret_cast<const uint4*>(src + e0);
    uint4* d = reinterpret_cast<uint4*>(dst + 2 * e0);
    const size_t nv = (e1 - e0) >> 3;
    size_t i = threadIdx.x;
    for (; i + 3 * kThreads < nv; i += 4 * kThreads) {  // streaming (evict-first): keep compute's L2 working set
      const uint4 v0 = __ldcs(s + i), v1 = __ldcs(s + i + kThreads), v2 = __ldcs(s + i + 2 * kThreads),
                  v3 = __ldcs(s + i + 3 * kThreads);
      __stcs(d + i, v0), __stcs(d + i + kThreads, v1), __stcs(d + i + 2 * kThreads, v2), __stcs(d + i + 3 * kThreads, v3);
    }
    for (; i < nv; i += kThreads) __stcs(d + i, __ldcs(s + i));
    e0 += nv << 3;
  }
  bf16* d = reinterpret_cast<bf16*>(dst);
  for (size_t e = e0 + threadIdx.x; e < e1; e += kThreads) d[e] = src[e];
}

// bf16(float(a) + float(b)): one rounding of the exact fp32 sum = correctly rounded bf16 add (= NCCL's sum)
__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
  __nv_bfloat162 x, y;
  memcpy(&x, &a, 4), memcpy(&y, &b, 4);
  const float2 fx = __bfloat1622float2(x), fy = __bfloat1622float2(y);
  const __nv_bfloat162 r = __floats2bfloat162_rn(__fadd_rn(fx.x, fy.x), __fadd_rn(fx.y, fy.y));
  uint32_t u;
  memcpy(&u, &r, 4);
  return u;
}
__device__ __forceinline__ uint4 add8(uint4 a, uint4 b) {
  return make_uint4(add2(a.x, b.x), add2(a.y, b.y), add2(a.z, b.z), add2(a.w, b.w));
}

// buf[e] += recv[e] for e in [e0, e1); recv is written by the NIC — plain loads (no read-only/nc path)
__device__ __forceinline__ void add_part(bf16* buf, const char* rcv, size_t e0, size_t e1, bool vec) {
  if (vec && e1 > e0) {
    uint4* b = reinterpret_cast<uint4*>(buf + e0);
    const uint4* r = reinterpret_cast<const uint4*>(rcv + 2 * e0);
    const size_t nv = (e1 - e0) >> 3;
    size_t i = threadIdx.x;
    for (; i + 3 * kThreads < nv; i += 4 * kThreads) {  // streaming (evict-first) like the copy
      const uint4 a0 = __ldcs(b + i), a1 = __ldcs(b + i + kThreads), a2 = __ldcs(b + i + 2 * kThreads),
                  a3 = __ldcs(b + i + 3 * kThreads);
      const uint4 r0 = __ldcs(r + i), r1 = __ldcs(r + i + kThreads), r2 = __ldcs(r + i + 2 * kThreads),
                  r3 = __ldcs(r + i + 3 * kThreads);
      __stcs(b + i, add8(a0, r0)), __stcs(b + i + kThreads, add8(a1, r1)), __stcs(b + i + 2 * kThreads, add8(a2, r2)),
          __stcs(b + i + 3 * kThreads, add8(a3, r3));
    }
    for (; i < nv; i += kThreads) __stcs(b + i, add8(__ldcs(b + i), __ldcs(r + i)));
    e0 += nv << 3;
  }
  const bf16* r = reinterpret_cast<const bf16*>(rcv);
  for (size_t e = e0 + threadIdx.x; e < e1; e += kThreads)
    buf[e] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(buf[e]), __bfloat162float(r[e])));
}

__device__ __forceinline__ void ring(const Args& a, int p, int k, size_t src, size_t dst, size_t bytes, long long flag,
                                     unsigned long long seq) {
  Db* d = &a.hdr()->db[p][k];
  d->src = src, d->dst = dst, d->bytes = bytes, d->flag = flag, d->last = (k == a.C - 1);
  st_release_sys(&d->seq, seq);  // release (system scope) covers the staging writes this thread has observed
}

// thread 0, after __syncthreads(): the last block to finish its part of chunk k rings the doorbell
__device__ void publish(const Args& a, int p, int k, unsigned long long seq, unsigned parts) {
  if (parts > 1) {
    __threadfence();
    if (atomicAdd(&a.ctl->cnt[k], 1u) != parts - 1) return;
    a.ctl->cnt[k] = 0;
    __threadfence();
  }
  const size_t c0 = size_t(k) * a.chunk, b = 2 * (min(a.n, c0 + a.chunk) - c0);
  ring(a, p, k, a.send_off(p) + 2 * c0, a.recv_off(p) + 2 * c0, b, p * kMaxChunks + k, seq);
}

// the last block of the kernel records the call as issued (every block read ctl->seq before counting itself done)
__device__ __forceinline__ void finish(const Args& a, unsigned long long seq) {
  if (threadIdx.x == 0) {
    if (gridDim.x == 1) {
      a.ctl->seq = seq;
      return;
    }
    __threadfence();
    if (atomicAdd(&a.ctl->done, 1u) == gridDim.x - 1) a.ctl->done = 0, a.ctl->seq = seq;
  }
}

// ---- LL (small calls): 16-byte lines {d0, flag, d1, flag} carry 8 data bytes; a line is valid when both flags match
// (the NIC writes 8-byte words atomically — NCCL's LL protocol relies on the same). No flag WRITE; the receiver polls
// the data itself (no flag → data round trip).
__device__ __forceinline__ uint4 ld_volatile4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint2 ld_line_data(const bf16* buf, size_t i, size_t n, bool vec) {  // elements 4i..4i+3
  if (vec && 4 * i + 4 <= n) return *reinterpret_cast<const uint2*>(buf + 4 * i);
  uint16_t h[4] = {0, 0, 0, 0};
  for (int j = 0; j < 4; j++)
    if (4 * i + j < n) h[j] = __bfloat16_as_ushort(buf[4 * i + j]);
  return make_uint2(h[0] | uint32_t(h[1]) << 16, h[2] | uint32_t(h[3]) << 16);
}
// Block b owns lines [b·per, …) and rings its own doorbell(s) (chunk b, or 2 chunks — one per rail — for a single
// block): no cross-block sync before sending; each block then polls exactly the lines it sent.
__global__ void __launch_bounds__(kThreads) k_ll(Args a) {
  // let a programmatic dependent launch now: it waits for this grid's completion (griddepcontrol.wait) before reading
  // the result, but can stream its own weights while this one waits on the network (P015; no effect otherwise)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ unsigned long long s_seq;
  if (threadIdx.x == 0) s_seq = a.ctl->seq + 1;
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int p = int(seq & 1);
  const uint32_t f = uint32_t(seq % 0xffffffffull) + 1;  // never 0
  const size_t lines = (a.n + 3) / 4;
  const int cpb = a.C / int(gridDim.x);                 // chunks per block (1 or 2)
  const size_t l0 = min(lines, size_t(blockIdx.x) * cpb * a.chunk), l1 = min(lines, l0 + cpb * a.chunk);
  const bool vec = (reinterpret_cast<uintptr_t>(a.buf) & 7) == 0;
  uint4* snd = reinterpret_cast<uint4*>(a.base + ll_send_off(p));
  for (size_t i = l0 + threadIdx.x; i < l1; i += kThreads) {
    const uint2 d = ld_line_data(a.buf, i, a.n, vec);
    snd[i] = make_uint4(d.x, f, d.y, f);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int c = 0; c < cpb; c++) {
      const int k = int(blockIdx.x) * cpb + c;
      const size_t c0 = min(lines, size_t(k) * a.chunk), c1 = min(lines, c0 + a.chunk);
      ring(a, p, k, ll_send_off(p) + 16 * c0, ll_recv_off(p) + 16 * c0, 16 * (c1 - c0), -1, seq);
    }
  uint4* rcv = reinterpret_cast<uint4*>(a.base + ll_recv_off(p));
  for (size_t i = l0 + threadIdx.x; i < l1; i += kThreads) {
    const uint2 mine = ld_line_data(a.buf, i, a.n, vec);  // before polling: off the critical path
    uint4 v = ld_volatile4(rcv + i);
    if (v.y != f || v.w != f) {
      const unsigned long long t0 = gtimer();
      for (unsigned it = 0; v.y != f || v.w != f; it++) {
        v = ld_volatile4(rcv + i);
        if (a.timeout_ns && (it & 1023) == 1023 && (long long)(gtimer() - t0) > a.timeout_ns) {
          printf("[rdma_ar] GPU wait timeout · LL call %llu line %llu\n", seq, (unsigned long long)i);
          __trap();
        }
      }
    }
    const uint2 r = make_uint2(add2(mine.x, v.x), add2(mine.y, v.z));
    if (vec && 4 * i + 4 <= a.n) {
      *reinterpret_cast<uint2*>(a.buf + 4 * i) = r;
    } else {
      const uint16_t h[4] = {uint16_t(r.x), uint16_t(r.x >> 16), uint16_t(r.y), uint16_t(r.y >> 16)};
      for (int j = 0; j < 4; j++)
        if (4 * i + j < a.n) a.buf[4 * i + j] = __ushort_as_bfloat16(h[j]);
    }
  }
  // Stale lines keep old flags; a false match needs a line untouched for 2^32-1 calls. Every 2^20 calls (per parity)
  // an LL call clears the lines beyond its own, so no stale flag gets that old. (The peer writes recv[p] again only
  // after we finished the next call — no race.)
  if (blockIdx.x == 0 && seq - a.ctl->ll_swept[p] >= (1u << 20)) {
    for (size_t i = lines + threadIdx.x; i < kLLRegion / 16; i += kThreads) rcv[i] = make_uint4(0, 0, 0, 0);
    __syncthreads();
    if (threadIdx.x == 0) a.ctl->ll_swept[p] = seq;
  }
  __syncthreads();
  finish(a, seq);
}

// mid-size calls: one kernel; block k owns chunk k — its own WRITE, so the NIC starts on a slice as soon as its writes
// have drained (no cross-block sync); chunks alternate rails; a block adds exactly what it copied (in place is safe)
__global__ void __launch_bounds__(kThreads) k_fused(Args a) {
  // let a programmatic dependent launch now: it waits for this grid's completion (griddepcontrol.wait) before reading
  // the result, but can stream its own weights while this one waits on the network (P015; no effect otherwise)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  __shared__ unsigned long long s_seq;
  if (threadIdx.x == 0) s_seq = a.ctl->seq + 1;
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int p = int(seq & 1), k = int(blockIdx.x);
  const bool vec = (reinterpret_cast<uintptr_t>(a.buf) & 15) == 0;
  const size_t e0 = size_t(k) * a.chunk, e1 = min(a.n, e0 + a.chunk);
  copy_part(a.base + a.send_off(p), a.buf, e0, e1, vec);
  __syncthreads();
  if (threadIdx.x == 0) {
    publish(a, p, k, seq, 1);
    wait_flag(a, p, k, seq, false);
  }
  __syncthreads();
  add_part(a.buf, a.base + a.recv_off(p), e0, e1, vec);
  __syncthreads();
  finish(a, seq);
}

// large calls: copy one group of chunks (one per rail) + doorbells; the first group's kernel issues the call …
__global__ void __launch_bounds__(kThreads) k_copy(Args a, int k0, int k1, int first) {
  __shared__ unsigned long long s_seq;
  if (threadIdx.x == 0) s_seq = a.ctl->seq + (first ? 1 : 0);
  __syncthreads();
  const unsigned long long seq = s_seq;
  const int p = int(seq & 1);
  const bool vec = (reinterpret_cast<uintptr_t>(a.buf) & 15) == 0;
  char* snd = a.base + a.send_off(p);
  for (int k = k0; k < k1; k++) {
    size_t e0, e1;
    block_part(a, size_t(k) * a.chunk, min(a.n, size_t(k + 1) * a.chunk), e0, e1);
    copy_part(snd, a.buf, e0, e1, vec);
    __syncthreads();
    if (threadIdx.x == 0) publish(a, p, k, seq, gridDim.x);
  }
  if (first) finish(a, seq);
}
// … then per group of chunks (one per rail): one sleeping warp waits for the flags …
__global__ void k_wait(Args a, int k0, int k1) {
  if (threadIdx.x) return;
  const unsigned long long seq = a.ctl->seq;
  for (int k = k0; k < k1; k++) wait_flag(a, int(seq & 1), k, seq, true);
}
// … and the add runs only once the data is there
__global__ void __launch_bounds__(kThreads) k_add(Args a, int k0, int k1) {
  const unsigned long long seq = a.ctl->seq;
  const int p = int(seq & 1);
  size_t e0, e1;
  block_part(a, size_t(k0) * a.chunk, min(a.n, size_t(k1) * a.chunk), e0, e1);
  add_part(a.buf, a.base + a.recv_off(p), e0, e1, (reinterpret_cast<uintptr_t>(a.buf) & 15) == 0);
}

// ---- host helpers ------------------------------------------------------------------------------------------------
void send_all(int fd, const void* p, size_t n) {
  const char* c = static_cast<const char*>(p);
  while (n) {
    ssize_t k = ::send(fd, c, n, 0);
    if (k <= 0) qlog::fail("rdma", "bootstrap send failed");
    c += k, n -= k;
  }
}
void recv_all(int fd, void* p, size_t n) {
  char* c = static_cast<char*>(p);
  while (n) {
    ssize_t k = ::recv(fd, c, n, 0);
    if (k <= 0) qlog::fail("rdma", "bootstrap receive failed");
    c += k, n -= k;
  }
}
int tcp_pair(int rank, const std::string& ip, int port) {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &a.sin_addr) != 1) qlog::fail("rdma", "bad address: %s", ip.c_str());
  if (rank == 0) {
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 1)) qlog::fail("rdma", "cannot listen on %s:%d", ip.c_str(), port);
    int fd = accept(ls, nullptr, nullptr);
    close(ls);
    return fd;
  }
  for (int i = 0; i < 3000; i++) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(fd, (sockaddr*)&a, sizeof a) == 0) return fd;
    close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  qlog::fail("rdma", "cannot reach the head %s:%d (300 s)", ip.c_str(), port);
}

std::vector<std::string> hca_names() {
  const char* e = std::getenv("Q38_RDMA_HCA");
  if (!e || !*e) e = std::getenv("NCCL_IB_HCA");
  if (!e || !*e) qlog::fail("rdma", "no HCA list (Q38_RDMA_HCA or NCCL_IB_HCA)");
  std::string s = e;
  if (s[0] == '=') s = s.substr(1);
  if (s[0] == '^') qlog::fail("rdma", "HCA exclude list (^) not supported: %s", e);
  std::vector<std::string> v;
  for (size_t i = 0; i <= s.size();) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    std::string h = s.substr(i, j - i);
    if (h.find(':') != std::string::npos) h = h.substr(0, h.find(':'));
    if (!h.empty()) v.push_back(h);
    i = j + 1;
  }
  if (v.empty() || v.size() > size_t(kMaxRails)) qlog::fail("rdma", "HCA list has %zu entries (1–%d): %s", v.size(), kMaxRails, e);
  return v;
}

int env_int(const char* k, int def) {
  const char* e = std::getenv(k);
  return (e && *e) ? std::atoi(e) : def;
}

// proxy CPU candidates: Q38_RDMA_CPU if set, else every CPU by capacity (big cores first; GB10: X925 > A725), ties by
// higher index (CPU 0 takes most interrupts)
std::vector<int> cpu_candidates() {
  const int c = env_int("Q38_RDMA_CPU", -2);
  if (c != -2) return c >= 0 ? std::vector<int>{c} : std::vector<int>{};
  std::vector<std::pair<int, int>> v;
  const long n = sysconf(_SC_NPROCESSORS_CONF);
  for (int i = 0; i < n; i++) {
    int cap = 0;
    char path[96];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
    if (FILE* f = std::fopen(path, "r")) {
      if (std::fscanf(f, "%d", &cap) != 1) cap = 0;
      std::fclose(f);
    }
    v.push_back({-cap, -i});
  }
  std::sort(v.begin(), v.end());
  std::vector<int> r;
  for (auto& x : v) r.push_back(-x.second);
  return r;
}

struct Wire {  // bootstrap exchange (both ranks)
  uint32_t magic, rails;
  uint64_t stride, base;  // staging bytes per parity · virtual address of the registered region
  struct { uint32_t qpn, psn, rkey, mtu; uint8_t gid[16]; } r[kMaxRails];
};

struct Rail {
  std::string name;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;
  int gidx = -1;
  ibv_gid gid{};
  ibv_mtu mtu = IBV_MTU_1024;
  uint32_t psn = 0, peer_rkey = 0;
  int outstanding = 0;
};

}  // namespace

// ---- RdmaAR ------------------------------------------------------------------------------------------------------
struct RdmaAR::Impl {
  int rank = 0, nr = 0, cpu = -1, copy_blocks = 2, add_blocks = 4;
  size_t max_elems = 0, stride = 0, bytes = 0, ll_max = 16 << 10, chunk_bytes = kBigChunk;
  long long timeout_ns = 0;
  Rail rail[kMaxRails];
  char* host = nullptr;    // region: Hdr | LL send[2] | LL recv[2] | send[2] | recv[2]
  char* dev = nullptr;     // GPU view of the same memory
  Ctl* ctl = nullptr;
  uint64_t peer_base = 0;
  std::thread th;
  std::atomic<bool> stop{false};

  Hdr* hdr() const { return reinterpret_cast<Hdr*>(host); }

  void reap(int r) {
    ibv_wc wc[16];
    int n = ibv_poll_cq(rail[r].cq, 16, wc);
    if (n < 0) qlog::fail("rdma", "%s CQ error", rail[r].name.c_str());
    for (int i = 0; i < n; i++) {
      if (wc[i].status != IBV_WC_SUCCESS)
        qlog::fail("rdma", "%s RDMA WRITE failed: %s", rail[r].name.c_str(), ibv_wc_status_str(wc[i].status));
      rail[r].outstanding -= int(wc[i].wr_id);  // WQEs this completion covers
    }
  }

  // data WRITE (src → peer dst); then, if flag ≥ 0, an inline 8-byte WRITE of seq to the peer's flag slot (same QP)
  void post(int r, uint64_t src, uint64_t dst, uint64_t len, long long flag, unsigned long long seq) {
    Rail& R = rail[r];
    while (R.outstanding + 2 > kSqDepth) reap(r);
    ibv_sge sd{reinterpret_cast<uint64_t>(host) + src, uint32_t(len), R.mr->lkey};
    unsigned long long v = seq;
    ibv_sge sf{reinterpret_cast<uint64_t>(&v), 8, R.mr->lkey};
    ibv_send_wr w[2]{}, *bad = nullptr;
    w[0].sg_list = &sd, w[0].num_sge = 1, w[0].opcode = IBV_WR_RDMA_WRITE;
    w[0].wr.rdma.remote_addr = peer_base + dst, w[0].wr.rdma.rkey = R.peer_rkey;
    int nw = 1;
    if (flag >= 0) {
      w[0].next = &w[1];
      w[1].sg_list = &sf, w[1].num_sge = 1, w[1].opcode = IBV_WR_RDMA_WRITE;
      w[1].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED, w[1].wr_id = 2;
      w[1].wr.rdma.remote_addr = peer_base + offsetof(Hdr, flag) + size_t(flag) * sizeof(Flag);
      w[1].wr.rdma.rkey = R.peer_rkey;
      nw = 2;
    } else {
      w[0].send_flags = IBV_SEND_SIGNALED, w[0].wr_id = 1;  // a lone WQE goes out by BlueFlame
    }
    if (int e = ibv_post_send(R.qp, w, &bad)) qlog::fail("rdma", "%s post_send failed: %s", R.name.c_str(), strerror(e));
    R.outstanding += nw;
  }

  // Follows the GPU's calls in order: per call, chunks 0.. until `last`; chunk k of call seq goes on rail (seq+k)%nr.
  void proxy() {
    using clk = std::chrono::steady_clock;
    auto active = clk::now();
    unsigned long long seq = 0;
    for (;;) {
      ++seq;
      const int p = int(seq & 1);
      for (int k = 0;; k++) {
        Db* d = &hdr()->db[p][k];
        for (unsigned spin = 0; __atomic_load_n(&d->seq, __ATOMIC_ACQUIRE) != seq; spin++) {
          if ((spin & 255) == 255) {
            if (stop.load(std::memory_order_relaxed)) return;
            for (int r = 0; r < nr; r++)
              if (rail[r].outstanding) reap(r);
            // idle: 200 ms → 20 µs naps; 1 s → 500 µs naps (an idle server near 0% CPU; the first all-reduce after a
            // long pause waits ≤ 0.5 ms more; while steps run, calls come within µs and the loop keeps spinning)
            const auto idle = clk::now() - active;
            if (idle > std::chrono::seconds(1)) std::this_thread::sleep_for(std::chrono::microseconds(500));
            else if (idle > std::chrono::milliseconds(200)) std::this_thread::sleep_for(std::chrono::microseconds(20));
          }
#if defined(__aarch64__)
          asm volatile("yield" ::: "memory");
#endif
        }
        const uint64_t src = d->src, dst = d->dst, len = d->bytes, last = d->last;
        const long long flag = d->flag;
        post(int((seq + k) % nr), src, dst, len, flag, seq);
        if (last) break;
      }
      active = clk::now();
    }
  }
};

RdmaAR::RdmaAR(int rank, const std::string& head_ip, int port, size_t max_elems) : d_(std::make_unique<Impl>()) {
  Impl& I = *d_;
  auto t0 = std::chrono::steady_clock::now();
  I.rank = rank;
  I.max_elems = std::max<size_t>(64, (max_elems + 63) & ~size_t(63));
  I.stride = (I.max_elems * 2 + (size_t(64) << 10) - 1) & ~((size_t(64) << 10) - 1);
  I.bytes = kBulk + 4 * I.stride;
  I.timeout_ns = (long long)(std::atof(std::getenv("Q38_RDMA_TIMEOUT_S") ? std::getenv("Q38_RDMA_TIMEOUT_S") : "0") * 1e9);
  I.copy_blocks = std::max(1, env_int("Q38_RDMA_COPY_BLOCKS", 2));  // few blocks: the copy only has to stay ahead of
  I.add_blocks = std::max(1, env_int("Q38_RDMA_ADD_BLOCKS", 4));    // the NIC; more starve its DMA (DRAM-bound)
  I.chunk_bytes = size_t(std::max(64, env_int("Q38_RDMA_CHUNK_KB", int(kBigChunk >> 10)))) << 10;
  I.ll_max = std::min<size_t>(kLLMaxCap, std::max(0, env_int("Q38_RDMA_LL_MAX", 16 << 10)));

  CUDA_CHECK(cudaHostAlloc(&I.host, I.bytes, cudaHostAllocMapped | cudaHostAllocPortable));
  std::memset(I.host, 0, I.bytes);
  void* dp = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&dp, I.host, 0));
  I.dev = static_cast<char*>(dp);
  CUDA_CHECK(cudaMalloc(&I.ctl, sizeof(Ctl)));
  CUDA_CHECK(cudaMemset(I.ctl, 0, sizeof(Ctl)));

  // rails: device, RoCE v2 IPv4 GID, PD, MR over the whole region, CQ, RC QP → INIT
  auto names = hca_names();
  I.nr = int(names.size());
  int ndev = 0;
  ibv_device** devs = ibv_get_device_list(&ndev);
  if (!devs) qlog::fail("rdma", "ibv_get_device_list failed (/dev/infiniband?)");
  std::mt19937 rng(std::random_device{}());
  const int gid_env = env_int("Q38_RDMA_GID_INDEX", -1);
  for (int r = 0; r < I.nr; r++) {
    Rail& R = I.rail[r];
    R.name = names[r];
    for (int i = 0; i < ndev && !R.ctx; i++)
      if (R.name == ibv_get_device_name(devs[i])) R.ctx = ibv_open_device(devs[i]);
    if (!R.ctx) qlog::fail("rdma", "HCA %s not found", R.name.c_str());
    ibv_port_attr pa{};
    if (ibv_query_port(R.ctx, 1, &pa) || pa.state != IBV_PORT_ACTIVE) qlog::fail("rdma", "%s port 1 not active", R.name.c_str());
    R.mtu = pa.active_mtu;
    if (gid_env >= 0) {
      R.gidx = gid_env;
    } else {
      ibv_gid_entry ge[64];
      ssize_t ng = ibv_query_gid_table(R.ctx, ge, 64, 0);
      for (ssize_t i = 0; i < ng && R.gidx < 0; i++) {
        const uint8_t* g = ge[i].gid.raw;
        bool v4 = g[10] == 0xff && g[11] == 0xff;
        for (int b = 0; b < 10; b++) v4 = v4 && g[b] == 0;
        if (ge[i].port_num == 1 && ge[i].gid_type == IBV_GID_TYPE_ROCE_V2 && v4) R.gidx = int(ge[i].gid_index);
      }
    }
    if (R.gidx < 0 || ibv_query_gid(R.ctx, 1, R.gidx, &R.gid)) qlog::fail("rdma", "%s has no RoCE v2 IPv4 GID", R.name.c_str());
    if (!(R.pd = ibv_alloc_pd(R.ctx))) qlog::fail("rdma", "%s PD failed", R.name.c_str());
    R.mr = ibv_reg_mr(R.pd, I.host, I.bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!R.mr) qlog::fail("rdma", "%s MR registration failed (%.0f MiB): %s", R.name.c_str(), I.bytes / 1048576.0, strerror(errno));
    if (!(R.cq = ibv_create_cq(R.ctx, kSqDepth, nullptr, nullptr, 0))) qlog::fail("rdma", "%s CQ failed", R.name.c_str());
    ibv_qp_init_attr qa{};
    qa.send_cq = qa.recv_cq = R.cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = kSqDepth, qa.cap.max_recv_wr = 1, qa.cap.max_send_sge = 1, qa.cap.max_recv_sge = 1;
    qa.cap.max_inline_data = 64;
    if (!(R.qp = ibv_create_qp(R.pd, &qa))) qlog::fail("rdma", "%s QP failed: %s", R.name.c_str(), strerror(errno));
    ibv_qp_attr at{};
    at.qp_state = IBV_QPS_INIT, at.pkey_index = 0, at.port_num = 1;
    at.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(R.qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
      qlog::fail("rdma", "%s QP INIT failed", R.name.c_str());
    R.psn = rng() & 0xffffff;
  }
  ibv_free_device_list(devs);

  // exchange QPs and the region address with the peer
  Wire me{}, pe{};
  me.magic = kMagic, me.rails = uint32_t(I.nr), me.stride = I.stride, me.base = reinterpret_cast<uint64_t>(I.host);
  for (int r = 0; r < I.nr; r++) {
    me.r[r].qpn = I.rail[r].qp->qp_num, me.r[r].psn = I.rail[r].psn, me.r[r].rkey = I.rail[r].mr->rkey;
    me.r[r].mtu = uint32_t(I.rail[r].mtu);
    std::memcpy(me.r[r].gid, I.rail[r].gid.raw, 16);
  }
  const int fd = tcp_pair(rank, head_ip, port);
  send_all(fd, &me, sizeof me);
  recv_all(fd, &pe, sizeof pe);
  if (pe.magic != kMagic || pe.rails != me.rails || pe.stride != me.stride)
    qlog::fail("rdma", "settings differ from the peer · rails %u/%u · buffer %llu/%llu B", me.rails, pe.rails, (unsigned long long)me.stride,
               (unsigned long long)pe.stride);
  I.peer_base = pe.base;
  for (int r = 0; r < I.nr; r++) {
    Rail& R = I.rail[r];
    if (std::memcmp(me.r[r].gid + 12, pe.r[r].gid + 12, 3))  // same /24 per rail (direct cables)
      qlog::fail("rdma", "rail %d subnets differ (%u.%u.%u.x ↔ %u.%u.%u.x) — check the HCA order", r, me.r[r].gid[12],
                 me.r[r].gid[13], me.r[r].gid[14], pe.r[r].gid[12], pe.r[r].gid[13], pe.r[r].gid[14]);
    R.peer_rkey = pe.r[r].rkey;
    ibv_qp_attr at{};
    at.qp_state = IBV_QPS_RTR;
    at.path_mtu = ibv_mtu(std::min<uint32_t>(uint32_t(R.mtu), pe.r[r].mtu));
    at.dest_qp_num = pe.r[r].qpn, at.rq_psn = pe.r[r].psn, at.max_dest_rd_atomic = 1, at.min_rnr_timer = 12;
    at.ah_attr.is_global = 1, at.ah_attr.port_num = 1;
    std::memcpy(at.ah_attr.grh.dgid.raw, pe.r[r].gid, 16);
    at.ah_attr.grh.sgid_index = uint8_t(R.gidx), at.ah_attr.grh.hop_limit = 64;
    if (int e = ibv_modify_qp(R.qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                             IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
      qlog::fail("rdma", "%s QP RTR failed: %s", R.name.c_str(), strerror(e));
    at = ibv_qp_attr{};
    at.qp_state = IBV_QPS_RTS, at.timeout = 14, at.retry_cnt = 7, at.rnr_retry = 7, at.sq_psn = R.psn;
    at.max_rd_atomic = 1;
    if (int e = ibv_modify_qp(R.qp, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                             IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
      qlog::fail("rdma", "%s QP RTS failed: %s", R.name.c_str(), strerror(e));
  }
  char b = 1;  // both sides are RTS before anyone writes
  send_all(fd, &b, 1);
  recv_all(fd, &b, 1);
  close(fd);

  I.th = std::thread([&I] { I.proxy(); });
  for (int c : cpu_candidates()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c, &set);
    if (pthread_setaffinity_np(I.th.native_handle(), sizeof set, &set) == 0) {
      I.cpu = c;
      break;
    }
  }
  std::string rl;
  for (int r = 0; r < I.nr; r++)
    rl += (r ? "," : "") + I.rail[r].name + "/" + std::to_string(I.rail[r].gidx);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  qlog::ok("rdma", "rank%d · RDMA AR · %d rails %s · staging %.0f MiB · proxy CPU %d · LL ≤ %zu KiB · %.0f ms", rank, I.nr,
           rl.c_str(), I.bytes / 1048576.0, I.cpu, I.ll_max >> 10, ms);
}

RdmaAR::~RdmaAR() {
  if (!d_) return;
  Impl& I = *d_;
  I.stop = true;
  if (I.th.joinable()) I.th.join();
  for (int r = 0; r < I.nr; r++) {
    Rail& R = I.rail[r];
    if (R.qp) ibv_destroy_qp(R.qp);
    if (R.cq) ibv_destroy_cq(R.cq);
    if (R.mr) ibv_dereg_mr(R.mr);
    if (R.pd) ibv_dealloc_pd(R.pd);
    if (R.ctx) ibv_close_device(R.ctx);
  }
  if (I.host) cudaFreeHost(I.host);
  if (I.ctl) cudaFree(I.ctl);
}

int RdmaAR::rails() const { return d_->nr; }
size_t RdmaAR::max_elems() const { return d_->max_elems; }

void RdmaAR::all_reduce_bf16(bf16* buf, size_t n, cudaStream_t s) {
  Impl& I = *d_;
  for (size_t o = 0; o < n; o += I.max_elems) {
    Args a{};
    a.buf = buf + o, a.n = std::min(I.max_elems, n - o);
    a.base = I.dev, a.stride = I.stride, a.ctl = I.ctl, a.timeout_ns = I.timeout_ns;
    const size_t bytes = a.n * 2;
    if (bytes <= I.ll_max) {  // LL: ~8 KiB of data per block (≤ 16 blocks), one chunk per block; one block → 2 rails
      const size_t lines = (a.n + 3) / 4;
      int G = int(std::min<size_t>(16, std::max<size_t>(1, bytes >> 13)));
      if (G == 1) {
        a.C = (I.nr > 1 && bytes > (4 << 10)) ? 2 : 1;
        a.chunk = (lines + a.C - 1) / a.C;
      } else {  // no empty chunks
        a.chunk = (lines + G - 1) / G;
        a.C = G = int((lines + a.chunk - 1) / a.chunk);
      }
      k_ll<<<G, kThreads, 0, s>>>(a);
    } else if (bytes <= kFusedMax) {
      // fused: ~16 KiB per block (2–16 blocks), each block's slice is its own chunk (no cross-block sync; the NIC
      // starts on a slice as soon as its writes have drained), chunks alternate rails
      const int G = int(std::min<size_t>(16, std::max<size_t>(I.nr, bytes >> 14)));
      a.chunk = ((a.n + G - 1) / G + 63) & ~size_t(63);
      a.C = int((a.n + a.chunk - 1) / a.chunk);
      k_fused<<<a.C, kThreads, 0, s>>>(a);
    } else {
      // large: groups of one chunk per rail. Copies run one group ahead of the waits/adds in short bursts — one big
      // copy up front saturates DRAM and starves the NIC's DMA (measured: almost no transfer until it ended).
      int C = std::min<int>(kMaxChunks, 2 * int((bytes + 2 * I.chunk_bytes - 1) / (2 * I.chunk_bytes)));
      a.chunk = ((a.n + C - 1) / C + 63) & ~size_t(63);
      a.C = int((a.n + a.chunk - 1) / a.chunk);
      const int P = (a.C + I.nr - 1) / I.nr;
      auto group = [&](int j, int& k0, int& k1) { k0 = j * I.nr, k1 = std::min(a.C, k0 + I.nr); };
      int k0, k1;
      group(0, k0, k1);
      k_copy<<<I.copy_blocks, kThreads, 0, s>>>(a, k0, k1, 1);
      for (int j = 0; j < P; j++) {
        if (j + 1 < P) {
          group(j + 1, k0, k1);
          k_copy<<<I.copy_blocks, kThreads, 0, s>>>(a, k0, k1, 0);
        }
        group(j, k0, k1);
        k_wait<<<1, 32, 0, s>>>(a, k0, k1);
        k_add<<<I.add_blocks, kThreads, 0, s>>>(a, k0, k1);
      }
    }
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace q38
