// mtp.cu — small kernels of the MTP draft head (contract: mtp.h). One CTA per row, fixed reduction order (per-thread
// strided fp32 sums → warp xor tree → the 8 warps in index order), so a row's result never depends on M.
#include "kernels/mtp/mtp.h"

#include "core/log.h"

namespace q38 {
namespace {

constexpr int kT = 256, kW = kT / 32;

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void unpack8(uint4 v, float* f) {
  const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float2 p = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u[j]));
    f[2 * j] = p.x, f[2 * j + 1] = p.y;
  }
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  uint32_t u[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const __nv_bfloat162 p = __floats2bfloat162_rn(f[2 * j], f[2 * j + 1]);
    u[j] = *reinterpret_cast<const uint32_t*>(&p);
  }
  return make_uint4(u[0], u[1], u[2], u[3]);
}
// block sum in a fixed order; every thread gets the result
__device__ __forceinline__ float block_sum(float v, float* sh) {
#pragma unroll
  for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(~0u, v, m);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < kW; i++) t += sh[i];
  __syncthreads();  // sh reusable
  return t;
}

// GemmaRMSNorm (vLLM forward_native): x fp32 → x·rsqrt(mean(x²)+eps) → ·(1 + w) → bf16. 8 bf16 per uint4.
__global__ void __launch_bounds__(kT) k_gemma_norm(const bf16* x, const bf16* w, bf16* y, int D) {
  __shared__ float sh[kW];
  const uint4* xr = reinterpret_cast<const uint4*>(x + size_t(blockIdx.x) * D);
  const uint4* wr = reinterpret_cast<const uint4*>(w);
  uint4* yr = reinterpret_cast<uint4*>(y + size_t(blockIdx.x) * D);
  const int n8 = D / 8;
  float ss = 0.f;
  for (int i = threadIdx.x; i < n8; i += kT) {
    float f[8];
    unpack8(xr[i], f);
#pragma unroll
    for (int e = 0; e < 8; e++) ss = fmaf(f[e], f[e], ss);
  }
  const float r = rsqrtf(block_sum(ss, sh) / float(D) + kEps);
  for (int i = threadIdx.x; i < n8; i += kT) {  // each element read and written by the same thread: in place is safe
    float f[8], g[8];
    unpack8(xr[i], f);
    unpack8(wr[i], g);
#pragma unroll
    for (int e = 0; e < 8; e++) f[e] = (f[e] * r) * (1.f + g[e]);
    yr[i] = pack8(f);
  }
}

// out ← bf16(out + bf16(bf16(σ(bf16(v·g))) · down))
__global__ void __launch_bounds__(kT) k_shared_add(const bf16* v, const bf16* g, const bf16* down, bf16* out) {
  __shared__ float sh[kW];
  const size_t o = size_t(blockIdx.x) * kHidden;
  const uint4* vr = reinterpret_cast<const uint4*>(v + o);
  const uint4* gr = reinterpret_cast<const uint4*>(g);
  float d = 0.f;
  for (int i = threadIdx.x; i < kHidden / 8; i += kT) {
    float a[8], b[8];
    unpack8(vr[i], a);
    unpack8(gr[i], b);
#pragma unroll
    for (int e = 0; e < 8; e++) d = fmaf(a[e], b[e], d);
  }
  const float sig = bfr(1.f / (1.f + expf(-bfr(block_sum(d, sh)))));
  const uint4* dr = reinterpret_cast<const uint4*>(down + o);
  uint4* orow = reinterpret_cast<uint4*>(out + o);
  for (int i = threadIdx.x; i < kHidden / 8; i += kT) {
    float a[8], b[8];
    unpack8(orow[i], a);
    unpack8(dr[i], b);
#pragma unroll
    for (int e = 0; e < 8; e++) a[e] = a[e] + bfr(sig * b[e]);
    orow[i] = pack8(a);
  }
}

__global__ void k_gather(const uint4* src, const int* idx, uint4* dst, int n16) {
  const uint4* s = src + size_t(idx[blockIdx.x]) * n16;
  uint4* d = dst + size_t(blockIdx.x) * n16;
  for (int i = threadIdx.x; i < n16; i += blockDim.x) d[i] = s[i];
}
__global__ void k_gather8(const uint2* src, const int* idx, uint2* dst, int n8) {
  const uint2* s = src + size_t(idx[blockIdx.x]) * n8;
  uint2* d = dst + size_t(blockIdx.x) * n8;
  for (int i = threadIdx.x; i < n8; i += blockDim.x) d[i] = s[i];
}

}  // namespace

void gemma_norm(const bf16* x, const bf16* w, bf16* y, int M, int D, cudaStream_t s) {
  if (M <= 0) return;
  if (D % 8 || D > 10240) qlog::fail("mtp", "gemma_norm D %d", D);
  k_gemma_norm<<<M, kT, 0, s>>>(x, w, y, D);
  CUDA_CHECK(cudaGetLastError());
}

void mtp_shared_add(const bf16* v, const bf16* g, const bf16* down, bf16* out, int M, cudaStream_t s) {
  if (M <= 0) return;
  k_shared_add<<<M, kT, 0, s>>>(v, g, down, out);
  CUDA_CHECK(cudaGetLastError());
}

void gather_rows(const bf16* src, const int* idx, bf16* dst, int M, int width, cudaStream_t s) {
  if (M <= 0) return;
  if (width % 8) qlog::fail("mtp", "gather width %d", width);
  k_gather<<<M, 256, 0, s>>>(reinterpret_cast<const uint4*>(src), idx, reinterpret_cast<uint4*>(dst), width / 8);
  CUDA_CHECK(cudaGetLastError());
}

void gather_rows4(const bf16* src, const int* idx, bf16* dst, int M, int width, cudaStream_t s) {
  if (M <= 0) return;
  if (width % 4) qlog::fail("mtp", "gather width %d", width);
  k_gather8<<<M, 64, 0, s>>>(reinterpret_cast<const uint2*>(src), idx, reinterpret_cast<uint2*>(dst), width / 4);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace q38
