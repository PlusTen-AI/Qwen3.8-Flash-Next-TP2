// p013_pack12.cuh — P013 lin_pack12: a lossless 12-bit copy of a bf16 weight matrix, read by the decode kernel
// (lin_kernel, M ≤ 64) when patch::on(lin_pack12). The bf16 copy stays: M > 64 (prefill kernels) and the patch off read
// it. Included by linear.cu inside its anonymous namespace (parallel_for, Fmt).
//
// Format. bf16 = s · e[8] · m[7]. Its low byte (e & 1 · m) is kept verbatim; its high byte (s · e >> 1) as a nibble
// [s · c], c = (e >> 1) − b ∈ 0…7, with one b per row: b = max(max over the row of (e >> 1) − 7, 0). The window is the 16
// exponents 2b … 2b+15, the row's largest at its top or one below. A value with e >> 1 < b (a small value, or ±0 or a
// subnormal when b > 0) is an exception: nibble [s · 0], and its exact high byte goes to a side list of its 16-row tile.
// Measured on this checkpoint's lm_head halves: 0.0128 % exceptions, 5.2 per tile (max 18), no zeros/subnormals/inf/NaN.
// Layout per (16-row tile T, k32 block kb): 768 B (bf16: 1024), same lanes as the bf16 copy — lane (g, t) holds row
// 16T+g, k 32kb+8t … +7 (values 0–7) and row 16T+g+8, same k (values 8–15):
//   [lane·16, +16)       low bytes of values 0 … 15
//   [512 + lane·8, +8)   nibbles: word 0 byte j = nibble(value j) | nibble(value j+4) << 4 (j = 0…3); word 1 the same
//                        for values 8 … 15
// Row bases: one byte b per row (npad). Exceptions: tile T owns entries xo[T] … xo[T+1]−1, sorted ascending, 32 bits
// kb << 17 | lane << 12 | slot << 8 | high byte (slot = value index 0…15 of the lane).
// Unpacking (unpack8) rebuilds the exact bf16 bits in registers — the high bytes of 4 values from a word of nibbles in 4
// integer ops, then one byte permute per bf16 pair — so the kernel feeds the same MMA operands, in the same order, as
// from the bf16 copy: every output is bitwise the bf16 path's, for every M and row. Exceptions: each warp keeps its
// tile's next 32 entries in registers (one per lane) and the next entry's kb warp-uniform; a k32 block with an entry
// replaces those high bytes after unpacking (≈ 6 % of the blocks of lm_head).
#pragma once

namespace p013 {

constexpr int TB = 768;               // bytes per (tile, k32 block)
constexpr uint32_t kEnd = 0xFFFFFFFFu;  // window filler past the list end: kb 0x7FFF, never a real block
constexpr int kMaxKb = 0x7FFF;

__host__ __device__ __forceinline__ uint32_t prm(uint32_t x, uint32_t y, uint32_t s) {
#ifdef __CUDA_ARCH__
  return __byte_perm(x, y, s);
#else
  const uint64_t v = uint64_t(y) << 32 | x;
  uint32_t r = 0;
  for (int i = 0; i < 4; i++) r |= uint32_t((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
  return r;
#endif
}

// 8 values of one row: lo / hi = low bytes of values 0–3 / 4–7, n = their nibbles (byte j: value j | value j+4 << 4),
// bb = b · 0x01010101 → bf16 pairs w[0] = (v0, v1) … w[3] = (v6, v7). High byte = [s·0x80 | b + c] (≤ 0xFF, no carry).
__host__ __device__ __forceinline__ void unpack8(uint32_t lo, uint32_t hi, uint32_t n, uint32_t bb, uint32_t* w) {
  const uint32_t x0 = n & 0x0F0F0F0Fu, x1 = (n >> 4) & 0x0F0F0F0Fu;
  const uint32_t h0 = (x0 & 0x08080808u) * 15u + x0 + bb, h1 = (x1 & 0x08080808u) * 15u + x1 + bb;
  w[0] = prm(lo, h0, 0x5140), w[1] = prm(lo, h0, 0x7362);
  w[2] = prm(hi, h1, 0x5140), w[3] = prm(hi, h1, 0x7362);
}

// Exception entry x → the high byte of value slot (pair w[slot / 2], half slot % 2) of a lane's 8 bf16 pairs.
__host__ __device__ __forceinline__ void put_hi(uint32_t (&w)[8], uint32_t x) {
  const int wi = (x >> 9) & 7, sh = (x & 0x100) ? 24 : 8;
#pragma unroll
  for (int i = 0; i < 8; i++)
    if (i == wi) w[i] = (w[i] & ~(0xFFu << sh)) | ((x & 0xFFu) << sh);
}

// Per warp: its tile's exception list through a window of 32 entries in registers (lane i holds entry w0 + i); nk = kb
// of the next entry (warp-uniform; 0x7FFF: none). Entries below this split's first block are skipped at init.
struct Exc {
  uint32_t e, nk;
  int w0, i, end;
  __device__ __forceinline__ void load(const uint32_t* xe, int lane) { e = w0 + lane < end ? xe[w0 + lane] : kEnd; }
  __device__ __forceinline__ void init(const uint32_t* xo, const uint32_t* xe, int T, int kb0, int lane) {
    w0 = int(xo[T]), end = int(xo[T + 1]);
    for (;;) {
      load(xe, lane);
      const int below = __popc(__ballot_sync(~0u, (e >> 17) < uint32_t(kb0)));  // sorted: a prefix
      if (below < 32) {
        i = below;
        break;
      }
      w0 += 32;
    }
    nk = __shfl_sync(~0u, e, i) >> 17;
  }
  // every entry of block kb (call when kb == nk, warp-uniform), in list order
  __device__ __forceinline__ void apply(const uint32_t* xe, int kb, int lane, uint32_t (&w)[8]) {
    do {
      const uint32_t x = __shfl_sync(~0u, e, i);
      if (int((x >> 12) & 31) == lane) put_hi(w, x);
      if (++i == 32) w0 += 32, i = 0, load(xe, lane);
      nk = __shfl_sync(~0u, e, i) >> 17;
    } while (nk == uint32_t(kb));
  }
};

// Tests: the packed copy unpacked to bf16 [n, k] row-major (one warp per (tile, k32 block); exceptions by a scan of
// the tile's list), through unpack8 and put_hi as in lin_kernel.
__global__ void unpack_kernel(const uint8_t* pk, const uint8_t* pb, const uint32_t* xo, const uint32_t* xe, bf16* out,
                              int n, int k, int tiles) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, KB = k / 32;
  const int blk = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (blk >= tiles * KB) return;
  const int T = blk / KB, kb = blk % KB;
  const uint8_t* b = pk + size_t(blk) * TB;
  const uint4 lo = *reinterpret_cast<const uint4*>(b + lane * 16);
  const uint2 nb = *reinterpret_cast<const uint2*>(b + 512 + lane * 8);
  uint32_t w[8];
  unpack8(lo.x, lo.y, nb.x, pb[16 * T + g] * 0x01010101u, w);
  unpack8(lo.z, lo.w, nb.y, pb[16 * T + g + 8] * 0x01010101u, w + 4);
  for (uint32_t j = xo[T]; j < xo[T + 1]; j++) {
    const uint32_t x = xe[j];
    if (int(x >> 17) == kb && int((x >> 12) & 31) == lane) put_hi(w, x);
  }
  for (int h = 0; h < 2; h++) {
    const int r = 16 * T + g + 8 * h;
    if (r < n) *reinterpret_cast<uint4*>(out + size_t(r) * k + kb * 32 + 8 * t) = make_uint4(w[4 * h], w[4 * h + 1], w[4 * h + 2], w[4 * h + 3]);
  }
}

// Builds L's packed copy from the bf16 rows w [n, k] (host, row-major; the bytes L.w came from), checks on the host
// that every value unpacks to its exact bits (fail otherwise), uploads.
void build(Lin& L, const uint16_t* w) {
  const int n = L.n, k = L.k, KB = k / 32, tiles = L.npad / 16;
  if (KB >= kMaxKb || (KB / L.split) % 2) qlog::fail("linear", "pack12: shape %d×%d split %d not supported", n, k, L.split);
  std::vector<uint8_t> blk(size_t(tiles) * KB * TB), base(L.npad, 0);
  parallel_for(n, [&](int r) {
    int mx = 0;
    for (int i = 0; i < k; i++) mx = std::max(mx, (w[size_t(r) * k + i] >> 8) & 0x7F);
    base[r] = uint8_t(std::max(mx - 7, 0));
  });
  std::vector<std::vector<uint32_t>> ex(tiles);  // per tile, pushed in (kb, lane, slot) order = sorted
  parallel_for(tiles, [&](int T) {
    for (int kb = 0; kb < KB; kb++) {
      uint8_t* b = blk.data() + (size_t(T) * KB + kb) * TB;
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane >> 2, t = lane & 3;
        uint8_t nib[16];
        for (int i = 0; i < 16; i++) {
          const int r = 16 * T + g + (i >= 8 ? 8 : 0);
          const uint16_t v = r < n ? w[size_t(r) * k + kb * 32 + 8 * t + (i & 7)] : 0;  // padding rows: +0, b = 0
          const int hb = v >> 8, e7 = hb & 0x7F, bs = base[r];
          b[lane * 16 + i] = uint8_t(v & 0xFF);
          nib[i] = uint8_t((hb >> 7) << 3 | (e7 >= bs ? e7 - bs : 0));
          if (e7 < bs) ex[T].push_back(uint32_t(kb) << 17 | uint32_t(lane) << 12 | uint32_t(i) << 8 | uint32_t(hb));
        }
        for (int h = 0; h < 2; h++)
          for (int j = 0; j < 4; j++) b[512 + lane * 8 + 4 * h + j] = uint8_t(nib[8 * h + j] | nib[8 * h + j + 4] << 4);
      }
    }
  });
  std::vector<uint32_t> xo(tiles + 1, 0), xe;
  for (int T = 0; T < tiles; T++) xo[T + 1] = xo[T] + uint32_t(ex[T].size());
  xe.reserve(xo[tiles]);
  for (auto& v : ex) xe.insert(xe.end(), v.begin(), v.end());
  // host check: every value of every row unpacks (unpack8 + put_hi, as on the device) to its bf16 bits
  std::vector<int> bad(tiles, 0);
  parallel_for(tiles, [&](int T) {
    uint32_t j = xo[T];
    for (int kb = 0; kb < KB; kb++) {
      const uint8_t* b = blk.data() + (size_t(T) * KB + kb) * TB;
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane >> 2, t = lane & 3;
        uint32_t lo[4], nb[2], wv[8];
        std::memcpy(lo, b + lane * 16, 16), std::memcpy(nb, b + 512 + lane * 8, 8);
        unpack8(lo[0], lo[1], nb[0], base[16 * T + g] * 0x01010101u, wv);
        unpack8(lo[2], lo[3], nb[1], base[16 * T + g + 8] * 0x01010101u, wv + 4);
        for (; j < xo[T + 1] && int(xe[j] >> 17) == kb && int((xe[j] >> 12) & 31) == lane; j++) put_hi(wv, xe[j]);
        for (int i = 0; i < 16; i++) {
          const int r = 16 * T + g + (i >= 8 ? 8 : 0);
          if (r >= n) continue;
          const uint16_t got = uint16_t(wv[i >> 1] >> (16 * (i & 1)));
          bad[T] += got != w[size_t(r) * k + kb * 32 + 8 * t + (i & 7)];
        }
      }
    }
    bad[T] += j != xo[T + 1];  // every entry consumed in order
  });
  size_t nbad = 0;
  for (int v : bad) nbad += v;
  if (nbad) qlog::fail("linear", "pack12: %zu values differ from bf16 (%d×%d)", nbad, n, k);
  auto up = [](auto*& dst, const auto& v) {
    CUDA_CHECK(cudaMalloc(&dst, std::max<size_t>(1, v.size() * sizeof(v[0]))));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(dst, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice));
  };
  up(L.pk, blk), up(L.pk_base, base), up(L.pk_xo, xo), up(L.pk_x, xe);
  L.pk_nx = int(xe.size());
}

}  // namespace p013
