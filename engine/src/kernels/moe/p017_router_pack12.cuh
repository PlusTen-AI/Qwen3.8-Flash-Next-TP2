// p017_router_pack12.cuh — P017 router_pack12: the decode router (k_route, M ≤ 64) reads a lossless 12-bit copy of the
// bf16 router [512 × 2560] when patch::on(router_pack12). The bf16 fragments (MoeW::rt) stay: the large-M path
// (moe_large.cuh), the patch off and an escaped span (below) read them. Included by moe.cu inside its anonymous
// namespace (TR, KS1, kRouteBlocks, ld_stream…).
//
// Format: P013 lin_pack12's (p013_pack12.cuh) on k_route's fragment layout. bf16 = s · e[8] · m[7]: the low byte
// (e & 1 · m) is kept verbatim, the high byte (s · e >> 1) as a nibble [s · c], c = (e >> 1) − b ∈ 0…7, one b per row:
// b = max(max over the row of (e >> 1) − 7, 0) — a 16-binade window with the row's largest value at its top or one below.
// A value with e >> 1 < b is an exception: nibble [s · 0] and its exact high byte in an entry (below).
//   Per (router tile i, k-step ks): 384 B (bf16: 512 B). Lane (r, q) holds the 8 values of its rt fragment —
//   values 0–3 = row 16i+r at k = 16ks+2q, +1, +8, +9; values 4–7 = row 16i+r+8, same k:
//     [lane·8, +8)        low bytes of values 0 … 7 (one 8-byte load per lane, 256 contiguous bytes per warp)
//     [256 + lane·4, +4)  nibbles: byte j = nibble(value j) | nibble(value j+4) << 4 (128 contiguous bytes per warp)
//   Row bases: [tile][r] 16-bit pairs (b of row 16i+r | b of row 16i+r+8 << 8) — one 16-bit load per lane.
//   Exceptions: per warp span of k_route (block = tile · 3 + K third, warp → k-steps ks0 … ks0+nks−1; span = block·8 +
//   warp, 768 spans) 8 slots of 32 bits, (ks − ks0) << 16 | lane << 11 | slot << 8 | high byte, in (k-step, lane,
//   slot) order, unused slots kNone. A span with more than 8 exceptions is escaped (slot 0 = kEsc): its warp reads the
//   bf16 fragments instead — the format's only dependent load. All loads of a span are issued where k_route issues the
//   bf16 ones (before griddepcontrol.wait); slot j sits in lane j (shuffled to the warp when applied).
// Unpacking rebuilds the exact bits of the rt fragment {W[r][k,k+1], W[r+8][k,k+1], W[r][k+8,k+9], W[r+8][k+8,k+9]}:
// high bytes of 4 values = (x & 0x08080808)·15 + x + b·0x01010101 (x = their nibbles spread to bytes; no carry), one
// byte permute per bf16 pair; then the span's entries replace their high bytes. k_route feeds the same A operands to
// the same MMAs in the same order → router partials, logits, top-k and everything after are bitwise the bf16 path's.
// This checkpoint (rank 1, 48 layers + MTP; replicated, so both ranks): no zeros, subnormals, inf or NaN; exceptions
// 0.0345 % (0.015–0.079 % per layer), 0.59 per span, at most 9 in a span (one span of 37,632 escaped). Bytes per layer
// 1,991,168 (bf16 2,621,440: 76.0 %, 12.15 bits per weight): 1,966,080 blocks + 512 bases + 24,576 slots.
#pragma once

namespace p017 {

constexpr int kBlk = 384;                                 // bytes per (router tile, k-step)
constexpr int kSlots = 8;                                 // exception slots per warp span
constexpr int kSpans = kRouteBlocks * 8;                  // 768
constexpr size_t kBytes = size_t(TR) * KS1 * kBlk;        // 1,966,080
constexpr uint32_t kNone = 0xFFFFFFFFu, kEsc = 0xFFFFFFFEu;  // never entries (an entry is < 2^19)
static_assert(TR == 32 && KS1 == 160 && kRouteBlocks == 96, "p017: router shape");

// k_route's K split: block part p (K third), warp w → first k-step, k-step count
__host__ __device__ __forceinline__ int span_nks(int part) { return part < 2 ? 7 : 6; }
__host__ __device__ __forceinline__ int span_ks0(int part, int warp) { return part * 56 + warp * span_nks(part); }

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

// one lane, one k-step: low bytes lo (x: values 0–3, y: 4–7), nibble word n, bases bb0 / bb1 (rows r / r+8,
// × 0x01010101) → the rt fragment {a0, a1, a2, a3}. High byte = [s·0x80 | b + c] (≤ 0xFF, no carry).
__host__ __device__ __forceinline__ uint4 unpack(uint2 lo, uint32_t n, uint32_t bb0, uint32_t bb1) {
  const uint32_t x0 = n & 0x0F0F0F0Fu, x1 = (n >> 4) & 0x0F0F0F0Fu;
  const uint32_t h0 = (x0 & 0x08080808u) * 15u + x0 + bb0, h1 = (x1 & 0x08080808u) * 15u + x1 + bb1;
  return make_uint4(prm(lo.x, h0, 0x5140), prm(lo.y, h1, 0x5140), prm(lo.x, h0, 0x7362), prm(lo.y, h1, 0x7362));
}

// entry x → the high byte of value `slot` of k-step x >> 16 in the owning lane's fragments (values 0,1 → .x, 2,3 → .z,
// 4,5 → .y, 6,7 → .w; odd values in the upper bf16)
__host__ __device__ __forceinline__ void put(uint4 (&w)[7], uint32_t x) {
  const int s = int(x >> 16), slot = int(x >> 8) & 7, c = slot >> 1, sh = (slot & 1) ? 24 : 8;
  const uint32_t m = ~(0xFFu << sh), v = (x & 0xFFu) << sh;
#pragma unroll
  for (int i = 0; i < 7; i++)
    if (i == s) {
      if (c == 0) w[i].x = (w[i].x & m) | v;
      else if (c == 1) w[i].z = (w[i].z & m) | v;
      else if (c == 2) w[i].y = (w[i].y & m) | v;
      else w[i].w = (w[i].w & m) | v;
    }
}

__device__ __forceinline__ uint32_t ld_u16(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}

// a warp span's loads (issued where k_route issues its bf16 fragment loads)
struct Span {
  uint2 lo[7];
  uint32_t nb[7];
  uint32_t e;   // slot (lane & 7) of the span
  uint32_t bs;  // bases of rows r (byte 0) and r+8 (byte 1)
};
__device__ __forceinline__ void load(const uint8_t* pk, const uint8_t* pb, const uint32_t* px, int tile, int span,
                                     int ks0, int nks, int lane, Span& sp) {
  const uint8_t* p = pk + (size_t(tile) * KS1 + ks0) * kBlk;
#pragma unroll
  for (int s = 0; s < 7; s++)
    if (s < nks) {
      sp.lo[s] = ld_stream2(p + s * kBlk + lane * 8);
      sp.nb[s] = ld_stream1(p + s * kBlk + 256 + lane * 4);
    } else {
      sp.lo[s] = make_uint2(0, 0), sp.nb[s] = 0;
    }
  sp.e = ld_stream1(px + span * kSlots + (lane & 7));
  sp.bs = ld_u16(pb + 16 * tile + 2 * (lane >> 2));
}
// → the bf16 fragments w[s] (s < nks) of the span; A = its rt fragments (+ lane), read only when the span is escaped
__device__ __forceinline__ void decode(const Span& sp, const uint4* A, int nks, int lane, uint4 (&w)[7]) {
  if (__shfl_sync(~0u, sp.e, 0) == kEsc) {
#pragma unroll
    for (int s = 0; s < 7; s++) w[s] = s < nks ? ld_stream(A + s * 32) : make_uint4(0, 0, 0, 0);
    return;
  }
  const uint32_t bb0 = (sp.bs & 0xFFu) * 0x01010101u, bb1 = ((sp.bs >> 8) & 0xFFu) * 0x01010101u;
#pragma unroll
  for (int s = 0; s < 7; s++) w[s] = unpack(sp.lo[s], sp.nb[s], bb0, bb1);
#pragma unroll
  for (int j = 0; j < kSlots; j++) {
    const uint32_t x = __shfl_sync(~0u, sp.e, j);
    if (x == kNone) break;  // warp-uniform
    if (int((x >> 11) & 31) == lane) put(w, x);
  }
}

// Tests: the copy decoded (load + decode, as k_route) to bf16 [512][2560] row-major; one warp per span.
__global__ void unpack_kernel(const uint8_t* pk, const uint8_t* pb, const uint32_t* px, const uint4* rt, uint32_t* out) {
  const int lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
  const int span = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (span >= kSpans) return;
  const int blk = span / 8, warp = span % 8, tile = blk / 3, part = blk % 3;
  const int nks = span_nks(part), ks0 = span_ks0(part, warp);
  Span sp;
  load(pk, pb, px, tile, span, ks0, nks, lane, sp);
  uint4 w[7];
  decode(sp, rt + (size_t(tile) * KS1 + ks0) * 32 + lane, nks, lane, w);
  for (int s = 0; s < nks; s++) {  // out as bf16 pairs: row-major [512][1280]
    const int k2 = (16 * (ks0 + s) + 2 * q) / 2;
    const size_t r0 = size_t(16 * tile + r) * (H / 2), r1 = size_t(16 * tile + r + 8) * (H / 2);
    out[r0 + k2] = w[s].x, out[r1 + k2] = w[s].y, out[r0 + k2 + 4] = w[s].z, out[r1 + k2 + 4] = w[s].w;
  }
}

// ---- host: build from the router's bf16 bits, check every value decodes to them ----------------------------------
struct Stats {
  long long values = 0, exc = 0, spans_exc = 0, esc_spans = 0, max_span = 0;
};
struct Copy {
  std::vector<uint8_t> blk, base;
  std::vector<uint32_t> slots;
};

// w: bf16 bits [512][2560] row-major (the bytes rt came from)
inline Copy build(const uint16_t* w, Stats& st) {
  Copy c;
  c.blk.assign(kBytes, 0), c.base.assign(16 * TR, 0), c.slots.assign(size_t(kSpans) * kSlots, kNone);
  std::vector<uint8_t> b(kExperts);
  for (int row = 0; row < kExperts; row++) {
    int mx = 0;
    for (int k = 0; k < H; k++) mx = std::max(mx, (w[size_t(row) * H + k] >> 8) & 0x7F);
    b[row] = uint8_t(std::max(mx - 7, 0));
  }
  for (int i = 0; i < TR; i++)
    for (int r = 0; r < 8; r++) c.base[16 * i + 2 * r] = b[16 * i + r], c.base[16 * i + 2 * r + 1] = b[16 * i + r + 8];
  auto val = [&](int i, int ks, int lane, int j) {  // value j of lane's fragment at (tile i, k-step ks)
    const int r = lane >> 2, q = lane & 3, k = 16 * ks + 2 * q + (j & 1) + ((j >> 1) & 1) * 8;
    return w[size_t(16 * i + r + (j >= 4 ? 8 : 0)) * H + k];
  };
  st.values += size_t(kExperts) * H;
  for (int i = 0; i < TR; i++)
    for (int part = 0; part < 3; part++)
      for (int wp = 0; wp < 8; wp++) {
        const int span = (3 * i + part) * 8 + wp, ks0 = span_ks0(part, wp), nks = span_nks(part);
        std::vector<uint32_t> ent;
        for (int s = 0; s < nks; s++) {
          uint8_t* p = c.blk.data() + (size_t(i) * KS1 + ks0 + s) * kBlk;
          for (int lane = 0; lane < 32; lane++) {
            const int r = lane >> 2;
            uint8_t nib[8];
            for (int j = 0; j < 8; j++) {
              const uint16_t v = val(i, ks0 + s, lane, j);
              const int hb = v >> 8, e7 = hb & 0x7F, bs = b[16 * i + r + (j >= 4 ? 8 : 0)];
              p[lane * 8 + j] = uint8_t(v & 0xFF);
              nib[j] = uint8_t((hb >> 7) << 3 | (e7 >= bs ? e7 - bs : 0));
              if (e7 < bs) ent.push_back(uint32_t(s) << 16 | uint32_t(lane) << 11 | uint32_t(j) << 8 | uint32_t(hb));
            }
            for (int j = 0; j < 4; j++) p[256 + lane * 4 + j] = uint8_t(nib[j] | nib[j + 4] << 4);
          }
        }
        uint32_t* sl = c.slots.data() + size_t(span) * kSlots;
        st.exc += ent.size(), st.spans_exc += !ent.empty(), st.max_span = std::max<long long>(st.max_span, ent.size());
        if (ent.size() > size_t(kSlots)) {
          sl[0] = kEsc, st.esc_spans++;
          continue;
        }
        std::copy(ent.begin(), ent.end(), sl);
        // check: unpack + entries (the device functions) == the bf16 bits of every value of the span
        for (int lane = 0; lane < 32; lane++) {
          const int r = lane >> 2;
          uint4 wv[7];
          for (int s = 0; s < 7; s++) {
            const uint8_t* p = c.blk.data() + (size_t(i) * KS1 + ks0 + std::min(s, nks - 1)) * kBlk;
            uint2 lo;
            uint32_t n;
            std::memcpy(&lo, p + lane * 8, 8), std::memcpy(&n, p + 256 + lane * 4, 4);
            wv[s] = unpack(lo, n, c.base[16 * i + 2 * r] * 0x01010101u, c.base[16 * i + 2 * r + 1] * 0x01010101u);
          }
          for (int j = 0; j < kSlots && sl[j] != kNone; j++)
            if (int((sl[j] >> 11) & 31) == lane) put(wv, sl[j]);
          for (int s = 0; s < nks; s++) {
            const uint32_t got[4] = {wv[s].x, wv[s].y, wv[s].z, wv[s].w};
            for (int j = 0; j < 8; j++) {  // value j: register (0,1 → x; 2,3 → z; 4,5 → y; 6,7 → w), half j & 1
              const int reg = (j >> 2) | ((j >> 1) & 1) << 1;
              if (uint16_t(got[reg] >> (16 * (j & 1))) != val(i, ks0 + s, lane, j))
                qlog::fail("moe", "p017: tile %d k-step %d lane %d value %d decodes differently", i, ks0 + s, lane, j);
            }
          }
        }
      }
  return c;
}

}  // namespace p017
