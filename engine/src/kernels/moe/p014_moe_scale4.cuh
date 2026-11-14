// p014_moe_scale4.cuh — P014 moe_scale4: the decode MoE kernel (k_experts, M ≤ 64) reads the routed experts' NVFP4 block
// scales from a lossless 4-bit copy when patch::on(moe_scale4). The expanded e4m3 bytes stay in the expert block (kS13,
// kS2): the large-M path (moe_large.cuh) and the patch off read them, and an escaped row (below) reads them too. Included
// by moe.cu inside its anonymous namespace (T13, KP, KS1, KS2, T2, kW13, kS13, kW2, ld_stream…).
//
// Format. Scales are e4m3 bit patterns s ∈ 0x00…0x7E (positive; 0x7F/≥0x80 refused by fix_scales). Per kernel item and
// per row (or projection) a base byte b and per scale a nibble d = s − b ∈ 0…15 (window b … b+15, b = the window that
// holds the most scales of the block, lowest b on ties); a scale outside the window is an exception: nibble 0 and a
// 16-bit entry in the item's 8 slots, r << 13 | word << 9 | byte << 7 | s (word / byte = where it sits among the lane's
// scale words below; entries in (r, word, byte) order, unused slots 0xFFFF — never an entry: s ≤ 0x7E). An item with
// more than 8 exceptions escapes rows (most exceptions first) until the rest fit: an escaped row has bit 7 set in its
// base bytes and its lanes read the row's plain bytes from the block — the only dependent load of the format.
//   gate/up item (tile i, K part kp): rows 8i+r of gate and up, k-steps 32kp … +31 (one base per row, projection and
//     item: 32 scales). Record 288 B: [chunk c / 4][r][c % 4] u32 nibble words (256 B; lane r loads 2 × 16 B, each warp
//     load covers 128 contiguous bytes), bases [r][gate, up] (16 B, lane r loads 2 B), slots (16 B).
//   down item (output tile n): rows 16n+r / 16n+r+8, all 20 k-steps (one base per row). Record 192 B: [chunk c][r] u32
//     nibble words (160 B), bases [r][lo, hi] (16 B), slots (16 B).
//   Per expert: 200 gate/up records then 160 down records = 88,320 B (expanded: 153,600 B; whole block 1,382,400 B).
// A nibble word holds the 8 scales of one chunk (4 k-steps × 2 rows/projections) of one lane: nibble ((ks&1)·4 | p·2 |
// ks>>1) = (k-step ks, row/projection p), so that (w & 0x0F0F0F0F) + b·0x00010001 and ((w >> 4) & 0x0F0F0F0F) + same are
// the two scale words of the base layout ([ks0 p0, ks0 p1, ks1 p0, ks1 p1], [ks2 …, ks3 …]) — no carries (s ≤ 0x7E).
// The kernel feeds these words to the unchanged sc_fp4 → the same bf16 scale broadcasts, the same A fragments, the same
// MMAs in the same order: out is bitwise the base path's for every M and row. Exceptions are patched into the lane's
// words before the first MMA; the nibble/base/slot loads are issued with the first fragments, so the decode adds no
// dependent load except for an escaped row.
// This checkpoint (rank 1, all 48 layers × 512 experts): exceptions 0.17 % of the scales (0.72 per item), 0.32 % of the
// items over 8, 0.05 % of the rows escaped; MTP experts (plain e4m3 in the checkpoint, built the same way) 0.42 % / 2.4 %.
// Candidates and numbers: results/runs/20261112-*-p014/README.txt.
#pragma once

namespace p014 {

constexpr int kRecU = 288, kRecD = 192;                // record bytes: gate/up item, down item
constexpr int kNibU = 256, kNibD = 160;                // their nibble words; bases follow, then 8 slots
constexpr int kSlots = 8;
constexpr size_t kUpB = size_t(T13) * KP * kRecU;      // 57,600: gate/up records of an expert
constexpr size_t kSC = kUpB + size_t(T2) * kRecD;      // 88,320 per expert
constexpr uint32_t kNone = 0xFFFFu;
static_assert(KS1 / KP == 32 && KS2 == 20, "p014: item shapes");

__host__ __device__ __forceinline__ int nib_pos(int ks4, int p) { return ((ks4 & 1) << 2) | (p << 1) | (ks4 >> 1); }

// N scale words of row r from N/2 nibble words + base pair bs (lo byte p0, hi byte p1) + slots
template <int N>
__host__ __device__ __forceinline__ void decode(const uint32_t* nw, uint32_t bs, const uint32_t (&sl)[4], int r,
                                                uint32_t (&sw)[N]) {
  const uint32_t bb = bs * 0x00010001u;
#pragma unroll
  for (int c = 0; c < N / 2; c++) {
    sw[2 * c] = (nw[c] & 0x0F0F0F0Fu) + bb;
    sw[2 * c + 1] = ((nw[c] >> 4) & 0x0F0F0F0Fu) + bb;
  }
#pragma unroll
  for (int j = 0; j < kSlots; j++) {
    const uint32_t x = (sl[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
    if (x == kNone) break;  // same slots in every lane: warp-uniform
    if (int(x >> 13) == r) {
      const int wi = (x >> 9) & 15, sh = 8 * ((x >> 7) & 3);
#pragma unroll
      for (int k = 0; k < N; k++)
        if (k == wi) sw[k] = (sw[k] & ~(0xFFu << sh)) | ((x & 0x7Fu) << sh);
    }
  }
}

__device__ __forceinline__ uint32_t ld_stream_u16(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}

// gate/up item record (lane row r): loads issued at item start, decoded before the first MMA
struct UpRec {
  uint4 n0, n1, sl;
  uint32_t bs;
};
__device__ __forceinline__ UpRec up_load(const uint8_t* rec, int r) {
  UpRec x;
  x.n0 = ld_stream(rec + 16 * r);
  x.n1 = ld_stream(rec + 128 + 16 * r);
  x.bs = ld_stream_u16(rec + kNibU + 2 * r);
  x.sl = ld_stream(rec + kNibU + 16);
  return x;
}
// → sw[2c], sw[2c+1] = the base path's uint2 scale word pair of chunk c; S = the item's plain scales (escape)
__device__ __forceinline__ void up_decode(const UpRec& x, const uint2* S, int r, uint32_t (&sw)[16]) {
  if (x.bs & 0x80u) {  // escaped row: its plain bytes
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint2 p = ld_stream2(S + c);
      sw[2 * c] = p.x, sw[2 * c + 1] = p.y;
    }
    return;
  }
  const uint32_t nw[8] = {x.n0.x, x.n0.y, x.n0.z, x.n0.w, x.n1.x, x.n1.y, x.n1.z, x.n1.w};
  const uint32_t sl[4] = {x.sl.x, x.sl.y, x.sl.z, x.sl.w};
  decode<16>(nw, x.bs, sl, r, sw);
}

struct DnRec {
  uint32_t nw[5];
  uint4 sl;
  uint32_t bs;
};
__device__ __forceinline__ DnRec dn_load(const uint8_t* rec, int r) {
  DnRec x;
#pragma unroll
  for (int c = 0; c < 5; c++) x.nw[c] = ld_stream1(rec + 32 * c + 4 * r);
  x.bs = ld_stream_u16(rec + kNibD + 2 * r);
  x.sl = ld_stream(rec + kNibD + 16);
  return x;
}
__device__ __forceinline__ void dn_decode(const DnRec& x, const uint2* S, int r, uint32_t (&sw)[10]) {
  if (x.bs & 0x80u) {
#pragma unroll
    for (int c = 0; c < 5; c++) {
      const uint2 p = ld_stream2(S + c);
      sw[2 * c] = p.x, sw[2 * c + 1] = p.y;
    }
    return;
  }
  const uint32_t sl[4] = {x.sl.x, x.sl.y, x.sl.z, x.sl.w};
  decode<10>(x.nw, x.bs, sl, r, sw);
}

// Tests: every expert's compressed scales decoded (up_decode / dn_decode, as k_experts) into the expanded layout
// [512][kS13 + kS2]. One warp per item; lanes q = 0 write their row's words.
__global__ void unpack_kernel(const uint8_t* sc, const uint8_t* ex, uint8_t* out) {
  constexpr int IU = T13 * KP, ID = T2;
  const int lane = threadIdx.x & 31, r = lane >> 2, q = lane & 3;
  const int item = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (item >= kExperts * (IU + ID)) return;
  const int e = item / (IU + ID), it = item % (IU + ID);
  const uint8_t* eb = ex + size_t(e) * kEB;
  uint8_t* o = out + size_t(e) * (kS13 + kS2);
  if (it < IU) {
    const int i = it / KP, kp = it % KP;
    const UpRec x = up_load(sc + size_t(e) * kSC + size_t(it) * kRecU, r);
    const uint2* S = reinterpret_cast<const uint2*>(eb + kW13 + size_t(i) * 8 * KS1 * 2 + r * KS1 * 2) + kp * 8;
    uint32_t sw[16];
    up_decode(x, S, r, sw);
    if (q == 0)
      for (int k = 0; k < 16; k++)
        *reinterpret_cast<uint32_t*>(o + size_t(i) * 8 * KS1 * 2 + r * KS1 * 2 + kp * 64 + 4 * k) = sw[k];
  } else {
    const int n = it - IU;
    const DnRec x = dn_load(sc + size_t(e) * kSC + kUpB + size_t(n) * kRecD, r);
    const uint2* S = reinterpret_cast<const uint2*>(eb + kW13 + kS13 + kW2 + size_t(n) * 8 * KS2 * 2 + r * KS2 * 2);
    uint32_t sw[10];
    dn_decode(x, S, r, sw);
    if (q == 0)
      for (int k = 0; k < 10; k++)
        *reinterpret_cast<uint32_t*>(o + kS13 + size_t(n) * 8 * KS2 * 2 + r * KS2 * 2 + 4 * k) = sw[k];
  }
}

// ---- host: build one expert's records from its expanded scales, check every item decodes to them -------------------
struct Stats {
  long long scales = 0, exc = 0, items = 0, esc_items = 0, esc_rows = 0;
};

// one item: v[r][p][k] (k < 4·NC) → record (nibble word of (r, c) at nib(r, c), bases at bo, slots at so); host check
template <int NC, class Nib>
void item(const uint8_t (&v)[8][2][32], uint8_t* rec, Nib nib, int bo, int so, Stats& st) {
  constexpr int K = 4 * NC;
  uint8_t b[8][2];
  for (int r = 0; r < 8; r++)
    for (int p = 0; p < 2; p++) {
      int h[128] = {0};
      for (int k = 0; k < K; k++) {
        if (v[r][p][k] > 0x7E) qlog::fail("moe", "p014: scale 0x%02x", v[r][p][k]);
        h[v[r][p][k]]++;
      }
      int best = -1, bi = 0, win = 0;
      for (int x = 0; x < 16; x++) win += h[x];
      for (int x = 0; x < 128; x++) {  // window x … x+15
        if (win > best) best = win, bi = x;
        win -= h[x];
        if (x + 16 < 128) win += h[x + 16];
      }
      b[r][p] = uint8_t(bi);
    }
  std::vector<uint16_t> ent[8];
  uint32_t nw[8][NC] = {};
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < NC; c++)
      for (int ks = 0; ks < 4; ks++)
        for (int p = 0; p < 2; p++) {
          const int s = v[r][p][4 * c + ks], d = s - b[r][p];
          if (d >= 0 && d < 16) nw[r][c] |= uint32_t(d) << (4 * nib_pos(ks, p));
          else ent[r].push_back(uint16_t(r << 13 | (2 * c + (ks >> 1)) << 9 | ((ks & 1) * 2 + p) << 7 | s));
        }
  // more than kSlots exceptions: escape rows, most exceptions first (lower r on ties), until the rest fit
  bool esc[8] = {};
  size_t tot = 0;
  for (auto& x : ent) tot += x.size();
  st.scales += 16 * K, st.items++, st.exc += tot, st.esc_items += tot > size_t(kSlots);
  while (tot > size_t(kSlots)) {
    int m = -1;
    for (int r = 0; r < 8; r++)
      if (!esc[r] && (m < 0 || ent[r].size() > ent[m].size())) m = r;
    esc[m] = true, tot -= ent[m].size(), st.esc_rows++;
  }
  uint16_t sl[kSlots];
  int ns = 0;
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < NC; c++) {
      const uint32_t x = esc[r] ? 0u : nw[r][c];
      std::memcpy(rec + nib(r, c), &x, 4);
    }
    rec[bo + 2 * r] = esc[r] ? 0x80 : b[r][0];
    rec[bo + 2 * r + 1] = esc[r] ? 0x80 : b[r][1];
    if (!esc[r])
      for (auto x : ent[r]) sl[ns++] = x;
  }
  for (int j = ns; j < kSlots; j++) sl[j] = uint16_t(kNone);
  std::memcpy(rec + so, sl, sizeof sl);
  // check: the device decode (same function) of every row == its scales; escaped rows: flag set (plain bytes read)
  uint32_t slw[4];
  std::memcpy(slw, rec + so, 16);
  for (int r = 0; r < 8; r++) {
    uint32_t w[NC], sw[2 * NC];
    for (int c = 0; c < NC; c++) std::memcpy(&w[c], rec + nib(r, c), 4);
    const uint32_t bs = rec[bo + 2 * r] | uint32_t(rec[bo + 2 * r + 1]) << 8;
    if (bool(bs & 0x80) != esc[r]) qlog::fail("moe", "p014: escape flag");
    if (esc[r]) continue;
    decode<2 * NC>(w, bs, slw, r, sw);
    for (int c = 0; c < NC; c++)
      for (int ks = 0; ks < 4; ks++)
        for (int p = 0; p < 2; p++)
          if (((sw[2 * c + (ks >> 1)] >> (8 * ((ks & 1) * 2 + p))) & 0xFF) != v[r][p][4 * c + ks])
            qlog::fail("moe", "p014: row %d k-step %d scale 0x%02x decodes differently", r, 4 * c + ks, v[r][p][4 * c + ks]);
  }
}
// s13 / s2: the expert block's expanded scales ([40][8][160][2], [160][8][20][2]) → out [kSC]
void build(const uint8_t* s13, const uint8_t* s2, uint8_t* out, Stats& st) {
  uint8_t v[8][2][32];
  for (int i = 0; i < T13; i++)
    for (int kp = 0; kp < KP; kp++) {
      for (int r = 0; r < 8; r++)
        for (int k = 0; k < 32; k++)
          for (int p = 0; p < 2; p++) v[r][p][k] = s13[((size_t(i) * 8 + r) * KS1 + 32 * kp + k) * 2 + p];
      item<8>(v, out + size_t(i * KP + kp) * kRecU, [](int r, int c) { return (c >> 2) * 128 + 16 * r + 4 * (c & 3); }, kNibU, kNibU + 16, st);
    }
  for (int n = 0; n < T2; n++) {
    for (int r = 0; r < 8; r++)
      for (int k = 0; k < KS2; k++)
        for (int p = 0; p < 2; p++) v[r][p][k] = s2[((size_t(n) * 8 + r) * KS2 + k) * 2 + p];
    item<5>(v, out + kUpB + size_t(n) * kRecD, [](int r, int c) { return 32 * c + 4 * r; }, kNibD, kNibD + 16, st);
  }
}

}  // namespace p014
