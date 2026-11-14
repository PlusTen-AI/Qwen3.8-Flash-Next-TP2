// moe.h — MoE block for one layer on this rank: spec §2.6. Router bf16 [512,2560] replicated → softmax fp32 →
// top-10 → renormalise; routed experts NVFP4 W4A16 (bf16 activations; w = e2m1·e4m3·g), TP-split intermediate
// (gate/up rows [320r,+320), down cols [320r,+320)): h = silu(g)·u, y_e = topk_w_e·(h·w2ᵀ), routed = Σ_e y_e;
// shared expert MXFP8 (TP-split, implemented here, fused — no dependency on linear.h) · sigmoid(x·shared_expert_gate). out = routed + shared = this rank's partial
// (caller all-reduces).
#pragma once
#include <string>

#include "core/common.h"
#include "kernels/linear/linear.h"
#include "core/weights.h"

namespace q38 {

struct MoeW;  // opaque, defined in moe.cu
MoeW* moe_upload(const Weights& W, int layer);
// Same by tensor prefix (".../mlp."). mtp = the MTP layer ("mtp.layers.0.mlp."): routed experts with plain E4M3 scales
// (no CSF); its bf16 shared expert is NOT included (uploaded as zeros → out = routed partial exactly), the caller adds it.
MoeW* moe_upload(const Weights& W, const std::string& mlp_prefix, bool mtp);
// With P014 moe_scale4 on at upload, also a 4-bit copy of the routed scales (+45 MB per layer) for the decode path;
// with P017 router_pack12 on at upload, a 12-bit copy of the router (+2.0 MB per layer) for the decode path.
// v [M,2560] → out [M,2560] partial. topk_out (optional): [M,10] int32 expert ids, topw_out [M,10] fp32 (tests).
// M ≤ 64: decode path, two launches (PDL); calls on one MoeW must be stream-ordered (per-layer scratch).
// 64 < M ≤ 8192: large-M path (moe_large.cuh), 8 launches; its workspace (~487 MB, allocated by the first moe_upload) is
// shared by all MoeW, so large-M calls must be stream-ordered with each other. Both paths graph-capturable, no
// allocation per call, and bitwise equal row by row (a row's result does not depend on M or on the other rows).
void moe_forward(const MoeW* w, const bf16* v, bf16* out, int M, int* topk_out, float* topw_out, cudaStream_t s);
void moe_free(MoeW* w);

// Tests only. Optional device outputs of the same call: router logits [M,512], routed sum / shared expert part
// [M,2560] (out = bf16(routed + shared)), shared-expert h [M,320] in B-fragment order (moe_frag_off), distinct experts.
struct MoeDbg {
  bf16 *logits = nullptr, *routed = nullptr, *shared = nullptr, *h_shared = nullptr;
  int* n_experts = nullptr;
};
void moe_forward_dbg(const MoeW* w, const bf16* v, bf16* out, int M, int* topk_out, float* topw_out, const MoeDbg& dbg,
                     cudaStream_t s);
void moe_silu_mul(const bf16* gu, bf16* h, int M, int n, cudaStream_t s);  // h = bf16(bf16(silu(g))·u), gu = [M, g‖u]
int moe_frag_off(int k);  // element offset of k in a B-fragment-ordered row
// Tests only: one large-M call (64 < M ≤ 8192) timed per launch with events: ms[7] = router GEMM, top-k, scan, place,
// routed gate/up, routed down (+ shared gate/up in its shadow), shared down + combine.
void moe_forward_prof(const MoeW* w, const bf16* v, bf16* out, int M, float* ms, cudaStream_t s);
// Tests only, P014 moe_scale4 (4-bit copy of the routed scales, decode path): built at upload when the patch is on;
// info v[5] = scales, exceptions, items, items over 8 exceptions, escaped rows; unpack: every expert's scales decoded
// as k_experts does → dec, and the block's expanded scales → ref (device, [512][102400 + 51200] each); get/set one
// expert's expanded scales (host [102400 + 51200]: gate/up then down, block layout); set rebuilds its 4-bit records
// (checked) and returns their stats in v[5].
bool moe_p014_built(const MoeW* w);
void moe_p014_info(const MoeW* w, long long* v);
void moe_p014_unpack(const MoeW* w, uint8_t* dec, uint8_t* ref, cudaStream_t s);
void moe_p014_get_scales(const MoeW* w, int e, uint8_t* s13s2);
void moe_p014_set_scales(MoeW* w, int e, const uint8_t* s13s2, long long* v);
// Tests only, P017 router_pack12 (12-bit copy of the bf16 router, decode path): built at upload when the patch is on;
// info v[5] = values, exceptions, spans with exceptions, escaped spans, most exceptions in one span; unpack: the copy
// decoded as k_route does → out bf16 [512, 2560] row-major (device); set_router: replace the router (host bf16 bits
// [512, 2560]): its bf16 fragments and, when built, the 12-bit copy (rebuilt, host-checked), stats → v[5]; roundtrip:
// the same from host bits without a layer (build, upload, decode into out, free).
bool moe_p017_built(const MoeW* w);
void moe_p017_info(const MoeW* w, long long* v);
void moe_p017_unpack(const MoeW* w, bf16* out, cudaStream_t s);
void moe_p017_set_router(MoeW* w, const uint16_t* bits, long long* v);
void moe_p017_roundtrip(const uint16_t* bits, bf16* out, long long* v, cudaStream_t s);
// Tests only, P022 route_fast: the routing state the last decode call (M ≤ 64) on w left — its Plan as raw ints
// (moe_p022_plan_ints() of them: U, uexp[512], ucnt[512], uoff[512], utok[640], uw[640] (fp32 bits), tpair[640], NT,
// tt_u[592], tt_t[592]), the shared gate sig [M] fp32 and v in B-fragment order xf [M, 2560] (device pointers).
void moe_p022_state(const MoeW* w, int M, int* plan, float* sig, bf16* xf, cudaStream_t s);
int moe_p022_plan_ints();

}  // namespace q38
