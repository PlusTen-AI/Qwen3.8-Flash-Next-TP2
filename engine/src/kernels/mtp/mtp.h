// mtp.h — small kernels of the MTP draft head (bundle 3; model/mtp.cc assembles the head from these and the layer kernels).
// Reference code: vllm/models/qwen4_exp/nvidia/mtp.py (Qwen4ExpMultiTokenPredictor.forward):
//   emb  = fc_embedding(GemmaRMSNorm_2560(embed(next token)))          → the layer's pending block output (unit inject)
//   X    = fc_hidden per stream(GemmaRMSNorm_10240(multi-stream hidden)) → the layer's multi-stream input
//   one QSA decoder layer (own experts, bf16 shared expert, own HC) → final HC mixer → shared lm_head.
// Numerics: fp32 inside each op, bf16 at the same materialisation points as vLLM; reductions in a fixed order
// (deterministic and batch-invariant like every kernel of this engine).
#pragma once
#include "core/common.h"

namespace q38 {

// y [M, D] = bf16(x · rsqrt(mean(x²) + eps) · (1 + w)), fp32 inside (vLLM GemmaRMSNorm.forward_native). D % 8 == 0,
// D ≤ 10240. In place allowed (y == x).
void gemma_norm(const bf16* x, const bf16* w, bf16* y, int M, int D, cudaStream_t s);

// MTP shared expert (bf16 checkpoint weights; the MoE kernel adds a zero shared term for this layer):
// out [M,2560] ← bf16(out + bf16(bf16(σ(bf16(v·g))) · down)), the main MoE's shared-expert rounding points.
// v [M,2560] block input, g [2560] shared_expert_gate, down [M,2560] = bf16 partial of the shared down projection.
void mtp_shared_add(const bf16* v, const bf16* g, const bf16* down, bf16* out, int M, cudaStream_t s);

// dst row j = src row idx[j] (idx device [M]), rows of `width` bf16 (width % 8 == 0). src and dst must not overlap.
void gather_rows(const bf16* src, const int* idx, bf16* dst, int M, int width, cudaStream_t s);
// the same for rows of width % 4 == 0 (8-byte units)
void gather_rows4(const bf16* src, const int* idx, bf16* dst, int M, int width, cudaStream_t s);

}  // namespace q38
