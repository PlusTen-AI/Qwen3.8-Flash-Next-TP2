// model_impl.h — Engine internals shared by model.cc (target step) and mtp.cc (MTP draft head). Not a public header.
#pragma once
#include <string>
#include <vector>

#include "core/log.h"
#include "model/model.h"

namespace q38 {

inline constexpr int kEos = 248044;
inline const std::string kP = "model.language_model.";
// prefix checkpoint layout: per GDN layer S (fp32 [24,128,128]) then conv (bf16 [3,5120]), then the PLE window
inline constexpr int kGdnLayers = 36;  // every 4th of the 48 layers is QSA
inline constexpr size_t kSnapS = sizeof(float) * 24 * 128 * 128, kSnapConv = sizeof(bf16) * 3 * 5120,
                        kSnapPle = sizeof(bf16) * 9 * 10240;
inline constexpr size_t kCkptGdn = kGdnLayers * (kSnapS + kSnapConv), kCkptBytes = kCkptGdn + kSnapPle;

struct Engine::Layer {
  bool qsa = false;
  HcW ahc, mhc;
  GdnW gdn;
  QsaW q;
  MoeW* moe = nullptr;
  GdnState* d_gst = nullptr;   // [slots] device structs
  QsaCache* d_qc = nullptr;    // [slots]
  GdnState** d_gp = nullptr;   // [max B] per-step pointers
  QsaCache** d_qp = nullptr;
  GdnRec* d_rec = nullptr;     // [2·slots] verify-step records (MTP): record (slot s, parity q) at d_rec + 2s + q
  GdnRec** d_rp = nullptr;     // [max B] records the verify writes
  GdnRec** d_pp = nullptr;     // [max B] P020: pending records the verify applies first
};

struct Engine::Slot {
  std::vector<GdnState> g;  // per GDN layer (host copies of device pointers)
  std::vector<QsaCache> q;  // per QSA layer
  bf16* ple_conv = nullptr; // [9, 10240]
  QsaCache mq;              // MTP layer's QSA cache (if the head is loaded)
  void* gdn_rec = nullptr;  // verify records of the 36 GDN layers, two parities (gdn_rec_bytes() each)
  int gdn_pn = 0;           // P020: kept rows of the last verify not yet applied to the GDN state (0 = none)
  int gdn_par = 0;          // P020: parity of the record holding them
  bf16* ple_snap = nullptr; // [9, 10240] PLE conv state before the last verify step
  bf16* ple_rec = nullptr;  // [16, 10240] PLE conv inputs of the last verify step's rows
  float* dq = nullptr;      // [15, vocab] logits the pending drafts were sampled from (q of the accept test); P019:
                            // row i holds only this rank's half (first 124160 floats)
  SparseCand* dqc = nullptr;  // [15][2][kSparseRec] P019: the pending drafts' candidate records (rank 0, rank 1)
  PenSeen pen{};            // penalties' seen tokens (allocated by the first pen_begin)
};

// MTP draft head weights (mtp.cc). Projections are bf16 in the checkpoint (not MXFP8 as in the decoder layers).
struct Engine::Mtp {
  Lin fc_e, fc_h;                          // fc_embedding / fc_hidden [2560, 2560], replicated: full outputs, no gather
  const bf16 *nw_e = nullptr, *nw_h = nullptr;  // pre_fc_norm_embedding [2560] / pre_fc_norm_hidden [10240]
  HcW ahc, mhc, mixer;
  QsaW q;                                  // q.qkv = q ‖ k ‖ v ‖ indexer, q.o (bf16, TP split as the decoder)
  MoeW* moe = nullptr;                     // routed experts only (zero shared term)
  Lin sh_gu, sh_dn;                        // bf16 shared expert: gate ‖ up (col split), down (row split)
  const bf16* sh_g = nullptr;              // shared_expert_gate [2560]
  QsaCache* d_qc = nullptr;                // [slots]
  QsaCache** d_qp = nullptr;               // [max B]
  Lin lm_q;                                // P006: NVFP4 / MXFP8 copy of this rank's lm_head half, draft passes only
};

// zeroed device array. The memset runs on the legacy stream, which does not order with the engine's non-blocking
// streams: wait for it, so work issued later on s_ / cs_ never races the zeroing.
template <class T>
T* dalloc(size_t n) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
  CUDA_CHECK(cudaStreamSynchronize(nullptr));
  return p;
}

}  // namespace q38
