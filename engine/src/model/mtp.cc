// mtp.cc — the MTP draft head (bundle 3): upload and one head pass over M rows.
// Reference code: vllm/models/qwen4_exp/nvidia/mtp.py. Per row (hidden of position p, token at p + 1):
//   bo = fc_embedding(GemmaRMSNorm_2560(embed(token)))                  pending block output, unit injection
//   X  = fc_hidden applied per stream to GemmaRMSNorm_10240(hidden)     multi-stream input [4 × 2560]
//   attn HC(X, bo) → QSA (own cache, position p) → o_proj → AR → mlp HC → MoE (own experts) + bf16 shared expert → AR
//   → final HC mixer: X materialised (the hidden of the next draft step) and y → shared lm_head.
// The hidden is the target's materialised multi-stream state after its last layer (X_ after the final mixer's
// combine) for the first pass, then the head's own output (Xb_). The draft is exact whatever it proposes (the target
// verifies), so the head only has to be good, not bitwise equal to the reference; it still follows the same operations.
#include "model/model_impl.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <thread>

#include "kernels/mtp/mtp.h"
#include "kernels/sample/sample.h"
#include "model/patches.h"

namespace q38 {

void Engine::mtp_upload() {
  const Weights& W = *w_;
  auto m = std::make_unique<Mtp>();
  const std::string p = "mtp.", lp = p + "layers.0.", a = lp + "self_attn.";
  auto rep = [&](const std::string& n) {  // replicated bf16 [2560, 2560] from the whole tensor
    const auto& ti = W.info(n);
    if (ti.dtype != "BF16" || ti.shape != std::vector<int64_t>{kHidden, kHidden}) qlog::fail("mtp", "%s: shape/format", n.c_str());
    auto b = W.raw(n);
    return lin_make(WMat::BF16, kHidden, kHidden, b.data(), nullptr, 0, true);  // + P013 12-bit copy (decode passes)
  };
  auto vec = [&](const std::string& n, size_t len) {
    auto b = W.raw(n);
    if (W.info(n).dtype != "BF16" || b.size() != len * 2) qlog::fail("mtp", "%s: %zu B (expected bf16 %zu)", n.c_str(), b.size(), len);
    bf16* d = dalloc<bf16>(len);
    CUDA_CHECK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice));
    return static_cast<const bf16*>(d);
  };
  m->fc_e = rep(p + "fc_embedding.weight");
  m->fc_h = rep(p + "fc_hidden.weight");
  m->nw_e = vec(p + "pre_fc_norm_embedding.weight", kHidden);
  m->nw_h = vec(p + "pre_fc_norm_hidden.weight", size_t(kStreams) * kHidden);
  m->ahc = hc_upload(W, lp + "attn_hyper_connection", true);
  m->mhc = hc_upload(W, lp + "mlp_hyper_connection", true);
  m->mixer = hc_upload(W, p + "hyper_connection_mixer", false);
  m->q = qsa_upload(W, a);
  // the head's bf16 projections also get the P013 12-bit copy (decode-sized passes read it; same bits)
  m->q.qkv = lin_upload(W, {a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "indexer.index_qk_proj.weight"},
                        true);
  m->q.o = lin_upload(W, {a + "o_proj.weight"}, true);
  m->moe = moe_upload(W, lp + "mlp.", true);
  m->sh_gu = lin_upload(W, {lp + "mlp.shared_expert.gate_proj.weight", lp + "mlp.shared_expert.up_proj.weight"}, true);
  m->sh_dn = lin_upload(W, {lp + "mlp.shared_expert.down_proj.weight"}, true);
  m->sh_g = vec(lp + "mlp.shared_expert_gate.weight", kHidden);
  // per rank: qkv 6144 + 256 + 256 + 640 · o K 3072 · shared gate ‖ up 320 + 320 · down K 320
  if (m->q.qkv.n != 7296 || m->q.o.k != 3072 || m->sh_gu.n != kSharedInter || m->sh_dn.k != kSharedInter / 2)
    qlog::fail("mtp", "projection shapes qkv %d · o.k %d · shared %d/%d", m->q.qkv.n, m->q.o.k, m->sh_gu.n, m->sh_dn.k);
  m->d_qc = dalloc<QsaCache>(c_.slots);
  m->d_qp = dalloc<QsaCache*>(c_.max_rows);
  for (int s = 0; s < c_.slots; s++) {
    QsaCache qc;  // with the shared pool: address ranges from kv_pool_init
    qc.cap = c_.ctx;
    if (!c_.kv_pool) {
      qc.kv = dalloc<bf16>(size_t(c_.ctx) * 512);
      qc.ik_raw = dalloc<bf16>(size_t(c_.ctx) * 128);
      qc.ik_comp = dalloc<bf16>(size_t(c_.ctx / 4) * 128);
    }
    CUDA_CHECK(cudaMemcpy(m->d_qc + s, &qc, sizeof qc, cudaMemcpyHostToDevice));
    slots_[s]->mq = qc;
  }
  // verify records: per GDN layer and slot (gdn_rec_bytes() each, carved from one block per slot), PLE window
  const size_t rb = gdn_rec_bytes();
  int ngdn = 0;
  for (auto& l : L_)
    if (!l->qsa) {
      l->d_rec = dalloc<GdnRec>(size_t(2) * c_.slots);
      l->d_rp = dalloc<GdnRec*>(c_.max_rows), l->d_pp = dalloc<GdnRec*>(c_.max_rows), ngdn++;
    }
  d_npend_ = dalloc<int>(c_.max_rows);
  for (int s = 0; s < c_.slots; s++) {
    auto& sl = *slots_[s];
    sl.gdn_rec = dalloc<uint8_t>(2 * rb * ngdn);  // two parities (P020 reads one while writing the other)
    int j = 0;
    for (auto& l : L_) {
      if (l->qsa) continue;
      for (int q = 0; q < 2; q++) {
        const GdnRec r = gdn_rec_at(static_cast<uint8_t*>(sl.gdn_rec) + rb * (2 * j + q));
        CUDA_CHECK(cudaMemcpy(l->d_rec + 2 * s + q, &r, sizeof r, cudaMemcpyHostToDevice));
      }
      j++;
    }
    sl.ple_snap = dalloc<bf16>(size_t(9) * 10240);
    sl.ple_rec = dalloc<bf16>(size_t(kGdnRecRows) * 10240);
    sl.dq = dalloc<float>(size_t(kGdnRecRows - 1) * kVocab);
    sl.dqc = dalloc<SparseCand>(size_t(kGdnRecRows - 1) * 2 * kSparseRec);
  }
  d_cst_ = dalloc<GdnState*>(size_t(ngdn) * c_.slots);
  d_crec_ = dalloc<GdnRec*>(size_t(ngdn) * c_.slots);
  d_n_ = dalloc<int>(c_.slots);
  const size_t LR = c_.logit_rows;
  qbuf_ = dalloc<float>(LR * kVocab);
  cand_send_ = dalloc<SparseCand>(LR * kSparseRec);  // P019 records
  cand_recv_ = dalloc<SparseCand>(2 * LR * kSparseRec);
  qcbuf_ = dalloc<SparseCand>(2 * LR * kSparseRec);
  d_nacc_ = dalloc<int>(LR), d_next_ = dalloc<int>(LR), d_pst_ = dalloc<int>(LR + 1), d_qst_ = dalloc<int>(LR + 1);
  d_drafts_ = dalloc<int>(LR), d_temp_ = dalloc<float>(LR), d_seed_ = dalloc<uint64_t>(LR), d_ctr_ = dalloc<uint64_t>(LR);
  d_topk_ = dalloc<int>(LR), d_topp_ = dalloc<float>(LR), d_minp_ = dalloc<float>(LR);
  d_pen_ = dalloc<PenSeq>(LR), d_last_ = dalloc<int>(LR), d_p01_ = dalloc<int>(2);
  {
    const int p01[2] = {0, 1};
    CUDA_CHECK(cudaMemcpy(d_p01_, p01, sizeof p01, cudaMemcpyHostToDevice));
  }
  const size_t R = c_.max_rows;
  Xa_ = dalloc<bf16>(R * 10240);
  Xb_ = dalloc<bf16>(R * 10240);
  Xh_ = dalloc<bf16>(R * 10240);
  d_src_ = dalloc<int>(R);
  mtp_ = std::move(m);
  draft_lm_upload();
}

// P006: quantized copy of this rank's lm_head half for the draft passes. NVFP4 (default, 179 MB: lin_make_nvfp4) or
// with Q38_DRAFT_LM=mxfp8 MXFP8 (328 MB; OCP MX: per row and 32-column block, scale 2^(⌊log2 amax⌋ − 8) as ue8m0,
// values e4m3 round-to-nearest, saturating). Uploaded with the head whatever the switch, so tests can compare both
// paths in one process. Only the draft distribution changes: the accept test uses exactly the logits drafts came from.
void Engine::draft_lm_upload() {
  std::vector<int64_t> sh;
  const auto w = w_->slice("lm_head.weight", &sh);  // this rank's [124160, 2560] bf16, the bytes lm_ came from
  if (sh != std::vector<int64_t>{kVocabHalf, kHidden} || w.size() != size_t(kVocabHalf) * kHidden * 2)
    qlog::fail("mtp", "lm_head slice %zu B", w.size());
  const bf16* x = reinterpret_cast<const bf16*>(w.data());
  const char* f = std::getenv("Q38_DRAFT_LM");
  if (!f || std::string(f) != "mxfp8") {
    if (f && std::string(f) != "nvfp4") qlog::fail("mtp", "Q38_DRAFT_LM=%s (nvfp4 | mxfp8)", f);
    mtp_->lm_q = lin_make_nvfp4(kVocabHalf, kHidden, x);
    return;
  }
  constexpr int KB = kHidden / 32;
  std::vector<uint8_t> q(size_t(kVocabHalf) * kHidden), sc(size_t(kVocabHalf) * KB);
  const int nt = std::max(1, std::min<int>(16, int(std::thread::hardware_concurrency())));
  std::vector<std::thread> th;
  for (int i = 0; i < nt; i++)
    th.emplace_back([&, i] {
      for (int r = i; r < kVocabHalf; r += nt)
        for (int kb = 0; kb < KB; kb++) {
          const size_t o = size_t(r) * kHidden + kb * 32;
          float amax = 0.f;
          for (int j = 0; j < 32; j++) amax = std::max(amax, std::fabs(__bfloat162float(x[o + j])));
          if (amax == 0.f) {
            sc[size_t(r) * KB + kb] = 0;  // 2^-127 · 0 = 0
            for (int j = 0; j < 32; j++) q[o + j] = 0;
            continue;
          }
          const int e = std::min(127, std::max(-126, int(std::floor(std::log2(amax))) - 8));
          sc[size_t(r) * KB + kb] = uint8_t(e + 127);
          const float inv = std::ldexp(1.f, -e);
          for (int j = 0; j < 32; j++)
            q[o + j] = __nv_cvt_float_to_fp8(__bfloat162float(x[o + j]) * inv, __NV_SATFINITE, __NV_E4M3);
        }
    });
  for (auto& t : th) t.join();
  mtp_->lm_q = lin_make(WMat::MXFP8, kVocabHalf, kHidden, q.data(), sc.data(), 0);
}

const float* Engine::mtp_step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                              const std::vector<int>& seq_start, const std::vector<int>& src, bool from_target,
                              bool all_logits) {
  const int M = int(ids.size()), B = int(slot.size());
  if (!mtp_) qlog::fail("mtp", "MTP head not loaded (EngineCfg.mtp)");
  if (M == 0 || M > c_.max_rows || int(pos.size()) != M || int(src.size()) != M || int(seq_start.size()) != B + 1 ||
      seq_start[B] != M)
    qlog::fail("mtp", "bad input · M %d · B %d", M, B);
  if ((all_logits ? M : B) > c_.logit_rows) qlog::fail("mtp", "logits rows %d > %d", all_logits ? M : B, c_.logit_rows);
  for (int j = 0; j < M; j++)
    if (pos[j] < 0 || pos[j] >= c_.ctx || src[j] < 0 || src[j] >= c_.max_rows)
      qlog::fail("mtp", "row %d: position %d / source row %d out of range", j, pos[j], src[j]);
  // One kernel call takes ≤ 8192 rows (the target step splits larger prefill chunks into P002 lanes). A longer pass of
  // one sequence runs as consecutive pieces: a piece's rows attend to the head's cache written by the earlier pieces,
  // the same values as one call (row-wise kernels). Logits of the last piece = the sequence's last row.
  constexpr int kPiece = 8192;
  if (M > kPiece) {
    if (B != 1 || all_logits) qlog::fail("mtp", "%d rows > %d: only one sequence with last-row logits can be split", M, kPiece);
    const float* lg = nullptr;
    for (int o = 0; o < M; o += kPiece) {
      const int n = std::min(kPiece, M - o);
      CUDA_CHECK(cudaMemcpyAsync(d_ids_, ids.data() + o, n * 4, cudaMemcpyHostToDevice, s_));
      lg = mtp_core({pos.begin() + o, pos.begin() + o + n}, slot, {0, n}, {src.begin() + o, src.begin() + o + n},
                    from_target, false);
    }
    return lg;
  }
  CUDA_CHECK(cudaMemcpyAsync(d_ids_, ids.data(), M * 4, cudaMemcpyHostToDevice, s_));
  return mtp_core(pos, slot, seq_start, src, from_target, all_logits);
}

const float* Engine::mtp_core(const std::vector<int>& pos, const std::vector<int>& slot,
                              const std::vector<int>& seq_start, const std::vector<int>& src, bool from_target,
                              bool all_logits) {
  const int M = int(pos.size()), B = int(slot.size());
  auto& m = *mtp_;
  for (int b = 0; b < B; b++) kv_ensure(slot[b], *std::max_element(pos.begin() + seq_start[b], pos.begin() + seq_start[b + 1]) + 1);
  CUDA_CHECK(cudaMemcpyAsync(d_pos_, pos.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_seq_, seq_start.data(), (B + 1) * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_src_, src.data(), M * 4, cudaMemcpyHostToDevice, s_));
  const int* rp = rope_rows(pos, slot, seq_start);  // images / videos: mRoPE positions (null: text)
  std::vector<QsaCache*> qp(B);
  for (int b = 0; b < B; b++) qp[b] = m.d_qc + slot[b];
  CUDA_CHECK(cudaMemcpyAsync(m.d_qp, qp.data(), B * sizeof(void*), cudaMemcpyHostToDevice, s_));

  // P023 mtp_last_rows: when a sequence has several rows (the first draft pass of a round, prefill head passes), only
  // its last row's output is used (logits, the next pass's hidden); the others only have to write the head's caches.
  // All rows run up to the QSA cache writes; then the last rows are gathered into rows 0 … B−1 (scratch in Xa_, free
  // after the hidden branch) and run the rest alone. Every kernel is row-wise, so those rows get the same bits.
  const bool last_only = M > B && !all_logits && patch::on(patch::mtp_last_rows);
  std::vector<int> lrow(B), lpos(B), lseq(B + 1);
  for (int b = 0; b < B; b++) lrow[b] = seq_start[b + 1] - 1, lpos[b] = pos[lrow[b]], lseq[b] = b;
  lseq[B] = B;
  const int Mb = last_only ? B : M;
  bf16* const Xc = last_only ? Xa_ : Xb_;  // the state, qkv and injection rows the rest of the pass runs on
  bf16* const t0c = last_only ? Xa_ + size_t(B) * 10240 : t0_;
  bf16* const iac = last_only ? t0c + size_t(B) * m.q.qkv.n : inj_a_;
  nmark_ = 0;
  auto enqueue = [&] {
  qlog::where_layer() = kLayers;  // the MTP layer
  mark("start");
  // token branch → pending block output bo_
  embed_lookup(embed_, c_.rank, d_ids_, part_, M, s_);
  ar(part_, size_t(M) * kHidden, s_);
  gemma_norm(part_, m.nw_e, E_, M, kHidden, s_);
  lin_forward(m.fc_e, E_, bo_, M, s_);
  // hidden branch → multi-stream state Xb_ (fc_hidden shared by the 4 streams: rows of the [4M, 2560] view)
  gather_rows(head_src_ ? head_src_ : from_target ? X_ : Xb_, d_src_, Xa_, M, 10240, s_);
  gemma_norm(Xa_, m.nw_h, Xa_, M, 10240, s_);
  for (int r = 0; r < 4 * M; r += kMaxRows)
    lin_forward(m.fc_h, Xa_ + size_t(r) * kHidden, Xb_ + size_t(r) * kHidden, std::min(kMaxRows, 4 * M - r), s_);
  mark("mtp.in");
  // the decoder layer (QSA), as in Engine::step with the injection of the token branch = unit weights
  hc_combine_mix(m.ahc, Xb_, bo_, nullptr, u_, inj_a_, M, s_);
  lin_forward(m.q.qkv, u_, t0_, M, s_);
  if (last_only) {  // P023: every row's cache writes, then the last rows alone (d_src_ / d_pos_ / d_seq_ reused in order)
    qsa_prepare(m.q, t0_, d_pos_, m.d_qp, d_seq_, B, M, s_, rp);
    CUDA_CHECK(cudaMemcpyAsync(d_src_, lrow.data(), B * 4, cudaMemcpyHostToDevice, s_));
    gather_rows(Xb_, d_src_, Xc, B, 10240, s_);
    gather_rows(t0_, d_src_, t0c, B, m.q.qkv.n, s_);
    gather_rows4(inj_a_, d_src_, iac, B, 4, s_);
    CUDA_CHECK(cudaMemcpyAsync(d_pos_, lpos.data(), B * 4, cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d_seq_, lseq.data(), (B + 1) * 4, cudaMemcpyHostToDevice, s_));
    if (rp) rope_rows(lpos, slot, lseq);  // the last rows' (same buffer, after qsa_prepare in stream order)
  }
  qsa_core(m.q, t0c, d_pos_, t2_, m.d_qp, d_seq_, B, Mb, nullptr, s_, *std::max_element(pos.begin(), pos.end()), rp);
  lin_forward(m.q.o, t2_, part_, Mb, s_);
  ar(part_, size_t(Mb) * kHidden, s_);
  mark("mtp.attn");
  hc_combine_mix(m.mhc, Xc, part_, iac, v_, inj_m_, Mb, s_);
  moe_forward(m.moe, v_, bo_, Mb, nullptr, nullptr, s_);  // routed partial (the kernel's shared term is zero)
  lin_forward(m.sh_gu, v_, t0_, Mb, s_);                  // [M, 320 gate ‖ 320 up] of this rank
  moe_silu_mul(t0_, t2_, Mb, kSharedInter / 2, s_);
  lin_forward(m.sh_dn, t2_, part_, Mb, s_);               // shared down, partial over the TP-split intermediate
  mtp_shared_add(v_, m.sh_g, part_, bo_, Mb, s_);
  ar(bo_, size_t(Mb) * kHidden, s_);
  mark("mtp.moe");
  hc_combine_mix(m.mixer, Xc, bo_, inj_m_, y_, nullptr, Mb, s_);  // Xc materialised: next draft step's hidden
  if (last_only) CUDA_CHECK(cudaMemcpyAsync(Xb_, Xc, size_t(B) * 10240 * sizeof(bf16), cudaMemcpyDeviceToDevice, s_));
  mark("mtp.hc");
  head_enqueue(Mb, B, last_only ? lseq : seq_start, all_logits, true);
  };
  enqueue();
  for (int b = 0; b < B; b++) head_last_[b] = last_only ? b : lrow[b];  // row of Xb_ holding sequence b's next hidden
  return head_finish(all_logits ? M : B);
}

const float* Engine::verify(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                            const std::vector<int>& seq_start) {
  const int B = int(slot.size()), M = int(ids.size());
  if (!mtp_) qlog::fail("verify", "MTP head not loaded (EngineCfg.mtp)");
  if (int(seq_start.size()) != B + 1 || B > c_.slots) qlog::fail("verify", "bad input · B %d", B);
  // the PLE record reads the decode path's conv inputs (ple_last_cin: M ≤ 64)
  if (M > kDecodeRows) qlog::fail("verify", "%d rows > %d (the verify step takes the decode path)", M, kDecodeRows);
  spec_pos0_.assign(B, 0), spec_rows_.assign(B, 0);
  for (int b = 0; b < B; b++) {
    const int rows = seq_start[b + 1] - seq_start[b];
    if (rows < 1 || rows > kGdnRecRows) qlog::fail("verify", "sequence %d: %d rows (1…%d)", b, rows, kGdnRecRows);
    auto& sl = *slots_[slot[b]];
    CUDA_CHECK(cudaMemcpyAsync(sl.ple_snap, sl.ple_conv, sizeof(bf16) * 9 * 10240, cudaMemcpyDeviceToDevice, s_));
    spec_pos0_[b] = pos[seq_start[b]], spec_rows_[b] = rows;
  }
  spec_ = true;
  const float* lg = step(ids, pos, slot, seq_start, true);
  spec_ = false;
  const bf16* cin = ple_last_cin(ple_);
  for (int b = 0; b < B; b++)
    CUDA_CHECK(cudaMemcpyAsync(slots_[slot[b]]->ple_rec, cin + size_t(seq_start[b]) * 10240,
                               sizeof(bf16) * spec_rows_[b] * 10240, cudaMemcpyDeviceToDevice, s_));
  spec_slot_ = slot;
  return lg;
}

// P020: apply the pending GDN rows of these slots (gdn_commit over all GDN layers, one launch) before any use of their
// state other than a verify. Nothing to do in the common case (no pending rows).
void Engine::gdn_flush_slots(const std::vector<int>& slots) {
  std::vector<int> fs, n;
  for (int s : slots)
    if (slots_[s]->gdn_pn > 0 && std::find(fs.begin(), fs.end(), s) == fs.end()) fs.push_back(s), n.push_back(slots_[s]->gdn_pn);
  if (fs.empty()) return;
  const int Bf = int(fs.size());
  std::vector<GdnState*> st;
  std::vector<GdnRec*> rc;
  for (auto& l : L_)
    if (!l->qsa)
      for (int j = 0; j < Bf; j++)
        st.push_back(l->d_gst + fs[j]), rc.push_back(l->d_rec + 2 * fs[j] + slots_[fs[j]]->gdn_par);
  CUDA_CHECK(cudaMemcpyAsync(d_cst_, st.data(), st.size() * sizeof(void*), cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_crec_, rc.data(), rc.size() * sizeof(void*), cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_n_, n.data(), Bf * sizeof(int), cudaMemcpyHostToDevice, s_));
  gdn_flush(d_cst_, d_crec_, d_n_, int(st.size()) / Bf, Bf, s_);
  for (int s : fs) slots_[s]->gdn_pn = 0;
}

void Engine::commit(const std::vector<int>& n) {
  const int B = int(spec_slot_.size());
  if (B == 0 || int(n.size()) != B) qlog::fail("commit", "no previous verify, or n has %zu entries ≠ %d sequences", n.size(), B);
  for (int b = 0; b < B; b++)
    if (n[b] < 0 || n[b] > spec_rows_[b]) qlog::fail("commit", "sequence %d: n %d (0…%d)", b, n[b], spec_rows_[b]);
  if (spec_p20_) {  // P020: the GDN rows stay pending; the slot's next verify (or a flush) applies them
    for (int b = 0; b < B; b++) slots_[spec_slot_[b]]->gdn_pn = n[b];
  } else {
    // GDN: replay the first n rows of every layer's record, one launch (pair = layer · B + b)
    std::vector<GdnState*> st;
    std::vector<GdnRec*> rc;
    for (auto& l : L_)
      if (!l->qsa)
        for (int b = 0; b < B; b++) st.push_back(l->d_gst + spec_slot_[b]), rc.push_back(l->d_rec + 2 * spec_slot_[b]);
    CUDA_CHECK(cudaMemcpyAsync(d_cst_, st.data(), st.size() * sizeof(void*), cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d_crec_, rc.data(), rc.size() * sizeof(void*), cudaMemcpyHostToDevice, s_));
    CUDA_CHECK(cudaMemcpyAsync(d_n_, n.data(), B * sizeof(int), cudaMemcpyHostToDevice, s_));
    gdn_commit(d_cst_, d_crec_, d_n_, int(st.size()) / B, B, s_);
  }
  // PLE: window = last 9 rows of (state before the step ‖ conv inputs of the kept rows); copies, so bit-exact
  constexpr size_t RW = 10240, RB = RW * sizeof(bf16);
  for (int b = 0; b < B; b++) {
    auto& sl = *slots_[spec_slot_[b]];
    const int k = n[b];
    if (k >= 9) {
      CUDA_CHECK(cudaMemcpyAsync(sl.ple_conv, sl.ple_rec + (k - 9) * RW, 9 * RB, cudaMemcpyDeviceToDevice, s_));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(sl.ple_conv, sl.ple_snap + k * RW, (9 - k) * RB, cudaMemcpyDeviceToDevice, s_));
      if (k) CUDA_CHECK(cudaMemcpyAsync(sl.ple_conv + (9 - k) * RW, sl.ple_rec, k * RB, cudaMemcpyDeviceToDevice, s_));
    }
    hist_[spec_slot_[b]].resize(spec_pos0_[b] + k);
  }
  spec_slot_.clear();  // (pageable host sources: cudaMemcpyAsync has staged them before returning, as in step())
}

// ---- speculative generation ---------------------------------------------------------------------------------------
// Random streams (kernels/sample: Philox keyed by seed, counter): a draft for position x uses (seed, x); the accept
// test of a round whose last token sits at position p uses (seed ^ kSeedAcc, p); the first token of a sequence
// (position L) uses (seed ^ kSeedFirst, L). A draft's uniform can come back in a later round for the same position,
// but only after that draft was discarded unverified (positions after the first rejection), so it is independent of
// everything emitted — exactness holds.
static constexpr uint64_t kSeedAcc = 0x9E3779B97F4A7C15ull, kSeedFirst = 0xD1B54A32D192ED03ull;

// sample one token per row of lg_ (rows 0 … B−1, one per sequence) into dst (device), keys and truncation per sequence
static void sample_keys(const std::vector<SpecSeq*>& q, uint64_t xr, const std::vector<uint64_t>& ctr, float* d_temp,
                        int* d_topk, float* d_topp, uint64_t* d_seed, uint64_t* d_ctr, cudaStream_t s,
                        float* d_minp = nullptr) {
  const int B = int(q.size());
  std::vector<float> t(B), tp(B), mp(B);
  std::vector<int> tk(B);
  std::vector<uint64_t> sd(B);
  for (int b = 0; b < B; b++)
    t[b] = q[b]->temp, tk[b] = q[b]->top_k, tp[b] = q[b]->top_p, sd[b] = q[b]->seed ^ xr, mp[b] = q[b]->min_p;
  if (d_minp) CUDA_CHECK(cudaMemcpyAsync(d_minp, mp.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_temp, t.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_topk, tk.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_topp, tp.data(), B * 4, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_seed, sd.data(), B * 8, cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_ctr, ctr.data(), B * 8, cudaMemcpyHostToDevice, s));
}

// The head has just run over each sequence's rows fss[b] … fss[b+1]−1 (its logits lg, row b = its last row). Draft k
// tokens: sample from lg, feed the sample as the next pass's token (d_ids_, no host round trip), k − 1 more passes.
bool Engine::sparse_seqs(const std::vector<SpecSeq*>& q) const {
  if (!patch::on(patch::sparse_logits)) return false;
  for (auto* x : q)
    if (!sparse_ok(x->temp, x->top_k, x->top_p, kVocab)) return false;
  return true;
}

void Engine::drafts(std::vector<SpecSeq*>& q, const std::vector<int>& fss,
                    const std::function<const float*(int)>& first) {
  bool sp = sparse_seqs(q);
  for (int attempt = 0;; attempt++) {
    sparse_ = sp;
    const float* lg = first(attempt);
    const bool ok = draft_chain(q, fss, lg, sp);
    sparse_ = false;
    if (ok) return;
    sp = false;  // P019: a head row held a NaN / inf — the same passes again with full rows (positional caches: same values)
  }
}

bool Engine::draft_chain(std::vector<SpecSeq*>& q, const std::vector<int>& fss, const float* lg, bool sp) {
  const int B = int(q.size());
  int k = 0;  // the chain runs to the largest depth; each sequence keeps its own depth's drafts (and q rows)
  for (auto* x : q) k = std::max(k, x->depth);
  bool any_t = false;
  for (auto* x : q) any_t |= x->temp > 0.f;
  std::vector<int> slot(B), seq(B + 1), pos(B), src(B);
  for (int b = 0; b <= B; b++) seq[b] = b;
  for (int b = 0; b < B; b++) slot[b] = q[b]->slot;
  std::vector<uint64_t> ctr(B);
  for (int i = 0; i < k; i++) {
    for (int b = 0; b < B; b++) ctr[b] = uint64_t(q[b]->pos + i + 1);
    sample_keys(q, 0, ctr, d_temp_, d_topk_, d_topp_, d_seed_, d_ctr_, s_);
    if (sp) sample_rows_sparse(cand_recv_, B, B, kVocab, d_temp_, d_topk_, d_topp_, d_seed_, d_ctr_, d_ids_, nullptr, s_);
    else sample_rows(lg, B, kVocab, d_temp_, d_topk_, d_topp_, d_seed_, d_ctr_, d_ids_, nullptr, s_);
    CUDA_CHECK(cudaMemcpy2DAsync(d_drafts_ + i, size_t(k) * 4, d_ids_, 4, 4, B, cudaMemcpyDeviceToDevice, s_));
    if (any_t)  // q of the accept test, kept with the sequence (its drafts outlive this call: spec_begin is per sequence)
      for (int b = 0; b < B; b++) {
        if (!(q[b]->temp > 0.f && i < q[b]->depth)) continue;
        auto& sl = *slots_[q[b]->slot];
        if (sp) {  // P019: both ranks' records of row b, and this rank's half (for a full-row fallback of the accept)
          for (int r = 0; r < 2; r++)
            CUDA_CHECK(cudaMemcpyAsync(sl.dqc + (size_t(i) * 2 + r) * kSparseRec, cand_recv_ + (size_t(r) * B + b) * kSparseRec,
                                       kSparseRec * sizeof(SparseCand), cudaMemcpyDeviceToDevice, s_));
          CUDA_CHECK(cudaMemcpyAsync(sl.dq + size_t(i) * kVocab, lg_half_ + (size_t(c_.rank) * B + b) * kVocabHalf,
                                     size_t(kVocabHalf) * 4, cudaMemcpyDeviceToDevice, s_));
        } else {
          CUDA_CHECK(cudaMemcpyAsync(sl.dq + size_t(i) * kVocab, lg + size_t(b) * kVocab, size_t(kVocab) * 4,
                                     cudaMemcpyDeviceToDevice, s_));
        }
      }
    if (i + 1 == k) break;
    for (int b = 0; b < B; b++) pos[b] = q[b]->pos + i, src[b] = i == 0 ? head_last_[b] : b;
    lg = mtp_core(pos, slot, seq, src, false, false);
  }
  std::vector<int> d(size_t(B) * k);
  CUDA_CHECK(cudaMemcpyAsync(d.data(), d_drafts_, d.size() * 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  if (sp && std::find(d.begin(), d.end(), kSparseDense) != d.end()) return false;  // redo with full rows
  for (int b = 0; b < B; b++) {
    q[b]->drafts.assign(d.begin() + size_t(b) * k, d.begin() + size_t(b) * k + q[b]->depth);
    q[b]->q_sparse = sp;
  }
  return true;
}

void Engine::q_full_rows(const std::vector<SpecSeq*>& q, const std::vector<int>& kb, const std::vector<int>& qst) {
  const int B = int(q.size());
  std::vector<int> sb;
  int Qs = 0;
  for (int b = 0; b < B; b++) {
    if (!(q[b]->temp > 0.f)) continue;  // greedy ignores q
    if (q[b]->q_sparse) {
      sb.push_back(b), Qs += kb[b];
    } else {
      CUDA_CHECK(cudaMemcpyAsync(qbuf_ + size_t(qst[b]) * kVocab, slots_[q[b]->slot]->dq, size_t(kb[b]) * kVocab * 4,
                                 cudaMemcpyDeviceToDevice, s_));
    }
  }
  if (!Qs) return;
  // P019 drafts: both ranks' stored halves → full rows (one all-gather through lg_half_, free at this point)
  float* mine = lg_half_ + size_t(c_.rank) * Qs * kVocabHalf;
  int j = 0;
  for (int b : sb) {
    CUDA_CHECK(cudaMemcpy2DAsync(mine + size_t(j) * kVocabHalf, kVocabHalf * 4, slots_[q[b]->slot]->dq, kVocab * 4,
                                 kVocabHalf * 4, kb[b], cudaMemcpyDeviceToDevice, s_));
    j += kb[b];
  }
  comm_->all_gather_f32(mine, lg_half_, size_t(Qs) * kVocabHalf, s_);
  j = 0;
  for (int b : sb) {
    for (int r = 0; r < 2; r++)
      CUDA_CHECK(cudaMemcpy2DAsync(qbuf_ + size_t(qst[b]) * kVocab + size_t(r) * kVocabHalf, kVocab * 4,
                                   lg_half_ + (size_t(r) * Qs + j) * kVocabHalf, kVocabHalf * 4, kVocabHalf * 4, kb[b],
                                   cudaMemcpyDeviceToDevice, s_));
    j += kb[b];
  }
}

int Engine::spec_begin(SpecSeq& q, int row0, int lrow, const std::vector<int>& next_ids) {
  if (!mtp_) qlog::fail("spec", "MTP head not loaded (EngineCfg.mtp)");
  const int n = int(next_ids.size()) + 1, p0 = q.pos - n;
  if (q.depth < 1 || q.depth + 1 > kGdnRecRows || q.depth > c_.logit_rows || p0 < 0 || row0 + n > c_.max_rows)
    qlog::fail("spec", "spec_begin: depth %d · chunk %d rows · position %d", q.depth, n, q.pos);
  kv_ensure(q.slot, q.pos + q.depth + 1);  // the drafts' head positions, before anything changes (KvFull)
  std::vector<SpecSeq*> one{&q};
  // first token from the step's logits row lrow → the head's token for the prompt's last row (d_ids_[n − 1])
  sample_keys(one, kSeedFirst, {uint64_t(q.pos)}, d_temp_, d_topk_, d_topp_, d_seed_, d_ctr_, s_, d_minp_);
  if (pen_upload(one))  // the prompt's last row: prompt marks, no generated tokens yet
    penalize_rows(lg_ + size_t(lrow) * kVocab, kVocab, 0, kVocab, kVocab, 1, d_p01_, nullptr, nullptr, 1, d_pen_, s_);
  sample_rows(lg_ + size_t(lrow) * kVocab, 1, kVocab, d_temp_, d_topk_, d_topp_, d_minp_, d_seed_, d_ctr_,
              d_ids_ + n - 1, nullptr, s_);
  if (n > 1) CUDA_CHECK(cudaMemcpyAsync(d_ids_, next_ids.data(), (n - 1) * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(&q.last, d_ids_ + n - 1, 4, cudaMemcpyDeviceToHost, s_));
  std::vector<int> pos(n), src(n);
  for (int j = 0; j < n; j++) pos[j] = p0 + j, src[j] = row0 + j;
  drafts(one, {0, n}, [&](int attempt) {
    if (attempt > 0) {  // the chain overwrote d_ids_: the chunk's next tokens and the first token again
      if (n > 1) CUDA_CHECK(cudaMemcpyAsync(d_ids_, next_ids.data(), (n - 1) * 4, cudaMemcpyHostToDevice, s_));
      CUDA_CHECK(cudaMemcpyAsync(d_ids_ + n - 1, &q.last, 4, cudaMemcpyHostToDevice, s_));
    }
    return mtp_core(pos, {q.slot}, {0, n}, src, true, false);  // syncs: q.last is valid after it
  });
  return q.last;
}

int Engine::spec_first(SpecSeq& q, int lrow) {
  if (!mtp_) qlog::fail("spec", "MTP head not loaded (EngineCfg.mtp)");
  if (q.depth < 1 || q.depth + 1 > kGdnRecRows || q.depth > c_.logit_rows || q.pos < 1)
    qlog::fail("spec", "spec_first: depth %d · position %d", q.depth, q.pos);
  std::vector<SpecSeq*> one{&q};
  sample_keys(one, kSeedFirst, {uint64_t(q.pos)}, d_temp_, d_topk_, d_topp_, d_seed_, d_ctr_, s_, d_minp_);
  if (pen_upload(one))  // the prompt's last row: prompt marks, no generated tokens yet
    penalize_rows(lg_ + size_t(lrow) * kVocab, kVocab, 0, kVocab, kVocab, 1, d_p01_, nullptr, nullptr, 1, d_pen_, s_);
  sample_rows(lg_ + size_t(lrow) * kVocab, 1, kVocab, d_temp_, d_topk_, d_topp_, d_minp_, d_seed_, d_ctr_, d_ids_, nullptr,
              s_);
  CUDA_CHECK(cudaMemcpyAsync(&q.last, d_ids_, 4, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  return q.last;
}

void Engine::spec_rest(SpecSeq& q, int src_row) {
  std::vector<SpecSeq*> one{&q};
  if (src_row < 0 || src_row >= c_.max_rows) qlog::fail("spec", "spec_rest: source row %d", src_row);
  kv_ensure(q.slot, q.pos + q.depth + 1);  // the drafts' head positions, before anything changes (KvFull)
  drafts(one, {0, 1}, [&](int) {
    CUDA_CHECK(cudaMemcpyAsync(d_ids_, &q.last, 4, cudaMemcpyHostToDevice, s_));  // head token of the last prompt row
    return mtp_core({q.pos - 1}, {q.slot}, {0, 1}, {src_row}, true, false);
  });
}

void Engine::hold_hidden(int n) {
  if (n < 1 || n > c_.max_rows) qlog::fail("mtp", "hold_hidden %d rows", n);
  CUDA_CHECK(cudaMemcpyAsync(Xh_, X_, size_t(n) * 10240 * sizeof(bf16), cudaMemcpyDeviceToDevice, s_));
}

const float* Engine::mtp_step_held(const std::vector<int>& ids, const std::vector<int>& pos,
                                   const std::vector<int>& slot, const std::vector<int>& seq_start,
                                   const std::vector<int>& src) {
  head_src_ = Xh_;
  const float* lg = mtp_step(ids, pos, slot, seq_start, src, true, false);
  head_src_ = nullptr;
  return lg;
}

std::vector<std::vector<int>> Engine::spec_round(std::vector<SpecSeq*>& q) {
  const int B = int(q.size());
  if (!mtp_ || B == 0) qlog::fail("spec", "spec_round: no MTP head or empty batch");
  // k_b = drafts verified for sequence b (the first min(depth, pending) — the rest are dropped unverified); kd = the
  // largest depth (the next draft chain's length)
  std::vector<int> kb(B);
  int kd = 0;
  for (int b = 0; b < B; b++) {
    auto* x = q[b];
    kb[b] = std::min(x->depth, int(x->drafts.size()));
    kd = std::max(kd, x->depth);
    if (x->depth < 1 || x->depth + 1 > kGdnRecRows || kb[b] < 1)
      qlog::fail("spec", "spec_round: slot %d depth %d · %zu drafts", x->slot, x->depth, x->drafts.size());
  }
  // every position the round can write (verify rows to pos + k_b, the head's drafts to pos + k_b + kd), mapped before
  // anything changes — a KvFull leaves every sequence as it was
  for (int b = 0; b < B; b++) kv_ensure(q[b]->slot, q[b]->pos + kb[b] + kd + 2);
  for (int b = 0; b < B; b++)  // this round writes cache positions up to pos + k_b + kd − 1 (verify rows, then the drafts)
    if (q[b]->pos + kb[b] + kd - 1 >= c_.ctx)
      qlog::fail("spec", "spec_round: slot %d position %d + %d + %d − 1 ≥ window %d", q[b]->slot, q[b]->pos, kb[b], kd, c_.ctx);
  // verify rows per sequence: last, d1 … d_kb at positions pos … pos + k_b
  std::vector<int> ids, pos, slot(B), seq(B + 1, 0), qst(B + 1, 0), dr;
  for (int b = 0; b < B; b++) {
    const int k = kb[b];
    seq[b + 1] = seq[b] + k + 1, qst[b + 1] = qst[b] + k, slot[b] = q[b]->slot;
    ids.push_back(q[b]->last), pos.push_back(q[b]->pos);
    for (int i = 0; i < k; i++) ids.push_back(q[b]->drafts[i]), pos.push_back(q[b]->pos + 1 + i), dr.push_back(q[b]->drafts[i]);
  }
  // profile mode: wall time per phase (stream synced at each boundary)
  auto t_prev = std::chrono::steady_clock::now();
  auto phase = [&](const char* g) {
    if (!prof_on_) return;
    CUDA_CHECK(cudaStreamSynchronize(s_));
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - t_prev).count();
    t_prev = now;
    auto it = std::find_if(spec_ms_.begin(), spec_ms_.end(), [&](auto& x) { return x.first == g; });
    if (it == spec_ms_.end()) spec_ms_.emplace_back(g, ms);
    else it->second += ms;
  };
  // P019: the verify leaves candidate records when every sequence's sampling allows it and its drafts kept records
  bool sp = sparse_seqs(q);
  for (auto* x : q) sp = sp && (!(x->temp > 0.f) || x->q_sparse);
  // row layout before the verify: the penalties of its target rows read it in the lm_head (and the accept after)
  CUDA_CHECK(cudaMemcpyAsync(d_pst_, seq.data(), (B + 1) * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_qst_, qst.data(), (B + 1) * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_drafts_, dr.data(), dr.size() * 4, cudaMemcpyHostToDevice, s_));
  const bool pen = pen_upload(q);
  if (pen) {  // each sequence's `last` counts from now on (as the caller left it — a replaced one counts as replaced)
    std::vector<int> lasts(B);
    for (int b = 0; b < B; b++) lasts[b] = q[b]->last;
    CUDA_CHECK(cudaMemcpyAsync(d_last_, lasts.data(), B * 4, cudaMemcpyHostToDevice, s_));
    penalty_add(d_pen_, B, nullptr, nullptr, nullptr, d_last_, kVocab, s_);
    pen_B_ = B;
  }
  sparse_ = sp;
  verify(ids, pos, slot, seq);
  sparse_ = false;
  pen_B_ = 0;
  phase("verify");
  // accept on the GPU (both ranks hold the same logits and keys → the same decisions)
  std::vector<uint64_t> ctr(B);
  for (int b = 0; b < B; b++) ctr[b] = uint64_t(q[b]->pos);
  sample_keys(q, kSeedAcc, ctr, d_temp_, d_topk_, d_topp_, d_seed_, d_ctr_, s_, d_minp_);
  std::vector<int> na(B), nx(B);
  auto read_acc = [&] {
    CUDA_CHECK(cudaMemcpyAsync(na.data(), d_nacc_, B * 4, cudaMemcpyDeviceToHost, s_));
    CUDA_CHECK(cudaMemcpyAsync(nx.data(), d_next_, B * 4, cudaMemcpyDeviceToHost, s_));
    CUDA_CHECK(cudaStreamSynchronize(s_));
  };
  auto dense_accept = [&] {  // q rows of this round in batch order (sampling sequences only; greedy ignores q)
    q_full_rows(q, kb, qst);
    spec_accept(lg_, d_pst_, qbuf_, d_qst_, d_drafts_, B, kVocab, d_temp_, d_topk_, d_topp_, d_minp_, nullptr, d_seed_,
                d_ctr_, d_nacc_, d_next_, s_);
    read_acc();
  };
  if (sp) {  // P019: records of the verify rows and of the drafts (gathered in q_start order)
    const int Q = qst[B];
    for (int b = 0; b < B; b++)
      if (q[b]->temp > 0.f)
        for (int r = 0; r < 2; r++)
          CUDA_CHECK(cudaMemcpy2DAsync(qcbuf_ + (size_t(r) * Q + qst[b]) * kSparseRec, kSparseRec * sizeof(SparseCand),
                                       slots_[q[b]->slot]->dqc + size_t(r) * kSparseRec, 2 * kSparseRec * sizeof(SparseCand),
                                       kSparseRec * sizeof(SparseCand), kb[b], cudaMemcpyDeviceToDevice, s_));
    spec_accept_sparse(cand_recv_, seq[B], d_pst_, qcbuf_, std::max(Q, 1), d_qst_, d_drafts_, B, kVocab, d_temp_,
                       d_topk_, d_topp_, d_minp_, nullptr, d_seed_, d_ctr_, d_nacc_, d_next_, s_);
    read_acc();
    if (std::find(na.begin(), na.end(), kSparseDense) != na.end()) {  // a NaN / inf row: the full rows, same keys
      gather_full(seq[B]);
      dense_accept();
    }
  } else {
    dense_accept();
  }
  if (pen) penalty_add(d_pen_, B, d_drafts_, d_qst_, d_nacc_, nullptr, kVocab, s_);  // the accepted drafts
  phase("accept");
  std::vector<int> keep(B);
  for (int b = 0; b < B; b++) keep[b] = na[b] + 1;
  commit(keep);
  phase("commit");
  // emitted tokens; the head over the kept rows (verify rows seq[b] … seq[b] + na[b], token = the one after each)
  std::vector<std::vector<int>> out(B);
  std::vector<int> hid, hpos, hsrc, fss(B + 1, 0);
  for (int b = 0; b < B; b++) {
    out[b].assign(q[b]->drafts.begin(), q[b]->drafts.begin() + na[b]);
    out[b].push_back(nx[b]);
    for (int j = 0; j <= na[b]; j++) hid.push_back(out[b][j]), hpos.push_back(q[b]->pos + j), hsrc.push_back(seq[b] + j);
    fss[b + 1] = int(hid.size());
    q[b]->pos += na[b] + 1, q[b]->last = nx[b];
  }
  drafts(q, fss, [&](int) {
    CUDA_CHECK(cudaMemcpyAsync(d_ids_, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice, s_));
    return mtp_core(hpos, slot, fss, hsrc, true, false);
  });
  phase("drafts");
  if (prof_on_) spec_rounds_++;
  return out;
}

// ---- penalties ------------------------------------------------------------------------------------------------------
bool Engine::pen_upload(const std::vector<SpecSeq*>& q) {
  bool any = false;
  std::vector<PenSeq> v(q.size());
  for (size_t b = 0; b < q.size(); b++) {
    const auto* x = q[b];
    const bool on = x->pen() && slots_[x->slot]->pen.cnt;
    v[b] = PenSeq{on ? slots_[x->slot]->pen : PenSeen{}, x->rep, x->freq, x->pres};
    any |= on;
  }
  if (any) CUDA_CHECK(cudaMemcpyAsync(d_pen_, v.data(), v.size() * sizeof(PenSeq), cudaMemcpyHostToDevice, s_));
  return any;
}

static void put_tokens(int*& buf, size_t& cap, const std::vector<int>& t, cudaStream_t s) {
  if (t.size() > cap) {
    if (buf) CUDA_CHECK(cudaFree(buf));
    cap = std::max<size_t>(t.size(), 4096);
    CUDA_CHECK(cudaMalloc(&buf, cap * sizeof(int)));
  }
  if (!t.empty()) CUDA_CHECK(cudaMemcpyAsync(buf, t.data(), t.size() * sizeof(int), cudaMemcpyHostToDevice, s));
}

void Engine::pen_begin(int slot, const std::vector<int>& prompt) {
  auto& st = slots_[slot]->pen;
  if (!st.cnt) st.cnt = dalloc<uint32_t>(kVocab), st.ids = dalloc<int>(kVocab), st.n = dalloc<int>(1);
  penalty_reset(st, kVocab, s_);
  put_tokens(d_ptok_, ptok_cap_, prompt, s_);
  if (!prompt.empty()) penalty_prompt(st, d_ptok_, int(prompt.size()), kVocab, s_);
  CUDA_CHECK(cudaStreamSynchronize(s_));  // the host vector may go before the copy otherwise
}

void Engine::pen_add(int slot, const std::vector<int>& toks) {
  auto& st = slots_[slot]->pen;
  if (!st.cnt || toks.empty()) return;
  put_tokens(d_ptok_, ptok_cap_, toks, s_);
  const PenSeq one{st, 1.f, 0.f, 0.f};
  const int ts[2] = {0, int(toks.size())};
  CUDA_CHECK(cudaMemcpyAsync(d_pen_, &one, sizeof one, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_last_, ts, sizeof ts, cudaMemcpyHostToDevice, s_));
  penalty_add(d_pen_, 1, d_ptok_, d_last_, nullptr, nullptr, kVocab, s_);
  CUDA_CHECK(cudaStreamSynchronize(s_));
}

}  // namespace q38
