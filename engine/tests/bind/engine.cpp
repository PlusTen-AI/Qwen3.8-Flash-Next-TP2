// Test binding for the whole engine (engine/src/model.cc): one TP2 rank per process (run through tools/tp2.sh).
#include <torch/extension.h>

#include <map>
#include <memory>

#include "core/log.h"
#include "model/model.h"
#include "model/patches.h"

namespace {
std::unique_ptr<q38::Engine> E;

void init(const std::string& model_dir, const std::string& table, int rank, const std::string& head_ip, int slots,
          int ctx, int max_rows, bool mtp, int port, double yarn_factor, int yarn_ramp, int checkpoints, bool kv_pool) {
  q38::EngineCfg c;
  c.model_dir = model_dir, c.table = table, c.rank = rank, c.head_ip = head_ip, c.port = port, c.slots = slots, c.ctx = ctx,
  c.max_rows = max_rows, c.logit_rows = std::min(max_rows, 64), c.mtp = mtp;
  c.yarn_factor = yarn_factor, c.yarn_ramp = yarn_ramp, c.checkpoints = checkpoints, c.kv_pool = kv_pool;
  E = std::make_unique<q38::Engine>(c);
}

// returns fp32 logits [R, 248320] (device tensor copy): R = M, or B (last row per sequence) when all_logits = false
torch::Tensor step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                   const std::vector<int>& seq_start, bool all_logits, int snap_row,
                   const std::vector<std::pair<int, int>>& snaps) {
  const float* lg = E->step(ids, pos, slot, seq_start, all_logits, snap_row, snaps);
  auto out = torch::empty({int64_t(E->logit_rows_last()), q38::kVocab}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  CUDA_CHECK(cudaMemcpy(out.data_ptr(), lg, out.numel() * 4, cudaMemcpyDeviceToDevice));
  return out;
}

// MTP draft head pass (Engine::mtp_step); logits as step()
torch::Tensor mtp_step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                       const std::vector<int>& seq_start, const std::vector<int>& src, bool from_target, bool all_logits) {
  const float* lg = E->mtp_step(ids, pos, slot, seq_start, src, from_target, all_logits);
  auto out = torch::empty({int64_t(E->logit_rows_last()), q38::kVocab}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  CUDA_CHECK(cudaMemcpy(out.data_ptr(), lg, out.numel() * 4, cudaMemcpyDeviceToDevice));
  return out;
}

torch::Tensor verify(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                     const std::vector<int>& seq_start) {
  const float* lg = E->verify(ids, pos, slot, seq_start);
  auto out = torch::empty({int64_t(E->logit_rows_last()), q38::kVocab}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  CUDA_CHECK(cudaMemcpy(out.data_ptr(), lg, out.numel() * 4, cudaMemcpyDeviceToDevice));
  return out;
}

std::map<int, torch::Tensor> MEDIA;  // rank 0: media embeddings per slot (Engine::media_src)

// speculative generation through Engine::spec_begin / spec_round; sequences kept here by slot
std::map<int, q38::SpecSeq> SQ;
// min_p and the penalties (rep / freq / pres) apply to the target rows; the penalties need pen_begin first
static void sampling(q38::SpecSeq& q, int top_k, double top_p, double min_p, double rep, double freq, double pres) {
  q.top_k = top_k, q.top_p = float(top_p), q.min_p = float(min_p);
  q.rep = float(rep), q.freq = float(freq), q.pres = float(pres);
}
int spec_begin(int slot, int pos, int depth, double temp, int64_t seed, int row0, int lrow, const std::vector<int>& next_ids,
               int top_k, double top_p, double min_p, double rep, double freq, double pres) {
  auto& q = SQ[slot];
  q = q38::SpecSeq{};
  q.slot = slot, q.pos = pos, q.depth = depth, q.temp = float(temp), q.seed = uint64_t(seed);
  sampling(q, top_k, top_p, min_p, rep, freq, pres);
  return E->spec_begin(q, row0, lrow, next_ids);
}
int spec_first(int slot, int pos, int depth, double temp, int64_t seed, int lrow, int top_k, double top_p, double min_p,
               double rep, double freq, double pres) {
  auto& q = SQ[slot];
  q = q38::SpecSeq{};
  q.slot = slot, q.pos = pos, q.depth = depth, q.temp = float(temp), q.seed = uint64_t(seed);
  sampling(q, top_k, top_p, min_p, rep, freq, pres);
  return E->spec_first(q, lrow);
}
std::vector<std::vector<int>> spec_round(const std::vector<int>& slots) {
  std::vector<q38::SpecSeq*> v;
  for (int s : slots) v.push_back(&SQ.at(s));
  return E->spec_round(v);
}

torch::Tensor tap(const std::string& name, int64_t rows, int64_t cols) {
  const q38::bf16* p = E->tap(name);
  if (!p) return torch::Tensor();
  auto out = torch::empty({rows, cols}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
  CUDA_CHECK(cudaMemcpy(out.data_ptr(), p, out.numel() * 2, cudaMemcpyDeviceToDevice));
  return out;
}
// Shutdown: wait for the GPU, then destroy the engine — RDMA all-reduce (proxy thread, QPs, MRs, pinned staging) and
// the NCCL communicator. Device memory goes with the process. Both ranks call it after their last engine call.
void close_engine() {
  if (!E) return;
  CUDA_CHECK(cudaDeviceSynchronize());
  E.reset();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Engine calls release the GIL (the server's HTTP thread keeps serving while a step runs — and answers 503 while
  // init loads the weights); arguments are converted before and the result after, with the GIL held. Only one thread
  // may drive the engine.
  using nogil = py::call_guard<py::gil_scoped_release>;
  m.def("init", &init, py::arg("model_dir"), py::arg("table"), py::arg("rank"), py::arg("head_ip"), py::arg("slots"),
        py::arg("ctx"), py::arg("max_rows"), py::arg("mtp") = false, py::arg("port") = 29611,
        py::arg("yarn_factor") = 1.5625, py::arg("yarn_ramp") = 262144, py::arg("checkpoints") = 0,
        py::arg("kv_pool") = false, nogil());
  m.def("step", &step, py::arg("ids"), py::arg("pos"), py::arg("slot"), py::arg("seq_start"), py::arg("all_logits") = true,
        py::arg("snap_row") = -1, py::arg("snaps") = std::vector<std::pair<int, int>>{}, nogil());
  m.def("mtp_step", &mtp_step, py::arg("ids"), py::arg("pos"), py::arg("slot"), py::arg("seq_start"), py::arg("src"),
        py::arg("from_target"), py::arg("all_logits") = false, nogil());
  m.def("verify", &verify, nogil());
  m.def("close", &close_engine, nogil());
  m.def("spec_begin", &spec_begin, py::arg("slot"), py::arg("pos"), py::arg("depth"), py::arg("temp"), py::arg("seed"),
        py::arg("row0"), py::arg("lrow"), py::arg("next_ids"), py::arg("top_k") = 0, py::arg("top_p") = 1.0,
        py::arg("min_p") = 0.0, py::arg("rep") = 1.0, py::arg("freq") = 0.0, py::arg("pres") = 0.0, nogil());
  m.def("spec_round", &spec_round, nogil());
  m.def("spec_first", &spec_first, py::arg("slot"), py::arg("pos"), py::arg("depth"), py::arg("temp"), py::arg("seed"),
        py::arg("lrow"), py::arg("top_k") = 0, py::arg("top_p") = 1.0, py::arg("min_p") = 0.0, py::arg("rep") = 1.0,
        py::arg("freq") = 0.0, py::arg("pres") = 0.0, nogil());
  m.def("pen_begin", [](int s, const std::vector<int>& prompt) { E->pen_begin(s, prompt); }, nogil());
  m.def("pen_add", [](int s, const std::vector<int>& toks) { E->pen_add(s, toks); }, nogil());
  m.def("spec_rest", [](int s, int src_row) { E->spec_rest(SQ.at(s), src_row); }, py::arg("slot"), py::arg("src_row") = 0,
        nogil());
  m.def("hold_hidden", [](int n) { E->hold_hidden(n); }, nogil());
  m.def("mtp_step_held", [](const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                            const std::vector<int>& seq_start, const std::vector<int>& src) {
    const float* lg = E->mtp_step_held(ids, pos, slot, seq_start, src);
    auto out = torch::empty({int64_t(E->logit_rows_last()), q38::kVocab}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    CUDA_CHECK(cudaMemcpy(out.data_ptr(), lg, out.numel() * 4, cudaMemcpyDeviceToDevice));
    return out;
  }, nogil());
  m.def("spec_drafts", [](int s) { return SQ.at(s).drafts; });
  // depth of the slot's next round (verifies min(depth, pending drafts), leaves `depth` new drafts; SpecSeq in model.h)
  m.def("spec_depth", [](int s, int depth) { SQ.at(s).depth = depth; }, py::arg("slot"), py::arg("depth"));
  // replace the slot's next input token (the last emitted one, not yet through the target) — the server's thinking
  // budget / loop guard put </think> there; the pending drafts stay (valid proposals for the exact accept test)
  m.def("spec_last", [](int s, int tok) { SQ.at(s).last = tok; }, py::arg("slot"), py::arg("token"));
  m.def("commit", [](const std::vector<int>& n) { E->commit(n); }, nogil());
  // dropping a slot's media embeddings: the engine stream first finishes every call that may read them
  auto drop_media = [](int s) {
    if (MEDIA.count(s)) {
      CUDA_CHECK(cudaStreamSynchronize(E->stream()));
      MEDIA.erase(s);
    }
  };
  m.def("reset", [drop_media](int s) { drop_media(s); E->reset(s); }, nogil());
  // images / videos (model.h): mRoPE positions [n·3] + delta; media positions (both ranks); embeddings (rank 0, kept
  // alive here until the slot's next set_media or reset)
  m.def("set_mrope", [](int s, const std::vector<int>& pos3, int delta) { E->set_mrope(s, pos3, delta); }, nogil());
  m.def("set_media", [drop_media](int s, const std::vector<int>& rows) { drop_media(s); E->set_media(s, rows); },
        nogil());
  m.def("media_src", [](int s, torch::Tensor emb) {
    TORCH_CHECK(emb.is_cuda() && emb.scalar_type() == torch::kBFloat16 && emb.dim() == 2 && emb.size(1) == q38::kHidden &&
                emb.is_contiguous(), "media embeddings: contiguous CUDA bf16 [K, 2560]");
    MEDIA[s] = emb;
    E->media_src(s, reinterpret_cast<const q38::bf16*>(emb.data_ptr()), int(emb.size(0)));
  });
  m.def("snapshot", [](int s) { E->snapshot(s); }, nogil());
  m.def("restore", [](int s) { return E->restore(s); }, nogil());
  // prefix checkpoints (model.h): buffers 0 … slots−1 the slots' own, then the pool (init's `checkpoints`)
  m.def("ckpt_save", [](int s, int id) { E->ckpt_save(s, id); }, py::arg("slot"), py::arg("id"), nogil());
  m.def("ckpt_load", [](int s, int id) { return E->ckpt_load(s, id); }, py::arg("slot"), py::arg("id"), nogil());
  m.def("ckpt_share", [](int s, int id) { return E->ckpt_share(s, id); }, py::arg("slot"), py::arg("id"), nogil());
  m.def("ckpt_drop", [](int id) { E->ckpt_drop(id); }, py::arg("id"), nogil());
  m.def("ckpt_buffers", [] { return E->ckpt_buffers(); });
  m.def("ckpt_chunks", [] { return E->ckpt_chunks(); });
  // shared KV pool (init's kv_pool = true): take it (both ranks), then chunk accounting for admission
  m.def("kv_pool_init", [](double reserve_gib, double rank0_gib, double max_gib) {
    return E->kv_pool_init(reserve_gib, rank0_gib, max_gib); }, nogil());
  m.def("kv_free", [] { return E->kv_free(); });
  m.def("kv_total", [] { return E->kv_total(); });
  m.def("kv_held", [](int s) { return E->kv_held(s); });
  m.def("kv_need", [](int tokens) { return E->kv_need(tokens); });
  m.def("kv_sole", [](int s) { return E->kv_sole(s); });
  m.def("kv_full", [](int tokens) { return E->kv_full(tokens); });
  m.def("history", [](int s) { return E->history(s); });
  m.def("set_taps", [](bool on) { E->set_taps(on); });
  m.def("tap", &tap);
  m.def("set_prof", [](bool on) { E->set_prof(on); });
  m.def("patch", [](const std::string& name, bool on) { q38::patch::set(name, on); });
  m.def("patch_on", [](const std::string& name) { return q38::patch::on(name); });
  m.def("prof_report", [](bool reset) { return E->prof_report(reset); }, py::arg("reset") = true);
  m.def("spec_report", [](bool reset) { return E->spec_report(reset); }, py::arg("reset") = true);
}
