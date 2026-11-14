// model.h — one decode step of Qwen3.8-Flash-Next on this rank (TP=2, plan P0), kernels from src/kernels/*.
// A step takes M = Σ_b T_b rows: rows of sequence b are consecutive (seq_start), each sequence owns a state slot
// (GDN/QSA/PLE caches) and advances by its T_b tokens (all kept; MTP rollback comes in a later bundle).
// MTP draft head (bundle 3, model/mtp.cc): one extra QSA layer fed with the target's multi-stream hidden state.
// Output: fp32 logits [M, 248320] (lm_head fp32 accumulate, no bf16 rounding) gathered on both ranks.
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "core/comm.h"
#include "core/kv_pool.h"
#include "kernels/comm/rdma_ar.h"
#include "core/common.h"
#include "kernels/gdn/gdn.h"
#include "kernels/hc/hc.h"
#include "kernels/linear/linear.h"
#include "kernels/moe/moe.h"
#include "kernels/qsa/qsa.h"
#include "kernels/sample/sample.h"
#include "core/weights.h"

namespace q38 {

// One sequence in speculative generation (MTP). pos = position of `last` (emitted, not yet through the target);
// drafts = the head's proposals for positions pos + 1 … (made by spec_begin / the previous spec_round).
// depth (1 … 15) may change between rounds: a round verifies the first min(depth, drafts) pending drafts (the rest are
// dropped unverified) and leaves `depth` new drafts. Sequences of one round may have different depths; the draft chain
// runs to the largest and each sequence keeps its own depth's worth, so a sequence's tokens for a given seed and depth
// schedule do not depend on the batch around it.
struct SpecSeq {
  int slot = 0, pos = 0, last = 0, depth = 4;  // depth: drafts per round (≤ 15)
  float temp = 0.f;                            // 0 = greedy
  int top_k = 0;                               // ≤ 0 = off (kernels/sample: truncation after temperature)
  float top_p = 1.f;                           // ≥ 1 = off
  float min_p = 0.f;                           // ≤ 0 = off (target rows only; drafts are drawn without it)
  float rep = 1.f, freq = 0.f, pres = 0.f;     // vLLM penalties (target rows only); state: Engine::pen_begin
  bool pen() const { return rep != 1.f || freq != 0.f || pres != 0.f; }
  uint64_t seed = 0;
  std::vector<int> drafts;
  bool q_sparse = false;  // P019: the pending drafts' q is kept as candidate records + this rank's half (not full rows)
};

// The shared KV pool cannot map what a call needs (the server admits requests by the pool, so only when its
// accounting was wrong). Thrown before the call changes any state; both ranks throw at the same call (their pools
// change in step). The server ends that slot's request and goes on.
struct KvFull : std::runtime_error {
  int slot;
  KvFull(int s, const std::string& m) : std::runtime_error(m), slot(s) {}
};

struct EngineCfg {
  std::string model_dir, table;
  int rank = 0;
  std::string head_ip;  // the head's CX7 address (NCCL bootstrap; the RDMA all-reduce uses port + 1)
  int port = 29611;
  int slots = 4;        // sequences with resident state
  int ctx = 32768;      // tokens per slot (QSA caches)
  int max_rows = 64;    // M per step (prefill chunks: up to 8192)
  int logit_rows = 64;  // rows that can get logits in one step (decode / verify); prefill returns the last row only
  bool mtp = true;      // load the MTP draft head (≈0.8 GB per rank, its own QSA cache per slot)
  // QSA RoPE (kernels/qsa/qsa.h QsaRope, table rows = ctx ≤ kQsaMaxPos): YaRN factor (≤ 1: plain RoPE) and the length
  // its correction range comes from (262,144: the checkpoint's original_max_position_embeddings, as HF; vLLM's
  // MRotaryEmbedding: max(262,144, window)).
  double yarn_factor = 1.5625;
  int yarn_ramp = 262144;
  int checkpoints = 0;  // prefix checkpoint buffers for the server, allocated at start (≈ 55 MiB each)
  bool kv_pool = false; // the QSA / MTP caches come from kv_pool_init's shared pool instead of ctx tokens per slot
};

class Engine {
 public:
  explicit Engine(const EngineCfg& c);
  ~Engine();
  // ids/pos [M], slot [B], seq_start [B+1] (host). logits: device fp32 [R, 248320] (valid until the next step),
  // R = M when all_logits (M ≤ logit_rows), else R = B: the last row of each sequence (prefill chunks).
  // snap_row ≥ 0 (one sequence, not a verify step): the step also takes the slot's prefix snapshot (as snapshot()) of
  // the state just before its row snap_row — the GDN and PLE calls are split there (same values: their rows give the
  // same bits however a sequence's rows are split over calls); the snapshot's token count is pos[snap_row].
  // snaps: checkpoints taken inside the step, (row, buffer id) — the state just before that row of its (single)
  // sequence, as ckpt_save there would; token count pos[row]. snap_row ≥ 0 is (snap_row, the slot's own buffer).
  const float* step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                    const std::vector<int>& seq_start, bool all_logits = true, int snap_row = -1,
                    const std::vector<std::pair<int, int>>& snaps = {});
  int logit_rows_last() const { return last_rows_; }
  // MTP draft head, M rows (rows of one sequence consecutive, as in step). Row j: the multi-stream hidden of position
  // pos[j] — row src[j] of the last target step (from_target) or of the last mtp_step (the head's own output; after a
  // pass over several rows of a sequence, the next draft continues from that sequence's LAST row) — and
  // ids[j] = the token at pos[j] + 1. Writes the head's KV at pos[j] (own QSA cache per slot). Logits as in step():
  // all rows, or the last row of each sequence. Must follow the step whose rows it reads (buffers are reused).
  // Speculative verify (MTP): step() with all logits and ≤ kGdnRecRows (16) rows per sequence, but the GDN and PLE
  // states only record the rows; commit(n) then keeps the first n[b] rows of each sequence of that step (GDN replay,
  // PLE window, token history; QSA/MTP caches are positional and get overwritten). verify + commit(n) leaves every
  // state bitwise as step() over those n rows would, so speculative decoding reproduces plain decoding exactly.
  const float* verify(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                      const std::vector<int>& seq_start);
  void commit(const std::vector<int>& n);
  // Speculative generation (exact for any temperature: speculative sampling, kernels/sample). spec_begin: right after
  // the step that ran the sequence's last prompt chunk as rows row0 … row0 + n − 1 (all_logits = false; the logits
  // row of this sequence is `lrow`), with next_ids = the chunk's prompt tokens 1 … n − 1: samples the first token, runs
  // the head over the chunk (its KV) and drafts. Returns the first token. spec_round: one round for B sequences —
  // verify, accept, commit, head over the kept rows, new drafts. Returns the tokens emitted per sequence (1 … depth+1).
  int spec_begin(SpecSeq& q, int row0, int lrow, const std::vector<int>& next_ids);
  // spec_begin in two parts, so a server can emit the first token before the head runs (time to first token): the
  // prompt's last row ran in the last target step (its logits row lrow, its hidden row src_row of X_). spec_first
  // samples the first token (as spec_begin); spec_rest — after any head passes over earlier rows (mtp_step from the
  // target, or mtp_step_held) — runs the head over the last row and drafts. spec_first + spec_rest ==
  // spec_begin(q, src_row, lrow, {}) bit for bit.
  int spec_first(SpecSeq& q, int lrow);
  void spec_rest(SpecSeq& q, int src_row = 0);
  // Keep rows 0 … n−1 of the last target step's multi-stream state (X_) for a later head pass (mtp_step_held reads
  // them instead of X_; other steps may run in between).
  void hold_hidden(int n);
  const float* mtp_step_held(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                             const std::vector<int>& seq_start, const std::vector<int>& src);
  std::vector<std::vector<int>> spec_round(std::vector<SpecSeq*>& seqs);
  const float* mtp_step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                        const std::vector<int>& seq_start, const std::vector<int>& src, bool from_target,
                        bool all_logits = false);
  void reset(int slot);                       // new sequence in this slot (zero states, clear history, media)
  // Images / videos. set_mrope: the slot's RoPE positions (mRoPE) — pos3 [n, 3] = (t, h, w) of positions 0 … n − 1 —
  // and delta: a position p ≥ n rotates by p + delta in all three. Every later step, verify and head pass of the slot
  // rotates by these (QSA layers, indexer keys, MTP head); without a call (or after reset) the rotation is the position.
  void set_mrope(int slot, const std::vector<int>& pos3, int delta);
  // set_media (both ranks): the positions of the slot's media tokens (ascending; {} clears). A step row at one of
  // these positions takes a media embedding instead of its token's: rank 0 writes it into its partial embedding row and
  // rank 1 zeros the row, so the all-reduce gives it exactly. The PLE n-gram branch still reads the token ids (vLLM).
  // The MTP head's token branch embeds the placeholder tokens as tokens (drafts only; the media embeddings there gained
  // nothing).
  // media_src (rank 0): the embeddings, device bf16 [K, 2560], row k for rows[k]. The caller keeps them alive and
  // finished (the engine stream does not wait for other streams) until the slot's prompt has run or set_media(slot, {}).
  void set_media(int slot, const std::vector<int>& rows);
  void media_src(int slot, const bf16* emb, int rows);  // rows: K (checked against set_media)
  // Penalties (SpecSeq rep / freq / pres): pen_begin gives the slot's sequence its seen-token state — the prompt's
  // tokens, no generated ones — before its first spec_first / spec_begin; the engine then counts every emitted token
  // (a round adds its `last` token before the verify and the accepted drafts after the accept, so a `last` the caller
  // replaces counts as replaced). pen_add counts tokens the caller ran itself (e.g. forced ones through step()).
  void pen_begin(int slot, const std::vector<int>& prompt);
  void pen_add(int slot, const std::vector<int>& toks);
  // Prefix checkpoints (server prefix reuse). A checkpoint is a copy of a slot's recurrent state (GDN S and conv of
  // the 36 GDN layers, PLE window) at some token count. Buffers 0 … slots−1 are the slots' own (snapshot / restore),
  // slots … slots + checkpoints − 1 a pool the caller assigns. ckpt_save: the state at the slot's history end into
  // buffer id; step(..., snaps) takes them before given rows of a prefill step; ckpt_load returns the slot to one
  // (history truncated) and gives its token count, −1 if the buffer is empty or holds another slot's state; reset()
  // empties the slot's buffers. The QSA / MTP caches are positional: the caller must not load a checkpoint after the
  // slot rewrote positions below it (taking another branch from an earlier point voids the later checkpoints).
  // With the shared KV pool the pool's buffers are address ranges whose chunks are mapped when a checkpoint goes in
  // (ckpt_save, step snaps: KvFull before anything changes if the pool has too few) and given back by ckpt_drop and by
  // reset of the slot that holds it: free pool memory is the checkpoints' room; ckpt_chunks() = chunks one takes.
  // A load that takes the slot back below positions whose caches it shares with another slot (ckpt_share) first gives
  // it its own copy of the chunk it goes on writing in (KvFull before anything changes if the pool has none).
  void ckpt_save(int slot, int id);
  int ckpt_load(int slot, int id);
  // Another slot's checkpoint (shared KV pool only; a prefix cache shared by all requests): slot `slot`, emptied by
  // reset(), continues from checkpoint `id` that another slot holds (its own buffer or a pool buffer). The recurrent
  // state is copied; the QSA / MTP caches below the checkpoint are the holder's chunks mapped again (the same memory,
  // nothing copied; the chunk the checkpoint position falls in is copied); the token history is the holder's first
  // tokens. The holder must still have those positions as they were (the caller's checkpoint bookkeeping, as for
  // ckpt_load), and the MTP head's cache at the position before was written with the token the caller's prompt has
  // there. Then both slots read the shared chunks and write only above them: the state is the one a cold prefill of
  // those tokens leaves, bit for bit. Returns the token count, −1 if the buffer is empty or the slot's own. KvFull
  // (before anything changes) if the pool has too few chunks for the copies.
  int ckpt_share(int slot, int id);
  void ckpt_drop(int id);
  int ckpt_buffers() const { return int(ck_.size()); }
  size_t ckpt_chunks() const;
  void snapshot(int slot) { ckpt_save(slot, slot); }
  // Shared KV pool (EngineCfg::kv_pool). kv_pool_init (both ranks, before the first step) takes the pool now: as many
  // 2 MiB chunks as the smaller MemAvailable of the two nodes allows less reserve_gib (rank 0 also less rank0_gib), at
  // most max_gib (0 = no cap); every slot's caches get address ranges for ctx tokens. A step maps chunks (zeroed) as a
  // slot's positions grow; reset() gives them back. Returns the chunk count. For admission: kv_need(n) = chunks one
  // slot holds for n tokens; kv_free / kv_total / kv_held(slot) in chunks. Without the pool these return 0.
  size_t kv_pool_init(double reserve_gib, double rank0_gib, double max_gib);
  size_t kv_free() const { return pool_ ? pool_->free_chunks() : 0; }
  size_t kv_total() const { return pool_ ? pool_->total_chunks() : 0; }
  size_t kv_held(int slot) const;
  size_t kv_need(int tokens) const;
  size_t kv_sole(int slot) const;    // chunks of the slot no other slot maps (what reset() gives back)
  size_t kv_full(int tokens) const;  // chunks wholly below a count (what ckpt_share at that count maps, not new memory)
  int restore(int slot) { return ckpt_load(slot, slot); }
  int history(int slot) const { return int(hist_[slot].size()); }
  // Debug taps (tests, only for steps with M ≤ logit_rows): device pointer of a named buffer after the last step.
  // "X.<i>" materialised multi-stream state after layer i, "u.<i>", "v.<i>", "attn.<i>", "moe.<i>", "y".
  const bf16* tap(const std::string& name) const;
  void set_taps(bool on) { taps_on_ = on; }
  // Profile mode: CUDA events between op groups; per-group ms accumulated over steps (prof_report → one line).
  void set_prof(bool on);
  std::string prof_report(bool reset = true);
  // Profile mode, spec_round: wall ms per phase (verify · accept · commit · head over kept rows · draft chain)
  std::string spec_report(bool reset = true);
  cudaStream_t stream() const { return s_; }

 private:
  struct Slot;
  struct Layer;
  struct Mtp;
  std::unique_ptr<Mtp> mtp_;
  bf16 *Xa_ = nullptr, *Xb_ = nullptr;  // MTP: gathered/normed hidden rows, head state (multi-stream) [max_rows, 10240]
  bf16* Xh_ = nullptr;                  // hold_hidden: target rows kept for a deferred head pass [max_rows, 10240]
  const bf16* head_src_ = nullptr;      // gather source override of the next mtp_core (mtp_step_held)
  std::vector<int> head_last_ = std::vector<int>(64);  // after mtp_core: row of Xb_ with each sequence's hidden (P023)
  int* d_src_ = nullptr;
  void mtp_upload();
  // the head over M rows whose token ids are already in d_ids_ (mtp_step copies them; the draft chain samples into it)
  const float* mtp_core(const std::vector<int>& pos, const std::vector<int>& slot, const std::vector<int>& seq_start,
                        const std::vector<int>& src, bool from_target, bool all_logits);
  // false: a draft came back as kSparseDense (P019, a NaN / inf in a head row) — the caller redoes it with full rows
  bool draft_chain(std::vector<SpecSeq*>& seqs, const std::vector<int>& first_seq_start, const float* lg, bool sparse);
  // the first head pass (first(): uploads its tokens, runs mtp_core) and the draft chain; P019 sparse when it may be
  void drafts(std::vector<SpecSeq*>& seqs, const std::vector<int>& first_seq_start,
              const std::function<const float*(int attempt)>& first);
  // P019 sparse_logits: lm_head passes leave candidate records (cand_recv_) instead of full rows
  bool sparse_ = false;
  SparseCand *cand_send_ = nullptr, *cand_recv_ = nullptr, *qcbuf_ = nullptr;  // [LR] / [2][LR] / [2][LR] records
  bool sparse_seqs(const std::vector<SpecSeq*>& q) const;  // P019 on and every sequence's sampling allows it
  void gather_full(int R);  // full rows of the last sparse lm_head pass into lg_ (its halves are still in lg_half_)
  // qbuf_ rows qst[b] + i = full q rows of the round's sampling sequences (copied, or rebuilt from both ranks' halves)
  void q_full_rows(const std::vector<SpecSeq*>& q, const std::vector<int>& kb, const std::vector<int>& qst);
  void draft_lm_upload();
  float* qbuf_ = nullptr;  // [logit_rows, vocab] q rows of one spec_round, gathered from the slots in batch order
  int *d_nacc_ = nullptr, *d_next_ = nullptr, *d_pst_ = nullptr, *d_qst_ = nullptr, *d_drafts_ = nullptr;
  float* d_temp_ = nullptr;
  int* d_topk_ = nullptr;
  float* d_topp_ = nullptr;
  float* d_minp_ = nullptr;
  PenSeq* d_pen_ = nullptr;           // [logit_rows] the penalised sequences of the current round / first draw
  int *d_last_ = nullptr, *d_p01_ = nullptr, *d_ptok_ = nullptr;  // lasts [logit_rows] · {0, 1} · tokens to count
  size_t ptok_cap_ = 0;
  int pen_B_ = 0;                     // > 0: the next target lm_head penalises its rows (B sequences, d_pst_ …)
  bool pen_upload(const std::vector<SpecSeq*>& q);  // d_pen_ for these sequences; false if none is penalised
  uint64_t *d_seed_ = nullptr, *d_ctr_ = nullptr;
  bool spec_ = false;                                // inside verify(): GDN records instead of advancing
  std::vector<int> spec_slot_, spec_pos0_, spec_rows_;  // the last verify step's sequences (for commit)
  // P020 gdn_fused_commit: commit(n) leaves the GDN rows pending (Slot::gdn_pn); the next verify of the slot applies
  // them inside its GDN kernel (one S read + write per round); any other use of the state flushes them first
  bool spec_p20_ = false;  // the last verify ran with P020
  int* d_npend_ = nullptr; // [max B] pending rows of the verify's sequences
  void gdn_flush_slots(const std::vector<int>& slots);
  GdnState** d_cst_ = nullptr;                       // commit: [36 · slots] states, records, [slots] n
  GdnRec** d_crec_ = nullptr;
  int* d_n_ = nullptr;
  void head_enqueue(int M, int B, const std::vector<int>& seq_start, bool all_logits, bool draft);
  const float* head_finish(int R);
  EngineCfg c_;
  std::unique_ptr<Weights> w_;
  std::unique_ptr<Comm> comm_;
  std::unique_ptr<RdmaAR> rdma_;  // P001: SM-free RDMA all-reduce (bf16 sums); NCCL keeps the logits gather
  void ar(bf16* p, size_t n, cudaStream_t s);
  cudaStream_t s_ = nullptr, cs_ = nullptr;  // compute, all-reduce (P002)
  cudaEvent_t ev_rdy_[2], ev_rdy2_[2], ev_part_[2], ev_bo_[2];
  std::vector<std::unique_ptr<Layer>> L_;
  HcW final_;
  PleW* ple_ = nullptr;
  Lin lm_;
  bf16* embed_ = nullptr;
  std::vector<std::unique_ptr<Slot>> slots_;
  // activations [max_rows, ...]
  bf16 *X_, *bo_, *inj_a_, *inj_m_, *u_, *v_, *t0_, *t1_, *t2_, *part_, *E_, *y_;
  float *lg_half_, *lg_;
  int *d_ids_, *d_pos_, *d_prev_, *d_seq_, *d_seq2_;
  // images / videos (set_mrope, set_media), per slot
  std::vector<std::vector<int>> mr_;     // [n, 3] RoPE positions of the prompt (empty: plain)
  std::vector<int> mdelta_;
  std::vector<std::vector<int>> mrows_;  // media token positions (ascending)
  std::vector<const bf16*> memb_;        // rank 0: their embeddings [K, 2560]
  int* d_rp_ = nullptr;                  // [max_rows, 6] qsa_core rp
  int* d_mrow_ = nullptr;                // [max_rows] step rows that take a media embedding
  const bf16** d_msrc_ = nullptr;        // [max_rows] their sources (rank 0)
  // rp rows for a call (qsa.h qsa_core): null when no sequence of it has RoPE positions; else uploaded to d_rp_
  const int* rope_rows(const std::vector<int>& pos, const std::vector<int>& slot, const std::vector<int>& seq_start);
  // media rows of a target step: (row, source) pairs uploaded; returns their count (the MTP head embeds media
  // placeholders as tokens: giving it the media embeddings changed nothing, 3.27 → 3.24 tokens a round)
  int media_rows(const std::vector<int>& pos, const std::vector<int>& slot, const std::vector<int>& seq_start);
  struct Ckpt {
    void* mem = nullptr;  // per GDN layer S then conv, then the PLE window
    int slot = -1, len = -1;
    int arr = -1;  // shared KV pool: its address range (chunks mapped while it holds a checkpoint); −1: own memory
  };
  void ck_map(int slot, int id);  // map a pool buffer's chunks (KvFull if the pool has too few)
  void ck_unmap(int id);          // give them back (after the stream is idle)
  std::vector<Ckpt> ck_;
  std::unique_ptr<KvPool> pool_;
  struct KvArr { int id, kind; };            // kind 0 K‖V 1,024 B/token · 1 raw indexer keys 256 · 2 compressed 256 per 4
  std::vector<std::vector<KvArr>> kv_arr_;   // per slot: the 13 caches' three arrays
  std::vector<int> kv_cover_;                // per slot: tokens mapped
  // per slot: positions below may lie in chunks shared with another slot (ckpt_share) — a step writes only at or
  // above it; kv_cut lowers it when the slot goes back
  std::vector<int> kv_floor_;
  static constexpr int kKvSlack = 64;        // mapped past the last position a step writes
  void kv_ensure(int slot, int tokens);      // map chunks so that positions < tokens (+ slack) exist, else KvFull
  void kv_cut(int slot, int tokens);         // the slot goes back to `tokens`: no chunk it rewrites stays shared
  int kv_cover_of(int slot) const;           // tokens its mapped chunks cover
  void ck_restore(int slot, const void* mem);  // the recurrent state from a checkpoint buffer
  static constexpr int kMaxSnaps = 8;  // checkpoints inside one step
  int* d_seqs_ = nullptr;              // [2 · (kMaxSnaps + 2)] one-sequence seq_start {0, n} per GDN / PLE piece
  void snap_part(int slot, int id, int gdn_layer);  // checkpoint pieces: one GDN layer's S and conv (−1: PLE window)
  bf16** d_pc_ = nullptr;
  std::vector<std::vector<int>> hist_;  // token history per slot (n-gram predecessors)
  int last_rows_ = 0;
  bool prof_on_ = false;
  std::vector<std::pair<std::string, cudaEvent_t>> marks_;  // events of the current step, in order
  size_t nmark_ = 0;
  std::vector<std::pair<std::string, double>> prof_ms_, spec_ms_;
  int prof_steps_ = 0, spec_rounds_ = 0;
  void mark(const char* group);
  bf16* ylast_ = nullptr;  // [logit_rows, 2560] last rows of y for prefill logits
  static constexpr int kSelCols = 2052;  // qsa_core's sel_out row: 2,051 token positions + count (qsa.h)
  int* d_sel_ = nullptr;   // taps: the QSA selection of the step's rows
  bool taps_on_ = false;
  std::vector<std::pair<std::string, bf16*>> taps_;
  void tap_copy(const std::string& name, const bf16* src, size_t n);
};

}  // namespace q38
