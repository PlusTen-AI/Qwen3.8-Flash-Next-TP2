#include "model/model_impl.h"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>

#include "kernels/mtp/mtp.h"
#include "model/patches.h"

namespace q38 {

Engine::Engine(const EngineCfg& c) : c_(c) {
  auto t0 = std::chrono::steady_clock::now();
  if (c.ctx > kQsaMaxPos)
    qlog::fail("engine", "ctx %d > %d: the QSA layers serve positions below %d (kernels/qsa/qsa.h kQsaMaxPos)", c.ctx,
               kQsaMaxPos, kQsaMaxPos);
  {  // P002's all-reduce stream at the highest priority: its copy/add kernels take SMs as soon as compute CTAs retire
    // (16K chunk: AR stalls 8.3 → 0.5 %). Compute one step below it and above the default (the vision tower runs on a
    // default-priority stream in the same process: decode CTAs go first as its CTAs retire)
    int lo = 0, hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&s_, cudaStreamNonBlocking, hi < lo ? hi + 1 : lo));
  }
  for (auto* e : {ev_rdy_, ev_rdy2_, ev_part_, ev_bo_})
    for (int k = 0; k < 2; k++) CUDA_CHECK(cudaEventCreateWithFlags(&e[k], cudaEventDisableTiming));
  w_ = std::make_unique<Weights>(c.model_dir, c.table, c.rank);
  comm_ = std::make_unique<Comm>(c.rank, c.head_ip, c.port);
  // P001 transport (164 MB pinned staging, one proxy core): set up when the patch is on or Q38_RDMA_INIT=1 (per-step
  // switching in patch_check)
  if (patch::on(patch::rdma_ar) || std::getenv("Q38_RDMA_INIT"))
    rdma_ = std::make_unique<RdmaAR>(c.rank, c.head_ip, c.port + 1, size_t(8192) * kHidden);
  const Weights& W = *w_;
  qsa_rope_set(QsaRope{c.yarn_factor, kQsaOrigPos, c.yarn_ramp, c.ctx});  // RoPE table of every QSA layer + MTP head

  {  // embedding half (vocab rows of this rank)
    auto b = W.slice(kP + "embed_tokens.weight");
    embed_ = dalloc<bf16>(b.size() / 2);
    CUDA_CHECK(cudaMemcpy(embed_, b.data(), b.size(), cudaMemcpyHostToDevice));
  }
  for (int i = 0; i < kLayers; i++) {
    auto l = std::make_unique<Layer>();
    const std::string lp = kP + "layers." + std::to_string(i) + ".";
    l->qsa = is_qsa_layer(i);
    l->ahc = hc_upload(W, lp + "attn_hyper_connection", true);
    l->mhc = hc_upload(W, lp + "mlp_hyper_connection", true);
    if (l->qsa) {
      l->q = qsa_upload(W, i);
      const std::string a = lp + "self_attn.";
      l->q.qkv = lin_upload(W, {a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "indexer.index_qk_proj.weight"});
      l->q.o = lin_upload(W, {a + "o_proj.weight"});
    } else {
      l->gdn = gdn_upload(W, i);
      const std::string a = lp + "linear_attn.";
      l->gdn.qkvz = lin_upload(W, {a + "in_proj_qkv.weight", a + "in_proj_z.weight"});
      l->gdn.ba = lin_upload(W, {a + "in_proj_b.weight", a + "in_proj_a.weight"});
      l->gdn.out = lin_upload(W, {a + "out_proj.weight"});
    }
    l->moe = moe_upload(W, i);
    if (l->qsa) {
      l->d_qc = dalloc<QsaCache>(c.slots);
      l->d_qp = dalloc<QsaCache*>(c.max_rows);
    } else {
      l->d_gst = dalloc<GdnState>(c.slots);
      l->d_gp = dalloc<GdnState*>(c.max_rows);
    }
    L_.push_back(std::move(l));
  }
  qsa_reserve(c.max_rows);  // QSA workspace (one per process) before any graph capture
  final_ = hc_upload(W, kP + "hyper_connection_mixer", false);
  ple_ = ple_upload(W, c.rank);
  lm_ = lin_upload(W, {"lm_head.weight"}, true);  // + P013 12-bit copy (decode passes, patch lin_pack12; same bits)

  // state slots
  for (int s = 0; s < c.slots; s++) {
    auto sl = std::make_unique<Slot>();
    for (int i = 0; i < kLayers; i++) {
      if (L_[i]->qsa) {
        QsaCache qc;  // with the shared pool: address ranges from kv_pool_init
        qc.cap = c.ctx;
        if (!c.kv_pool) {
          qc.kv = dalloc<bf16>(size_t(c.ctx) * 512);
          qc.ik_raw = dalloc<bf16>(size_t(c.ctx) * 128);
          qc.ik_comp = dalloc<bf16>(size_t(c.ctx / 4) * 128);
        }
        CUDA_CHECK(cudaMemcpy(L_[i]->d_qc + s, &qc, sizeof qc, cudaMemcpyHostToDevice));
        sl->q.push_back(qc);
      } else {
        GdnState g;
        g.S = dalloc<float>(size_t(24) * 128 * 128);
        g.conv = dalloc<bf16>(size_t(3) * 5120);
        CUDA_CHECK(cudaMemcpy(L_[i]->d_gst + s, &g, sizeof g, cudaMemcpyHostToDevice));
        sl->g.push_back(g);
      }
    }
    sl->ple_conv = dalloc<bf16>(size_t(9) * 10240);
    slots_.push_back(std::move(sl));
  }
  hist_.resize(c.slots);
  mr_.resize(c.slots), mdelta_.assign(c.slots, 0), mrows_.resize(c.slots), memb_.assign(c.slots, nullptr);
  // checkpoint buffers: the slots' own (allocated on first use), then the server's pool — static caches: allocated now
  // (part of the budget checked at start); shared KV pool: address ranges from kv_pool_init, mapped as they are used
  ck_.resize(c.slots + std::max(0, c.checkpoints));
  if (!c.kv_pool)
    for (int i = c.slots; i < int(ck_.size()); i++) CUDA_CHECK(cudaMalloc(&ck_[i].mem, kCkptBytes));

  const size_t R = c.max_rows;
  X_ = dalloc<bf16>(R * 10240);
  bo_ = dalloc<bf16>(R * kHidden);
  inj_a_ = dalloc<bf16>(R * 4);
  inj_m_ = dalloc<bf16>(R * 4);
  u_ = dalloc<bf16>(R * kHidden);
  v_ = dalloc<bf16>(R * kHidden);
  t0_ = dalloc<bf16>(R * 8192);   // qkvz / qkv
  t1_ = dalloc<bf16>(R * 64);     // ba
  t2_ = dalloc<bf16>(R * 3072);   // core / attn
  part_ = dalloc<bf16>(R * kHidden);
  E_ = dalloc<bf16>(R * kHidden);
  y_ = dalloc<bf16>(R * kHidden);
  const size_t LR = c.logit_rows;
  lg_half_ = dalloc<float>(LR * kVocabHalf * 2);  // [2][R][124160] after the gather
  lg_ = dalloc<float>(LR * kVocab);
  ylast_ = dalloc<bf16>(LR * kHidden);
  d_ids_ = dalloc<int>(R);
  d_pos_ = dalloc<int>(R);
  d_rp_ = dalloc<int>(R * 6);
  d_mrow_ = dalloc<int>(R);
  d_msrc_ = reinterpret_cast<const bf16**>(dalloc<int>(R * 2));
  d_prev_ = dalloc<int>(R * 2);
  d_seq_ = dalloc<int>(R + 1);
  d_seq2_ = dalloc<int>(4);
  d_seqs_ = dalloc<int>(2 * (kMaxSnaps + 2));
  // per-step PLE conv-state pointers. Allocated here, not on the first step: dalloc's memset runs on the legacy
  // stream, which does not order with s_ (non-blocking) — a lazily allocated array could be zeroed after the step's
  // copy into it (the first prefill then read address 0: Xid 31, 2026-11-11/12)
  d_pc_ = dalloc<bf16*>(c.max_rows);
  if (c.mtp) mtp_upload();
  CUDA_CHECK(cudaDeviceSynchronize());
  w_->drop_cache();  // readahead left shard pages behind
  size_t fr, tot;
  CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
  if (fr < (size_t(4) << 30))
    qlog::warn("engine", "free memory %.1f GiB — page cache of other files is holding it (more slots may stall allocation)", fr / 1073741824.0);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  qlog::ok("engine", "rank%d · weights and state loaded%s · %d slots × %dK · %.0f s · GPU free %.1f GiB · patches %s", c.rank,
           c.mtp ? " (with MTP head)" : "", c.slots, c.ctx / 1024, sec, fr / 1073741824.0, patch::active().c_str());
}

Engine::~Engine() {
  // process exit frees device memory; nothing persistent to flush
  if (s_) cudaStreamDestroy(s_);
}

void Engine::reset(int s) {
  auto& sl = *slots_[s];
  for (auto& g : sl.g) {
    CUDA_CHECK(cudaMemsetAsync(g.S, 0, sizeof(float) * 24 * 128 * 128, s_));
    CUDA_CHECK(cudaMemsetAsync(g.conv, 0, sizeof(bf16) * 3 * 5120, s_));
  }
  if (pool_) {  // the slot's chunks back to the pool (zeroed again when mapped)
    CUDA_CHECK(cudaStreamSynchronize(s_));
    for (auto& a : kv_arr_[s]) pool_->shrink(a.id, 0);  // chunks another slot still maps stay with it
    kv_cover_[s] = 0, kv_floor_[s] = 0;
  }
  for (auto& q : sl.q) {
    if (pool_) break;
    CUDA_CHECK(cudaMemsetAsync(q.kv, 0, sizeof(bf16) * size_t(q.cap) * 512, s_));
    CUDA_CHECK(cudaMemsetAsync(q.ik_raw, 0, sizeof(bf16) * size_t(q.cap) * 128, s_));
    CUDA_CHECK(cudaMemsetAsync(q.ik_comp, 0, sizeof(bf16) * size_t(q.cap / 4) * 128, s_));
  }
  CUDA_CHECK(cudaMemsetAsync(sl.ple_conv, 0, sizeof(bf16) * 9 * 10240, s_));
  if (sl.mq.kv && !pool_) {
    CUDA_CHECK(cudaMemsetAsync(sl.mq.kv, 0, sizeof(bf16) * size_t(sl.mq.cap) * 512, s_));
    CUDA_CHECK(cudaMemsetAsync(sl.mq.ik_raw, 0, sizeof(bf16) * size_t(sl.mq.cap) * 128, s_));
    CUDA_CHECK(cudaMemsetAsync(sl.mq.ik_comp, 0, sizeof(bf16) * size_t(sl.mq.cap / 4) * 128, s_));
  }
  hist_[s].clear();
  mr_[s].clear(), mdelta_[s] = 0, mrows_[s].clear(), memb_[s] = nullptr;
  for (int id = 0; id < int(ck_.size()); id++)
    if (ck_[id].slot == s) {
      ck_[id].len = -1, ck_[id].slot = -1;
      if (id >= c_.slots) ck_unmap(id);  // the stream is idle here (synchronised above with the pool)
    }
  sl.gdn_pn = 0;       // P020: pending GDN rows are dropped with the state (never applied to the zeroed one)
  spec_slot_.clear();  // a pending verify of any slot is void after a reset
}

// one piece of a checkpoint of slot s into buffer id (stream-ordered copy): GDN layer gi's S and conv state, or
// (gi = −1) the PLE window
void Engine::snap_part(int s, int id, int gi) {
  auto& sl = *slots_[s];
  auto& k = ck_[id];
  if (!k.mem) CUDA_CHECK(cudaMalloc(&k.mem, kCkptBytes));
  uint8_t* p = static_cast<uint8_t*>(k.mem);
  if (gi < 0) {
    CUDA_CHECK(cudaMemcpyAsync(p + kCkptGdn, sl.ple_conv, kSnapPle, cudaMemcpyDeviceToDevice, s_));
    return;
  }
  p += size_t(gi) * (kSnapS + kSnapConv);
  CUDA_CHECK(cudaMemcpyAsync(p, sl.g[gi].S, kSnapS, cudaMemcpyDeviceToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(p + kSnapS, sl.g[gi].conv, kSnapConv, cudaMemcpyDeviceToDevice, s_));
}

size_t Engine::ckpt_chunks() const { return pool_ ? KvPool::chunks_for(kCkptBytes) : 0; }

void Engine::ck_map(int s, int id) {
  auto& k = ck_[id];
  if (k.arr < 0 || pool_->mapped(k.arr) >= kCkptBytes) return;
  if (!pool_->grow(k.arr, kCkptBytes, s_)) {
    char m[200];
    std::snprintf(m, sizeof m, "KV pool full: slot %d checkpoint %d needs %zu chunks, %zu of %zu free", s, id,
                  KvPool::chunks_for(kCkptBytes), pool_->free_chunks(), pool_->total_chunks());
    qlog::warn("kv pool", "%s", m);
    throw KvFull(s, m);
  }
}

void Engine::ck_unmap(int id) {
  auto& k = ck_[id];
  if (k.arr < 0 || !pool_->mapped(k.arr)) return;
  CUDA_CHECK(cudaStreamSynchronize(s_));  // no queued copy still reads or writes it
  pool_->shrink(k.arr, 0);
}

void Engine::ckpt_save(int s, int id) {
  if (id < 0 || id >= int(ck_.size())) qlog::fail("ckpt", "buffer %d of %zu", id, ck_.size());
  ck_map(s, id);         // KvFull before anything changes
  gdn_flush_slots({s});  // P020: the checkpoint holds the committed state
  for (int gi = 0; gi < kGdnLayers; gi++) snap_part(s, id, gi);
  snap_part(s, id, -1);
  ck_[id].slot = s, ck_[id].len = int(hist_[s].size());
}

int Engine::ckpt_load(int s, int id) {
  if (id < 0 || id >= int(ck_.size())) qlog::fail("ckpt", "buffer %d of %zu", id, ck_.size());
  auto& k = ck_[id];
  if (k.len < 0 || k.slot != s) return -1;
  kv_cut(s, k.len);  // KvFull before anything changes
  ck_restore(s, k.mem);
  hist_[s].resize(k.len);
  return k.len;
}

void Engine::ck_restore(int s, const void* mem) {
  auto& sl = *slots_[s];
  const uint8_t* p = static_cast<const uint8_t*>(mem);
  for (auto& g : sl.g) {
    CUDA_CHECK(cudaMemcpyAsync(g.S, p, kSnapS, cudaMemcpyDeviceToDevice, s_));
    p += kSnapS;
    CUDA_CHECK(cudaMemcpyAsync(g.conv, p, kSnapConv, cudaMemcpyDeviceToDevice, s_));
    p += kSnapConv;
  }
  CUDA_CHECK(cudaMemcpyAsync(sl.ple_conv, p, kSnapPle, cudaMemcpyDeviceToDevice, s_));
  sl.gdn_pn = 0;  // P020: pending GDN rows belonged to the overwritten state — dropped, never applied
  spec_slot_.clear();
}

void Engine::ckpt_drop(int id) {
  if (id < 0 || id >= int(ck_.size())) qlog::fail("ckpt", "buffer %d of %zu", id, ck_.size());
  ck_[id].len = -1, ck_[id].slot = -1;
  ck_unmap(id);
}

// ---- shared KV pool (EngineCfg::kv_pool) --------------------------------------------------------------------------
namespace {
size_t kv_bytes(int kind, int tokens) {  // one cache array's bytes for positions < tokens
  return kind == 0 ? size_t(tokens) * 1024 : kind == 1 ? size_t(tokens) * 256 : size_t((tokens + 3) / 4) * 256;
}
// the first byte a write at position p touches: its K‖V row, its raw indexer key, its group's compressed key (written
// when the group's last position runs, qsa.cu) — every write at p' ≥ p lies at or past it
size_t kv_wstart(int kind, int p) {
  return kind == 0 ? size_t(p) * 1024 : kind == 1 ? size_t(p) * 256 : size_t(p / 4) * 256;
}
double mem_available_gib() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) qlog::fail("kv pool", "cannot read /proc/meminfo");
  char k[64];
  long v = -1, x;
  char u[8];
  while (std::fscanf(f, "%63s %ld %7s", k, &x, u) == 3)
    if (!std::strcmp(k, "MemAvailable:")) v = x;
  std::fclose(f);
  if (v < 0) qlog::fail("kv pool", "no MemAvailable in /proc/meminfo");
  return v / 1048576.0;
}
}  // namespace

size_t Engine::kv_need(int tokens) const {
  const int t = std::min(c_.ctx, tokens + kKvSlack);
  const size_t caches = kLayers / 4 + (c_.mtp ? 1 : 0);
  size_t n = 0;
  for (int kind = 0; kind < 3; kind++) n += KvPool::chunks_for(kv_bytes(kind, t));
  return caches * n;
}

size_t Engine::kv_held(int s) const {
  if (!pool_) return 0;
  size_t n = 0;
  for (auto& a : kv_arr_[s]) n += pool_->mapped(a.id) / KvPool::kChunk;
  return n;
}

size_t Engine::kv_sole(int s) const {
  if (!pool_) return 0;
  size_t n = 0;
  for (auto& a : kv_arr_[s]) n += pool_->sole(a.id);
  return n;
}

size_t Engine::kv_full(int tokens) const {
  const size_t caches = kLayers / 4 + (c_.mtp ? 1 : 0);
  size_t n = 0;
  for (int kind = 0; kind < 3; kind++) n += kv_wstart(kind, std::min(c_.ctx, tokens)) / KvPool::kChunk;
  return caches * n;
}

int Engine::kv_cover_of(int s) const {
  int t = c_.ctx;
  for (auto& a : kv_arr_[s]) {
    const size_t b = pool_->mapped(a.id);
    t = std::min<int64_t>(t, a.kind == 0 ? b / 1024 : a.kind == 1 ? b / 256 : b / 256 * 4);
  }
  return t;
}

void Engine::kv_cut(int s, int n) {
  if (!pool_ || n >= kv_floor_[s]) return;  // nothing at or above n is shared
  // per array: the chunk holding position n's first write and the ones after it — any shared? Then the slot keeps
  // its own copy of that chunk when it also holds positions below n, and gives the later ones back (it rewrites them;
  // mapped again zeroed as it grows)
  std::vector<std::pair<size_t, bool>> plan(kv_arr_[s].size(), {0, false});  // (chunks to keep, copy the last)
  size_t copies = 0;
  bool any = false;
  for (size_t j = 0; j < kv_arr_[s].size(); j++) {
    const auto& a = kv_arr_[s][j];
    const size_t w = kv_wstart(a.kind, n), i0 = w / KvPool::kChunk, have = pool_->mapped(a.id) / KvPool::kChunk;
    bool sh = false;
    for (size_t i = i0; i < have && !sh; i++) sh = pool_->shared(a.id, i);
    if (!sh) {
      plan[j] = {have, false};
      continue;
    }
    any = true;
    const bool part = w % KvPool::kChunk != 0;
    plan[j] = {part ? i0 + 1 : i0, part && pool_->shared(a.id, i0)};
    copies += plan[j].second;
  }
  if (any) {
    if (copies > pool_->free_chunks()) {
      char m[200];
      std::snprintf(m, sizeof m, "KV pool full: slot %d going back to %d tokens needs %zu chunks, %zu of %zu free", s,
                    n, copies, pool_->free_chunks(), pool_->total_chunks());
      qlog::warn("kv pool", "%s", m);
      throw KvFull(s, m);
    }
    CUDA_CHECK(cudaStreamSynchronize(s_));  // no queued kernel still uses the chunks unmapped below
    for (size_t j = 0; j < kv_arr_[s].size(); j++) {
      const int id = kv_arr_[s][j].id;
      pool_->shrink(id, plan[j].first * KvPool::kChunk);
      if (plan[j].second && !pool_->own(id, plan[j].first - 1, s_))
        qlog::fail("kv pool", "slot %d: no chunk for a copy (counted before)", s);
    }
    kv_cover_[s] = kv_cover_of(s);
  }
  kv_floor_[s] = n;  // chunks at or above n's writes are the slot's own now
}

int Engine::ckpt_share(int s, int id) {
  if (!pool_) qlog::fail("ckpt", "ckpt_share needs the shared KV pool");
  if (id < 0 || id >= int(ck_.size())) qlog::fail("ckpt", "buffer %d of %zu", id, ck_.size());
  const auto& k = ck_[id];
  if (k.len < 0 || k.slot < 0 || k.slot == s) return -1;
  const int d = k.slot, n = k.len;
  if (!hist_[s].empty() || kv_held(s)) qlog::fail("ckpt", "ckpt_share: slot %d is not empty (reset it first)", s);
  if (n > int(hist_[d].size()) || n > kv_cover_[d])
    qlog::fail("ckpt", "ckpt_share: checkpoint %d (%d tokens) is past slot %d's history (%zu, %d mapped)", id, n, d,
               hist_[d].size(), kv_cover_[d]);
  // per array: the chunks wholly below position n's first write are mapped again; the one it falls in (if it holds
  // positions below n) is copied — counted first: KvFull before anything changes
  size_t copies = 0;
  for (auto& a : kv_arr_[d]) copies += kv_bytes(a.kind, n) > kv_wstart(a.kind, n) / KvPool::kChunk * KvPool::kChunk;
  if (copies > pool_->free_chunks()) {
    char m[200];
    std::snprintf(m, sizeof m, "KV pool full: slot %d taking %d tokens from slot %d needs %zu chunks, %zu of %zu free", s,
                  n, d, copies, pool_->free_chunks(), pool_->total_chunks());
    qlog::warn("kv pool", "%s", m);
    throw KvFull(s, m);
  }
  for (size_t j = 0; j < kv_arr_[s].size(); j++) {
    const auto& a = kv_arr_[d][j];
    const size_t full = kv_wstart(a.kind, n) / KvPool::kChunk;
    pool_->share(kv_arr_[s][j].id, a.id, full);
    if (kv_bytes(a.kind, n) > full * KvPool::kChunk && !pool_->grow_copy(kv_arr_[s][j].id, a.id, s_))
      qlog::fail("kv pool", "slot %d: no chunk for a copy (counted before)", s);
  }
  kv_cover_[s] = kv_cover_of(s);
  kv_floor_[s] = n, kv_floor_[d] = std::max(kv_floor_[d], n);
  ck_restore(s, k.mem);
  hist_[s].assign(hist_[d].begin(), hist_[d].begin() + n);
  return n;
}

void Engine::kv_ensure(int s, int tokens) {
  if (!c_.kv_pool) return;
  if (!pool_) qlog::fail("kv pool", "a step before kv_pool_init");
  const int t = std::min(c_.ctx, tokens + kKvSlack);
  if (t <= kv_cover_[s]) return;
  for (auto& a : kv_arr_[s])
    if (!pool_->grow(a.id, kv_bytes(a.kind, t), s_)) {
      char m[200];
      std::snprintf(m, sizeof m, "KV pool full: slot %d needs %d tokens, %zu of %zu chunks free", s, tokens,
                    pool_->free_chunks(), pool_->total_chunks());
      qlog::warn("kv pool", "%s", m);
      throw KvFull(s, m);
    }
  kv_cover_[s] = t;
}

size_t Engine::kv_pool_init(double reserve_gib, double rank0_gib, double max_gib) {
  if (!c_.kv_pool || pool_) qlog::fail("kv pool", "kv_pool_init: the engine was not made for a pool, or it has one");
  CUDA_CHECK(cudaDeviceSynchronize());
  // both ranks take the same number of chunks: the smaller room of the two
  const double avail = mem_available_gib();
  const float room = float(avail - reserve_gib - (c_.rank == 0 ? rank0_gib : 0.0));
  float* d = dalloc<float>(3);
  CUDA_CHECK(cudaMemcpy(d, &room, 4, cudaMemcpyHostToDevice));
  comm_->all_gather_f32(d, d + 1, 1, s_);
  float both[2];
  CUDA_CHECK(cudaMemcpyAsync(both, d + 1, 8, cudaMemcpyDeviceToHost, s_));
  CUDA_CHECK(cudaStreamSynchronize(s_));
  cudaFree(d);
  double gib = std::min(both[0], both[1]);
  if (max_gib > 0) gib = std::min(gib, max_gib);
  const size_t chunks = gib > 0 ? size_t(gib * 1073741824.0 / KvPool::kChunk) : 0;
  if (chunks < kv_need(16384))
    qlog::fail("kv pool", "room for %.1f GiB of caches (rank0 %.1f, rank1 %.1f GiB after keeping %.1f GiB free) — not "
               "even one 16K-token sequence", gib, both[0], both[1], reserve_gib);
  pool_ = std::make_unique<KvPool>(chunks);
  for (int i = c_.slots; i < int(ck_.size()); i++) {  // the checkpoint pool: address ranges, mapped when used
    ck_[i].arr = pool_->add(kCkptBytes);
    ck_[i].mem = pool_->ptr(ck_[i].arr);
  }
  kv_arr_.assign(c_.slots, {});
  kv_cover_.assign(c_.slots, 0);
  kv_floor_.assign(c_.slots, 0);
  auto make = [&](int s, QsaCache& qc) {
    const int a0 = pool_->add(kv_bytes(0, c_.ctx)), a1 = pool_->add(kv_bytes(1, c_.ctx)), a2 = pool_->add(kv_bytes(2, c_.ctx));
    qc.cap = c_.ctx;
    qc.kv = static_cast<bf16*>(pool_->ptr(a0));
    qc.ik_raw = static_cast<bf16*>(pool_->ptr(a1));
    qc.ik_comp = static_cast<bf16*>(pool_->ptr(a2));
    kv_arr_[s].push_back({a0, 0}), kv_arr_[s].push_back({a1, 1}), kv_arr_[s].push_back({a2, 2});
  };
  for (int s = 0; s < c_.slots; s++) {
    auto& sl = *slots_[s];
    int k = 0;
    for (int i = 0; i < kLayers; i++) {
      if (!L_[i]->qsa) continue;
      make(s, sl.q[k]);
      CUDA_CHECK(cudaMemcpy(L_[i]->d_qc + s, &sl.q[k], sizeof(QsaCache), cudaMemcpyHostToDevice));
      k++;
    }
    if (mtp_) {
      make(s, sl.mq);
      CUDA_CHECK(cudaMemcpy(mtp_->d_qc + s, &sl.mq, sizeof(QsaCache), cudaMemcpyHostToDevice));
    }
  }
  const double tok = double(chunks) / double(kv_need(c_.ctx)) * c_.ctx;
  qlog::ok("kv pool", "rank%d · %zu chunks · %.1f GiB · ≈ %.2fM tokens · MemAvailable now %.1f GiB (rank0 room %.1f, "
           "rank1 room %.1f, %.1f GiB kept free)", c_.rank, chunks, chunks * double(KvPool::kChunk) / 1073741824.0,
           tok / 1e6, mem_available_gib(), both[0], both[1], reserve_gib);
  return chunks;
}

// every bf16 all-reduce goes through here, in the same order on both ranks
void Engine::ar(bf16* p, size_t n, cudaStream_t s) {
  if (rdma_ && patch::on(patch::rdma_ar)) rdma_->all_reduce_bf16(p, n, s);
  else comm_->all_reduce_bf16(p, n, s);
}

void Engine::set_prof(bool on) { prof_on_ = on; }

// mark(g): the time since the previous mark is attributed to group g
void Engine::mark(const char* g) {
  qlog::where() = g;
  if (!prof_on_) return;
  if (nmark_ == marks_.size()) {
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreate(&e));
    marks_.emplace_back(g, e);
  }
  marks_[nmark_].first = g;
  CUDA_CHECK(cudaEventRecord(marks_[nmark_].second, s_));
  nmark_++;
}

std::string Engine::prof_report(bool reset) {
  double tot = 0;
  for (auto& [g, ms] : prof_ms_) tot += ms;
  std::string r;
  char buf[96];
  auto v = prof_ms_;
  std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
  for (auto& [g, ms] : v) {
    std::snprintf(buf, sizeof buf, "%s%s %.1f%%", r.empty() ? "" : " · ", g.c_str(), tot > 0 ? 100 * ms / tot : 0.0);
    r += buf;
  }
  std::snprintf(buf, sizeof buf, " (%.2f ms per step, %d steps)", prof_steps_ ? tot / prof_steps_ : 0.0, prof_steps_);
  r += buf;
  if (reset) prof_ms_.clear(), prof_steps_ = 0;
  return r;
}

std::string Engine::spec_report(bool reset) {
  std::string r;
  char buf[96];
  double tot = 0;
  for (auto& [g, ms] : spec_ms_) tot += ms;
  for (auto& [g, ms] : spec_ms_) {
    std::snprintf(buf, sizeof buf, "%s%s %.2f", r.empty() ? "" : " · ", g.c_str(), spec_rounds_ ? ms / spec_rounds_ : 0.0);
    r += buf;
  }
  std::snprintf(buf, sizeof buf, " (%.2f ms per round, %d rounds)", spec_rounds_ ? tot / spec_rounds_ : 0.0, spec_rounds_);
  r += buf;
  if (reset) spec_ms_.clear(), spec_rounds_ = 0;
  return r;
}

void Engine::tap_copy(const std::string& name, const bf16* src, size_t n) {
  bf16* d = nullptr;
  for (auto& [k, p] : taps_) if (k == name) d = p;
  if (!d) {
    d = dalloc<bf16>(size_t(c_.logit_rows) * 10240);
    taps_.emplace_back(name, d);
  }
  CUDA_CHECK(cudaMemcpyAsync(d, src, n * sizeof(bf16), cudaMemcpyDeviceToDevice, s_));
}

const bf16* Engine::tap(const std::string& name) const {
  for (auto& [k, p] : taps_) if (k == name) return p;
  return nullptr;
}

// ---- images / videos (model.h set_mrope, set_media) ---------------------------------------------------------------
void Engine::set_mrope(int s, const std::vector<int>& pos3, int delta) {
  if (s < 0 || s >= c_.slots || pos3.size() % 3) qlog::fail("mrope", "slot %d · %zu values (n × 3)", s, pos3.size());
  for (int v : pos3)
    if (v < 0 || v >= c_.ctx) qlog::fail("mrope", "slot %d: RoPE position %d outside 0 … %d", s, v, c_.ctx - 1);
  mr_[s] = pos3, mdelta_[s] = delta;
}

void Engine::set_media(int s, const std::vector<int>& rows) {
  if (s < 0 || s >= c_.slots) qlog::fail("media", "slot %d", s);
  for (size_t k = 1; k < rows.size(); k++)
    if (rows[k] <= rows[k - 1]) qlog::fail("media", "slot %d: positions not ascending at %zu", s, k);
  mrows_[s] = rows, memb_[s] = nullptr;
}

void Engine::media_src(int s, const bf16* emb, int rows) {
  if (s < 0 || s >= c_.slots || (emb && reinterpret_cast<uintptr_t>(emb) % 16))
    qlog::fail("media", "slot %d: embeddings must be 16-byte aligned", s);
  if (rows != int(mrows_[s].size()))
    qlog::fail("media", "slot %d: %d embedding rows for %zu media positions", s, rows, mrows_[s].size());
  memb_[s] = emb;
}

const int* Engine::rope_rows(const std::vector<int>& pos, const std::vector<int>& slot,
                             const std::vector<int>& seq_start) {
  bool any = false;
  for (int sl : slot) any = any || !mr_[sl].empty() || mdelta_[sl] != 0;
  if (!any) return nullptr;
  std::vector<int> h(pos.size() * 6);
  for (size_t b = 0; b < slot.size(); b++) {
    const auto& t = mr_[slot[b]];
    const int n = int(t.size() / 3), d = mdelta_[slot[b]];
    auto at = [&](int p, int* o) {
      if (p >= 0 && p < n) o[0] = t[3 * p], o[1] = t[3 * p + 1], o[2] = t[3 * p + 2];
      else o[0] = o[1] = o[2] = p + d;
      if (o[0] < 0 || o[1] < 0 || o[2] < 0 || std::max({o[0], o[1], o[2]}) >= c_.ctx)
        qlog::fail("mrope", "slot %d position %d: RoPE position outside the table", slot[b], p);
    };
    for (int j = seq_start[b]; j < seq_start[b + 1]; j++) {
      at(pos[j], &h[6 * j]);
      if ((pos[j] & 3) == 3) at(pos[j] - 3, &h[6 * j + 3]);
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(d_rp_, h.data(), h.size() * 4, cudaMemcpyHostToDevice, s_));
  return d_rp_;
}

int Engine::media_rows(const std::vector<int>& pos, const std::vector<int>& slot, const std::vector<int>& seq_start) {
  std::vector<int> rows;
  std::vector<const bf16*> src;
  for (size_t b = 0; b < slot.size(); b++) {
    const auto& mr = mrows_[slot[b]];
    if (mr.empty() || seq_start[b + 1] == seq_start[b]) continue;
    // a sequence's positions are consecutive (checked against the history)
    const int j0 = seq_start[b], j1 = seq_start[b + 1], p0 = pos[j0], p1 = pos[j1 - 1];
    if (c_.rank == 0 && !memb_[slot[b]] &&
        std::lower_bound(mr.begin(), mr.end(), p0) != std::lower_bound(mr.begin(), mr.end(), p1 + 1))
      qlog::fail("media", "slot %d: media rows in this call but no embeddings (media_src)", slot[b]);
    for (auto it = std::lower_bound(mr.begin(), mr.end(), p0); it != mr.end() && *it <= p1; ++it) {
      const size_t k = it - mr.begin();
      rows.push_back(j0 + *it - p0);
      src.push_back(c_.rank == 0 ? memb_[slot[b]] + k * kHidden : nullptr);
    }
  }
  if (rows.empty()) return 0;
  CUDA_CHECK(cudaMemcpyAsync(d_mrow_, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_msrc_, src.data(), src.size() * sizeof(void*), cudaMemcpyHostToDevice, s_));
  return int(rows.size());
}

const float* Engine::step(const std::vector<int>& ids, const std::vector<int>& pos, const std::vector<int>& slot,
                          const std::vector<int>& seq_start, bool all_logits, int snap_row,
                          const std::vector<std::pair<int, int>>& snaps) {
  const int M = int(ids.size()), B = int(slot.size());
  if (M == 0 || M > c_.max_rows || int(seq_start.size()) != B + 1 || seq_start[B] != M)
    qlog::fail("step", "bad input · M %d · B %d", M, B);
  std::vector<std::pair<int, int>> sn(snaps);  // checkpoints inside the step: (row, buffer), row order
  if (snap_row >= 0) sn.push_back({snap_row, slot[0]});
  std::sort(sn.begin(), sn.end());
  for (auto [row, id] : sn)
    if (B != 1 || row < 0 || row >= M || spec_ || id < 0 || id >= int(ck_.size()) || int(sn.size()) > kMaxSnaps)
      qlog::fail("step", "checkpoint row %d buffer %d: needs one sequence, a row within %d, not verify, ≤ %d per step",
                 row, id, M, kMaxSnaps);
  const int R = all_logits ? M : B;
  if (R > c_.logit_rows) qlog::fail("step", "logits rows %d > %d (prefill chunks use all_logits=false)", R, c_.logit_rows);
  for (int b = 0; b < B; b++) {
    if (pool_ && pos[seq_start[b]] < kv_floor_[slot[b]])  // below it the caches may be another slot's too (read-only)
      qlog::fail("step", "slot %d writes position %d below its shared caches (%d)", slot[b], pos[seq_start[b]],
                 kv_floor_[slot[b]]);
    kv_ensure(slot[b], pos[seq_start[b + 1] - 1] + 1);  // KvFull before any state changes
  }
  for (auto [row, id] : sn) ck_map(slot[0], id);
  // P020: a verify step applies its slots' pending GDN rows in its GDN kernels; any other step flushes them first
  const bool p20 = spec_ && mtp_ && patch::on(patch::gdn_fused_commit);
  if (!p20) gdn_flush_slots(slot);
  if (spec_) spec_p20_ = p20;
  const bool taps = taps_on_ && M <= c_.logit_rows;
  // n-gram predecessors per row (tokens at p-2, p-1; EOS before the sequence start)
  std::vector<int> prev(2 * M);
  for (int b = 0; b < B; b++) {
    auto& h = hist_[slot[b]];
    for (int j = seq_start[b]; j < seq_start[b + 1]; j++) {
      if (pos[j] != int(h.size())) qlog::fail("step", "slot %d position %d ≠ history %zu", slot[b], pos[j], h.size());
      if (pos[j] >= c_.ctx) qlog::fail("step", "slot %d position %d ≥ window %d (outside the QSA cache)", slot[b], pos[j], c_.ctx);
      int p = pos[j];
      prev[2 * j] = p >= 2 ? h[p - 2] : kEos;
      prev[2 * j + 1] = p >= 1 ? h[p - 1] : kEos;
      h.push_back(ids[j]);
    }
  }
  CUDA_CHECK(cudaMemcpyAsync(d_ids_, ids.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_pos_, pos.data(), M * 4, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_prev_, prev.data(), M * 8, cudaMemcpyHostToDevice, s_));
  CUDA_CHECK(cudaMemcpyAsync(d_seq_, seq_start.data(), (B + 1) * 4, cudaMemcpyHostToDevice, s_));
  const int* rp = rope_rows(pos, slot, seq_start);  // images / videos: mRoPE positions (null: text)
  const int n_media = media_rows(pos, slot, seq_start);
  // per-layer state pointers for the B sequences of this step
  std::vector<GdnState*> gp(B);
  std::vector<QsaCache*> qp(B);
  std::vector<bf16*> pc(B);
  for (int i = 0; i < kLayers; i++) {
    auto& l = *L_[i];
    for (int b = 0; b < B; b++) {
      if (l.qsa) qp[b] = l.d_qc + slot[b];
      else gp[b] = l.d_gst + slot[b];
    }
    if (l.qsa) CUDA_CHECK(cudaMemcpyAsync(l.d_qp, qp.data(), B * sizeof(void*), cudaMemcpyHostToDevice, s_));
    else CUDA_CHECK(cudaMemcpyAsync(l.d_gp, gp.data(), B * sizeof(void*), cudaMemcpyHostToDevice, s_));
    if (spec_ && !l.qsa) {
      std::vector<GdnRec*> rp(B), pp(B);
      for (int b = 0; b < B; b++) {
        const int par = p20 ? slots_[slot[b]]->gdn_par : 1;  // without P020 always parity 0
        rp[b] = l.d_rec + 2 * slot[b] + (par ^ 1), pp[b] = l.d_rec + 2 * slot[b] + par;
      }
      CUDA_CHECK(cudaMemcpyAsync(l.d_rp, rp.data(), B * sizeof(void*), cudaMemcpyHostToDevice, s_));
      if (p20) CUDA_CHECK(cudaMemcpyAsync(l.d_pp, pp.data(), B * sizeof(void*), cudaMemcpyHostToDevice, s_));
    }
  }
  if (p20) {  // pending rows per sequence (consumed by this verify), then the new records become the pending parity
    std::vector<int> np(B);
    for (int b = 0; b < B; b++) np[b] = slots_[slot[b]]->gdn_pn;
    CUDA_CHECK(cudaMemcpyAsync(d_npend_, np.data(), B * sizeof(int), cudaMemcpyHostToDevice, s_));
    for (int b = 0; b < B; b++) slots_[slot[b]]->gdn_par ^= 1, slots_[slot[b]]->gdn_pn = 0;
  }
  for (int b = 0; b < B; b++) pc[b] = slots_[slot[b]]->ple_conv;
  CUDA_CHECK(cudaMemcpyAsync(d_pc_, pc.data(), B * sizeof(void*), cudaMemcpyHostToDevice, s_));

  // P002 only for chunks above one kernel call: two lanes read every weight twice (the MoE reads ~all experts at
  // these M), which outweighs the hidden all-reduces for smaller chunks (next turn 130K + 6,695: 1,644 → 1,558 ms
  // without the split; a 16K chunk as two 8K lanes stays ~7 % faster than 8K chunks)
  constexpr int kMaxCall = 8192;  // rows per kernel call (workspaces); larger prefill chunks need two lanes (P002)
  const bool overlap = patch::on(patch::ar_overlap) && B == 1 && M > kMaxCall && !taps;
  // P008: chunks of 2048 … 8192 rows — attention in two lanes (each lane's all-reduce under the other's compute), the
  // MoE once over all rows (expert weights read once), its all-reduce in two halves
  const bool split_attn = patch::on(patch::ar_split_attn) && B == 1 && M >= 2048 && M <= kMaxCall && !taps;
  if ((overlap ? (M + 1) / 2 : M) > kMaxCall) qlog::fail("step", "%d rows > %d rows per kernel call (up to twice that with P002 ar_overlap)", M, kMaxCall);
  const int hlf = overlap ? M / 2 : M;  // row split used for the pre/post stages in overlap mode (P008: one stage)

  // checkpoints inside the step: the lane holding a row — lanes are [0, M/2) and [M/2, M) in P002/P008 modes — runs its
  // GDN and PLE calls in pieces between the rows (one-sequence seq_start {0, n} each), the checkpoint copies in
  // between (same values: those calls give the same bits however a sequence's rows are split over calls)
  struct Piece { int r0, n, id; const int* seq; };  // rows [r0, r0 + n), or the checkpoint into buffer id (n = 0)
  std::vector<Piece> pieces;
  if (!sn.empty()) {
    std::vector<int> bounds = {0, M}, hs;
    if (overlap || split_attn) bounds.insert(bounds.begin() + 1, M / 2);
    auto rows = [&](int r0, int n) {
      pieces.push_back({r0, n, -1, d_seqs_ + hs.size()});
      hs.push_back(0), hs.push_back(n);
    };
    for (size_t li = 0; li + 1 < bounds.size(); li++) {
      int cur = bounds[li];
      for (auto [row, id] : sn) {
        if (row < cur || row >= bounds[li + 1]) continue;
        if (row > cur) rows(cur, row - cur), cur = row;
        pieces.push_back({row, 0, id, nullptr});
        ck_[id].slot = slot[0], ck_[id].len = pos[row];
      }
      if (cur < bounds[li + 1]) rows(cur, bounds[li + 1] - cur);
    }
    CUDA_CHECK(cudaMemcpyAsync(d_seqs_, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice, s_));
  }
  auto by_pieces = [&](int o, int n, int gi, const std::function<void(int, int, const int*)>& run) {
    for (auto& p : pieces)
      if (p.r0 >= o && p.r0 < o + n) {
        if (p.id >= 0) snap_part(slot[0], p.id, gi);
        else run(p.r0, p.n, p.seq);
      }
  };
  nmark_ = 0;
  auto enqueue = [&] {
  qlog::where_layer() = -1;
  mark("start");
  // embedding → 4 streams
  embed_lookup(embed_, c_.rank, d_ids_, part_, M, s_);
  embed_media(part_, d_mrow_, d_msrc_, n_media, s_);
  mark("embed");
  ar(part_, size_t(M) * kHidden, s_);
  mark("AR");
  hc_expand(part_, X_, hlf, s_);
  if (hlf < M) hc_expand(part_ + size_t(hlf) * kHidden, X_ + size_t(hlf) * 10240, M - hlf, s_);
  mark("hc");

  // One lane = rows [off, off+n) of the step. Normal path: one lane with all rows. P002 ar_overlap (prefill, one
  // sequence): two lanes (halves); compute stays on s_ in order (kernel workspaces are shared), all-reduces go to cs_
  // so one half's all-reduce runs under the other half's compute. Same per-row arithmetic → bitwise equal.
  struct Lane { int off, n; const int* seq; int B; };
  auto R2 = [](bf16* p, int off, int w) { return p + size_t(off) * w; };
  auto attn_part = [&](int i, const Lane& ln, bool pend) {
    qlog::where_layer() = i;
    auto& l = *L_[i];
    const int o = ln.off, n = ln.n;
    hc_combine_mix(l.ahc, R2(X_, o, 10240), pend ? R2(bo_, o, kHidden) : nullptr, pend ? R2(inj_m_, o, 4) : nullptr,
                   R2(u_, o, kHidden), R2(inj_a_, o, 4), n, s_);
    mark("hc");
    if (l.qsa) {
      lin_forward(l.q.qkv, R2(u_, o, kHidden), R2(t0_, o, l.q.qkv.n), n, s_);
      mark("proj");
      if (taps && !d_sel_) d_sel_ = dalloc<int>(size_t(c_.logit_rows) * kSelCols);
      qsa_core(l.q, R2(t0_, o, l.q.qkv.n), d_pos_ + o, R2(t2_, o, 3072), l.d_qp, ln.seq, ln.B, n, taps ? d_sel_ : nullptr,
               s_, *std::max_element(pos.begin() + o, pos.begin() + o + n), rp ? rp + size_t(o) * 6 : nullptr);
      mark("qsa");
      // tap "sel.<i>": the selected token positions [n, 2052] int32 (stored in bf16 pairs; count in the last column)
      if (taps) tap_copy("sel." + std::to_string(i), reinterpret_cast<const bf16*>(d_sel_), size_t(n) * kSelCols * 2);
      lin_forward(l.q.o, R2(t2_, o, 3072), R2(part_, o, kHidden), n, s_);
      mark("proj");
    } else {
      lin_forward(l.gdn.qkvz, R2(u_, o, kHidden), R2(t0_, o, 8192), n, s_);
      lin_forward(l.gdn.ba, R2(u_, o, kHidden), R2(t1_, o, 48), n, s_);
      mark("proj");
      auto gdn = [&](int r0, int nr, const int* seq, int Bn) {
        gdn_core(l.gdn, R2(t0_, r0, 8192), R2(t1_, r0, 48), R2(t2_, r0, 3072), l.d_gp, seq, Bn, nr, s_);
      };
      if (spec_ && p20) {
        gdn_core_spec_pending(l.gdn, R2(t0_, o, 8192), R2(t1_, o, 48), R2(t2_, o, 3072), l.d_gp, l.d_rp, l.d_pp,
                              d_npend_, ln.seq, ln.B, n, s_);
      } else if (spec_) {
        gdn_core_spec(l.gdn, R2(t0_, o, 8192), R2(t1_, o, 48), R2(t2_, o, 3072), l.d_gp, l.d_rp, ln.seq, ln.B, n, s_);
      } else if (!pieces.empty()) {  // B = 1: the lane's rows in pieces, the checkpoint copies between them
        by_pieces(o, n, i - (i + 1) / 4, [&](int r0, int nr, const int* sq) { gdn(r0, nr, sq, 1); });
      } else {
        gdn(o, n, ln.seq, ln.B);
      }
      mark("gdn");
      lin_forward(l.gdn.out, R2(t2_, o, 3072), R2(part_, o, kHidden), n, s_);
      mark("proj");
    }
  };
  auto mlp_hc = [&](int i, const Lane& ln) {
    qlog::where_layer() = i;
    auto& l = *L_[i];
    const int o = ln.off, n = ln.n;
    hc_combine_mix(l.mhc, R2(X_, o, 10240), R2(part_, o, kHidden), R2(inj_a_, o, 4), R2(v_, o, kHidden),
                   R2(inj_m_, o, 4), n, s_);
    mark("hc");
  };
  auto moe_part = [&](int i, const Lane& ln) {
    moe_forward(L_[i]->moe, R2(v_, ln.off, kHidden), R2(bo_, ln.off, kHidden), ln.n, nullptr, nullptr, s_);
    mark("moe");
  };
  auto mlp_part = [&](int i, const Lane& ln) {
    mlp_hc(i, ln);
    moe_part(i, ln);
  };
  // all collectives of a step must run in the same order on both ranks: in overlap mode every all-reduce goes to cs_
  auto ar_sync = [&](bf16* p, size_t n, bool on_cs) {
    if (!on_cs) return ar(p, n, s_);
    CUDA_CHECK(cudaEventRecord(ev_rdy_[0], s_));
    CUDA_CHECK(cudaStreamWaitEvent(cs_, ev_rdy_[0]));
    ar(p, n, cs_);
    CUDA_CHECK(cudaEventRecord(ev_rdy2_[0], cs_));
    CUDA_CHECK(cudaStreamWaitEvent(s_, ev_rdy2_[0]));
  };
  auto ple_part = [&](const Lane& ln, bool on_cs) {  // layer 1: PLE sees the materialised state
    const int o = ln.off, n = ln.n;
    hc_combine(R2(X_, o, 10240), R2(bo_, o, kHidden), R2(inj_m_, o, 4), n, s_);
    mark("hc");
    ple_lookup(ple_, d_ids_ + o, d_prev_ + 2 * o, R2(E_, o, kHidden), n, s_);
    mark("ple");
    ar_sync(R2(E_, o, kHidden), size_t(n) * kHidden, on_cs);
    mark("AR");
    if (!pieces.empty()) {  // B = 1: as the GDN calls
      by_pieces(o, n, -1, [&](int r0, int nr, const int* sq) {
        ple_apply(ple_, R2(X_, r0, 10240), R2(E_, r0, kHidden), d_pc_, sq, 1, nr, s_);
      });
    } else {
      ple_apply(ple_, R2(X_, o, 10240), R2(E_, o, kHidden), d_pc_, ln.seq, ln.B, n, s_);
    }
    mark("ple");
  };

  if (split_attn) {
    const int h = M / 2;  // (next turn 130K + 6,695: lanes ½ 1,550 · ¼ 1,553 · 0.35 1,552 ms; full P002 lanes 1,546)
    const int hs[4] = {0, h, 0, M - h};
    CUDA_CHECK(cudaMemcpyAsync(d_seq2_, hs, sizeof hs, cudaMemcpyHostToDevice, s_));
    const Lane ln[2] = {{0, h, d_seq2_, 1}, {h, M - h, d_seq2_ + 2, 1}}, all{0, M, d_seq_, B};
    auto ar_async = [&](bf16* p, int lane, cudaEvent_t ready, cudaEvent_t done) {
      CUDA_CHECK(cudaEventRecord(ready, s_));
      CUDA_CHECK(cudaStreamWaitEvent(cs_, ready));
      ar(R2(p, ln[lane].off, kHidden), size_t(ln[lane].n) * kHidden, cs_);
      CUDA_CHECK(cudaEventRecord(done, cs_));
    };
    for (int i = 0; i < kLayers; i++) {
      for (int k = 0; k < 2; k++) {
        if (i > 0) {
          CUDA_CHECK(cudaStreamWaitEvent(s_, ev_bo_[k]));
          mark("AR.wait");
        }
        if (i == 1) ple_part(ln[k], true);
        attn_part(i, ln[k], i > 1);
        ar_async(part_, k, ev_rdy_[k], ev_part_[k]);
      }
      for (int k = 0; k < 2; k++) {
        CUDA_CHECK(cudaStreamWaitEvent(s_, ev_part_[k]));
        mark("AR.wait");
        mlp_hc(i, ln[k]);
      }
      moe_part(i, all);
      for (int k = 0; k < 2; k++) ar_async(bo_, k, ev_rdy2_[k], ev_bo_[k]);
    }
    for (int k = 0; k < 2; k++) CUDA_CHECK(cudaStreamWaitEvent(s_, ev_bo_[k]));
    mark("AR.wait");
  } else if (!overlap) {
    const Lane all{0, M, d_seq_, B};
    for (int i = 0; i < kLayers; i++) {
      if (i == 1) ple_part(all, false);
      attn_part(i, all, i > 1);
      ar(part_, size_t(M) * kHidden, s_);
      mark("AR");
      if (taps) tap_copy("attn." + std::to_string(i), part_, size_t(M) * kHidden);
      if (taps) tap_copy("u." + std::to_string(i), u_, size_t(M) * kHidden);
      mlp_part(i, all);
      ar(bo_, size_t(M) * kHidden, s_);
      mark("AR");
      if (taps) {
        const std::string si = std::to_string(i);
        tap_copy("v." + si, v_, size_t(M) * kHidden);
        tap_copy("moe." + si, bo_, size_t(M) * kHidden);
        tap_copy("X." + si, X_, size_t(M) * 10240);
        hc_combine(const_cast<bf16*>(tap("X." + si)), bo_, inj_m_, M, s_);  // materialised (golden `.resid`)
      }
    }
  } else {
    const int h = M / 2;
    const int hs[4] = {0, h, 0, M - h};
    CUDA_CHECK(cudaMemcpyAsync(d_seq2_, hs, sizeof hs, cudaMemcpyHostToDevice, s_));
    const Lane ln[2] = {{0, h, d_seq2_, 1}, {h, M - h, d_seq2_ + 2, 1}};
    auto ar_async = [&](bf16* p, int lane, cudaEvent_t ready, cudaEvent_t done) {
      CUDA_CHECK(cudaEventRecord(ready, s_));
      CUDA_CHECK(cudaStreamWaitEvent(cs_, ready));
      ar(R2(p, ln[lane].off, kHidden), size_t(ln[lane].n) * kHidden, cs_);
      CUDA_CHECK(cudaEventRecord(done, cs_));
    };
    for (int i = 0; i < kLayers; i++) {
      for (int k = 0; k < 2; k++) {
        if (i > 0) {
          CUDA_CHECK(cudaStreamWaitEvent(s_, ev_bo_[k]));
          mark("AR.wait");  // profile: compute stalled on this lane's all-reduce (not hidden under the other lane)
        }
        if (i == 1) ple_part(ln[k], true);
        attn_part(i, ln[k], i > 1);
        ar_async(part_, k, ev_rdy_[k], ev_part_[k]);
      }
      for (int k = 0; k < 2; k++) {
        CUDA_CHECK(cudaStreamWaitEvent(s_, ev_part_[k]));
        mark("AR.wait");
        mlp_part(i, ln[k]);
        ar_async(bo_, k, ev_rdy2_[k], ev_bo_[k]);
      }
    }
    for (int k = 0; k < 2; k++) CUDA_CHECK(cudaStreamWaitEvent(s_, ev_bo_[k]));
    mark("AR");
  }
  if (taps) mark("taps");
  hc_combine_mix(final_, X_, bo_, inj_m_, y_, nullptr, hlf, s_);
  if (hlf < M)
    hc_combine_mix(final_, X_ + size_t(hlf) * 10240, bo_ + size_t(hlf) * kHidden, inj_m_ + size_t(hlf) * 4,
                   y_ + size_t(hlf) * kHidden, nullptr, M - hlf, s_);
  mark("hc");
  if (taps) tap_copy("y", y_, size_t(M) * kHidden);
  head_enqueue(M, B, seq_start, all_logits, false);
  };
  enqueue();
  return head_finish(all_logits ? M : B);
}

// lm_head over y_ (all M rows, or the last row of each sequence) → fp32 logits gathered on both ranks (stream work
// only, graph-capturable); head_finish ends the step (sync, profile accounting). Shared by the target step and the head.
void Engine::head_enqueue(int M, int B, const std::vector<int>& seq_start, bool all_logits, bool draft) {
  const int R = all_logits ? M : B;
  const bf16* yl = y_;
  if (!all_logits) {  // last row of each sequence only
    for (int b = 0; b < B; b++)
      CUDA_CHECK(cudaMemcpyAsync(ylast_ + size_t(b) * kHidden, y_ + size_t(seq_start[b + 1] - 1) * kHidden,
                                 kHidden * sizeof(bf16), cudaMemcpyDeviceToDevice, s_));
    yl = ylast_;
  }
  // P006: the MTP head's passes read the quantized copy (draft logits only — the accept test keeps them as q)
  const Lin& lm = draft && mtp_ && patch::on(patch::draft_lm_q) ? mtp_->lm_q : lm_;
  float* mine = lg_half_ + size_t(c_.rank) * R * kVocabHalf;
  lin_forward_f32(lm, yl, mine, R, s_);
  // penalties (spec_round's verify rows): this rank's vocab half, before the candidates / the gather — both see them
  if (pen_B_ > 0 && !draft)
    penalize_rows(mine, kVocabHalf, c_.rank * kVocabHalf, kVocabHalf, kVocab, R, d_pst_, d_qst_, d_drafts_, pen_B_,
                  d_pen_, s_);
  if (sparse_) {  // P019: each rank's top candidates per row, exchanged (R × 520 B per rank); the half stays in mine
    sample_cands(mine, R, kVocabHalf, c_.rank * kVocabHalf, cand_send_, s_);
    comm_->all_gather_f32(reinterpret_cast<const float*>(cand_send_), reinterpret_cast<float*>(cand_recv_),
                          size_t(R) * kSparseRec * 2, s_);
  } else {
    comm_->all_gather_f32(mine, lg_half_, size_t(R) * kVocabHalf, s_);
    for (int r = 0; r < 2; r++)
      CUDA_CHECK(cudaMemcpy2DAsync(lg_ + size_t(r) * kVocabHalf, kVocab * 4, lg_half_ + size_t(r) * R * kVocabHalf,
                                   kVocabHalf * 4, kVocabHalf * 4, R, cudaMemcpyDeviceToDevice, s_));
  }
  mark(draft ? "mtp.lm_head" : "lm_head");
}

void Engine::gather_full(int R) {
  float* mine = lg_half_ + size_t(c_.rank) * R * kVocabHalf;
  comm_->all_gather_f32(mine, lg_half_, size_t(R) * kVocabHalf, s_);
  for (int r = 0; r < 2; r++)
    CUDA_CHECK(cudaMemcpy2DAsync(lg_ + size_t(r) * kVocabHalf, kVocab * 4, lg_half_ + size_t(r) * R * kVocabHalf,
                                 kVocabHalf * 4, kVocabHalf * 4, R, cudaMemcpyDeviceToDevice, s_));
}

const float* Engine::head_finish(int R) {
  last_rows_ = R;
  CUDA_CHECK(cudaStreamSynchronize(s_));
  if (prof_on_ && nmark_ > 1) {
    for (size_t k = 1; k < nmark_; k++) {
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, marks_[k - 1].second, marks_[k].second));
      const std::string& g = marks_[k].first;
      auto it = std::find_if(prof_ms_.begin(), prof_ms_.end(), [&](auto& x) { return x.first == g; });
      if (it == prof_ms_.end()) prof_ms_.emplace_back(g, ms);
      else it->second += ms;
    }
    prof_steps_++;
  }
  return lg_;
}

}  // namespace q38
