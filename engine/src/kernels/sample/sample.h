// sample.h — token selection from fp32 logits rows (V = 248320 in the engine; both ranks hold the gathered row): greedy
// argmax, temperature sampling, and exact speculative acceptance for the MTP verify step (standard speculative
// sampling), with optional top-k / top-p truncation. Deterministic and batch-invariant: a row's (sequence's) result
// depends only on its own logits, temperature, top_k, top_p, seed and counter — never on R / B, the other rows, the
// cluster size or the stream.
//
// Distribution of a row x at temperature T > 0 (the one sample_rows samples and spec_accept uses for p and q):
//   m    = max_v x_v                                    exact
//   c    = fl(log2(e) / T)                              IEEE fp32 division, once per row
//   e_v  = 2^((x_v − m) · c) ∈ [0, 1]                   fp32 product, hardware ex2 (ex2.approx.ftz, the MUFU op behind
//                                                       exp2f); one device function for both kernels
//   w_v  = round(e_v · 2^44)        uint64              exact for e_v ≥ 2^-20; below that the 2^-44 grid (≤ 2.9e-14 of
//                                                       the top token's weight per token; e_v < 2^-45 → never drawn)
//   W    = Σ_v w_v                  exact integer       < V · 2^44 < 2^64 (V < 2^20)
//   P(v) = w_v / W
// e_v differs from exp((x_v − m) / T) by ≤ ~2^-22 + |ln e_v|·2^-23 relative (ex2's error, the scale's and the
// product's rounding) — the size of an fp32 softmax's own rounding. One MUFU op per token: an IEEE division + expf per
// token (three MUFU-pipe ops) and a 64-bit float→int conversion made the weight pass ~4× slower.
// Why integer weights instead of an fp32 Σexp: every partial sum is exact, so the total and the CDF do not depend on
// the summation order, chunking or batch (determinism by construction), and the CDF the search walks is exactly the
// one the weights define. An fp32 prefix scan rounds each of its V steps by up to ½ ulp of the running sum — that moves
// mass between tail tokens (bounded by V·2^-24 of the total), and spec_accept's p/q would then not use the q the
// drafts were really drawn from.
// Sampling (inverse CDF, one uniform): U = 64 random bits, t = ⌊U·W / 2^64⌋ ∈ [0, W), token = the v with
// Σ_{u<v} w_u ≤ t < Σ_{u≤v} w_u (exact prefix scan in index order). P(token = v) = w_v / W to within 2^-64.
// Degenerate rows (W = 0: the max is +inf or NaN, or every logit is −inf) give the argmax.
// Greedy (T == 0; also T < 0 or NaN): argmax in torch.argmax order — the first maximum, a NaN counts as the maximum.
//
// Truncation (top-k, then top-p; T > 0 only — greedy ignores both), per row / per sequence k = top_k, p = top_p:
//   rank order  v above u iff x_v > x_u, or x_v == x_u and v < u (−0 == +0): a total order, ties to the lower index.
//               As a key: Z_v = key(x_v)·2^20 + 2^20 − 1 − v (52 bits; key = the fp32 bits made order-preserving: sign
//               set → ~bits, else bits | 2^31; −0 taken as +0), v above u iff Z_v > Z_u.
//   top-k       1 ≤ k < V: K = the k highest-ranked tokens (exactly k; vLLM keeps every token tied with the k-th
//               logit). k ≤ 0 or k ≥ V: K = all V.
//   top-p       p < 1: S = the shortest rank-order prefix of K whose weight reaches p·W_K, W_K = Σ_{v∈K} w_v — the
//               last kept token is the first whose inclusive prefix is ≥ p·W_K, in integers ≥ ⌈p·W_K⌉ (p's exact
//               value); i.e. v ∈ K is kept iff Σ_{u∈K above v} w_u < p·W_K. Always ≥ 1 token (p ≤ 0: the top token
//               only). p ≥ 1 or NaN: S = K.
//   w'_v = w_v on S, 0 elsewhere;  W' = Σ_v w'_v ≥ 2^44 (the top token has e = 1);  P(v) = w'_v / W'.
// vLLM's apply_top_k_top_p in exact integers (it sorts the logits, masks below the k-th, then keeps the tokens whose
// ascending fp32 cumsum of softmax exceeds 1 − p, at least one): kept sets differ only where vLLM's order of tied
// logits or its fp32 rounding at the 1 − p boundary decides. Everything above and below holds with w', W' in place of
// w, W: the draw is t = ⌊U·W' / 2^64⌋ over the kept tokens in index order, so P(token = v) = w'_v / W' to within
// 2^-64 (and k = 1 gives the argmax). The truncated distribution is a function of the row alone — no path, cluster
// size or batch changes it.
//
// min_p (per row / per sequence mp, applied before top-k; T > 0 only — greedy ignores it): vLLM keeps v iff
// p_v ≥ mp·p_max; here p_v / p_max = w_v / 2^44 (the top token has e = 1), so
//   M     = {v : w_v ≥ t_mp},  t_mp = ⌈mp·2^44⌉ (mp's exact fp32 value) for 0 < mp < 1, 2^44 for mp ≥ 1;
//           mp ≤ 0 or NaN: off (M = all V). The top token is always in M.
//   K, S  top-k / top-p as above over M: K = the min(k, |M|) highest-ranked tokens of M, S = top-p's prefix of K
//         with W_K = Σ_{v∈K} w_v (vLLM's order: min_p, then top-k, then top-p).
// M is a rank-order prefix: weight(x, m, c) is non-decreasing in x — fl(x − m) and fl(·c) round monotonically, and
// round(ex2.approx.ftz(y)·2^44) is non-decreasing over every fp32 argument y ≤ 0 (checked exhaustively on GB10:
// sample_weight_monotone, all 2^31 arguments −0 … −inf, 0 violations; tests/test_penalty.py). So M = {Z_v ≥ zm} with
// zm = key(x*)·2^20, x* the lowest float whose weight reaches t_mp, and K ∩ M is the cut max(z_K, zm): kept sets stay
// rank prefixes, and every statement above holds with M's cut as a floor. A row whose top weight is below t_mp (c not
// finite: W = 0) ignores min_p.
//
// spec_accept, sequence b with k = k_b drafts d_i at temperature T (p_i / q_i: target / draft row i, weights as above
// with each row's own max, wp / wq, totals Wp / Wq):
//   T > 0: for i = 0 … k−1: A = wp_i(d_i)·Wq_i, B = wq_i(d_i)·Wp_i (exact 128-bit); accept iff ⌊U_i·B / 2^64⌋ < A, i.e.
//     with probability min(1, A/B) = min(1, p_i(d_i) / q_i(d_i)) to within 2^-64 (q(d) = 0 < p(d) accepts). At the
//     first rejection i: next ~ r_v = max(0, wp_v·Wq − wq_v·Wp) = Wp·Wq·(p_i(v) − q_i(v))⁺ (exact 128-bit, Σr < 2^124),
//     drawn like a row (t = ⌊U·Σr / 2^64⌋, exact prefix scan); Σr = 0 (p_i == q_i exactly) → next ~ p_i. All k
//     accepted: next ~ p_k exactly as sample_rows draws it. The tokens of a round are therefore distributed exactly as
//     drawing them from the target rows one by one.
//   T == 0: accept while d_i == argmax p_i; next = argmax of the first mismatching row (or of row k).
//   Truncation: p_i and q_i are both truncated with the sequence's top_k / top_p (each row its own kept set, w', W');
//     the tokens of a round are then distributed exactly as drawing them from the truncated target rows. A draft
//     outside p_i's kept set has A = 0: rejected with probability 1. min_p: the target rows with the sequence's min_p,
//     the draft rows with q_min_p (each row its own M).
// q_logits must be the logits each draft was drawn from by sample_rows, with bitwise the same T, top_k, top_p and
// min_p (= q_min_p here; 0 / null when the drafts were drawn without min_p): q is then exactly the distribution the
// draft came from. Penalties (below) apply to the target rows only.
//
// P016 block_verify (patches.def; on by default): for T > 0 and 1 ≤ k ≤ 15 the round is decided by block verification
// (Sun et al. 2024, arXiv 2403.10444) instead — the same output distribution (the tokens of a round are distributed as
// drawing them from the target rows one by one), at least as many tokens per round on average. 1-indexed drafts
// x_i = d_{i−1}, P_i / Q_i = (truncated) target / draft row i−1, P_{k+1} = target row k:
//   c_0 = 1;  c_i = min(1, c_{i−1}·P_i(x_i) / Q_i(x_i))   (c_i = 0 once some P_j(x_j) = 0; Q_i(x_i) = 0 < P_i(x_i) → 1)
//   R_i = Σ_v max(c_i·P_{i+1}(v) − Q_{i+1}(v), 0),  h_i = R_i / (R_i + 1 − c_i) (1 if that is 0)  for 1 ≤ i < k;  h_k = c_k
//   τ = the largest i ≥ 1 with η_i ≤ h_i, 0 if none;  n_acc = τ
//   next ~ P_{k+1} if τ = k, else ~ max(c_τ·P_{τ+1} − Q_{τ+1}, 0) normalised (all 0 → ~ P_{τ+1})
// In integers (every quantity exact, kernels/sample/p016_block_verify.cuh): a_i = wp(x_i)·Wq, b_i = wq(x_i)·Wp of row
// i−1; c_i = N_i / D_i (N_i = N_{i−1}·a_i, D_i = D_{i−1}·b_i, reset to 1 / 1 when N_i ≥ D_i): big integers (≤ 1620 bits
// at k = 15). Row i of the residual (0-indexed row i, c_i): α = N_i·Wq, β = D_i·Wp, r_v = max(α·wp_v − β·wq_v, 0) =
// D_i·Wp·Wq·max(c_i·P(v) − Q(v), 0); num = Σr = α·S_p − β·S_q (S = Σ wp, Σ wq over the tokens with r_v > 0, exact u64);
// h_i = num / den, den = num + (D_i − N_i)·Wp·Wq. Tests: η_i ≤ h_i ⟺ ⌊U_i·den / 2^64⌋ < num (den = 0: passed),
// η_k ≤ c_k ⟺ ⌊U_k·D_k / 2^64⌋ < N_k — each with probability h to within 2^-64, as the base test. The draw:
// t = ⌊U·num / 2^64⌋, the v with Σ_{u<v} r_u ≤ t < Σ_{u≤v} r_u (index order); P_{k+1} / P_{τ+1} drawn as sample_rows
// draws a row. τ = 0 uses α = Wq, β = Wp: the base residual weights. Greedy (T = 0) and k > 15 keep the base rule.
// A degenerate row (W = 0) follows the same integers: Wp·Wq = 0 gives num = den = 0, so h_i = 1 (τ = i if reached,
// next from p_i — its argmax if Wp = 0), and a_i = 0 ends the chain (c = 0 after it).
// With k = 1 the rule is the base rule and the integers, uniforms and draws are the same: identical results.
//
// Random numbers: Philox4x32-10 (Random123 / curand constants), key = (lo32 seed, hi32 seed), counter =
// (lo32 ctr, hi32 ctr, j, tag), U = out0 | out1 << 32:
//   sample_rows   tag 0, j = 0
//   spec_accept   tag 1, j = 0: the residual / bonus draw; j = 1 + i: the accept draw of draft i
//                 (P016: η_i, the test of h_i, is j = i — the base's accept draw of draft x_i; j = 0 the draw of next)
// The caller derives ctr from the sequence position and a purpose tag (a token never depends on batching); the drafts
// of one round must come from distinct (seed, ctr). The tag keeps spec_accept's draws apart from the draws that chose
// the drafts even for an equal ctr.
//
// Rows: fp32, 16-B aligned, V % 4 == 0, V < 2^20. All pointer arguments are device memory; prob may be null (top_k = 0,
// top_p = 1: no truncation). min_p / q_min_p may be null (0: off; ≤ 65536 rows / sequences); the overloads without them
// are the same calls with null — bitwise what they gave before min_p existed. The first call of the process must not be
// inside a graph capture (one-time device queries and the zeros array).
//
// Penalties (penalize_rows, kernels/sample/penalty.cu) — vLLM's sampler, per row, before temperature and truncation:
// for the row's sequence (repetition r > 0, 1 = off; frequency f, presence p ∈ [−2, 2], 0 = off), n_v = the count of v
// among the generated tokens (every emitted token, thinking included; the prompt does not count) plus the row's extra
// tokens (below), prompt_v = v occurs in the prompt:
//   l ← (prompt_v or n_v > 0) ? (l > 0 ? fl(l · fl(1 / r)) : fl(l · r)) : l      repetition (vLLM: × the reciprocal)
//   l ← n_v > 0 ? fl(fl(l − fl(f · fl(n_v))) − p) : l                             frequency, then presence
// each operation one IEEE fp32 rounding (no fused multiply-add; fl(n_v) exact below 2^24). vLLM subtracts f·0 and p·0
// from every other token too: the same values, except that a −0 logit can turn into +0 there (the same rank key and
// weight). Rows whose sequence has r == 1, f == 0 and p == 0 are not touched.
// Speculative verify: a sequence's verify rows are [last, d_1 … d_k] — row i (0-based) predicts the token after d_i,
// so its counts are the sequence's (every emitted token, `last` included) plus d_1 … d_i: the extras of row i. Draft
// rows (q) are never penalised. Deterministic: a logit's value depends only on itself, its count and its prompt bit.
// Seen tokens of a sequence (PenSeen, device memory owned by the caller, V = the full vocabulary):
//   cnt [V] uint32   bit 31: v is in the prompt · bits 0–30: v's count among the generated tokens
//   ids [V] int32    the tokens with cnt ≠ 0, each once, in no particular order: ids[0 … *n − 1]
//   n   [1] int32
// Maintained on the GPU: penalty_reset → penalty_prompt (any number of chunks) → penalty_add after every emitted
// token(s). penalize_rows reads ids once per sequence and updates the sequence's rows at those ids: O(distinct seen
// tokens × rows), independent of V.
#pragma once
#include "core/common.h"

namespace q38 {

// Per row r of logits [R, V]: token[r] (temp[r] == 0: argmax), prob[r] = P(token[r]) of that row's (truncated)
// distribution as fp32 (1 for greedy). temp, top_k, top_p, min_p, seed, ctr: [R].
void sample_rows(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                 const float* min_p, const uint64_t* seed, const uint64_t* ctr, int* token, float* prob, cudaStream_t s);
void sample_rows(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                 const uint64_t* seed, const uint64_t* ctr, int* token, float* prob, cudaStream_t s);

// B sequences; sequence b has k_b = q_start[b+1] − q_start[b] drafts drafts[q_start[b] + i] (draft logits: rows
// q_start[b] + i of q_logits) and k_b + 1 target rows p_start[b] + i of p_logits (row i: the target distribution at
// the position of draft i; row k_b: the one after the last draft). p_start, q_start: [B+1]; temp, top_k, top_p, min_p
// (target rows), q_min_p (draft rows: the min_p the drafts were drawn with), seed, ctr: [B]. Out: n_acc[b] ∈ [0, k_b]
// accepted drafts and next[b], the token that follows them (both −1 if p_start[b+1] − p_start[b] ≠ k_b + 1).
void spec_accept(const float* p_logits, const int* p_start, const float* q_logits, const int* q_start,
                 const int* drafts, int B, int V, const float* temp, const int* top_k, const float* top_p,
                 const float* min_p, const float* q_min_p, const uint64_t* seed, const uint64_t* ctr, int* n_acc,
                 int* next, cudaStream_t s);
void spec_accept(const float* p_logits, const int* p_start, const float* q_logits, const int* q_start,
                 const int* drafts, int B, int V, const float* temp, const int* top_k, const float* top_p,
                 const uint64_t* seed, const uint64_t* ctr, int* n_acc, int* next, cudaStream_t s);

// ---- penalties (kernels/sample/penalty.cu; contract above) -----------------------------------------------------------
constexpr int kPenMaxX = 64;  // extra tokens per row at most (more are not counted)
struct PenSeen {
  uint32_t* cnt;  // [V]
  int* ids;       // [V]
  int* n;         // [1]
};
struct PenSeq {   // one sequence of a penalize_rows / penalty_add call (a device array [B] of these)
  PenSeen seen;   // cnt == null: no state — penalize_rows leaves its rows alone, penalty_add skips it
  float rep, freq, pres;
};
// In place on rows [R][ld] of fp32 logits x; row r holds the logits of tokens off … off + Vb − 1 (full rows: off = 0,
// Vb = V; a rank's vocab half: off = rank·Vh, Vb = Vh — then only the listed / extra ids inside it are touched). Sequence
// b owns rows p_start[b] … p_start[b+1] − 1; row p_start[b] + i gets the extras drafts[q_start[b] … q_start[b] +
// min(i, k_b) − 1] (k_b = q_start[b+1] − q_start[b] ≤ kPenMaxX; q_start = drafts = null: no extras). p_start, q_start
// [B+1], drafts, seqs [B]: device. R = the rows of x (rows ≥ R, e.g. past p_start[B], are not touched); extra ids
// outside [0, V) count for nothing. Graph-capturable (no host sync; the list length is read on the device). Not
// idempotent: penalise each row once.
void penalize_rows(float* x, int ld, int off, int Vb, int V, int R, const int* p_start, const int* q_start,
                   const int* drafts, int B, const PenSeq* seqs, cudaStream_t s);
// The empty state (cnt = 0, n = 0): before a sequence's first penalty_prompt.
void penalty_reset(const PenSeen& st, int V, cudaStream_t s);
// Marks the prompt tokens tok[0 … L) (device; ids outside [0, V) ignored). Repeats are harmless.
void penalty_prompt(const PenSeen& st, const int* tok, int L, int V, cudaStream_t s);
// Counts generated tokens: sequence b (seqs [B], device) adds tok[t_start[b] + j] for j < n_b — n_b = min(n_tok[b],
// t_start[b+1] − t_start[b]), or t_start[b+1] − t_start[b] if n_tok is null; tok null: none — then last[b] if last is
// not null. n_b < 0 (spec_accept's −1 / kSparseDense) adds nothing for b, last included. After spec_accept:
// penalty_add(seqs, B, drafts, q_start, n_acc, next, V) — the accepted drafts and the next token.
void penalty_add(const PenSeq* seqs, int B, const int* tok, const int* t_start, const int* n_tok, const int* last, int V,
                 cudaStream_t s);

// ---- P019 sparse_logits: sampling and spec_accept from each vocab half's top candidates (kernels/sample/p019_sparse.cuh)
// lm_head is vocab-split (rank q owns ids [q·Vh, (q+1)·Vh)). Instead of gathering full rows, each rank sends its half's
// kSparseK highest-ranked tokens (rank order Z above: larger logit, ties to the lower id), and the sparse kernels give
// exactly what sample_rows / spec_accept give on the full rows — bitwise (token, prob, n_acc, next) — whenever:
//   greedy  temp ≤ 0 (or NaN): any top_k / top_p;
//   T > 0   the normalised top-k kk (trunc_params: top_p ≤ 0 → 1; else top_k if 1 ≤ top_k < V, else none) satisfies
//           1 ≤ kk ≤ kSparseK, and c = fl(log2(e) / T) is finite (T ≥ ~4.2e-39);
//   data    no record of the row carries the dense flag (a NaN or ±inf anywhere in that rank's half);
//   accept  k_b ≤ kSparseMaxK drafts (the engine's depth limit; greedy sequences: any k).
// Why it is exact: with 1 ≤ kk ≤ 64 the truncated distribution depends only on the kk highest-ranked tokens (their ids
// and logits) and the row maximum — the top one. A token among the row's top kk has fewer than kk tokens above it, so
// it is among its own half's top 64: the two lists hold them all, and merging them by Z gives their exact ranks. Every
// later quantity (weights, W_K, the top-p cut, W', the draws in index order, the accept tests, the residuals — nonzero
// only on the target's kept set — and P016's big-integer rule) is then computed from these tokens with the same
// integer formulas and the same Philox draws.
// Outside the conditions: sample_rows_sparse writes token = kSparseDense (prob 0), spec_accept_sparse n_acc = next =
// kSparseDense (−1 / −1 stays the malformed-starts code) — the caller must then use the full rows (the gather and the
// full kernels). sparse_ok() is the parameter part of the test, for the host. Switch: patches.def P019 (off until the
// engine side uses it; the kernels themselves do not read it). Deterministic and batch-invariant as the full kernels.
// min_p needs no further condition: it only removes tokens, so the kept set stays inside the top kk ≤ kSparseK the
// records hold, and M is decided by each token's own weight against the row maximum (the top candidate) — K ∩ M is the
// first |{j < kk : w_j ≥ t_mp}| ranks (a prefix: weights are non-increasing in rank order). Penalties change the logits
// before sample_cands (each rank penalises its own half), so the candidates are those of the penalised rows.
constexpr int kSparseK = 64;                // candidates per rank and row
constexpr int kSparseRec = kSparseK + 1;     // entries per record: kSparseK candidates, then the flag entry
constexpr int kSparseMaxK = 15;              // spec_accept_sparse: drafts per sequence at most (T > 0)
constexpr int kSparseDense = -2;             // marker: this row / sequence needs the full rows
struct SparseCand {
  int id;   // global token id
  float x;  // its logit
};
// Records: one per (rank, row) — entries [0, kSparseK) the half's candidates sorted from the top (Z), entry kSparseK =
// {flag, 0}: flag 1 = the half holds a NaN or ±inf (dense needed). The sparse kernels read both ranks' records of a row
// from a [2][ld][kSparseRec] buffer: rank q's record of row r at cands + (q·ld + r)·kSparseRec (ld = R for an all-gather
// of the ranks' [R][kSparseRec] blocks).

// One rank's half x [R, Vh] (Vh % 4 == 0, kSparseK ≤ Vh, ids off … off + Vh − 1, off + Vh < 2^20) → out [R][kSparseRec].
void sample_cands(const float* x, int R, int Vh, int off, SparseCand* out, cudaStream_t s);
// sample_rows on the rows' records (cands [2][ld][kSparseRec], ld ≥ R); V = the full vocabulary (top_k's range).
void sample_rows_sparse(const SparseCand* cands, int ld, int R, int V, const float* temp, const int* top_k,
                        const float* top_p, const float* min_p, const uint64_t* seed, const uint64_t* ctr, int* token,
                        float* prob, cudaStream_t s);
void sample_rows_sparse(const SparseCand* cands, int ld, int R, int V, const float* temp, const int* top_k,
                        const float* top_p, const uint64_t* seed, const uint64_t* ctr, int* token, float* prob,
                        cudaStream_t s);
// spec_accept on records: target rows p_cands [2][p_ld][kSparseRec] (row p_start[b] + i), draft rows q_cands
// [2][q_ld][kSparseRec] (row q_start[b] + i); the other arguments as spec_accept (P016 followed as spec_accept does).
void spec_accept_sparse(const SparseCand* p_cands, int p_ld, const int* p_start, const SparseCand* q_cands, int q_ld,
                        const int* q_start, const int* drafts, int B, int V, const float* temp, const int* top_k,
                        const float* top_p, const float* min_p, const float* q_min_p, const uint64_t* seed,
                        const uint64_t* ctr, int* n_acc, int* next, cudaStream_t s);
void spec_accept_sparse(const SparseCand* p_cands, int p_ld, const int* p_start, const SparseCand* q_cands, int q_ld,
                        const int* q_start, const int* drafts, int B, int V, const float* temp, const int* top_k,
                        const float* top_p, const uint64_t* seed, const uint64_t* ctr, int* n_acc, int* next,
                        cudaStream_t s);
// The parameter part of the sparse conditions (host): greedy, or T > 0 with a finite c and 1 ≤ kk ≤ kSparseK.
inline bool sparse_ok(float temp, int top_k, float top_p, int V) {
  if (!(temp > 0.f)) return true;
  const float c = 1.44269504088896341f / temp;  // IEEE fp32 division, as the kernels' temp_scale
  const int kk = top_p <= 0.f ? 1 : top_k > 0 && top_k < V ? top_k : 0;
  return c - c == 0.f && kk >= 1 && kk <= kSparseK;  // (c − c: NaN unless c is finite)
}

// ---- test hooks --------------------------------------------------------------------------------------------------
// CTAs per row cluster for later calls (0 = automatic: the rows' clusters spread over the SMs, at least 4096 logits
// per CTA, at most the device's largest cluster — 12 on GB10). Scheduling only — results never depend on it. Returns
// the previous setting.
int sample_force_cluster(int c);
// Tokens resolved in shared memory at most when finding a truncation cut (0 = the default 1024; 1 … 1024, and at most
// the kernel's threads per CTA): smaller sends more rows down the general path (more histogram passes, the draw by a
// masked pass). Scheduling only. Returns the previous setting.
int sample_force_cap(int cap);
// P016: 1 = every token with wp, wq > 0 is classified by the exact big-integer comparison instead of its 64-bit bounds
// (the slow path of real rows' near-ties). Results never depend on it. Returns the previous setting.
int sample_force_exact(int on);
// The truncation sample_rows applies to each row (temp[r] > 0): cut[2r] = z, cut[2r+1] = W' with S = {v : Z_v ≥ z}
// (z = 0, W' = W without truncation; z = 2^52, W' = 0 for a degenerate row). temp[r] ≤ 0: both 0.
void sample_cut(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                const float* min_p, uint64_t* cut, cudaStream_t s);
void sample_cut(const float* logits, int R, int V, const float* temp, const int* top_k, const float* top_p,
                uint64_t* cut, cudaStream_t s);
// min_p's premise (synchronous, < 1 s): weight(y, 0, 1) = round(ex2.approx.ftz(y)·2^44) over every fp32 y ∈ [−inf, −0]
// → the count of y whose next lower float has a larger weight (0: weights are non-decreasing in the logit); descent (if
// not null): the summed decreases, 2^44 = w(−0) − w(−inf) exactly when the walk covered the whole range.
uint64_t sample_weight_monotone(uint64_t* descent = nullptr);
// The integer weights w_v of rows [R, V] (finite rows, temp[r] > 0) → w [R, V].
void sample_weights(const float* logits, int R, int V, const float* temp, uint64_t* w, cudaStream_t s);
// P019: CTAs of sample_cands that took its slow path (survivors > 1024, θ raised by bisection) since the last reset
// (synchronous). sample_force_cluster also sets sample_cands' cluster size.
unsigned sample_cands_slow(bool reset);
// P019: the cluster size sample_cands uses for R rows of Vh (scheduling only).
int sample_cands_cluster(int Vh, int R);

}  // namespace q38
