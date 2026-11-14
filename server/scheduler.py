"""Scheduler (rank 0): the only thread that drives the engine. Requests wait for a slot; a slot keeps its sequence's
state after the request ends so the next request can continue from it (prefix reuse):
  - the slot's current history (tokens through the target) is a prefix of the new prompt and the prompt's next token
    is the reply's last one → continue from there;
  - else the longest of its checkpoints the new prompt shares → load it (positions above it are rewritten);
  - else reset. The new prompt always has ≥ 1 token to run (its last one), which gives the first token's logits.
  - A longer prefix in a checkpoint that a busy slot holds (another request with the same text, e.g. the same long
    context sent several times at once) → an emptied free slot takes it (engine ckpt_share): the recurrent state is
    copied and the caches below it are the busy slot's memory mapped again, as a prefix cache's shared blocks are
    (vLLM, SGLang) — nothing read again, no second copy of the text's caches. It needs the token after the checkpoint
    to be the prompt's (the MTP head's cache before it was written with that token; each checkpoint remembers it).
Checkpoints (copies of the recurrent state, engine ckpt_*): the slot's own buffer before the last two prompt tokens of
every prefill (the next turn: a reply sent back without its thinking renders as <think>\n\n</think>, which leaves the
generation prompt's <think>\n at its last token — one before it the next turn still has the token after; a prompt that
runs only its last token keeps it before that one); from a pool shared by all slots, while a prompt is read: the start of its last message (an
edited or replaced last message), CKPT_TAIL tokens before its end (the same text with another question at the end) and
the last chunk boundary (a prompt that leaves the text further back). A checkpoint — or the slot's history — is used
only when the prompt also has the token after it: the MTP head's cache at the position before was written with that
token (each checkpoint remembers it, the slot its reply's last token, which was not run), so a reused prompt gives the
same bits as a cold one (e.g. the same messages with the generation prompt's last token changed by thinking off). The pool is kept by recency, as a prefix cache is: with the shared KV pool a
pool checkpoint's memory is pool chunks (mapped when it is taken, given back when it is dropped), so the free pool is
its room; when room is needed — for a checkpoint or for a request's caches — the least recently used of the pool
checkpoints (saved or loaded longest ago, whichever slot holds it, busy or idle) and the idle slots (an emptied slot
takes its checkpoints along) goes first — never the boundary checkpoint a prompt being read is rewriting, nor the ones
taken or loaded in the same step. server.checkpoints is then only the most buffers there can be.
KV pool (server.kv_pool; engine kv_*): the slots' caches share one pool of 2 MiB chunks, mapped as a sequence grows.
A request is placed only when the pool can hold its prompt + min(max_tokens, server.output_reserve) on top of what the
running requests may still map within theirs; checkpoints and idle slots (least recently used first) make room, else it
waits. A reply that grows past its reservation takes more as it goes (another output_reserve at a time) and ends with
finish "length" only if the pool has none left.
Each loop iteration: at most one prefill chunk, then one speculative round over every decoding sequence
(Engine::spec_round: verify, exact accept, commit, new drafts). Prompts being read share the chunks so a long one does
not starve the others: a prompt whose rest fits in one chunk cuts in (the fewest tokens left first) — it gets its first
token after at most the chunk running now; prompts needing more chunks take turns, one chunk each (the one served
longest ago first). A long read waits only for the others' tokens. One screen line per finished request.
Draft depth: server.draft_depth drafts every round (a round ends earlier only at a thinking budget or a forced tool
call). Depth by load or by each request's acceptance gained nothing measurable (results/runs/20261112-2000-depth,
20261113-0930-decode-p021-p023) and was removed."""
import collections
import os
import queue
import time

import qlog


LOOP_SPAN, LOOP_PERIOD = 256, 8  # loop guard: the last 256 thinking tokens repeat a pattern of ≤ 8 tokens
CKPT_TAIL = 1024  # a pool checkpoint this many tokens before a prompt's end
CKPT_GAP = 512    # pool checkpoints closer than this to another one of the slot (or to the reuse start) are not taken
TOWER_SLICE_S = 0.25      # vision tower blocks per scheduler loop: about this long (replies keep flowing meanwhile)
TOWER_SLICE_MAX_S = 4.0   # … or as long as the last prefill chunk while a prompt is read (equal turns), at most this
TOWER_S_PER_ROW = 4.3e-6  # first guess of one block's time per patch row (4096² picture: 65,536 rows, 0.28 s a block)


def looping(tail):
    """Is the full tail (LOOP_SPAN tokens) periodic with a period of at most LOOP_PERIOD?"""
    if len(tail) < LOOP_SPAN:
        return False
    t = list(tail)
    return any(all(t[i] == t[i + p] for i in range(LOOP_SPAN - p)) for p in range(1, LOOP_PERIOD + 1))


class Req:
    _n = 0

    def __init__(self, prompt, max_tokens, temp, top_k, top_p, seed, push, label=""):
        Req._n += 1
        self.id, self.label = Req._n, label
        self.prompt, self.max_tokens = prompt, max_tokens
        self.temp, self.top_k, self.top_p, self.seed = temp, top_k, top_p, seed
        self.push = push  # push(tokens: list[int], finish: str | None) from the scheduler thread
        self.cancelled = False
        self.slot = None
        self.state = "wait"  # wait → prefill → decode → done
        self.pf = 0  # next prompt position to run
        self.cached = 0
        self.last = None  # last emitted token (not yet through the target)
        self.hlast = None  # the token the MTP head's cache at the last position run was written with (≠ last when a
        # round's last token was replaced afterwards: </think> forced in, tool_choice forcing)
        self.n_out = 0
        self.rounds = self.accepted = 0
        self.t0, self.t_first = time.time(), None
        self.t_place = self.t_pf = self.t_sse = None  # timing marks (app.timing)
        self.held = None  # (c0, c1): last prompt chunk whose head pass waits for the first token
        self.t_chunk = 0.0  # when this request's last prefill chunk ran (prompts needing several chunks take turns)
        self.pend = 0  # drafts pending for the next round (= the depth they were made with)
        self.verified = 0  # drafts verified over all rounds
        # thinking (set by the HTTP front for chat requests that think): budget in thinking tokens (None = none, 0 =
        # answer at once), tokens emitted while thinking, the last LOOP_SPAN of them (loop guard), why the server
        # closed the thinking (if it did)
        self.thinking, self.budget, self.n_think, self.forced = False, None, 0, None
        self.tail = collections.deque(maxlen=LOOP_SPAN)
        # tool_choice required / named (vLLM forces the answer to start with a tool call — after </think> when
        # thinking): force = the tokens to put right after </think> (thinking on), prefix_out = the same tokens already
        # appended to the prompt, emitted first (thinking off), stop_tok = end the reply after this token (named)
        self.force, self.prefix_out, self.stop_tok, self.inj = None, None, None, None
        self.min_p, self.pen = 0.0, (1.0, 0.0, 0.0)  # min_p; (repetition, frequency, presence) penalties
        self.n_prompt = len(prompt)  # the prompt without tokens forced onto its end (tool_choice, thinking off)
        # images / videos (app: vision.expand + Tower.embed): (pos3, delta, rows, embeddings) or None; key = the prompt
        # for prefix reuse with each media token replaced by a value from its file's hash (the token ids are the same
        # for any picture: without this a different picture would reuse the cached one)
        self.media, self.key = None, prompt
        self.reuse, self.ck_plan, self.ck_prog = "reset", [], None  # how the prompt was reused; pool checkpoints
        self.kv_tok = 0  # KV pool: tokens this request's slot may map (its reservation)
        self.error = None  # why the server refused it (sent to the client)

    @property
    def penalised(self):
        return self.pen != (1.0, 0.0, 0.0)

    def sampling(self):
        """spec_first's sampling arguments after top_p (positional: the command link forwards no keywords)"""
        return (self.min_p, *self.pen)


class Slot:
    def __init__(self, i):
        self.i, self.hist, self.req, self.used = i, [], None, 0.0  # hist: Req.key values of the tokens run here
        self.media = False  # the engine holds RoPE positions / media rows for this slot (set_mrope, set_media)
        self.ck = {}  # checkpoint buffer → token count (the slot's own buffer is buffer i); hist[:count] is its prefix
        self.nxt = {}  # checkpoint buffer → the token after its count (the MTP head's cache before it has it)
        self.tail = None  # free slot: the token after its history (its reply's last one, or the prompt's next one)


def _lcp(a, b):
    """length of the common prefix of two token lists (compared in blocks: list equality runs in C)"""
    n, k, B = min(len(a), len(b)), 0, 4096
    while k < n and a[k:k + B] == b[k:k + B]:
        k += B
    if k >= n:
        return n
    while k < n and a[k] == b[k]:
        k += 1
    return k


class Scheduler:
    """run() is the engine loop; call it on the thread that initialised the engine."""

    def __init__(self, eng, eos, slots, ctx, chunk, depth, think_end=None, loop_guard=True, im_start=None,
                 checkpoints=0, kv=None, output_reserve=32768, tower_jobs=None):
        self.eng, self.eos, self.ctx, self.chunk = eng, eos, ctx, chunk
        self.im_start = im_start  # <|im_start|>: the last message of a prompt starts at its second-to-last one
        self.ck_free = list(range(slots, slots + checkpoints))  # pool buffers (engine ids after the slots' own)
        self.ck_owner, self.ck_time = {}, {}  # pool buffer → slot, last use
        self.ck_evicted = 0  # pool checkpoints dropped for room (/metrics)
        self.kv, self.out_reserve = kv, output_reserve  # kv: rank 0's engine for pool queries (None: no shared pool)
        self.think_end, self.loop_guard = think_end, loop_guard  # </think> token id; close looping thinking
        self.depth = int(depth)  # MTP drafts per round
        self.last_in_chunk = eng.patch_on("last_row_in_chunk")  # P011
        self.slots = [Slot(i) for i in range(slots)]
        self.inbox = queue.Queue()
        self.waiting, self.running = [], []
        # running totals (also read by /metrics from the HTTP thread: plain int updates, no lock needed)
        self.stats = {"requests": 0, "prompt": 0, "cached": 0, "out": 0, "drafts": 0, "draft_tokens": 0,
                      "accepted": 0}
        # per (sequences, depth) of a round: [rounds, tokens out of spec_round, ms inside spec_round, ms from this round's
        # start to the next one's (only when no prefill chunk ran in between), rounds counted in that wall total]
        self.rstats = {}
        self._prev = None  # (key, start) of the last round, None after a prefill chunk
        self.draining, self.deadline, self.error = False, None, None
        # vision tower items (media.TowerJobs, rank 0) run here in slices, in the engine's call order (_tower); with a
        # prompt being read the two take turns, a slice as long as the last chunk
        self.jobs, self.tower_turn, self.chunk_s = tower_jobs, False, 0.0
        if tower_jobs is not None:
            tower_jobs.wake = lambda: self.inbox.put(None)

    def submit(self, r):
        self.inbox.put(r)

    # ---- stopping (called from other threads) ------------------------------------------------------------------
    def drain(self, deadline):
        """Stop when every request has ended; at `deadline` (time.time()) the rest are cut (finish as aborted). run()
        returns then. The loop is woken from a helper thread (a queue put must not run inside a signal handler)."""
        self.deadline = deadline
        self.draining = True
        self.inbox.put(None)

    def fail_all(self, msg):
        """The engine cannot go on (the other rank is gone): every request ends with an error."""
        self.error = msg
        if self.jobs is not None:
            self.jobs.fail_all(msg)
        for r in list(self.waiting) + list(self.running):
            r.push([], "error")
        while True:
            try:
                r = self.inbox.get_nowait()
            except queue.Empty:
                break
            if r is not None:
                r.push([], "error")

    def busy(self):
        return len(self.waiting) + len(self.running)

    def counts(self):
        return len(self.running), len(self.waiting) + self.inbox.qsize()

    # ---- slot choice and prefix reuse -------------------------------------------------------------------------
    def _place(self, r):
        free = [s for s in self.slots if s.req is None]
        if not free:
            return False
        L = len(r.prompt)
        best, how, n = None, "reset", 0
        for s in free:
            m = _lcp(s.hist, r.key)
            if m == len(s.hist) and m < L and m > n and s.tail is not None and r.key[m] == s.tail:
                best, how, n = s, "cur", m
            for cid, c in s.ck.items():
                # its tokens the prompt's, and the token after it too (see the module doc)
                if c <= m and c < L and c > n and s.nxt.get(cid) == r.key[c]:
                    best, how, n = s, cid, c
        src = None  # a longer prefix in a busy slot's checkpoint (its history up to it the prompt's, the token after too)
        if self.kv is not None:
            for d in self.slots:
                if d.req is None or not d.ck:
                    continue
                m = None
                for cid, c in d.ck.items():
                    if c <= n or c >= L or d.nxt.get(cid) != r.key[c]:
                        continue
                    if m is None:
                        m = _lcp(d.hist, r.key)
                    if c <= m:
                        src, n = (d, cid), c
            if src is not None:
                best, how = None, "share"
        if best is None:
            best = min(free, key=lambda s: s.used)
        s = best
        if self.kv is not None:
            T = min(self.ctx, L + min(r.max_tokens, self.out_reserve) + 2 * self.depth + 1)
            if self.kv.kv_need(L + 2 * self.depth + 1) > self.kv.kv_total():
                r.error = (f"the prompt ({L:,} tokens) does not fit in the server's cache memory "
                           f"({self.kv.kv_total() * 2 // 1024:,} GiB)")
                r.state = "done"
                r.push([], "error")
                return None
            while self.kv.kv_need(T) > self.kv.kv_total():
                T -= 1024
            if how in ("reset", "share"):  # the slot is emptied: what only it maps comes back
                held, extra = (self.kv.kv_full(n) if how == "share" else 0), self.kv.kv_sole(s.i)
                avoid = [src[1]] if how == "share" else []
            else:
                held, extra, avoid = self.kv.kv_held(s.i), 0, [how] if isinstance(how, int) else []
                if how != "cur" and self.kv.kv_sole(s.i) < held:  # going back below shared caches: copies
                    held -= self.kv.kv_need(0)
            if not self._kv_room(max(0, self.kv.kv_need(T) - held), s, extra, avoid):
                return False
            r.kv_tok = T
        if how in ("reset", "share"):
            self.eng.reset(s.i)
            self._ck_keep(s, -1)
            s.hist = []
        if how == "share":
            d, cid = src
            got = self._kv_try(self.eng.ckpt_share, s.i, cid)
            if got == n:
                s.hist = d.hist[:n]
                if cid in self.ck_time:
                    self.ck_time[cid] = time.time()
            else:  # the pool had no room for the copies (both ranks threw before any change): read it all
                how, n = "reset", 0
        elif how != "reset" and how != "cur":
            got = self._kv_try(self.eng.ckpt_load, s.i, how)
            if got is None:  # no room for the copies of shared caches it goes back below: read it all
                self.eng.reset(s.i)
                self._ck_keep(s, -1)
                s.hist = []
                how, n = "reset", 0
            else:
                assert got == n, (got, n)
                s.hist = s.hist[:n]
                self._ck_keep(s, n)  # the ones above n are about to be overwritten
                if how in self.ck_time:
                    self.ck_time[how] = time.time()
        r.reuse = (how if how in ("cur", "reset") else "shared" if how == "share" else
                   "own" if how == s.i else "checkpoint")
        if r.media is not None or s.media:  # the request's RoPE positions and media rows (or back to plain)
            pos3, delta, rows, emb = r.media or ([], 0, [], None)
            self.eng.set_mrope(s.i, pos3, delta)
            self.eng.set_media(s.i, rows)
            if emb is not None:  # rank 0 only: the tensor stays here
                getattr(self.eng, "eng", self.eng).media_src(s.i, emb)
            s.media = r.media is not None
        s.req, s.tail, r.slot, r.state, r.pf, r.cached = s, None, s, "prefill", n, n
        r.t_place = time.time()
        r.ck_plan, r.ck_prog = self._ck_plan(r, s), None
        if r.penalised:  # the seen-token state: the prompt (forced tokens at its end count as generated ones)
            self.eng.pen_begin(s.i, r.prompt[:r.n_prompt])
            if r.n_prompt < len(r.prompt):
                self.eng.pen_add(s.i, r.prompt[r.n_prompt:])
        return True

    @staticmethod
    def _kv_try(fn, *a):
        """fn(*a), or None when the engine's KvFull came (thrown before anything changed, on both ranks)"""
        try:
            return fn(*a)
        except RuntimeError as e:
            if not str(e).startswith("KV pool full"):
                raise
            return None

    # ---- KV pool ---------------------------------------------------------------------------------------------------
    def _kv_promised(self):
        """chunks the running requests may still map within their reservations"""
        return sum(max(0, self.kv.kv_need(x.kv_tok) - self.kv.kv_held(x.slot.i)) for x in self.running)

    def _kv_room(self, need, keep, extra=0, avoid=()):
        """`need` chunks free beyond the running requests' promises (+ `extra` about to be freed): pool checkpoints and
        idle slots other than `keep` make room, least recently used first (a checkpoint by its last save / load, a slot
        by its last request; an emptied slot takes its checkpoints along) — not the boundary checkpoints of prompts
        being read nor `avoid` → whether it fits"""
        while self.kv.kv_free() + extra - self._kv_promised() < need:
            busy = {x.ck_prog for x in self.running if x.state == "prefill"} | set(avoid)
            cands = [(self.ck_time[c], 0, c) for c in self.ck_owner if c not in busy]
            # an idle slot whose caches all are another's too frees nothing by itself: after the others
            cands += [(s.used if self.kv.kv_sole(s.i) else s.used + 1e12, 1, s.i) for s in self.slots
                      if s.req is None and s is not keep and self.kv.kv_held(s.i) > 0]
            if not cands:
                return False
            _, kind, x = min(cands)
            if kind == 0:
                del self.slots[self.ck_owner[x]].ck[x]
                self._ck_free(x)
                self.ck_evicted += 1
            else:
                s = self.slots[x]
                self.eng.reset(s.i)
                self._ck_keep(s, -1)
                s.hist = []
        return True

    def _kv_grow(self, rs, ks):
        """before a round: a reply about to pass its reservation takes another output_reserve tokens, or ends (length)
        → the requests that go on"""
        if self.kv is None or os.environ.get("Q38_TEST_KV_NOGROW") == "1":  # test: let the engine meet a full pool
            return rs, ks
        keep = []
        for r, k in zip(rs, ks):
            want = len(r.slot.hist) + 2 * k + 2
            if want > r.kv_tok:
                T = min(self.ctx, max(want, r.kv_tok + self.out_reserve))
                while T > want and self.kv.kv_need(T) > self.kv.kv_total():
                    T -= 1024
                r.kv_tok = 0  # out of the promises: it asks for all it still has to map
                if not self._kv_room(max(0, self.kv.kv_need(T) - self.kv.kv_held(r.slot.i)), r.slot):
                    qlog.warn("req", f"#{r.id}: the cache memory is full — reply ends at {r.n_out:,} tokens")
                    r.push([], "length")
                    self._finish(r, "length")
                    continue
                r.kv_tok = T
            keep.append((r, k))
        return [r for r, _ in keep], [k for _, k in keep]

    def _kv_guard(self, fn, arg):
        """run fn(arg); the engine's KvFull (a call needed chunks the pool did not have — thrown before the call changed
        anything, on both ranks) ends that slot's request (length if it has output, else an error) and empties the
        slot; the others go on"""
        try:
            fn(arg)
        except RuntimeError as e:
            msg = str(e)
            if not msg.startswith("KV pool full"):
                raise
            i = int(msg.split("slot ", 1)[1].split()[0])
            r = next((x for x in self.running if x.slot is not None and x.slot.i == i), None)
            qlog.warn("req", f"#{r.id if r else '?'}: {msg} — the request ends, the server goes on")
            if r is not None:
                fin = "length" if r.n_out else "error"
                r.error = "the server's cache memory is full"
                r.push([], fin)
                self._finish(r, fin)
            self.eng.reset(i)
            self._ck_keep(self.slots[i], -1)
            self.slots[i].hist = []

    # ---- checkpoints -------------------------------------------------------------------------------------------
    def _ck_keep(self, s, n):
        """drop the slot's checkpoints above n tokens (n < 0: all); pool buffers go back to the pool"""
        for cid, c in list(s.ck.items()):
            if n < 0 or c > n:
                del s.ck[cid]
                if cid != s.i:
                    self._ck_free(cid)

    def _ck_free(self, cid):
        self.eng.ckpt_drop(cid)
        self.ck_owner.pop(cid, None)
        self.ck_time.pop(cid, None)
        self.ck_free.append(cid)

    def _ck_alloc(self, s, avoid=()):
        """a pool buffer for slot s: a free one, else the least recently used one of the pool whoever holds it (a busy
        slot's checkpoints serve a later request as an idle slot's do), except the boundary buffers of the prompts being
        read (rewritten every chunk) and `avoid` (taken for the same step); with the shared KV pool also its memory
        (_kv_room) → its id (None: none)"""
        keep = {x.ck_prog for x in self.running if x.state == "prefill"} | set(avoid)
        if not self.ck_free:
            old = [c for c in self.ck_owner if c not in keep]
            if not old:
                return None
            cid = min(old, key=lambda c: self.ck_time[c])
            del self.slots[self.ck_owner[cid]].ck[cid]
            self._ck_free(cid)
            self.ck_evicted += 1
        cid = self.ck_free.pop()
        if self.kv is not None and not self._kv_room(self.kv.ckpt_chunks(), s, 0, keep | {cid}):
            self.ck_free.append(cid)
            return None
        self.ck_owner[cid], self.ck_time[cid] = s.i, time.time()
        return cid

    def ck_status(self):
        """(read from the HTTP thread, /metrics) the slots and the pool checkpoints they hold: [(slot, busy, tokens of its
        history, [(buffer, tokens, seconds since its last use), ...])], free pool buffers, checkpoints dropped for room"""
        now, out = time.time(), []
        for s in self.slots:
            ck = sorted((c, n, now - self.ck_time.get(c, now)) for c, n in list(s.ck.items()) if c != s.i)
            out.append((s.i, s.req is not None, len(s.hist), ck))
        return out, len(self.ck_free), self.ck_evicted

    def _ck_plan(self, r, s):
        """token counts at which this prefill takes pool checkpoints: the last message's start, CKPT_TAIL before the
        end — each at least CKPT_GAP past the reuse start, before the last token and from the slot's others"""
        if not self.ck_free and not self.ck_owner:
            return []
        P, L = r.key, len(r.prompt)
        cand = [L - CKPT_TAIL]
        if self.im_start is not None:  # the second-to-last <|im_start|> after the reuse start (list.index: C speed)
            rev = P[r.pf:][::-1]
            try:
                j = rev.index(self.im_start, rev.index(self.im_start) + 1)
                cand.append(L - 1 - j)
            except ValueError:
                pass
        have, want = list(s.ck.values()) + [r.pf, L - 1], []
        for d in sorted(cand):
            if r.pf + CKPT_GAP <= d <= L - 1 - CKPT_GAP and all(abs(d - h) >= CKPT_GAP for h in have + want):
                want.append(d)
        return want

    def _ck_snaps(self, r, c0, c1):
        """(row, buffer) of this chunk's planned checkpoints (rows c0 … c1 − 1), registered with the slot"""
        out = []
        for d in r.ck_plan:
            if c0 <= d < c1:
                cid = self._ck_alloc(r.slot, [c for _, c in out])
                if cid is not None:
                    out.append((d - c0, cid))
                    r.slot.ck[cid], r.slot.nxt[cid] = d, r.key[d]
        return out

    def _ck_progress(self, r, c1):
        """the prefill's last chunk boundary (one pool buffer per request, rewritten at every boundary)"""
        s = r.slot
        if r.ck_prog is None:
            if any(abs(c1 - h) < CKPT_GAP for h in s.ck.values()) or c1 < r.cached + CKPT_GAP:
                return
            r.ck_prog = self._ck_alloc(s)
            if r.ck_prog is None:
                return
        self.eng.ckpt_save(s.i, r.ck_prog)
        s.ck[r.ck_prog], s.nxt[r.ck_prog] = c1, r.key[c1]
        self.ck_time[r.ck_prog] = time.time()

    # ---- vision tower ----------------------------------------------------------------------------------------------
    def _tower(self, reading):
        """one slice of the head tower item on both ranks (tower_* through the command link): its input first, then
        blocks for about TOWER_SLICE_S (or the last prefill chunk's time while a prompt is being read, so the two take
        equal turns), the merger after the last block — rank 0's TowerJobs then hands the embeddings over"""
        j = self.jobs.head()
        t0 = time.perf_counter()
        if j.k < 0:
            self.eng.tower_begin(j.id, j.grids, j.shape)
            j.k, j.n = 0, max(1, min(self.jobs.depth, int(TOWER_SLICE_S / (j.L * TOWER_S_PER_ROW))))
            t0 = time.perf_counter()
        if j.k < self.jobs.depth:
            k1 = min(self.jobs.depth, j.k + j.n)
            self.eng.tower_blocks(j.k, k1)
            dt = max(time.perf_counter() - t0, 1e-3)
            want = max(TOWER_SLICE_S, min(TOWER_SLICE_MAX_S, self.chunk_s)) if reading else TOWER_SLICE_S
            j.n = max(1, min(self.jobs.depth, round((k1 - j.k) * want / dt)))
            j.k = k1
        if j.k >= self.jobs.depth:
            self.eng.tower_end(j.id)

    # ---- prefill ---------------------------------------------------------------------------------------------
    def _next_prefill(self, pf):
        """the prompt that reads the next chunk: one whose last chunk's head pass waits (the path without P011 keeps
        that hidden in the engine until its next call); else a prompt whose rest fits in one chunk, the fewest tokens
        left first; else the one needing more chunks served longest ago (turns)"""
        held = [r for r in pf if r.held]
        if held:
            return held[0]
        short = [r for r in pf if len(r.prompt) - r.pf <= self.chunk]
        if short:
            return min(short, key=lambda r: (len(r.prompt) - r.pf, r.t_place))
        return min(pf, key=lambda r: (r.t_chunk, r.t_place))

    def _prefill_chunk(self, r):
        if self.last_in_chunk:
            return self._prefill_chunk_p011(r)
        s, P, L = r.slot.i, r.prompt, len(r.prompt)
        if r.pf < L - 1:  # rows before the last prompt token, with the head over them (next token known)
            c0 = r.pf
            c1 = min(c0 + self.chunk, L - 1)
            n = c1 - c0
            self.eng.step(P[c0:c1], list(range(c0, c1)), [s], [0, n], False, -1, self._ck_snaps(r, c0, c1))
            if c1 < L - 1:
                self.eng.mtp_step(P[c0 + 1:c1 + 1], list(range(c0, c1)), [s], [0, n], list(range(n)), True, False)
            else:  # the head over this last chunk waits until the first token is out (its hidden kept meanwhile)
                self.eng.hold_hidden(n)
                r.held = (c0, c1)
            r.slot.hist += r.key[c0:c1]
            r.pf = c1
            if c1 < L - 1:
                self._ck_progress(r, c1)
            return
        # snapshot before the last token (the next turn shares at least this prefix), then the last token and the first
        # draw out to the client; then the deferred head pass and the drafts — spec_begin's work, reordered (same bits).
        # Always run, even when the request ended at its first token: the slot's head cache serves the next turn.
        r.t_pf = time.time()
        self.eng.snapshot(s)
        r.slot.ck[s], r.slot.nxt[s] = len(r.slot.hist), r.key[L - 1]
        self.eng.step([P[L - 1]], [L - 1], [s], [0, 1], False)
        r.slot.hist.append(r.key[L - 1])
        t = self.eng.spec_first(s, L, self._first_depth(r), r.temp, r.seed, 0, r.top_k, r.top_p, *r.sampling())
        t = self._think(r, [t])[0]
        r.state, r.last, r.t_first = "decode", t, time.time()
        self._emit(r, [t])
        self._after_think(r, [t])
        if r.held:
            c0, c1 = r.held
            n = c1 - c0
            self.eng.mtp_step_held(P[c0 + 1:c1 + 1], list(range(c0, c1)), [s], [0, n], list(range(n)))
            r.held = None
        self.eng.spec_rest(s)
        r.hlast = r.last
        self._media_done(r)

    def _prefill_chunk_p011(self, r):
        """P011 last_row_in_chunk: the last prompt token runs inside the last chunk (no separate one-row step); that step
        takes the slot snapshot before it. Then the first token out, the head over the chunk's other rows (their hidden
        is still the step's), the head over the last row and the drafts — the same values as the path above."""
        s, P, L = r.slot.i, r.prompt, len(r.prompt)
        c0 = r.pf
        c1 = min(c0 + self.chunk, L)
        n = c1 - c0
        if c1 < L:  # an earlier chunk: target rows, then the head over them (next token known)
            self.eng.step(P[c0:c1], list(range(c0, c1)), [s], [0, n], False, -1, self._ck_snaps(r, c0, c1))
            self.eng.mtp_step(P[c0 + 1:c1 + 1], list(range(c0, c1)), [s], [0, n], list(range(n)), True, False)
            r.slot.hist += r.key[c0:c1]
            r.pf = c1
            self._ck_progress(r, c1)
            return
        r.t_pf = time.time()
        sr = max(0, n - 2)  # the slot's own checkpoint before the last two tokens (one when only the last runs)
        self.eng.step(P[c0:L], list(range(c0, L)), [s], [0, n], False, sr, self._ck_snaps(r, c0, L))
        r.slot.ck[s], r.slot.nxt[s] = c0 + sr, r.key[c0 + sr]
        r.slot.hist += r.key[c0:L]
        r.pf = L
        t = self.eng.spec_first(s, L, self._first_depth(r), r.temp, r.seed, 0, r.top_k, r.top_p, *r.sampling())
        t = self._think(r, [t])[0]
        r.state, r.last, r.t_first = "decode", t, time.time()
        self._emit(r, [t])
        self._after_think(r, [t])
        if n > 1:
            self.eng.mtp_step(P[c0 + 1:L], list(range(c0, L - 1)), [s], [0, n - 1], list(range(n - 1)), True, False)
        self.eng.spec_rest(s, n - 1)
        r.hlast = r.last
        self._media_done(r)

    def _media_done(self, r):
        """The prompt and the head passes over it have run: free the media embeddings (the binding waits for the engine
        stream before it drops them). The RoPE positions stay (the reply rotates by position + delta)."""
        if r.media is not None and r.media[2]:
            self.eng.set_media(r.slot.i, [])
            r.media = (r.media[0], r.media[1], [], None)

    # ---- decode ----------------------------------------------------------------------------------------------
    def _first_depth(self, r):
        """depth of the drafts spec_first leaves (the request's first round)"""
        r.pend = self.depth
        return r.pend

    def _think(self, r, toks):
        """Thinking budget and loop guard on a round's tokens (before they go out). The round's last token has not run
        through the target yet (it is the next round's input), so it can be replaced by </think> without touching any
        state — the pending drafts stay valid proposals (the accept test is exact for any proposal distribution)."""
        if not r.thinking or self.think_end is None:
            return toks
        if self.think_end in toks:
            r.n_think += toks.index(self.think_end)
            r.thinking = False
            return toks
        r.n_think += len(toks)
        r.tail.extend(toks)
        why = ("budget" if r.budget is not None and r.n_think >= r.budget else
               "loop" if self.loop_guard and looping(r.tail) else None)
        if why and toks[-1] not in self.eos:
            toks = toks[:-1] + [self.think_end]
            self.eng.spec_last(r.slot.i, self.think_end)
            r.thinking, r.forced = False, why
        return toks

    def _after_think(self, r, toks):
        """tool_choice forcing: once </think> is out, the forced tokens follow it at once (vLLM's grammar starts right
        after </think>). The round's last token is the only one not yet through the target: </think> as the last token
        → inject all; </think> just before it → the last token becomes force[0], inject the rest (rounds of such
        requests are cut at a pending </think> draft, so </think> is never deeper in a round)."""
        if not r.force or r.thinking or r.state != "decode" or self.think_end not in toks:
            return toks
        j, T = toks.index(self.think_end), r.force
        r.force = None
        if j == len(toks) - 1:
            r.state, r.inj = "inject", list(T)
        elif j == len(toks) - 2:
            self.eng.spec_last(r.slot.i, T[0])
            toks[-1] = r.last = T[0]
            if len(T) > 1:
                r.state, r.inj = "inject", list(T[1:])
        else:
            qlog.warn("req", f"#{r.id}: </think> inside a round — tool_choice not forced for this reply")
        return toks

    def _inject(self, r):
        """Forced tokens T after the current last token: they run like a prompt chunk — the target over [last] +
        T[:-1], the first-token draw (replaced by T[-1], so the sampler state and drafts start from it), the head over
        the rows, new drafts — then decoding goes on. Exact: nothing sampled is changed except the forced positions."""
        s, T = r.slot.i, r.inj
        p = len(r.slot.hist)  # position of r.last
        ids = [r.last] + T[:-1]
        n = len(ids)
        self.eng.step(ids, list(range(p, p + n)), [s], [0, n], False)
        r.slot.hist += ids
        if r.penalised:  # the old last and the forced tokens before the new last are generated tokens
            self.eng.pen_add(s, ids)
        self.eng.spec_first(s, p + n, self._first_depth(r), r.temp, r.seed, 0, r.top_k, r.top_p, *r.sampling())
        self.eng.spec_last(s, T[-1])
        if n > 1:
            self.eng.mtp_step(T[:-1], list(range(p, p + n - 1)), [s], [0, n - 1], list(range(n - 1)), True, False)
        self.eng.spec_rest(s, n - 1)
        r.last, r.hlast, r.state, r.inj = T[-1], T[-1], "decode", None
        self._emit(r, list(T))

    def _round(self, rs):
        ks = [self.depth] * len(rs)
        # tool_choice forcing while thinking: verify no draft past a pending </think> (it then ends the round)
        for i, r in enumerate(rs):
            if r.force and r.thinking:
                d = self.eng.spec_drafts(r.slot.i)[:ks[i]]
                if self.think_end in d:
                    ks[i] = d.index(self.think_end) + 1
        # a thinking budget: no round runs past it, so the round that reaches it ends there (its last token is replaced)
        ks = [min(k, max(1, r.budget - r.n_think)) if r.thinking and r.budget is not None else k for r, k in zip(rs, ks)]
        rs, ks = self._kv_grow(rs, ks)
        if not rs:
            return
        k = max(ks)
        t0 = time.perf_counter()
        if self._prev is not None:
            x = self.rstats[self._prev[0]]
            x[3] += 1e3 * (t0 - self._prev[1])
            x[4] += 1
        for r, kr in zip(rs, ks):
            if r.pend != kr:
                self.eng.spec_depth(r.slot.i, kr)
        res = self.eng.spec_round([r.slot.i for r in rs])
        key = (len(rs), k)
        x = self.rstats.setdefault(key, [0, 0, 0.0, 0.0, 0])
        x[0] += 1
        x[1] += sum(len(t) for t in res)
        x[2] += 1e3 * (time.perf_counter() - t0)
        self._prev = (key, t0)
        st = self.stats
        st["drafts"] += len(rs)
        st["accepted"] += sum(len(t) - 1 for t in res)
        for r, kr, toks in zip(rs, ks, res):
            v = min(kr, r.pend)  # drafts this round verified (Engine::spec_round)
            st["draft_tokens"] += v
            r.verified += v
            r.pend = kr
            r.slot.hist += [r.last] + toks[:-1]  # kept verify rows: the old last token and the accepted drafts
            r.hlast = toks[-1]  # the round's head pass ran with it (a replacement below does not rewrite that)
            toks = self._think(r, toks)
            r.last = toks[-1]
            toks = self._after_think(r, toks)
            r.rounds += 1
            r.accepted += len(toks) - 1
            if r.state in ("decode", "inject"):
                self._emit(r, toks)

    def _emit(self, r, toks):
        if r.prefix_out:  # tool_choice forcing, thinking off: the forced tokens ended the prompt; they go out first
            toks, r.prefix_out = list(r.prefix_out) + list(toks), None
        out, fin = [], None
        for t in toks:
            if t in self.eos:  # not sent, but a generated token (usage counts it, as vLLM)
                r.n_out += 1
                fin = "stop"
                break
            out.append(t)
            r.n_out += 1
            if t == r.stop_tok:  # named tool_choice: one call, nothing after it
                fin = "stop"
                break
            if r.n_out >= r.max_tokens:
                fin = "length"
                break
        r.push(out, fin)
        if fin:
            self._finish(r, fin)

    def _finish(self, r, why):
        r.state = "done"
        if r in self.running:
            self.running.remove(r)
        r.slot.req, r.slot.used = None, time.time()
        # the token the MTP head's cache at the history's last position was written with: the reply's last one (not
        # run), or a prompt cut off mid-read: its next token (the head over each chunk has it); unknown otherwise
        h = len(r.slot.hist)
        r.slot.tail = (r.hlast if r.t_first is not None else
                       r.key[h] if r.pf > r.cached and h == r.pf and h < len(r.key) else None)
        dt = time.time() - (r.t_first or time.time())
        tps = (r.n_out - 1) / dt if r.n_out > 1 and dt > 0 else 0.0
        st = self.stats
        st["requests"] += 1
        st["prompt"] += len(r.prompt)
        st["cached"] += r.cached
        st["out"] += r.n_out
        ttft = (r.t_first - r.t0) if r.t_first else float("nan")
        via = f" {r.reuse}" if r.cached else ""
        line = (f"#{r.id}{' ' + r.label if r.label else ''} · prompt {len(r.prompt):,} (reused {r.cached:,}{via}) · "
                f"first token {ttft:.2f} s · output {r.n_out:,} · {tps:.1f} tok/s · "
                f"{(r.accepted + r.rounds) / max(r.rounds, 1):.2f} tokens/round · depth {r.verified / max(r.rounds, 1):.1f} · {why}"
                + (f" · thinking closed ({r.forced})" if r.forced else ""))
        (qlog.ok if why in ("stop", "length") else qlog.warn)("req", line)

    # ---- loop ------------------------------------------------------------------------------------------------
    def run(self):
        """The engine loop. Returns after drain() once no request is left (or the deadline cut them)."""
        while True:
            try:
                block = not (self.waiting or self.running or self.draining or (self.jobs and self.jobs.head()))
                while True:
                    r = self.inbox.get(block=block, timeout=None)
                    if r is not None:  # None: a wake-up (drain)
                        self.waiting.append(r)
                    block = False
            except queue.Empty:
                pass
            if self.draining and time.time() >= self.deadline:
                if self.jobs is not None:
                    self.jobs.fail_all("the server is stopping")
                for r in list(self.waiting) + list(self.running):
                    r.cancelled = True  # cut below, like a client that went away (finish: abort)
            for r in [r for r in self.waiting + self.running if r.cancelled and r.state != "done"]:
                if r in self.waiting:
                    self.waiting.remove(r)
                    r.state = "done"
                    r.push([], "abort")
                else:
                    r.push([], "abort")
                    self._finish(r, "abort")
            if self.draining and not (self.waiting or self.running):
                if self.jobs is not None:
                    self.jobs.fail_all("the server is stopping")
                return
            while self.waiting:
                got = self._place(self.waiting[0])
                if got is None:  # refused (it got its error)
                    self.waiting.pop(0)
                    continue
                if not got:
                    break
                self.running.append(self.waiting.pop(0))
            pf = [r for r in self.running if r.state == "prefill"]
            if self.jobs is not None and self.jobs.head() and (not pf or self.tower_turn):
                self._tower(bool(pf))
                self.tower_turn = False
                self._prev = None
            elif pf:
                r = self._next_prefill(pf)
                r.t_chunk = time.time()
                self._kv_guard(self._prefill_chunk, r)
                self.chunk_s = time.time() - r.t_chunk
                self.tower_turn = True
                self._prev = None
            for r in [r for r in self.running if r.state == "inject"]:
                self._kv_guard(self._inject, r)
                self._prev = None
            dec = [r for r in self.running if r.state == "decode"]
            if dec:
                self._kv_guard(self._round, dec)
