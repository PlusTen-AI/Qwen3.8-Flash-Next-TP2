"""Scheduler policy with a fake engine (no GPU, no model; plain Python): which prompt reads the next prefill chunk
(a short prompt cuts in, long ones take turns) and the shared checkpoint pool kept by recency (a busy slot's checkpoint
makes room — the soak's finding 2: a long text asked again with another question while agent slots were busy came back
cold). Run: python3 server/check_sched.py"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "tools")]
import scheduler as S  # noqa: E402

EOS = 2


class FakeEngine:
    """records prefill steps (slot, first position, rows); a request decodes `marker - 1000` rounds (its prompt's first
    token is the marker), then EOS; checkpoints remember their token counts. pool=N: a shared KV pool of N chunks (a
    slot maps tokens // 100 + 1 chunks as it grows, a pool checkpoint CK chunks while it holds one)"""
    CK = 28

    def __init__(self, pool=0, slots=4):
        self.steps, self.rounds, self.ck, self.pos = [], {}, {}, {}
        self.total, self.slots, self.held, self.mapped = pool, slots, {}, set()

    # ---- the KV pool (rank 0's queries) ----
    def kv_need(self, n):
        return n // 100 + 1

    def kv_total(self):
        return self.total

    def kv_held(self, s):
        return self.held.get(s, 0)

    def kv_free(self):
        return self.total - sum(self.held.values()) - self.CK * len(self.mapped)

    def ckpt_chunks(self):
        return self.CK

    def _map_ck(self, cid):
        if self.total and cid >= self.slots and cid not in self.mapped:
            assert self.kv_free() >= self.CK, "the server took a checkpoint without room"
            self.mapped.add(cid)

    def _grow(self, s, n):
        if self.total:
            need = self.kv_need(n)
            assert need - self.held.get(s, 0) <= self.kv_free(), "the server grew a slot without room"
            self.held[s] = max(self.held.get(s, 0), need)

    def reset(self, s):
        self.held[s] = 0
        for c in [c for c, v in self.ck.items() if v[1] == s]:
            self.mapped.discard(c)
            del self.ck[c]

    def ckpt_drop(self, cid):
        self.mapped.discard(cid)
        self.ck.pop(cid, None)

    def patch_on(self, name):
        return True  # P011 last_row_in_chunk (the default)

    def step(self, ids, pos, slots, cu, *a):
        s = slots[0]
        if pos[0] == 0:
            self.rounds[s] = max(0, ids[0] - 1000)
        self._grow(s, pos[-1] + 1)
        self.steps.append((s, pos[0], len(ids)))
        self.pos[s] = pos[-1] + 1
        time.sleep(0.005)  # a chunk takes time: requests arrive while a long read is going
        snaps = a[2] if len(a) > 2 else []
        for row, cid in snaps:
            self._map_ck(cid)
            self.ck[cid] = (pos[0] + row, s)

    def ckpt_save(self, s, cid):
        self._map_ck(cid)
        self.ck[cid] = (self.pos[s], s)

    def ckpt_load(self, s, cid):
        self.pos[s] = self.ck[cid][0]
        return self.ck[cid][0]

    def snapshot(self, s):
        pass

    def spec_first(self, s, *a):
        return EOS if self.rounds.get(s, 0) == 0 else 7

    def spec_round(self, slots):
        out = []
        for s in slots:
            self._grow(s, self.pos[s] + 8)
            self.pos[s] += 1
            self.rounds[s] = self.rounds.get(s, 0) - 1
            out.append([EOS] if self.rounds[s] <= 0 else [7])
            time.sleep(0.002)
        return out

    def __getattr__(self, name):  # reset, ckpt_drop, mtp_step, spec_rest, spec_depth, set_mrope, ...: nothing to do
        return lambda *a, **k: None


def serve(sched):
    t = threading.Thread(target=sched.run, daemon=True)
    t.start()
    return t


def req(prompt, label):
    done = threading.Event()

    def push(toks, fin):
        if fin:
            done.set()
    r = S.Req(prompt, 64, 1.0, 20, 0.95, 0, push, label)
    r.done = done
    return r


def text(marker, n, seed):
    return [marker] + [10 + (seed * 7919 + i * 31) % 5000 for i in range(n - 1)]


bad = 0


def check(ok, what):
    global bad
    bad += not ok
    print(("ok   " if ok else "FAIL ") + what)


# 1. prefill order: a long read (10 chunks) and a short prompt arriving after its first chunk; then two long ones
eng = FakeEngine()
sc = S.Scheduler(eng, {EOS}, 4, 1 << 20, 1024, "4")
long_a = req(text(1000, 10 * 1024, 1), "long")
sc.submit(long_a)
th = serve(sc)
while not any(s == 0 for s, _, _ in eng.steps):
    time.sleep(0.001)
short = req(text(1000, 300, 2), "short")
sc.submit(short)
short.done.wait(5)
long_a.done.wait(5)
order = [s for s, _, _ in eng.steps]
k = order.index(1)
check(short.done.is_set() and k <= 2, f"short prompt cut in after ≤ 2 long chunks (after {k}), order {order[:6]}…")
eng.steps.clear()
a, b = req(text(1000, 4 * 1024, 3), "a"), req(text(1000, 4 * 1024, 4), "b")
sc.submit(a)
sc.submit(b)
a.done.wait(5)
b.done.wait(5)
order = [s for s, _, _ in eng.steps]
sa, sb = a.slot.i, b.slot.i  # a's last chunk fits in one chunk while b still needs two: a cuts in
check(order == [sa, sb, sa, sb, sa, sa, sb, sb], f"two long reads take turns, a rest that fits in one chunk cuts in: {order}")
sc.drain(time.time() + 5)
th.join(5)

# 2. checkpoints kept by recency: 2 pool buffers both held by a busy slot; a long text read on another slot still gets
# its tail checkpoint (the busy slot's oldest makes room), so the same text with another question reuses it
for fixed in (True, False):
    eng = FakeEngine()
    sc = S.Scheduler(eng, {EOS}, 4, 1 << 20, 1024, "4", checkpoints=2)
    if not fixed:  # the old rule: only idle slots' checkpoints make room
        def old_alloc(s, avoid=(), sc=sc):
            if not sc.ck_free:
                idle = [c for c, o in sc.ck_owner.items() if sc.slots[o].req is None]
                if not idle:
                    return None
                cid = min(idle, key=lambda c: sc.ck_time[c])
                del sc.slots[sc.ck_owner[cid]].ck[cid]
                sc._ck_free(cid)
            cid = sc.ck_free.pop()
            sc.ck_owner[cid], sc.ck_time[cid] = s.i, time.time()
            return cid
        sc._ck_alloc = old_alloc
    th = serve(sc)
    agent = req(text(1000 + 400, 3000, 5), "agent")  # decodes 400 rounds: busy throughout
    sc.submit(agent)
    while not agent.slot or agent.state != "decode":
        time.sleep(0.001)
    held = [len(s.ck) - 1 for s in sc.slots]
    doc = text(1000, 3000, 6)
    q1 = req(doc, "doc+q1")
    sc.submit(q1)
    q1.done.wait(5)
    q2 = req(doc[:2900] + text(1000, 100, 7)[1:] + [11], "doc+q2")
    sc.submit(q2)
    q2.done.wait(5)
    busy = agent.state != "done"
    label = "global LRU" if fixed else "old rule (idle slots only)"
    want = 3000 - S.CKPT_TAIL  # the tail checkpoint (or the later chunk boundary, 2048 — both precede the new question)
    ok = (want <= q2.cached <= 2900) if fixed else (q2.cached == 0)
    check(ok and busy, f"{label}: agent held {held[agent.slot.i]} pool checkpoints while busy · same text + another "
                       f"question reused {q2.cached} (want {f'{want}…2900' if fixed else 0})")
    st, free, ev = sc.ck_status()
    check(isinstance(free, int) and all(len(x) == 4 for x in st), f"ck_status: {[(i, b, n, [(c, m) for c, m, _ in ck]) for i, b, n, ck in st if ck]}")
    sc.drain(time.time() + 5)
    th.join(5)

# 3. checkpoints in the shared KV pool: their room is free pool memory. Agents keep their slots busy and take a
# checkpoint every turn; with room the long text's checkpoint stays (64 buffers, none dropped), with little room the
# least recently used checkpoints go and the long text's own is dropped only when it is the oldest
for room, want_reuse in ((4000, True), (400, False)):
    eng = FakeEngine(pool=room)
    sc = S.Scheduler(eng, {EOS}, 4, 1 << 20, 1024, "4", checkpoints=64, kv=eng, output_reserve=64)
    th = serve(sc)
    doc = text(1000, 6000, 8)
    q1 = req(doc, "doc+q1")
    sc.submit(q1)
    q1.done.wait(5)
    turns = []
    for t in range(12):  # agent turns on the other slots: each a longer prompt → a tail checkpoint per turn
        rs = [req(text(1000 + 2, 3000 + 400 * t, 20 + a)[:2600 + 400 * t] + text(1000, 400, 40 + t)[1:], f"a{a}t{t}")
              for a in range(2)]
        for r in rs:
            sc.submit(r)
        for r in rs:
            r.done.wait(5)
        turns.append(eng.kv_free())
    q2 = req(doc[:5900] + text(1000, 100, 9)[1:] + [11], "doc+q2")
    sc.submit(q2)
    q2.done.wait(5)
    st, free, ev = sc.ck_status()
    held = sum(len(ck) for _, _, _, ck in st)
    ok = (q2.cached >= 6000 - S.CKPT_TAIL) if want_reuse else (eng.kv_free() >= 0)
    check(ok, f"pool {room} chunks: {held} checkpoints held ({held * eng.CK} chunks), {ev} dropped for room · same text "
              f"+ another question reused {q2.cached} · free {eng.kv_free()} (never negative: room asked first)")
    sc.drain(time.time() + 5)
    th.join(5)

# 4. vision tower items in slices between engine calls: every block once and in order, replies keep flowing meanwhile,
# a prompt being read takes turns with it
sys.path.insert(0, HERE)
import types  # noqa: E402
from media import TowerJobs  # noqa: E402


class TowerEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.calls = []

    def tower_begin(self, jid, grids, shape):
        self.calls.append(("begin", jid))

    def tower_blocks(self, k0, k1):
        self.calls.append(("blocks", k0, k1))
        time.sleep(0.004 * (k1 - k0))  # 4 ms a block

    def tower_end(self, jid):
        self.calls.append(("end", jid))
        jobs.finish(jid, "emb")

    def spec_round(self, slots):
        self.calls.append(("round",))
        return super().spec_round(slots)

    def step(self, ids, pos, slots, cu, *a):
        self.calls.append(("chunk",))
        return super().step(ids, pos, slots, cu, *a)


eng = TowerEngine()
jobs = TowerJobs(27)
S.TOWER_SLICE_S, S.TOWER_S_PER_ROW = 0.02, 4e-5  # small slices for the test: 5 blocks (20 ms) at a time
sc = S.Scheduler(eng, {EOS}, 4, 1 << 20, 1024, "4", tower_jobs=jobs)
th = serve(sc)
talker = req(text(1000 + 300, 800, 50), "talker")  # decodes 300 rounds
sc.submit(talker)
while talker.state != "decode":
    time.sleep(0.001)
m = types.SimpleNamespace(grid=(1, 10, 10), pixels=types.SimpleNamespace(shape=(100, 1536)))
out = []
t = threading.Thread(target=lambda: out.append(jobs.run(m)))
t.start()
t.join(10)
blocks = [c for c in eng.calls if c[0] == "blocks"]
covered = [k for _, k0, k1 in blocks for k in range(k0, k1)]
i0 = next(i for i, c in enumerate(eng.calls) if c[0] == "begin")
i1 = next(i for i, c in enumerate(eng.calls) if c[0] == "end")
rounds_between = sum(1 for c in eng.calls[i0:i1] if c[0] == "round")
check(out == ["emb"] and covered == list(range(27)) and len(blocks) > 1 and rounds_between >= len(blocks) - 1,
      f"tower item: {len(blocks)} slices {[(k0, k1) for _, k0, k1 in blocks]} · every block once, in order · "
      f"{rounds_between} decode rounds in between")
eng.calls.clear()
reader = req(text(1000, 6 * 1024, 51), "reader")  # 6 prefill chunks
sc.submit(reader)
while reader.state == "wait":
    time.sleep(0.001)
t = threading.Thread(target=lambda: out.append(jobs.run(m)))
t.start()
t.join(10)
reader.done.wait(10)
order = "".join("T" if c[0] == "blocks" else "P" for c in eng.calls if c[0] in ("blocks", "chunk"))
check(len(out) == 2 and out[1] == "emb" and reader.done.is_set() and "TP" in order and "PT" in order,
      f"with a prompt being read: item done, prompt done, turns {order}")
sc.drain(time.time() + 5)
th.join(5)

print("PASS" if not bad else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
