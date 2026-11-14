"""TP2 command link. Both ranks must make the same engine calls in the same order (every step has collectives), and
only rank 0 runs the scheduler. Rank 0 sends each call (name, args) over one TCP connection on the CX7 link before
making it itself; rank 1 replays the stream. Results are not sent back: both ranks compute the same logits and draw
the same tokens (sampling keys depend only on seed and position), so rank 1 needs nothing from rank 0 but the calls.
The link is opened before the engine loads: the ranks first compare their settings and sources (hello).
Frames: 4-byte length + pickle (both ends are this file; rank 1 binds its CX7 address only)."""
import os
import pickle
import queue
import socket
import struct
import sys
import threading
import time

import qlog


def _send(s, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    s.sendall(struct.pack("<I", len(b)) + b)


def _recv(f):
    h = f.read(4)
    if len(h) < 4:
        return None
    return pickle.loads(f.read(struct.unpack("<I", h)[0]))


def _alive(s):
    """A peer whose machine or cable is gone sends no FIN: keepalive probes (idle 5 s, every 2 s, 3 misses) and a 15 s
    limit on unacknowledged sends end the connection, so the watchers below see it within ~20 s."""
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, v in (("TCP_KEEPIDLE", 5), ("TCP_KEEPINTVL", 2), ("TCP_KEEPCNT", 3), ("TCP_USER_TIMEOUT", 15000)):
        if hasattr(socket, opt):
            s.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), v)


def connect(peer, port, timeout=600):
    """Rank 0: connect to rank 1 (retries while it starts)."""
    t0 = time.time()
    while True:
        try:
            s = socket.create_connection((peer, port), timeout=5)
            break
        except OSError:
            if time.time() - t0 > timeout:
                qlog.fail("link", f"rank 1 at {peer}:{port} not reachable ({timeout} s)")
            time.sleep(0.5)
    _alive(s)
    s.settimeout(None)
    qlog.ok("link", f"connected to rank 1 {peer}:{port} · {time.time() - t0:.1f} s")
    return s


def accept(bind, port):
    """Rank 1: wait for rank 0 on this node's CX7 address."""
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind((bind, port))
    ls.listen(1)
    s, addr = ls.accept()
    ls.close()
    _alive(s)
    qlog.ok("link", f"rank 0 {addr[0]} connected on {bind}:{port}")
    return s


def hello(s, rank, digest):
    """Both ranks: compare digests (settings, patches, sources). A mismatch stops both with the differing parts.
    Returns the link's reader (rank 1 keeps reading calls from it)."""
    if rank == 0:
        _send(s, ("__hello__", digest))
        r = _recv(s.makefile("rb"))
        if r != ("__ok__",):
            qlog.fail("link", f"rank 1 runs with different {r[1] if r else 'settings (link closed)'} — "
                              "same repo and serve.local.toml on both nodes?")
    else:
        f = s.makefile("rb")
        r = _recv(f)
        if not r or r[0] != "__hello__":
            qlog.fail("link", "rank 0 closed the link before hello")
        diff = [k for k in digest if digest[k] != r[1].get(k)]
        _send(s, ("__ok__",) if not diff else ("__diff__", ", ".join(diff)))
        if diff:
            qlog.fail("link", f"rank 0 runs with different {', '.join(diff)} — same repo and serve.local.toml on both?")
        return f


def gone(msg):
    """The other rank is gone: say so and leave at once — this rank may be stuck in a collective with it, and the
    process exit releases the GPU, NCCL and RDMA resources."""
    qlog.error("link", msg)
    sys.stdout.flush()
    os._exit(1)


def watch(s, closing, on_gone):
    """Rank 0: rank 1 sends nothing after hello, so a read returns only when the link closes. Unless we are closing
    it ourselves, call on_gone() (in this watcher thread)."""
    def run():
        try:
            d = s.recv(1)
        except OSError:
            d = b""
        if not closing.is_set():
            on_gone("rank 1 is gone (link closed)" if not d else "rank 1 sent unexpected data")
    threading.Thread(target=run, daemon=True, name="link-watch").start()


def reader(f):
    """Rank 1: a thread reads rank 0's calls into a queue (so a dropped link is seen even while the main thread is
    inside an engine call). Ends with ("__close__", ()) on a clean close; leaves the process if the link drops."""
    calls = queue.Queue()

    def run():
        while True:
            try:
                m = _recv(f)
            except (OSError, EOFError, pickle.UnpicklingError):
                m = None
            if m is None:
                gone("rank 0 is gone (link dropped) — stopping")
            calls.put(m)
            if m[0] == "__close__":
                return
    threading.Thread(target=run, daemon=True, name="link-read").start()
    return calls


class Mirror:
    """Rank 0: eng.<call>(...) through m.<call>(...) — forwarded to rank 1, then run here."""

    def __init__(self, eng, s):
        self.eng, self.s = eng, s

    def __getattr__(self, name):
        fn = getattr(self.eng, name)

        def call(*args):
            _send(self.s, (name, args))
            return fn(*args)
        return call

    def close(self):
        _send(self.s, ("__close__", ()))
        self.s.close()


def replay(eng, calls):
    """Rank 1: run rank 0's calls (from reader()) until it closes the link. Returns the number of calls."""
    n = 0
    while True:
        name, args = calls.get()
        if name == "__close__":
            return n
        try:
            getattr(eng, name)(*args)
        except RuntimeError as e:  # the shared KV pool was full: rank 0 got the same at the same call and goes on
            if not str(e).startswith("KV pool full"):
                raise
        n += 1
