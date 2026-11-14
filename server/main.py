"""Server entry (./start.sh runs it in the image on the head = rank 0 and the worker = rank 1).
Settings: serve.toml + serve.local.toml (+ --set); the link addresses come from the launcher's probe (Q38_HEAD_IP =
the head's CX7 address for NCCL / RDMA bootstrap, Q38_PEER_IP = the worker's, for the command link).

Start: rank 0's HTTP front answers at once (/health 503 "loading"); both ranks open the command link and compare
settings and sources, load the engine with the same settings (TP2 collectives pair up), and rank 0 turns ready
(/health 200). Rank 1 replays rank 0's engine calls (link.py); rank 0 runs the scheduler — every engine call — in the
main thread, the thread that initialised the engine.
Stop (SIGTERM / SIGINT to rank 0, e.g. ./stop.sh → docker stop): new requests get 503, running ones get
server.drain_seconds to finish and are then cut with a normal ending; rank 0 closes the link, both ranks destroy the
engine (RDMA proxy and QPs, NCCL) and exit. A second signal leaves at once. A signal while loading leaves at once.
If the other rank goes away, a watcher thread fails the running requests (503 / an error event) and leaves at once."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # the vision tower's memory: segments grow
# in place instead of piling up per input size (its peak is taken at start, before the KV pool)
import argparse
import glob
import hashlib
import json
import logging
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "engine", "tests"), os.path.join(ROOT, "tools")]
import qlog  # noqa: E402

import link  # noqa: E402
import settings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--model", default="/models/Qwen3.8-Flash-Next-FP4-Local", help="checkpoint folder in the container")
ap.add_argument("--table", default="/work/results/tensor_table.tsv")
ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a setting")
a = ap.parse_args()
try:
    S = settings.load(ROOT, a.set)
except settings.SettingsError as e:
    qlog.fail("serve", str(e))
sv, cl, md = S["server"], S["cluster"], S["model"]
if not sv["kv_pool"]:  # every slot's caches up front: refuse what cannot fit instead of allocating until it fails
    import glob
    with open("/proc/meminfo") as f:
        avail = next(int(x.split()[1]) for x in f if x.startswith("MemAvailable:")) / 1048576
    weights = sum(os.path.getsize(x) for x in glob.glob(os.path.join(a.model, "tensors", "*.safetensors"))) / 2 / 2**30
    need = sv["slots"] * sv["context"] * 17472 / 2**30  # 13 attention caches × 1,344 B per token per rank
    room = avail - weights - 17 - sv["memory_free_gib"]  # 17 GiB: workspaces, runtime, the server (measured ~16.6)
    if need > room:
        qlog.fail("serve", f"server.kv_pool false: {sv['slots']} slots × {sv['context']:,} tokens need {need:.1f} GiB of "
                           f"caches per machine, about {room:.1f} GiB fit — fewer slots, a smaller context or kv_pool true")
if sv["context"] > 262144 * max(1.0, float(md["yarn_factor"])):
    qlog.fail("serve", f"server.context {sv['context']:,} is past 262,144 × model.yarn_factor {md['yarn_factor']} — raise "
                       "the factor (4 for 1,048,576) or lower the context")
head, peer = os.environ.get("Q38_HEAD_IP", ""), os.environ.get("Q38_PEER_IP", "")
if not head or not peer:
    qlog.fail("serve", "Q38_HEAD_IP / Q38_PEER_IP not set — start the server with ./start.sh")
LOG = "/work/results/serve"
os.makedirs(LOG, exist_ok=True)
LOG_HOST = os.environ.get("Q38_LOG_HOST", LOG)  # the same folder as the host sees it (for the screen)
if a.rank == 0:
    commit = os.environ.get("Q38_COMMIT", "")
    rails = len([x for x in os.environ.get("NCCL_IB_HCA", "").split(",") if x])
    qlog.banner([f"plus {settings.version()}{' · ' + commit if commit else ''} · Qwen3.8-Flash-Next engine",
                 f"model    {a.model}",
                 f"image    {os.environ.get('Q38_IMAGE', '?')}",
                 f"cluster  TP2 · head {head} ↔ worker {peer}" + (f" · {rails} rail{'s' if rails > 1 else ''}" if rails else "")])


def load_engine(rank):
    """The engine module: built here on the first start (or after its sources change), cached in /work/cache/ext."""
    import ext
    why = ext.stale("engine", ext.engine_sources())
    if why and rank == 0:
        qlog.info("build", f"engine · compiling ({why}, a few minutes) · log {LOG_HOST}/build0.log")
    t0 = time.time()
    m = ext.load("engine", ext.engine_sources(), log=f"{LOG}/build{rank}.log", ldflags=ext.ENGINE_LDFLAGS)
    if why and rank == 0:
        qlog.ok("build", f"engine · built · {time.time() - t0:.0f} s")
    return m


def digest():
    """What both ranks must agree on: settings, engine switches from the environment, and the sources."""
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(ROOT, "engine", "src", "**", "*"), recursive=True) +
                    glob.glob(os.path.join(HERE, "*.py")) + [os.path.join(ROOT, "engine", "tests", "bind", "engine.cpp")]):
        if os.path.isfile(f):
            h.update(f.encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
    env = {k: os.environ.get(k, "") for k in ("Q38_PATCHES", "Q38_DRAFT_LM", "Q38_RDMA_INIT")}
    st = {k: v for k, v in S.items() if k != "paths"}
    st["server"] = {k: v for k, v in st["server"].items() if k != "api_key"}  # rank 0 only (HTTP)
    return {"settings": json.dumps(st, sort_keys=True),
            "engine switches": json.dumps(env, sort_keys=True), "sources": h.hexdigest(), "model": a.model}


class Calls:
    """What both ranks run in the same order (rank 0 through link.Mirror, rank 1 by link.replay): the engine's calls,
    and the vision tower's slices — so the tower's collectives (its own NCCL communicator) never overlap the engine's.
    jobs (rank 0): media.TowerJobs, which holds the item's pixels and takes its embeddings back."""
    def __init__(self, eng, tower=None, jobs=None):
        self.eng, self.tower, self.jobs = eng, tower, jobs

    def __getattr__(self, name):
        return getattr(self.eng, name)

    def tower_begin(self, jid, grids, shape):
        px = None
        if self.jobs is not None:
            j = self.jobs.head()
            assert j is not None and j.id == jid, (jid, j and j.id)
            px = j.px
        self.tower.job_begin(grids, shape, px)

    def tower_blocks(self, k0, k1):
        self.tower.job_blocks(k0, k1)

    def tower_end(self, jid):
        out = self.tower.job_end()
        if self.jobs is not None:
            self.jobs.finish(jid, out)


def close_engine(eng, who):
    """Destroy the engine; if that hangs (a peer stuck mid-collective), leave anyway after 20 s."""
    t = threading.Timer(20, lambda: (qlog.error(who, "engine close hung — leaving"), os._exit(1)))
    t.daemon = True
    t.start()
    eng.close()
    t.cancel()


# ---- rank 1 ------------------------------------------------------------------------------------------------------
if a.rank == 1:
    sock = link.accept(peer, cl["link_port"])
    calls = link.reader(link.hello(sock, 1, digest()))
    eng = load_engine(1)
    eng.init(a.model, a.table, 1, head, sv["slots"], sv["context"], sv["chunk"], mtp=True, port=cl["nccl_port"],
             yarn_factor=float(md["yarn_factor"]), yarn_ramp=md["yarn_ramp"], checkpoints=sv["checkpoints"],
             kv_pool=sv["kv_pool"])
    vs = S["vision"]
    tower = None
    if vs["enabled"]:  # this rank's half of the vision tower: its peak taken now (with rank 0); its slices then come
        # in the command stream with the engine's calls
        from vision_tower import TP, Tower
        t0 = time.time()
        tower = Tower(a.model, "cuda", tp=TP.connect(1, head, peer, None, cl["nccl_port"] + 2))
        peak, dt = tower.profile()
        qlog.ok("vision", f"rank 1 · tower half {tower.nbytes() / 2**30:.2f} GiB · peak {peak:.1f} GiB held · "
                          f"{time.time() - t0:.1f} s")
    n = link.replay(Calls(eng, tower), calls)
    close_engine(eng, "serve")
    qlog.ok("serve", f"rank 1 stopped · {n:,} engine calls")
    sys.stdout.flush()
    os._exit(0)

# ---- rank 0 ------------------------------------------------------------------------------------------------------
# signals go to one waiting thread, never into the middle of an engine call; threads made later inherit this mask.
# (Rank 1 keeps the default: a signal ends it at once.)
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
import uvicorn  # noqa: E402

from app import Front, build  # noqa: E402

front = Front(sv["api_key"])
logging.basicConfig(filename=f"{LOG}/http.log", level=logging.INFO)  # uvicorn's lines: log file only
http = uvicorn.Server(uvicorn.Config(front, host=sv["host"], port=sv["port"], log_config=None, access_log=False,
                                     lifespan="off"))
threading.Thread(target=http.run, daemon=True, name="http").start()
t0 = time.time()
while not http.started:
    if time.time() - t0 > 20 or not any(t.name == "http" and t.is_alive() for t in threading.enumerate()):
        qlog.fail("serve", f"HTTP front did not start on {sv['host']}:{sv['port']} (port in use?) — {LOG}/http.log")
    time.sleep(0.05)

sched = None
closing = threading.Event()


def peer_gone(msg):
    front.state = "failed"
    if sched is not None:
        sched.fail_all(f"the server stopped: {msg}")
        time.sleep(1.0)  # let the HTTP thread send the errors
    link.gone(msg + " — stopping")


def on_signal():
    signal.sigwait({signal.SIGTERM, signal.SIGINT})
    if sched is None:
        qlog.warn("serve", "stopped while loading")
        sys.stdout.flush()
        os._exit(0)
    front.state = "stopping"
    run, wait = sched.counts()
    qlog.info("serve", f"stopping · {run + wait} request(s) get up to {sv['drain_seconds']} s to finish")
    sched.drain(time.time() + sv["drain_seconds"])
    signal.sigwait({signal.SIGTERM, signal.SIGINT})
    qlog.warn("serve", "second signal — leaving now")
    sys.stdout.flush()
    os._exit(1)


threading.Thread(target=on_signal, daemon=True, name="signals").start()

sock = link.connect(peer, cl["link_port"])
link.hello(sock, 0, digest())
link.watch(sock, closing, peer_gone)

from chat import Chat  # noqa: E402
from scheduler import Scheduler  # noqa: E402

eng = load_engine(0)
eng.init(a.model, a.table, 0, head, sv["slots"], sv["context"], sv["chunk"], mtp=True, port=cl["nccl_port"],
         yarn_factor=float(md["yarn_factor"]), yarn_ramp=md["yarn_ramp"], checkpoints=sv["checkpoints"],
         kv_pool=sv["kv_pool"])
chat = Chat(a.model)
vs = S["vision"]
mf = None
if vs["enabled"]:  # the vision tower over both ranks; its peak held (profile) before the KV pool takes the rest
    from media import MediaFront
    from vision_tower import TP
    t0 = time.time()
    mf = MediaFront(vs, a.model, chat.tok, tp=TP.connect(0, head, peer, None, cl["nccl_port"] + 2))
    qlog.ok("vision", f"tower over 2 ranks · {mf.tower.nbytes() / 2**30:.2f} GiB here · peak {mf.peak_gib:.1f} GiB held "
                      f"(largest picture and video with the embeddings' room, {mf.profile_s:.1f} s) · {time.time() - t0:.1f} s · up to "
                      f"{vs['max_items']} images / videos and {vs['max_media_tokens']:,} media tokens per request · "
                      f"{vs['max_concurrent']} decoded at a time · URLs {'on' if vs['fetch_urls'] else 'off (data: only)'}")
m = link.Mirror(Calls(eng, mf.tower, mf.jobs) if mf else eng, sock)
if sv["kv_pool"]:  # both ranks (rank 1 replays the call): the same number of chunks from the smaller room (the vision
    # tower's memory, embeddings included, is already held)
    m.kv_pool_init(float(sv["memory_free_gib"]), 0.0, float(sv["kv_pool_gib"]))
sched = Scheduler(m, chat.eos, sv["slots"], sv["context"], sv["chunk"], sv["draft_depth"], think_end=chat.think_end,
                  loop_guard=S["thinking"]["loop_guard"], im_start=chat.im_start, checkpoints=sv["checkpoints"],
                  kv=eng if sv["kv_pool"] else None, output_reserve=sv["output_reserve"],
                  tower_jobs=mf.jobs if mf else None)
front.app = build(chat, sched, sv["served_name"], sv["context"], sched.depth, S["thinking"], sv["plain_specials"], mf)
d = chat.defaults
qlog.ok("serve", f"http://{sv['host']}:{sv['port']}/v1 · {sv['served_name']} · {sv['slots']} slots × "
                 f"{sv['context'] // 1024}K · chunk {sv['chunk']} · MTP depth {sv['draft_depth']} · default sampling "
                 f"T {d['temperature']} · top_k {d['top_k']} · top_p {d['top_p']}")
front.state = "ready"

sched.run()  # returns after a drain
front.state = "stopped"
time.sleep(0.5)  # the last chunks of cut requests
closing.set()
m.close()  # rank 1 closes its engine and exits
close_engine(eng, "serve")
st = sched.stats
qlog.ok("serve", f"stopped · {st['requests']:,} requests served")
sys.stdout.flush()
os._exit(0)
