"""Host-side control of the two-node server. Runs on the head node (rank 0) and reaches the worker (rank 1) by ssh.
Stdlib only (Python >= 3.11 on the host). ./start.sh and ./stop.sh call it.

  start [--set section.key=value ...]   copy the repo to the worker, read the link on both nodes, build the image
                                        and the tensor table if missing (first start), start both ranks, wait until
                                        /health answers
  restart                               stop, then start
  check                                 everything start needs, one line each (nothing is changed)
  download                              the checkpoint from Hugging Face (paths.weights_repo @ weights_revision) into
                                        paths.weights on the head, then copied to the worker (HF_TOKEN used if set)
  stop                                  both ranks down; their logs are kept in <work>/results/serve/
  status                                containers and /health
  logs [N]                              rank 0's last N screen lines
  pair NAME [--timeout S] [--models-parent] -- CMD...
        (development) run CMD (+ --rank R) on both ranks in the image with the same link setup, wait for both,
        keep both logs in <work>/results/runs/tp2-NAME-<time>/"""
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "tools")]
import qlog  # noqa: E402
import settings  # noqa: E402

LABEL = "project=qwen38-engine"
SERVE = "qwen38-tp2-plus"  # model-tp<nodes>-engine
PASS_ENV = ("Q38_RDMA_INIT", "Q38_DRAFT_LM", "Q38_RDMA_TIMEOUT_S", "CUDA_LAUNCH_BLOCKING", "NCCL_DEBUG",
            "Q38_TEST_KV_NOGROW")  # if set here
SYNC_EXCLUDE = (".git", ".claude", ".venv", "__pycache__", ".DS_Store", "results/raw", "build")
SCREEN = r"^(.\[36m)?\[[^]]+\]"  # the `[tag] line` screen lines in a container log


def q(s):
    return shlex.quote(str(s))


class Ctl:
    def __init__(self, overrides):
        env_over = []
        for sec, kv in settings.load(REPO).items():  # Q38_<SECTION>_<KEY> from this shell → --set for the containers
            for k in kv:
                v = os.environ.get(f"Q38_{sec.upper()}_{k.upper()}")
                if sec != "patches" and v is not None and (sec, k) != ("server", "api_key"):  # the key goes as -e only
                    env_over.append(f"{sec}.{k}={v}")
        self.overrides = env_over + list(overrides)
        self.S = settings.load(REPO, self.overrides)
        c, p = self.S["cluster"], self.S["paths"]
        self.worker, self.repo = c["worker"], {0: REPO, 1: p["worker_repo"] or REPO}
        w = p["work"]
        self.work = {0: w, 1: (self.repo[1] + w[len(REPO):]) if w.startswith(REPO + os.sep) else w}
        with open(os.path.join(REPO, "docker", "dev.Dockerfile"), "rb") as f:
            self.tag = f"{p['image']}:{hashlib.sha256(f.read()).hexdigest()[:8]}"
        self.host = socket.gethostname()

    # ---- shell on either node ------------------------------------------------------------------------------------
    def sh(self, rank, cmd, timeout=None):
        a = ["bash", "-c", cmd] if rank == 0 else \
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.worker, cmd]
        try:
            return subprocess.run(a, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(a, 124, "", f"timed out after {timeout} s")

    def node(self, rank):
        return self.host if rank == 0 else self.worker

    def commit(self):
        """This checkout's commit (+modified when tracked files differ); Q38_COMMIT from the shell wins (a copy
        without .git, as in development); '' when unknown."""
        if os.environ.get("Q38_COMMIT"):
            return os.environ["Q38_COMMIT"]
        try:
            r = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            if r.returncode:
                return ""
            d = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
            return r.stdout.strip() + (" +modified" if d.stdout.strip() else "")
        except OSError:
            return ""

    def require(self):
        c, p = self.S["cluster"], self.S["paths"]
        miss = [k for k, v in (("cluster.worker", c["worker"]), ("paths.weights", p["weights"])) if not v]
        if miss:
            qlog.fail("start", f"set {' and '.join(miss)} in serve.local.toml (see serve.local.toml.example)")
        if not os.path.isfile(os.path.join(p["weights"], "config.json")):
            qlog.fail("start", f"no checkpoint at {p['weights']} (paths.weights)")
        r = self.sh(1, "true", timeout=20)
        if r.returncode:
            qlog.fail("start", f"ssh {self.worker} failed: {r.stderr.strip()[-200:]} (cluster.worker; key login needed)")

    def sync(self):
        """Copy this repo to the worker (rsync --delete inside the repo folder only, never the work folder)."""
        dst = self.repo[1]
        r = self.sh(1, f"test ! -e {q(dst)} || test -z \"$(ls -A {q(dst)})\" || test -f {q(dst)}/start.sh")
        if r.returncode:
            qlog.fail("sync", f"{self.worker}:{dst} exists and is not this repo — set paths.worker_repo")
        ex = list(SYNC_EXCLUDE)
        for rk in (0, 1):
            if self.work[rk].startswith(self.repo[rk] + os.sep):
                ex.append("/" + os.path.relpath(self.work[rk], self.repo[rk]))
        self.sh(1, f"mkdir -p {q(dst)}")
        a = ["rsync", "-a", "--delete", "-e", "ssh -o BatchMode=yes"] + [f"--exclude={e}" for e in ex] + \
            [REPO + "/", f"{self.worker}:{dst}/"]
        r = subprocess.run(a, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            qlog.fail("sync", f"copy to {self.worker}:{dst} failed: {r.stderr.strip()[-300:]}")

    def link(self):
        """Rails on both nodes (rail k of the head and of the worker on one subnet)."""
        c = self.S["cluster"]
        try:
            head = settings.probe(settings.worker_addrs(self.S), c["rails"], c["gid_index"])
        except settings.ProbeError as e:
            qlog.fail("link", f"{self.host}: {e}")
        # the probe runs on the worker from this file through ssh stdin: nothing has to be on the worker yet
        with open(os.path.join(HERE, "settings.py")) as f:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", self.worker,
                                f"python3 - probe {' '.join(h['ip'] for h in head)} --rails {len(head)} "
                                f"--gid {c['gid_index']}"], stdin=f, capture_output=True, text=True, timeout=30)
        if r.returncode:
            qlog.fail("link", f"{self.worker}: {(r.stderr or r.stdout).strip().removeprefix('error: ')}")
        wk = json.loads(r.stdout)
        if len(wk) != len(head):
            qlog.fail("link", f"{len(head)} rails on {self.host}, {len(wk)} on {self.worker}")
        for k, (h, w) in enumerate(zip(head, wk)):
            if not settings.same_wire(h, w):
                qlog.fail("link", f"rail {k}: {h['ip']}/{h['prefix']} and {w['ip']} are not on one subnet")
        for rails, n in ((head, self.host), (wk, self.worker)):
            if len({x["gid"] for x in rails}) > 1:
                qlog.warn("link", f"{n}: GID index differs between rails ({[x['gid'] for x in rails]}); NCCL uses "
                                  f"{rails[0]['gid']} for all — set cluster.gid_index or cluster.rails = 1")
        self.rails = {0: head, 1: wk}
        return head, wk

    def link_line(self):
        h, w = self.rails[0], self.rails[1]
        return (f"{len(h)} rail{'s' if len(h) > 1 else ''} · " +
                " · ".join(f"{a['hca']} {a['ip']} ↔ {b['hca']} {b['ip']}" for a, b in zip(h, w)) +
                f" · GID {h[0]['gid']}/{w[0]['gid']} · MTU {h[0]['mtu']}")

    # ---- containers ----------------------------------------------------------------------------------------------
    def ids(self, rank):
        r = self.sh(rank, "echo $(id -u):$(id -g)")
        return r.stdout.strip()

    def run(self, rank, name, cmd, models_parent=False):
        p = self.S["paths"]
        work, repo, w = self.work[rank], self.repo[rank], p["weights"]
        mounts = ["-v", f"{work}:/work"]
        mk = [f"{work}/results/serve"]
        if os.path.normpath(os.path.join(work, "src")) != os.path.normpath(repo):
            mounts += ["-v", f"{repo}:/work/src"]
            mk.append(f"{work}/src")  # the mount point, made by us (not by docker as root)
        mounts += ["-v", f"{os.path.dirname(w)}:/models:ro"] if models_parent else \
                  ["-v", f"{w}:/models/{os.path.basename(w)}:ro"]
        rails = self.rails[rank]
        env = {"HOME": "/tmp", "HF_HOME": "/tmp/hf", "NCCL_IB_DISABLE": "0",
               "NCCL_IB_HCA": ",".join(r["hca"] for r in rails)}
        if len(rails) > 1:
            env.update(NCCL_IB_MERGE_NICS="1", NCCL_CROSS_NIC="1")
        env.update(NCCL_IB_GID_INDEX=str(rails[0]["gid"]), NCCL_IB_ROCE_VERSION_NUM="2", NCCL_IB_ADDR_FAMILY="AF_INET",
                   NCCL_SOCKET_IFNAME=rails[0]["dev"], GLOO_SOCKET_IFNAME=rails[0]["dev"], NCCL_CUMEM_ENABLE="0",
                   NCCL_NVLS_ENABLE="0", NCCL_IGNORE_CPU_AFFINITY="1",
                   Q38_HEAD_IP=self.rails[0][0]["ip"], Q38_PEER_IP=self.rails[1][0]["ip"],
                   Q38_IMAGE=self.tag, Q38_COMMIT=self.commit(), Q38_LOG_HOST=f"{work}/results/serve")
        pt = ",".join(x for x in (settings.patches_env(self.S), os.environ.get("Q38_PATCHES", "")) if x)
        if pt:
            env["Q38_PATCHES"] = pt
        for k in PASS_ENV:
            if os.environ.get(k):
                env[k] = os.environ[k]
        key = ["-e", "Q38_SERVER_API_KEY"] if rank == 0 and os.environ.get("Q38_SERVER_API_KEY") else []  # value from env
        a = ["docker", "run", "-d", "--init", "--name", name, "--label", LABEL, "--user", self.ids(rank),
             "--gpus", "all", "--ipc=host", "--network", "host", "--device", "/dev/infiniband", "--cap-add", "IPC_LOCK",
             "--ulimit", "memlock=-1:-1", *mounts, "-w", "/work/src", *key]
        for k, v in env.items():
            a += ["-e", f"{k}={v}"]
        a += [self.tag, *cmd, "--rank", str(rank)]
        r = self.sh(rank, f"mkdir -p {' '.join(q(x) for x in mk)} && docker rm -f {name} >/dev/null 2>&1; "
                          + " ".join(q(x) for x in a))
        if r.returncode:
            qlog.fail("start", f"{self.node(rank)}: docker run failed: {r.stderr.strip()[-300:]}")

    def state(self, rank, name):
        """'' (none) | running | exited (+ code)"""
        r = self.sh(rank, f"docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {name} 2>/dev/null")
        return r.stdout.strip()

    def screen(self, rank, name, n=None, pat=SCREEN):
        r = self.sh(rank, f"docker logs {name} 2>&1 | grep -aE {q(pat)}" + (f" | tail -{n}" if n else ""))
        return r.stdout.rstrip()

    def keep_logs(self, rank, name, folder):
        """docker logs → <work>/results/<folder>/rank<R>-<time>.log on that node; returns the path."""
        d = f"{self.work[rank]}/results/{folder}"
        f = f"{d}/rank{rank}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        self.sh(rank, f"mkdir -p {q(d)} && docker logs {name} > {q(f)} 2>&1")
        return f

    def busy(self):
        """Requests waiting or running (from /health), None if it does not answer."""
        sv = self.S["server"]
        h = "127.0.0.1" if sv["host"] in ("0.0.0.0", "") else sv["host"]
        try:
            with urllib.request.urlopen(f"http://{h}:{sv['port']}/health", timeout=2) as r:
                return json.loads(r.read()).get("busy")
        except (OSError, ValueError):
            return None

    def health(self, timeout=2):
        sv = self.S["server"]
        h = "127.0.0.1" if sv["host"] in ("0.0.0.0", "") else sv["host"]
        try:
            with urllib.request.urlopen(f"http://{h}:{sv['port']}/health", timeout=timeout) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except (OSError, ValueError):
            return 0

    # ---- first start: image and tensor table ----------------------------------------------------------------------
    def weights_id(self, rank):
        """index + config checksums, shard count and bytes of the checkpoint on that node ('' if missing)."""
        w = q(self.S["paths"]["weights"])
        r = self.sh(rank, f"cd {w} 2>/dev/null && echo $(md5sum < model.safetensors.index.json | cut -c1-8) "
                          f"$(md5sum < config.json | cut -c1-8) $(ls tensors/*.safetensors 2>/dev/null | wc -l) "
                          f"$(stat -c %s tensors/*.safetensors 2>/dev/null | awk '{{s+=$1}} END{{print s+0}}')")
        return r.stdout.strip() if r.returncode == 0 else ""

    def prepare(self):
        """First start: the image where it is missing, the tensor table when it is missing or the checkpoint changed."""
        self.build_image()
        self.table()

    def build_image(self, ranks=(0, 1)):
        """Build the image where it is missing (both nodes at once)."""
        need = [r for r in ranks if self.sh(r, f"docker image inspect {self.tag} >/dev/null 2>&1").returncode]
        if need:
            qlog.info("setup", f"building image {self.tag} on {', '.join(self.node(r) for r in need)} "
                               "(first start: pulls nvcr.io/nvidia/pytorch, ~25 GB, can take 10–30 min)")
            procs = {}
            for r in need:
                log = f"{self.work[r]}/results/setup/image-{time.strftime('%Y%m%d-%H%M%S')}.log"
                cmd = (f"mkdir -p {q(os.path.dirname(log))} && cd {q(self.repo[r])}/docker && docker build --label {LABEL} "
                       f"-t {self.tag} -f dev.Dockerfile . > {q(log)} 2>&1")
                a = ["bash", "-c", cmd] if r == 0 else ["ssh", "-o", "BatchMode=yes", self.worker, cmd]
                procs[r] = (subprocess.Popen(a, stdin=subprocess.DEVNULL), log)
            for r, (p, log) in procs.items():
                if p.wait():
                    qlog.fail("setup", f"{self.node(r)}: image build failed · {log}")
            qlog.ok("setup", f"image {self.tag} built")

    def table_rel(self):
        """The checkpoint's tensor table under <work>/results (one per checkpoint folder: switching between, e.g., the
        aligned and the abliterated checkpoint does not rebuild it)."""
        return f"tables/{os.path.basename(self.S['paths']['weights'].rstrip('/'))}.tsv"

    def table(self):
        """The tensor table for this checkpoint (on the head, stdlib only; copied to the worker)."""
        ids = {r: self.weights_id(r) for r in (0, 1)}
        if not ids[0] or ids[0] != ids[1]:
            qlog.fail("setup", f"the checkpoint differs between the nodes ({self.host}: {ids[0] or 'missing'} · "
                               f"{self.worker}: {ids[1] or 'missing'}) — same files at paths.weights on both")
        stamp = f"{ids[0]} {self.S['paths']['weights']}"
        tbl = {r: f"{self.work[r]}/results/{self.table_rel()}" for r in (0, 1)}
        have = {r: self.sh(r, f"test -s {q(tbl[r])} && cat {q(tbl[r])}.stamp 2>/dev/null").stdout.strip() for r in (0, 1)}
        if have[0] != stamp:
            qlog.info("setup", "tensor table for this checkpoint")
            os.makedirs(os.path.dirname(tbl[0]), exist_ok=True)
            r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "tensor_table.py"), self.S["paths"]["weights"],
                                "--out", tbl[0]], capture_output=True, text=True)
            if r.returncode:
                qlog.fail("setup", f"tensor table failed: {(r.stdout + r.stderr).strip()[-300:]}")
            with open(tbl[0] + ".stamp", "w") as f:
                f.write(stamp + "\n")
        if have[1] != stamp or have[0] != stamp:
            self.sh(1, f"mkdir -p {q(os.path.dirname(tbl[1]))}")
            r = subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", tbl[0], tbl[0] + ".stamp",
                                f"{self.worker}:{os.path.dirname(tbl[1])}/"], capture_output=True, text=True)
            if r.returncode:
                qlog.fail("setup", f"copying the tensor table to {self.worker} failed: {r.stderr.strip()[-200:]}")
            qlog.ok("setup", f"tensor table · {self.host} and {self.worker}")

    def check(self):
        """Everything start needs, one line each; nothing is changed. Exit 1 if anything is missing."""
        bad = []

        def line(ok, what, msg):
            (qlog.ok if ok is True else qlog.warn if ok is None else qlog.error)("check", f"{what} · {msg}")
            if ok is False:
                bad.append(what)
        c, p, sv = self.S["cluster"], self.S["paths"], self.S["server"]
        line(bool(c["worker"]), "settings", f"worker {c['worker'] or '(cluster.worker not set)'} · weights "
                                            f"{p['weights'] or '(paths.weights not set)'} · work {p['work']}")
        if bad:
            return 1
        r = self.sh(1, "python3 -c 'import sys; print(sys.version.split()[0]); sys.exit(sys.version_info < (3, 11))'",
                    timeout=20)
        line(r.returncode == 0, "ssh", f"{self.worker}: " + (f"python {r.stdout.strip()}" if r.returncode == 0 else
                                                              (r.stderr.strip() or r.stdout.strip())[-160:]))
        if bad:
            return 1
        try:
            self.link()
            ok = True
        except SystemExit:
            ok = False
        if not ok:
            bad.append("link")
        else:
            pings = []
            for h, w in zip(self.rails[0], self.rails[1]):
                big = min(h["mtu"], w["mtu"]) - 28
                rr = self.sh(0, f"ping -c1 -W1 -M do -s {big} -I {h['dev']} {w['ip']} >/dev/null 2>&1")
                pings.append(rr.returncode == 0)
            line(all(pings), "link", self.link_line() + f" · ping at MTU {'ok' if all(pings) else 'FAILED'}")
        running = {r: self.state(r, SERVE).startswith("running") for r in (0, 1)}
        for r in (0, 1):
            n = self.node(r)
            d = self.sh(r, "docker info --format '{{.ServerVersion}}' 2>&1; command -v nvidia-container-cli nvidia-ctk")
            ok = d.returncode == 0 and "nvidia" in d.stdout
            line(ok, f"docker {n}", f"Docker {d.stdout.split()[0]} · NVIDIA container toolkit" if ok else
                 (d.stdout + d.stderr).strip()[-160:] + " (needs Docker for this user and the NVIDIA container toolkit)")
            img = self.sh(r, f"docker image inspect {self.tag} >/dev/null 2>&1").returncode == 0
            line(True if img else None, f"image {n}", self.tag + ("" if img else " — built on the first start"))
            if running[r]:
                line(True, f"GPU {n}", "used by this server (running)")
                continue
            g = self.sh(r, "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1; "
                           "awk '/MemAvailable/{printf \"mem %d\\n\", $2/1048576}' /proc/meminfo")
            apps = [x for x in g.stdout.splitlines() if x.strip() and not x.startswith("mem ")]
            mem = int(([x for x in g.stdout.splitlines() if x.startswith("mem ")] or ["mem 0"])[0].split()[1])
            line(not apps and mem >= 100, f"GPU {n}", f"{'no other GPU processes' if not apps else 'in use: ' + '; '.join(apps)}"
                 f" · MemAvailable {mem} GiB (needs ~100)")
        if any(running.values()):
            line(None, "server", "running on " + ", ".join(self.node(r) for r in (0, 1) if running[r]) +
                 f" · /health {self.health() or 'no answer'}")
            busy = self.busy()
            if busy == 0:  # idle server: its CPU use should stay near zero (a few % of one core at most)
                cpu = {r: self.sh(r, f"docker stats --no-stream --format '{{{{.CPUPerc}}}}' {SERVE}").stdout.strip()
                       for r in (0, 1) if running[r]}
                hi = [r for r, v in cpu.items() if float(v.rstrip("%") or 0) > 3.0]
                line(not hi, "idle CPU", " · ".join(f"{self.node(r)} {v}" for r, v in cpu.items()) + " (idle; ≤ 3% each)")
        else:
            ports = {0: [sv["port"], c["nccl_port"], c["nccl_port"] + 1], 1: [c["link_port"]]}
            if self.S["vision"]["enabled"]:  # the vision tower's NCCL store (head)
                ports[0].append(c["nccl_port"] + 2)
            for r, ps in ports.items():
                used = self.sh(r, " ".join(f"ss -Htln 'sport = :{x}' | grep -q . && echo {x};" for x in ps)).stdout.split()
                line(not used, f"ports {self.node(r)}", f"{', '.join(map(str, ps))} " +
                     ("free" if not used else f"in use: {', '.join(used)}"))
        ids = {r: self.weights_id(r) for r in (0, 1)}
        line(bool(ids[0]) and ids[0] == ids[1], "weights", f"{p['weights']} · " + (
            f"index {ids[0].split()[0]} · config {ids[0].split()[1]} · {ids[0].split()[2]} shards · "
            f"{int(ids[0].split()[3]) / 2**30:.0f} GiB · same on both" if ids[0] and ids[0] == ids[1] else
            f"{self.host}: {ids[0] or 'missing'} · {self.worker}: {ids[1] or 'missing'}"))
        if bad:
            qlog.error("check", f"not ready: {', '.join(bad)}")
        else:
            qlog.ok("check", "all good — the server is running" if any(running.values()) else "ready to start")
        return 1 if bad else 0

    def download(self, only=None):
        """only: a file pattern (e.g. config.json) — fetch just that, to try the whole path (revision lookup, download,
        copy to the worker, comparison) without the 96 GiB."""
        p = self.S["paths"]
        w, repo, rev = p["weights"], p["weights_repo"], p["weights_revision"]
        if not w:
            qlog.fail("download", "set paths.weights in serve.local.toml (where the checkpoint goes, same path on both nodes)")
        if only or not os.path.isfile(os.path.join(w, "config.json")):
            if os.path.exists(w) and os.listdir(w):
                qlog.fail("download", f"{w} exists, is not empty and has no config.json — choose an empty folder")
            self.build_image((0,))
            os.makedirs(w, exist_ok=True)
            code = ("import shutil, sys\n"
                    "from huggingface_hub import HfApi, snapshot_download\n"
                    f"repo, pre = {repo!r}, {rev!r}\n"
                    "rev = next((c.commit_id for c in HfApi().list_repo_commits(repo) if c.commit_id.startswith(pre)), None)\n"
                    "if not rev: sys.exit(f'no commit {pre} in {repo}')\n"
                    "print('revision', rev, flush=True)\n"
                    f"snapshot_download(repo, revision=rev, local_dir='/m', allow_patterns={[only] if only else None!r})\n"
                    "shutil.rmtree('/m/.cache', ignore_errors=True)\n")
            log = f"{self.work[0]}/results/setup/download-{time.strftime('%Y%m%d-%H%M%S')}.log"
            os.makedirs(os.path.dirname(log), exist_ok=True)
            a = ["docker", "run", "--rm", "--label", LABEL, "--user", self.ids(0), "-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf",
                 "-v", f"{w}:/m"] + (["-e", "HF_TOKEN"] if os.environ.get("HF_TOKEN") else []) + \
                [self.tag, "python3", "-c", code]
            qlog.info("download", f"{repo} @ {rev} → {w} ({only or '~96 GiB'}) · log {log}")
            with open(log, "w") as f:
                pr = subprocess.Popen(a, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            t0, shown = time.time(), time.time()
            while pr.poll() is None:
                time.sleep(1)
                if time.time() - shown >= 30:  # a progress line every 30 s
                    shown = time.time()
                    got = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(w) for x in fs) / 2**30
                    qlog.info("download", f"{got:.1f} GiB · {time.time() - t0:.0f} s")
            if pr.returncode:
                qlog.fail("download", f"failed · {log}")
            qlog.ok("download", f"{self.host}: done · {time.time() - t0:.0f} s")
        else:
            qlog.info("download", f"{self.host}: {w} already has a checkpoint")
        if only:  # a trial: copy what came, compare checksums, done
            self.sh(1, f"mkdir -p {q(w)}")
            r = subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", w + "/", f"{self.worker}:{w}/"],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True)
            sums = {k: self.sh(k, f"cd {q(w)} && find . -type f | sort | xargs md5sum").stdout.strip() for k in (0, 1)}
            if r.returncode or not sums[0] or sums[0] != sums[1]:
                qlog.fail("download", f"trial copy / comparison failed: {r.stderr.strip()[-200:]} · {sums}")
            qlog.ok("download", f"trial ok · {len(sums[0].splitlines())} file(s) on both nodes · "
                                f"{' · '.join(x.split()[1] + ' ' + x.split()[0][:8] for x in sums[0].splitlines())}")
            return
        if self.weights_id(1) != self.weights_id(0):
            qlog.info("download", f"copying to {self.worker}:{w}")
            self.sh(1, f"mkdir -p {q(w)}")
            r = subprocess.run(["rsync", "-a", "-e", "ssh -o BatchMode=yes", w + "/", f"{self.worker}:{w}/"],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True)
            if r.returncode:
                qlog.fail("download", f"copy to {self.worker} failed: {r.stderr.strip()[-200:]}")
        ids = {r: self.weights_id(r) for r in (0, 1)}
        if not ids[0] or ids[0] != ids[1]:
            qlog.fail("download", f"the nodes differ: {self.host} {ids[0]} · {self.worker} {ids[1]}")
        qlog.ok("download", f"checkpoint on both nodes · {ids[0].split()[2]} shards · {int(ids[0].split()[3]) / 2**30:.0f} GiB")

    # ---- commands ------------------------------------------------------------------------------------------------
    def start(self):
        self.require()
        st = {r: self.state(r, SERVE) for r in (0, 1)}
        if any(s.startswith("running") for s in st.values()):
            qlog.fail("start", "already running (" + ", ".join(f"{self.node(r)} {s.split()[0]}" for r, s in st.items()
                                                            if s) + ") — ./start.sh restart, or ./stop.sh")
        for r, s in st.items():
            if s:  # a stopped container from an earlier run: keep its log, remove it
                qlog.info("start", f"{self.node(r)}: earlier run's log kept · {self.keep_logs(r, SERVE, 'serve')}")
                self.sh(r, f"docker rm -f {SERVE} >/dev/null 2>&1")
        self.sync()
        self.link()
        self.prepare()
        sv = self.S["server"]
        qlog.info("start", f"link · {self.link_line()}")
        cmd = ["python3", "/work/src/server/main.py", "--model",
               f"/models/{os.path.basename(self.S['paths']['weights'])}", "--table", f"/work/results/{self.table_rel()}"] + \
              [x for o in self.overrides for x in ("--set", o)]
        self.run(1, SERVE, cmd)
        self.run(0, SERVE, cmd)
        c = self.commit()
        qlog.info("start", f"plus {settings.version()}{' · ' + c if c else ''} · rank 0 {self.host} · rank 1 "
                           f"{self.worker} · image {self.tag} · loading (about 2 min; the first start on a machine "
                           "also compiles the engine)")
        t0 = time.time()
        while self.health() != 200:
            for r in (0, 1):
                s = self.state(r, SERVE)
                if not s.startswith("running"):
                    if s:
                        print(self.screen(r, SERVE, 6, SCREEN + "|Error|error"))
                    self.down(quiet=True)
                    qlog.fail("start", f"{self.node(r)}: rank {r} stopped ({s or 'gone'}) — both ranks are down; "
                                       f"logs in {self.work[r]}/results/serve/")
            if time.time() - t0 > sv["start_timeout"]:
                self.down(quiet=True)
                qlog.fail("start", f"not ready in {sv['start_timeout']} s (server.start_timeout) — both ranks are down")
            time.sleep(3)
        print(self.screen(0, SERVE, pat=r"^(.\[36m)?\[(build|engine|comm|rdma|link|serve)\]"))
        qlog.ok("start", f"ready · {time.time() - t0:.0f} s · http://{self.host}:{sv['port']}/v1")

    def down(self, quiet=False):
        """Both ranks down: SIGTERM to rank 0 (it drains, then closes the link — rank 1 then leaves by itself), rank 1
        stopped if it is still there after 30 s; logs kept, containers removed, leftovers checked. Safe to repeat;
        a worker that cannot be reached is reported and the head is still stopped."""
        drain = self.S["server"]["drain_seconds"]
        reach = self.sh(1, "true", timeout=20).returncode == 0
        if not reach:
            qlog.warn("stop", f"{self.worker} not reachable — stopping the head only")
        ranks = (0, 1) if reach else (0,)
        st = {r: self.state(r, SERVE) for r in ranks}
        if st[0].startswith("running"):
            if not quiet:
                qlog.info("stop", f"stopping · running requests get up to {drain} s (server.drain_seconds)")
            t0 = time.time()
            self.sh(0, f"docker stop -t {drain + 30} {SERVE} >/dev/null")
            if not quiet:
                qlog.info("stop", f"rank 0 down · {time.time() - t0:.0f} s")
        if reach and st[1].startswith("running"):
            t0 = time.time()
            while self.state(1, SERVE).startswith("running") and time.time() - t0 < 30:
                time.sleep(1)
            if self.state(1, SERVE).startswith("running"):
                qlog.warn("stop", "rank 1 still running 30 s after rank 0 — stopping it")
                self.sh(1, f"docker stop -t 10 {SERVE} >/dev/null")
        kept = []
        for r in ranks:
            if st[r]:
                kept.append(f"{self.node(r)}:{self.keep_logs(r, SERVE, 'serve')}")
                self.sh(r, f"docker rm -f {SERVE} >/dev/null 2>&1")
        left = []
        for r in ranks:
            c = self.sh(r, f"docker ps -aq --filter name=^{SERVE}$").stdout.split()
            g = self.sh(r, "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null").stdout
            if c:
                left.append(f"{self.node(r)} container")
            if g.strip():
                left.append(f"{self.node(r)} GPU processes: {' '.join(g.split())}")
        if left:
            qlog.warn("stop", "left over: " + "; ".join(left))
        if not quiet:
            qlog.ok("stop", ("both ranks down" if reach else "head down") + f" · nothing left over · logs {', '.join(kept)}"
                    if kept and not left else ("nothing was running" if not kept else "down (see above)"))

    def status(self):
        for r in (0, 1):
            s = self.sh(r, f"docker ps -a --filter name=^{SERVE}$ --format '{{{{.Status}}}}'").stdout.strip()
            qlog.info("status", f"rank {r} {self.node(r)}: {s or 'no container'}")
        h = self.health()
        (qlog.ok if h == 200 else qlog.warn)("status", f"/health {h or 'no answer'}")

    def logs(self, n):
        print(self.screen(0, SERVE, n))

    def pair(self, name, cmd, timeout, models_parent):
        cname = f"qwen38-engine-{name}"
        self.sync()
        self.link()
        logd = f"{self.work[0]}/results/runs/tp2-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.sh(0, f"mkdir -p {q(logd)}")
        self.run(0, cname, cmd, models_parent)
        self.run(1, cname, cmd, models_parent)
        qlog.info("tp2", f"{name} · rank 0 {self.host} · rank 1 {self.worker} · {self.link_line()}")
        t0 = time.time()
        while any(self.state(r, cname).startswith("running") for r in (0, 1)):
            if time.time() - t0 > timeout:
                qlog.warn("tp2", f"over {timeout} s — stopping both")
                for r in (0, 1):
                    self.sh(r, f"docker stop -t 5 {cname} >/dev/null 2>&1")
                break
            time.sleep(2)
        rc = {r: self.state(r, cname).split()[-1] for r in (0, 1)}
        self.sh(0, f"docker logs {cname} > {q(logd)}/rank0.log 2>&1; docker rm -f {cname} >/dev/null")
        with open(f"{logd}/rank1.log", "w") as f:
            subprocess.run(["ssh", "-o", "BatchMode=yes", self.worker, f"docker logs {cname} 2>&1"], stdout=f,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.sh(1, f"docker rm -f {cname} >/dev/null")
        print(self.sh(0, f"grep -aE {q(SCREEN)} {q(logd)}/rank0.log").stdout.rstrip())
        if rc[0] != "0" or rc[1] != "0":
            print(self.sh(0, f"grep -aE {q(SCREEN + '|Error|error')} {q(logd)}/rank1.log | tail -3").stdout.rstrip())
            qlog.fail("tp2", f"exit codes rank 0 {rc[0]} · rank 1 {rc[1]} · logs {logd}")
        qlog.ok("tp2", f"done · logs {logd}")


def main(argv):
    over, rest = [], []
    i = 0
    while i < len(argv):
        if argv[i] == "--":
            rest += argv[i:]
            break
        if argv[i] == "--set" and i + 1 < len(argv):
            over.append(argv[i + 1])
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    cmd = rest[0] if rest else "start"
    try:
        c = Ctl(over)
    except settings.SettingsError as e:
        qlog.fail("start", str(e))
    if cmd == "start":
        c.start()
    elif cmd == "restart":
        c.down()
        c.start()
    elif cmd == "stop":
        c.down()
    elif cmd == "check":
        return c.check()
    elif cmd == "download":
        c.download(rest[rest.index("--only") + 1] if "--only" in rest else None)
    elif cmd == "status":
        c.status()
    elif cmd == "logs":
        c.logs(int(rest[1]) if len(rest) > 1 else 40)
    elif cmd == "pair":
        if "--" not in rest or len(rest) < 2:
            qlog.fail("tp2", "usage: control.py pair NAME [--timeout S] [--models-parent] -- CMD...")
        name, opts, run = rest[1], rest[2:rest.index("--")], rest[rest.index("--") + 1:]
        to = int(opts[opts.index("--timeout") + 1]) if "--timeout" in opts else 1800
        c.pair(name, run, to, "--models-parent" in opts)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
