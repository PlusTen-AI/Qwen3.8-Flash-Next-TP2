"""P001 rdma_ar check — run: tools/tp2.sh rdma python3 engine/tests/rdma_check.py
RdmaAR (engine/src/kernels/comm/rdma_ar.cu) next to NCCL on C (rank 0) and D (rank 1):
  1. bitwise vs NCCL and vs bf16(float(a)+float(b)) on random data, [M, 2560] for M in 1 4 16 64 2048 8192 (+ odd/misaligned)
  2. latency eager / CUDA graph (best · median µs per call, back-to-back), bandwidth at 8192 rows
  3. overlap: a compute-bound bf16 GEMM on another stream — its slowdown while all-reduces run (RdmaAR vs NCCL)
  4. 1000 back-to-back calls eager + 1000 in a CUDA graph (sizes mixed, every result checked on the GPU)
Writes /work/results/tests/rdma/rdma.json (rank 0); build logs next to it."""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
import qlog  # noqa: E402

import ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--skip", default="", help="comma list of parts to skip: bits,lat,overlap,stress")
ap.add_argument("--env", default="", help="K=V,… set before init (tuning: Q38_RDMA_LL_MAX, Q38_RDMA_COPY_BLOCKS, …)")
ap.add_argument("--rows", default="", help="latency rows subset, e.g. 1,8192")
ap.add_argument("--prio", action="store_true", help="all-reduce stream at the highest priority (overlap test)")
a = ap.parse_args()
for kv in filter(None, a.env.split(",")):
    os.environ[kv.split("=")[0]] = kv.split("=", 1)[1]
SKIP = set(a.skip.split(","))
OUT = "/work/results/tests/rdma"
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("Q38_RDMA_TIMEOUT_S", "30")  # a lost peer traps instead of hanging the run
H, ROWS = 2560, (1, 4, 16, 64, 2048, 8192)
R0 = a.rank == 0
I16 = torch.int16

m = ext.load("rdma", ["src/core/comm.cc", "src/kernels/comm/rdma_ar.cu"], log=f"{OUT}/build{a.rank}.log",
             ldflags=["-lnccl", "-libverbs"])
m.init(a.rank, os.environ["Q38_HEAD_IP"], 29641, 8192 * H, a.prio)
res = {"rails": m.rails(), "ar_stream_high_priority": a.prio}


def both(counts):
    """per-rank integer lists → (rank0, rank1), each value capped at 256 (exact in bf16), via NCCL"""
    k = len(counts)
    t = torch.zeros(2 * k, dtype=torch.bfloat16, device="cuda")
    t[a.rank * k:(a.rank + 1) * k] = torch.tensor([min(int(c), 256) for c in counts], dtype=torch.bfloat16)
    m.ar(t, 0)
    torch.cuda.synchronize()
    v = [int(x) for x in t.float().tolist()]
    return v[:k], v[k:]


def gen(n, r, j):
    """rank r's input (variant j): mixed magnitudes so roundings and cancellations happen — same on both ranks"""
    g = torch.Generator(device="cuda").manual_seed(7919 * n + 31 * j + r)
    x = torch.randn(n, generator=g, device="cuda")
    e = torch.randint(-12, 12, (n,), generator=g, device="cuda").float()
    return (x * torch.exp2(e)).bfloat16()


def ref(n, j):
    return (gen(n, 0, j).float() + gen(n, 1, j).float()).bfloat16()


# ---- 1. bitwise ----------------------------------------------------------------------------------------------------
if "bits" not in SKIP:
    cases = [(f"[{r},2560]", r * H, 0) for r in ROWS] + [("odd n · 2-byte misaligned", 3 * H + 5, 1)]
    bad = []
    for name, n, off in cases:
        base = gen(n + off, a.rank, 0)
        y = base[off:]  # RdmaAR's buffer; off = 1 → data_ptr % 16 == 2 (scalar path)
        x = y.clone()   # NCCL's
        rf = ref(n + off, 0)[off:]
        m.ar(x, 0)
        m.ar(y, 1)
        torch.cuda.synchronize()
        bad += [int((x.view(I16) != y.view(I16)).sum()), int((y.view(I16) != rf.view(I16)).sum())]
    b0, b1 = both(bad)
    ok = not any(b0) and not any(b1)
    res["bitwise"] = {c[0]: {"vs_nccl": [b0[2 * i], b1[2 * i]], "vs_ref": [b0[2 * i + 1], b1[2 * i + 1]]}
                      for i, c in enumerate(cases)}
    if R0:
        (qlog.ok if ok else qlog.warn)(
            "rdma bits", f"{len(cases)} sizes · bitwise equal to NCCL {'yes' if ok else 'no'} · "
                         f"equal to bf16(a+b) {'yes' if ok else 'no'} (both ranks)")
        if not ok:
            qlog.warn("rdma bits", f"mismatches rank0 {b0} · rank1 {b1}")

# ---- 2. latency / bandwidth ---------------------------------------------------------------------------------------
if "lat" not in SKIP:
    lat = {}
    for rows in ([int(x) for x in a.rows.split(",")] if a.rows else ROWS):
        it, reps = (50, 15) if rows <= 64 else (5, 9)
        row = {}
        for which, nm in ((1, "rdma"), (0, "nccl")):
            for graph in (False, True):
                b, md = m.time_ar(rows * H, it, graph, which, reps)
                row[f"{nm}_{'graph' if graph else 'eager'}"] = [round(b, 2), round(md, 2)]
        lat[rows] = row
        if R0:
            r_, n_ = row["rdma_graph"], row["nccl_graph"]
            bw = f" · {rows * H * 2 / r_[0] / 1e3:.1f} GB/s (NCCL {rows * H * 2 / n_[0] / 1e3:.1f})" if rows > 64 else ""
            qlog.ok("rdma latency", f"[{rows},2560] graph {r_[0]:.1f} µs (median {r_[1]:.1f}) · "
                    f"eager {row['rdma_eager'][0]:.1f} · NCCL graph {n_[0]:.1f} (median {n_[1]:.1f}) · "
                    f"eager {row['nccl_eager'][0]:.1f}{bw}")
    res["latency_us"] = lat

# ---- 3. overlap with compute-bound GEMMs ----------------------------------------------------------------------------
# Two loads on a second stream: 2× bf16 GEMM 8192³ (cuBLAS, also streams operands from DRAM) and 400× GEMM 1536³
# (operands stay in L2: pure SM work). Each runs alone and under a running all-reduce loop (NCCL / RdmaAR, a graph on
# its own stream, started first and ≈2× longer, so the GEMMs run entirely under transfers). Best of 3 (shared GPUs).
if "overlap" not in SKIP:
    s2 = torch.cuda.Stream()
    big = [torch.randn(8192, 8192, device="cuda").bfloat16() for _ in range(3)]
    sml = [torch.randn(1536, 1536, device="cuda").bfloat16() for _ in range(3)]
    loads = {"GEMM 8192³×2": (big, 2), "GEMM 1536³×400": (sml, 400)}
    xb = {r: torch.zeros(r * H, dtype=torch.bfloat16, device="cuda") for r in (8192, 2048)}
    COUNT = {8192: 32, 2048: 128}

    def gemm(name):
        (A, B, Cm), n = loads[name]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(s2):
            e0.record()
            for _ in range(n):
                torch.mm(A, B, out=Cm)
            e1.record()
        return e0, e1

    def run(name, rows, which, with_gemm):  # → (GEMM ms, all-reduce loop ms); which None = no all-reduce loop
        m.barrier()
        if which is not None:
            m.loop_start(xb[rows], COUNT[rows], which)
        e = gemm(name) if with_gemm else None
        tl = m.loop_wait() if which is not None else 0.0
        torch.cuda.synchronize()
        return (e[0].elapsed_time(e[1]) if e else 0.0), tl

    for name in loads:
        gemm(name)
    torch.cuda.synchronize()
    ov = {}
    for name in loads:
        for rows in (8192, 2048):
            best = {}
            for _ in range(3):
                for key, w, wg in (("alone", None, 1), ("nccl", 0, 1), ("rdma", 1, 1), ("nccl_loop", 0, 0), ("rdma_loop", 1, 0)):
                    g, t = run(name, rows, w, wg)
                    o = best.setdefault(key, [1e9, 1e9])
                    o[0], o[1] = min(o[0], g or 1e9), min(o[1], t or 1e9)
            g0 = best["alone"][0]
            d = {k: [round(v[0], 3) if v[0] < 1e9 else None, round(v[1], 3) if v[1] < 1e9 else None] for k, v in best.items()}
            d["slowdown_pct"] = {k: round(100 * (best[k][0] / g0 - 1), 1) for k in ("nccl", "rdma")}
            ov[f"{name} · [{rows},2560]"] = d
            if R0:
                per = lambda k: best[k][1] / COUNT[rows] * 1e3  # noqa: E731  µs per all-reduce
                qlog.ok("rdma overlap", f"{name} {g0:.1f} ms alone · [{rows},2560] AR concurrent → slowdown RDMA "
                        f"{d['slowdown_pct']['rdma']:+.1f}% · NCCL {d['slowdown_pct']['nccl']:+.1f}% · "
                        f"one AR alone→concurrent RDMA {per('rdma_loop'):.0f}→{per('rdma'):.0f} µs · "
                        f"NCCL {per('nccl_loop'):.0f}→{per('nccl'):.0f} µs")
    res["overlap"] = ov
    del big, sml, xb

# ---- 4. 1000 back-to-back calls, every result checked --------------------------------------------------------------
if "stress" not in SKIP:
    plan = [1, 4, 1, 16, 64, 1, 256, 2048, 1, 8192]  # LL · fused · large paths, both parities
    K = 3
    sizes = sorted(set(plan))
    inp = {r: [gen(r * H, a.rank, j + 1) for j in range(K)] for r in sizes}
    refs = {r: [ref(r * H, j + 1) for j in range(K)] for r in sizes}
    buf = {r: torch.empty(r * H, dtype=torch.bfloat16, device="cuda") for r in sizes}
    mis_e = torch.zeros((), dtype=torch.int64, device="cuda")
    for i in range(1000):
        r, j = plan[i % len(plan)], (i // len(plan)) % K
        buf[r].copy_(inp[r][j])
        m.ar(buf[r], 1)
        mis_e += (buf[r].view(I16) != refs[r][j].view(I16)).sum()
    # same size back to back: each parity's receive staging is reused every 2nd call with other data (stale reads)
    for r in (1, 16, 64, 2048):
        for i in range(150 if r < 2048 else 60):
            buf[r].copy_(inp[r][i % K])
            m.ar(buf[r], 1)
            mis_e += (buf[r].view(I16) != refs[r][i % K].view(I16)).sum()
    torch.cuda.synchronize()
    # graph: 10 calls captured once, replayed 100× with new inputs copied in between
    sin = {r: torch.empty_like(buf[r]) for r in sizes}
    sref = {r: torch.empty_like(buf[r]) for r in sizes}
    mis_g = torch.zeros((), dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for r in plan:
            buf[r].copy_(sin[r])
            m.ar(buf[r], 1)
            mis_g += (buf[r].view(I16) != sref[r].view(I16)).sum()
    for rep in range(100):
        for r in sizes:
            sin[r].copy_(inp[r][rep % K])
            sref[r].copy_(refs[r][rep % K])
        g.replay()
    torch.cuda.synchronize()
    b0, b1 = both([int(mis_e), int(mis_g)])
    ok = not any(b0) and not any(b1)
    res["stress"] = {"eager_mismatch": [b0[0], b1[0]], "graph_mismatch": [b0[1], b1[1]], "calls": [1000, 1000]}
    if R0:
        (qlog.ok if ok else qlog.warn)(
            "rdma back-to-back", f"1000+510 eager + 1000 graph calls (1–8192 rows mixed·same size in a row) · "
                                 f"mismatches eager {b0[0]}/{b1[0]} · graph {b0[1]}/{b1[1]} (rank0/rank1)")

m.barrier()
if R0:
    json.dump(res, open(f"{OUT}/rdma.json", "w"), indent=1)
    qlog.ok("rdma", f"done · {OUT}/rdma.json")
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()  # flushes an nsys trace (no-op otherwise) before the hard exit
sys.stdout.flush()
os._exit(0)  # skip NCCL/ibverbs teardown at interpreter exit (it can block once the peer is gone)
