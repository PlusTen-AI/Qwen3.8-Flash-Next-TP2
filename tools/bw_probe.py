"""GPU probe (torch) for the byte-floor calculator: memory bandwidth (read = sum, copy), dense GEMM throughput
(bf16, fp8 e4m3 via _scaled_mm) and kernel launch cost (back-to-back tiny kernels, eager and in a CUDA graph).
One line per measurement; JSON to argv[1]. FP4 throughput is not measured here (no torch path) — the floor marks it."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import qlog  # noqa: E402

out = sys.argv[1]
ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731


def timeit(fn, it=20, warm=3):
    for _ in range(warm):
        fn()
    s, e, ts = ev(), ev(), []
    for _ in range(it):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


p = torch.cuda.get_device_properties(0)
r = {"device": p.name, "sm": p.multi_processor_count, "cc": f"{p.major}.{p.minor}"}

x = torch.empty(2 * 2**30, dtype=torch.bfloat16, device="cuda").uniform_()  # 4 GiB
y = torch.empty_like(x)
b, m = timeit(lambda: x.sum(dtype=torch.float32))
r["read_best"], r["read_med"] = x.nbytes / b / 1e9, x.nbytes / m / 1e9
b, m = timeit(lambda: y.copy_(x))
r["copy_best"], r["copy_med"] = 2 * x.nbytes / b / 1e9, 2 * x.nbytes / m / 1e9
del x, y
qlog.ok("bw", f"{p.name} · SM {p.multi_processor_count} · read {r['read_best']:.0f} (median {r['read_med']:.0f}) · "
              f"copy {r['copy_best']:.0f} GB/s")

n = 8192
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
b, _ = timeit(lambda: a @ w, it=10)
r["bf16_tflops"] = 2 * n**3 / b / 1e12
try:
    a8, w8 = a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn).t()
    one = torch.ones((), device="cuda")
    b, _ = timeit(lambda: torch._scaled_mm(a8, w8, one, one, out_dtype=torch.bfloat16), it=10)
    r["fp8_tflops"] = 2 * n**3 / b / 1e12
except Exception as e:  # noqa: BLE001
    r["fp8_tflops"] = None
    qlog.warn("gemm", f"fp8 not measurable ({type(e).__name__})")
fp8 = f"{r['fp8_tflops']:.0f}" if r["fp8_tflops"] else "–"
qlog.ok("gemm", f"{n}³ · bf16 {r['bf16_tflops']:.0f} · fp8 {fp8} TFLOPS (torch, measured)")
del a, w

t = torch.zeros(1, device="cuda")
K = 1000
b, _ = timeit(lambda: [t.add_(1) for _ in range(K)], it=5)
r["launch_eager_us"] = b / K * 1e6
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    t.add_(1)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    for _ in range(K):
        t.add_(1)
b, _ = timeit(g.replay, it=10)
r["launch_graph_us"] = b / K * 1e6
qlog.ok("launch", f"per small kernel · eager {r['launch_eager_us']:.1f} µs · CUDA graph {r['launch_graph_us']:.2f} µs")
json.dump(r, open(out, "w"), indent=1)
