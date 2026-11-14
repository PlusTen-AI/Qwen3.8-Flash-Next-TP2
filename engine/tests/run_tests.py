"""Batch test runner — every `test_*` function in engine/tests/test_*.py, one screen line each.

Run in the dev container on the node of the rank under test (rank0 = C, rank1 = D):
  python3 engine/tests/run_tests.py --rank 0 --dump /work/results/raw/dump/<id>/rank0 [-k pattern] [--out DIR]
A test gets a Ctx and returns a dict {"msg": str, "ok": bool} (or raises). Builds, tracebacks and per-test details go to
<out>/tests.log; the screen shows `[test name] point` only.
"""
import argparse
import glob
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))
import qlog  # noqa: E402

import ext  # noqa: E402
from golden import Golden  # noqa: E402


class Ctx:
    def __init__(self, a):
        self.rank, self.dump, self.out = a.rank, a.dump, a.out
        self.log = os.path.join(a.out, "tests.log")
        self.golden = Golden(a.dump, os.path.join(os.path.dirname(a.dump), "tokens.json"))
        self._w = None
        self.model_dir = a.model
        self.table = a.table

    @property
    def w(self):
        """Production loader (engine/src/weights.cc) through its test binding: w.slice(name) -> (u8, shape, dtype, tp)."""
        if self._w is None:
            self._w = ext.load("loader", ["src/core/weights.cc"], log=self.log)
            self._w.open(self.model_dir, self.table, self.rank)
        return self._w

    def ext(self, family, sources):
        return ext.load(family, sources, log=self.log)

    def detail(self, text):
        with open(self.log, "a") as f:
            f.write(text.rstrip() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--model", default="/models/Qwen3.8-Flash-Next-FP4-Local")
    ap.add_argument("--table", default="/work/results/tensor_table.tsv")
    ap.add_argument("--out", default="/work/results/tests")
    ap.add_argument("-k", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ctx = Ctx(a)
    n_ok = n_bad = 0
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            full = f"{mod.__name__[5:]}.{name[5:]}"
            if a.k and a.k not in full:
                continue
            t0 = time.time()
            try:
                r = getattr(mod, name)(ctx) or {"ok": True, "msg": ""}
            except Exception as e:  # noqa: BLE001
                ctx.detail(f"== {full}\n{traceback.format_exc()}")
                r = {"ok": False, "msg": f"{type(e).__name__}: {str(e)[:120]} · {ctx.log}"}
            dt = time.time() - t0
            (qlog.ok if r["ok"] else qlog.warn)(f"test {full}", f"{r['msg']} · {dt:.1f}s")
            n_ok += r["ok"]
            n_bad += not r["ok"]
    (qlog.ok if n_bad == 0 else qlog.fail)("test", f"passed {n_ok} · failed {n_bad} · rank{a.rank} · details {ctx.log}")


if __name__ == "__main__":
    main()
