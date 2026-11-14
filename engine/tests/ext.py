"""Build/load test bindings: one torch extension per kernel family (tests/bind/<family>.cpp + engine/src/kernels/<family>.cu
[+ extra sources]). Built inside the dev container; cache in /work/cache/ext (project). Compiler noise goes to the build log."""
import contextlib
import io
import os

from torch.utils import cpp_extension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # engine/
CACHE = os.environ.get("Q38_EXT_CACHE", "/work/cache/ext")
FLAGS = ["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_121a,code=sm_121a", "-lineinfo", "-std=c++17"]
_mods = {}


def engine_sources():
    """All engine sources, like CMakeLists: src/core/*.cc, src/model/*.cc, src/kernels/<family>/*.cu (patches included)."""
    import glob
    pats = ("src/core/*.cc", "src/model/*.cc", "src/kernels/*/*.cu")
    return sorted(os.path.relpath(f, ROOT) for p in pats for f in glob.glob(os.path.join(ROOT, p)))


ENGINE_LDFLAGS = ["-lnccl", "-libverbs", "-lcuda"]  # what the whole engine links against (as CMakeLists)


def stale(family, sources=()):
    """Why load() will compile: "first build" (nothing built here yet), "sources changed" (a source or header newer
    than the built module), or None (the cached module is used)."""
    import glob
    so = os.path.join(CACHE, family, f"q38_{family}.so")
    if not os.path.exists(so):
        return "first build"
    t = os.path.getmtime(so)
    files = [os.path.join(ROOT, "tests", "bind", f"{family}.cpp")] + [os.path.join(ROOT, s) for s in sources] + \
        [f for p in ("*.h", "*.cuh") for f in glob.glob(os.path.join(ROOT, "src", "**", p), recursive=True)]
    return "sources changed" if any(os.path.getmtime(f) > t for f in files) else None


def load(family, sources=(), log=None, ldflags=()):
    """family: name of tests/bind/<family>.cpp; sources: engine-relative paths (e.g. 'src/kernels/hc.cu')."""
    if family in _mods:
        return _mods[family]
    os.makedirs(os.path.join(CACHE, family), exist_ok=True)
    srcs = [os.path.join(ROOT, "tests", "bind", f"{family}.cpp")] + [os.path.join(ROOT, s) for s in sources]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        m = cpp_extension.load(name=f"q38_{family}", sources=srcs, extra_include_paths=[os.path.join(ROOT, "src")],
                               extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=FLAGS,
                               build_directory=os.path.join(CACHE, family), extra_ldflags=list(ldflags), with_cuda=True,
                               verbose=False)
    if log:
        with open(log, "a") as f:
            f.write(buf.getvalue())
    _mods[family] = m
    return m
