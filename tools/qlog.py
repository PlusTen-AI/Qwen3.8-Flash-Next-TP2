"""Common logger: one line per step, `[tag] point`. Green = OK, red = fail, yellow = warn.

Screen gets summaries only. Anything long (tracebacks, library noise, tables) goes to the detail
file and the screen shows its path. Repeated warnings print once and are counted at the end.
NO_COLOR=1 turns colors off.
"""
import atexit
import os
import sys
import time

_C = not os.environ.get("NO_COLOR")
_CYAN, _GREEN, _RED, _YEL, _RST = ("\033[36m", "\033[32m", "\033[31m", "\033[33m", "\033[0m") if _C else ("",) * 5
_MAG = "\033[35m" if _C else ""
# The mark: two linked rings, the two machines as one (a = the head's ring, b = the worker's; each passes over the
# other once)
_LOGO = (("╭────╮   ", "aaaaaa   "),
         ("│  ╭─┼──╮", "a  bbabbb"),
         ("╰──┼─╯  │", "aaabaa  b"),
         ("   ╰────╯", "   bbbbbb"))
_detail = None
_warn_counts = {}
_t0 = time.time()


def _out(tag, msg, color=""):
    sys.stdout.write(f"{_CYAN}[{tag}]{_RST} {color}{msg}{_RST if color else ''}\n")
    sys.stdout.flush()


def banner(lines):
    """The mark with up to four lines of text beside it (the server's first screen lines)."""
    lines = list(lines)[:len(_LOGO)]
    for i, (art, mask) in enumerate(_LOGO):
        s, cur = "", None
        for ch, m in zip(art, mask):
            want = {"a": _CYAN, "b": _MAG}.get(m, "")
            if want != cur:
                s += (_RST if cur else "") + want
                cur = want
            s += ch
        s += _RST if cur else ""
        sys.stdout.write(f"{s}   {lines[i] if i < len(lines) else ''}".rstrip() + "\n")
    sys.stdout.flush()


def info(tag, msg):
    _out(tag, msg)


def ok(tag, msg):
    _out(tag, msg, _GREEN)


def warn(tag, msg, key=None):
    """Print once per key; later repeats are only counted."""
    k = key or (tag, msg)
    n = _warn_counts.get(k, 0)
    _warn_counts[k] = n + 1
    if n == 0:
        _out(tag, msg, _YEL)


def error(tag, msg):
    """A red line without exiting (the caller decides how to leave)."""
    _out(tag, msg, _RED)


def fail(tag, msg, code=1):
    _out(tag, msg, _RED)
    sys.exit(code)


def check(tag, cond, msg_ok, msg_bad):
    """ok() if cond else fail()."""
    if cond:
        ok(tag, msg_ok)
    else:
        fail(tag, msg_bad)


def detail_file(path):
    """Open the detail file (append). Returns the file object; detail() writes to it."""
    global _detail
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _detail = open(path, "a", buffering=1)
    return _detail


def detail(text):
    if _detail is not None:
        _detail.write(text if text.endswith("\n") else text + "\n")


def elapsed():
    return time.time() - _t0


@atexit.register
def _summary():
    rep = {k: n for k, n in _warn_counts.items() if n > 1}
    for (tag, msg), n in ((k if isinstance(k, tuple) else (k, k), n) for k, n in rep.items()):
        _out(tag, f"{msg} ×{n}", _YEL)
