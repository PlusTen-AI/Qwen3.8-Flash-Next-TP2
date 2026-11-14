"""(development, this site; not in the public export) Mixed-load soak test against a running server, with a verdict.

  python3 tools/soak.py URL --corpus C.jsonl --books DIR --media DIR --out DIR [--minutes 60]
  python3 tools/soak.py --judge DIR          # the verdict again from a finished run's files

Load (all at once, for --minutes; a lane starts no new request after that, running ones finish):
  trace    recorded Hermes subagent sessions (tools/soak_corpus.py), every assistant step sent as Hermes sent it (the
           recorded messages before it, the session's tools, streaming, thinking on); our reply is checked and dropped
  live     the same sessions, but our own replies go back (reasoning and tool calls) and each of our tool calls gets the
           next recorded tool result — the path where the slot's own history continues
  window   long main sessions (telegram / cli / local): a cold read from a step in the middle, then 20 steps
  brief    the cron sessions as they ran: the main session's steps, and after each delegate_task step its
           subagents (the recorded ones, at the same time), then the title call
  aux      title calls (short, thinking off) and context-compression summaries (~40K tokens in, thinking off)
  media    pictures and a video in turn: a 12-megapixel photo-like picture with a number and a coloured square (checked),
           a small one (checked), the dump's chart (checked: 4721), scene and 12 s video
  long     a ~640K-token text read cold (thinking off, records asked: checked), a second question on the same text right
           after (reuse), and at 2/3 of the run another ~600K-token text read cold
  cut      a thinking stream closed by the client after 20 chunks, every 6 min
  probe    the same seeded request before the load, every 10 min during it and after it: the same text every time
           (the engine is deterministic for a seed and batch-invariant)
Before the load: one 12 MP picture and the video (the vision tower's memory is taken), then 2 min idle (baseline).
After it: 2 min idle. /metrics every 5 s. Node samples (tools/soak_nodes.sh: MemAvailable, GPU use / power / SM clock /
temperature / clock-event reasons every 2 s, container CPU and memory every 10 s) are read from OUT/nodes/.

Verdict (OUT/verdict.txt, last line PASS / FAIL): no error or broken stream · longest gap between two text chunks of
a stream ≤ 30 s, and the same for the gaps before tool-call pieces (calls stream as vLLM does since 2026-11-13: the
name, then argument pieces; a typed value — array, object, number … — goes out whole once written, as in vLLM, so the
wait before it is reported apart, not judged) · prompt reuse on continued steps (median ≥ 0.98 of the previous prompt, ≥ 90% of steps ≥ 0.9) · every
checked answer right · probes identical · tool calls well formed (a call the reply ended inside — max_tokens or the
model's end — stays unclosed as in vLLM: counted apart; the bad ones are kept in the record) · server idle at the end · per node MemAvailable after
the load within 1 GiB of before it and never under 4 GiB. Rank balance (C = rank 0, D = rank 1) is reported: GPU use,
power, SM clock, clock-event reasons, memory and CPU per node over the load, and both GPUs while the vision tower runs
(rank 0 only)."""
import argparse
import base64
import bisect
import hashlib
import http.client
import json
import os
import random
import re
import statistics
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://localhost:8888/v1")
ap.add_argument("--corpus", help="tools/soak_corpus.py output")
ap.add_argument("--books", help="folder with pg2600.txt and pg1399.txt (checks/needle.py's books)")
ap.add_argument("--media", help="folder with chart.png, scene.png, long.mp4 (tools/dump/media.py)")
ap.add_argument("--out", help="run folder")
ap.add_argument("--minutes", type=float, default=60)
ap.add_argument("--max-tokens", type=int, default=8192, help="per agent step")
ap.add_argument("--long", default="640,600", help="the two cold long reads, K tokens (1K = 1000)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--idle", type=float, default=120, help="idle seconds before and after the load")
ap.add_argument("--tick", type=float, default=1.0, help="scales every lane's waits (a short trial run: 0.1)")
ap.add_argument("--judge", help="only the verdict, from this finished run folder")
a = ap.parse_args()

GAP_MAX, REUSE_MED, REUSE_LOW, MEM_DROP, MEM_FLOOR = 30.0, 0.98, 0.9, 1.0, 4.0
GiB = 1024 ** 3


# ---- verdict ------------------------------------------------------------------------------------------------------
def judge(out):
    lines = []

    def say(ok, tag, msg):
        s = f"{'ok  ' if ok is True else 'FAIL' if ok is False else 'info'} {tag:10s} {msg}"
        lines.append((ok, s))
        print(s, flush=True)

    reqs = [json.loads(x) for x in open(os.path.join(out, "requests.jsonl"))]
    ph = json.load(open(os.path.join(out, "phases.json")))
    L0, L1 = ph["load_start"], ph["load_end"]
    kinds = {}
    for r in reqs:
        kinds.setdefault(r["lane"], []).append(r)
    say(None, "requests", " · ".join(f"{k} {len(v)}" for k, v in sorted(kinds.items())) +
        f" · load {(ph['drained'] - L0) / 60:.1f} min")
    tok_in = sum(r.get("prompt") or 0 for r in reqs)
    tok_c = sum(r.get("cached") or 0 for r in reqs)
    tok_out = sum(r.get("completion") or 0 for r in reqs)
    say(None, "tokens", f"prompt {tok_in:,} (reused {tok_c:,} = {tok_c / max(tok_in, 1):.0%}) · output {tok_out:,} · "
                        f"output {tok_out / max(ph['drained'] - L0, 1):.1f} tok/s over the load")
    errs = [r for r in reqs if r["lane"] != "cut" and (r["status"] != 200 or r.get("error"))]
    say(not errs, "errors", f"{len(errs)} of {len(reqs)}" +
        "".join(f"\n            {r['lane']} {r.get('sid', '')} {r['status']} {str(r.get('error'))[:160]}" for r in errs[:8]))
    broken = [r for r in reqs if r.get("stream") and r["status"] == 200 and r["lane"] != "cut" and
              not (r.get("done") and r.get("finish"))]
    say(not broken, "streams", f"{len(broken)} without finish_reason + [DONE] · finish " +
        " · ".join(f"{k} {v}" for k, v in sorted(_count(r.get("finish") for r in reqs if r["lane"] not in ("cut", "check")).items(), key=str)))
    for r in reqs:  # a call cut by max_tokens comes out as text at the end (runs before this rule logged it as text)
        if r.get("finish") == "length" and "<tool_call>" in (r.get("text") or "") and r.get("max_gap"):
            r["tool_gap"], r["max_gap"], r["cut_call"] = max(r.get("tool_gap") or 0, r["max_gap"]), None, True
    gaps = [r["max_gap"] for r in reqs if r.get("max_gap") is not None]
    if gaps:
        worst = max(reqs, key=lambda r: r.get("max_gap") or 0)
        say(max(gaps) <= GAP_MAX, "gaps", f"longest gap between text chunks {max(gaps):.1f} s ({worst['lane']} "
            f"{worst.get('sid', '')} step {worst.get('step', '')}) · p99 of per-stream max {_pct(gaps, 99):.1f} s · "
            f"median {_pct(gaps, 50):.2f} s (limit {GAP_MAX:.0f})")
    hg = [r for r in reqs if r.get("typed_gap")]
    if hg:
        w = max(hg, key=lambda r: r["typed_gap"])
        say(None, "typed args", f"wait before a typed argument value (array / object / number, sent whole once written, "
            f"as vLLM): {len(hg)} replies · median {_pct([r['typed_gap'] for r in hg], 50):.1f} s · longest "
            f"{w['typed_gap']:.1f} s ({w['lane']} {w.get('sid', '')} step {w.get('step', '')}, output {w.get('completion')})")
    tg = [r for r in reqs if r.get("tool_gap")]
    if tg:
        w = max(tg, key=lambda r: r["tool_gap"])
        say(w["tool_gap"] <= GAP_MAX or bool(w.get("cut_call")), "call gaps",
            f"longest silence before a tool-call piece (name, then argument pieces): {len(tg)} "
            f"replies · median {_pct([r['tool_gap'] for r in tg], 50):.1f} s · p90 {_pct([r['tool_gap'] for r in tg], 90):.1f}"
            f" s · longest {w['tool_gap']:.1f} s ({w['lane']} {w.get('sid', '')} step {w.get('step', '')}, output "
            f"{w.get('completion')}) · {sum(bool(r.get('cut_call')) for r in tg)} cut by max_tokens inside the call · "
            f"over 180 s (Hermes' cloud stream stale limit; 900 s for a private address): "
            f"{sum(r['tool_gap'] > 180 for r in tg)}")
    ttft = [r["ttft"] for r in reqs if r.get("ttft") is not None]
    if ttft:
        say(None, "first tok", f"median {_pct(ttft, 50):.2f} s · p90 {_pct(ttft, 90):.1f} s · max {max(ttft):.1f} s")
    cont = [r for r in reqs if r["lane"] in ("trace", "live", "window", "brief") and r.get("prev_prompt")
            and r["status"] == 200 and r.get("prompt")]
    if cont:
        rat = [min(r["cached"], r["prev_prompt"]) / r["prev_prompt"] for r in cont]
        low = sum(x < REUSE_LOW for x in rat)
        hist = sum(r["cached"] > r["prev_prompt"] for r in cont if r["lane"] == "live")
        nlive = sum(r["lane"] == "live" for r in cont)
        say(_pct(rat, 50) >= REUSE_MED and low <= 0.1 * len(rat), "reuse",
            f"{len(cont)} continued steps: reused / previous prompt median {_pct(rat, 50):.3f} · p10 "
            f"{_pct(rat, 10):.3f} · {low} under {REUSE_LOW} · live steps past the previous prompt (own history) "
            f"{hist}/{nlive}")
    chk = [r for r in reqs if r.get("check") is not None]
    bad = [r for r in chk if not r["check"]]
    say(not bad, "answers", f"{len(chk) - len(bad)}/{len(chk)} right" +
        "".join(f"\n            {r['lane']} {r.get('what', '')}: {str(r.get('text'))[:120]!r}" for r in bad[:6]))
    pr = [r for r in reqs if r["lane"] == "probe"]
    same = len({r.get("text") for r in pr}) == 1 and all(r["status"] == 200 for r in pr)
    say(same and len(pr) >= 2, "probes", f"{len(pr)} probes, {'all the same text' if same else 'DIFFERENT texts'}")
    tc = [r for r in reqs if r.get("tools_n")]
    mal = sum(r.get("malformed") or 0 for r in tc)
    unc = [r for r in tc if r.get("unclosed")]
    unk = [n for r in tc for n in (r.get("unknown_tool") or [])]

    def _is_obj(t):
        try:
            return isinstance(json.loads(t), dict)
        except ValueError:
            return False
    for r in tc:  # records judged before the split: a closed call with valid JSON under a name not in the tools
        for c in r.get("bad_calls") or []:
            if not c["unclosed"] and c["len"] <= 300 and _is_obj(c["head"]):
                r["malformed"] -= 1
                c["unclosed"] = True
                unk.append(c["name"])
    agree = [r for r in tc if r.get("ref_tool") is not None and r["status"] == 200]
    mal = sum(r.get("malformed") or 0 for r in tc)  # after the records judged before the split
    say(mal == 0, "tool calls", f"{sum(r.get('n_calls') or 0 for r in tc)} calls in {len(tc)} replies · malformed "
        f"{sum(r.get('malformed') or 0 for r in tc)} · unclosed (the reply ended inside the call) {len(unc)}: finish "
        + " · ".join(f"{k} {v}" for k, v in sorted(_count(r.get("finish") for r in unc).items(), key=str))
        + f" · a name not in the request's tools (the model's; passed on as vLLM) {len(unk)}: {sorted(set(unk))}"
        + f" · tool call where the recording had one {sum(bool(r.get('n_calls')) == r['ref_tool'] for r in agree)}"
        f"/{len(agree)} agree" + "".join(f"\n            {r['lane']} {r.get('sid', '')} step {r.get('step', '')}: "
                                        f"{[(c['name'], c['len'], c['tail'][-80:]) for c in r.get('bad_calls', []) if not c['unclosed']]}"
                                        for r in tc if r.get("malformed"))[:1500])
    lng = [r for r in reqs if r["lane"] == "long"]
    for r in lng:
        say(None, "long", f"{r.get('what')}: prompt {r.get('prompt') or 0:,} · reused {r.get('cached') or 0:,} · "
                          f"first token {r.get('ttft') or 0:.1f} s · {r.get('text')!r}")
    end = ph.get("end_state", {})
    say(end.get("running") == 0 and end.get("waiting") == 0 and end.get("health") == "ok", "end state",
        f"running {end.get('running')} · waiting {end.get('waiting')} · health {end.get('health')}")
    # nodes
    nd = os.path.join(out, "nodes")
    media_win = [(r["t0"], r["t0"] + r["ttft"]) for r in reqs if r["lane"] == "media" and r.get("ttft")
                 and r.get("what", "").startswith(("photo12", "video"))]
    per = {}
    for node in ("C", "D"):
        f = os.path.join(nd, f"{node}.csv")
        if not os.path.exists(f):
            say(False, "memory", f"node {node}: no samples ({f})")
            continue
        rows = []
        for x in open(f):
            p = x.strip().split(",")
            try:
                rows.append((float(p[0]), int(p[1]) * 1024 / GiB, float(p[2]), float(p[3]), float(p[4]), float(p[5]),
                             int(p[6].strip(), 16)))
            except (ValueError, IndexError):
                pass
        base = [m for t, m, *_ in rows if ph["base_start"] <= t <= ph["base_end"]]
        after = [m for t, m, *_ in rows if ph["after_start"] <= t <= ph["after_end"]]
        load = [r for r in rows if L0 <= r[0] <= ph["drained"]]
        if not (base and after and load):
            say(False, "memory", f"node {node}: samples missing in a phase")
            continue
        b, af, lo = statistics.median(base), statistics.median(after), min(r[1] for r in load)
        t10 = [r for r in load if r[0] >= L0 + 600]
        slope = _slope([(r[0], r[1]) for r in t10]) * 3600 if len(t10) > 10 else 0.0
        say(b - af <= MEM_DROP and lo >= MEM_FLOOR, "memory",
            f"node {node}: MemAvailable before {b:.2f} GiB · after {af:.2f} ({af - b:+.2f}) · lowest under load "
            f"{lo:.2f} · trend after 10 min {slope:+.2f} GiB/h")
        thr = [r for r in load if r[6] & 0xE8]  # SW power cap, HW slowdown, SW / HW thermal, power brake
        inw = [r for r in load if any(s <= r[0] <= e for s, e in media_win)]
        outw = [r for r in load if not any(s <= r[0] <= e for s, e in media_win)]
        per[node] = dict(util=_mean(r[2] for r in load), power=_mean(r[3] for r in load),
                         clk=_mean(r[4] for r in load), clk_min=min(r[4] for r in load), temp=max(r[5] for r in load),
                         thr=len(thr) / len(load), mem=_mean(r[1] for r in load),
                         util_tower=_mean(r[2] for r in inw), util_rest=_mean(r[2] for r in outw),
                         power_tower=_mean(r[3] for r in inw), n_tower=len(inw))
        dk = os.path.join(nd, f"{node}_docker.csv")
        if os.path.exists(dk):
            cpu = [float(p.split(",")[2].rstrip("%")) for p in open(dk)
                   if p.count(",") >= 3 and L0 <= float(p.split(",")[0]) <= ph["drained"]]
            per[node]["cpu"] = _mean(cpu)
    if len(per) == 2:
        c, d = per["C"], per["D"]
        for k, lab, fmt in (("util", "GPU use %", "{:.0f}"), ("power", "power W", "{:.1f}"), ("clk", "SM clock MHz", "{:.0f}"),
                            ("clk_min", "lowest SM clock", "{:.0f}"), ("temp", "hottest °C", "{:.0f}"),
                            ("thr", "time clock-limited", "{:.1%}"), ("mem", "MemAvailable GiB", "{:.1f}"),
                            ("cpu", "container CPU %", "{:.0f}"), ("util_tower", "GPU use % while tower runs", "{:.0f}"),
                            ("util_rest", "GPU use % otherwise", "{:.0f}"),
                            ("power_tower", "power W while tower runs", "{:.1f}")):
            if c.get(k) is not None and d.get(k) is not None:
                say(None, "ranks", f"{lab:28s} C {fmt.format(c[k]):>8s} · D {fmt.format(d[k]):>8s}")
        say(None, "ranks", f"tower samples {c['n_tower']} (12 MP picture and video, from request start to first token)")
    ok = all(x is not False for x, _ in lines)
    lines.append((ok, f"{'PASS' if ok else 'FAIL'} — soak {os.path.basename(out.rstrip('/'))}"))
    print(lines[-1][1])
    with open(os.path.join(out, "verdict.txt"), "w") as f:
        f.write("\n".join(s for _, s in lines) + "\n")
    return ok


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _slope(pts):
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den if den else 0.0


if a.judge:
    sys.exit(0 if judge(a.judge) else 1)

# ---- client -------------------------------------------------------------------------------------------------------
BASE = a.url.rstrip("/")
U = urllib.parse.urlparse(BASE)
ROOT = BASE.rsplit("/v1", 1)[0]
os.makedirs(a.out, exist_ok=True)
LOG = open(os.path.join(a.out, "requests.jsonl"), "a")
LOCK = threading.Lock()
PH = {}
STOP_AT = [float("inf")]


def say(tag, msg):
    print(f"{time.strftime('%H:%M:%S')} [{tag}] {msg}", flush=True)


def get(path, timeout=30):
    with urllib.request.urlopen(ROOT + path, timeout=timeout) as r:
        return r.read().decode()


def post(path, body, timeout=900):
    req = urllib.request.Request(ROOT + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


MODEL = json.loads(get("/v1/models"))["data"][0]["id"]
CTX = json.loads(get("/v1/models"))["data"][0]["max_model_len"]


def call(lane, body, cut=None, **meta):
    """One chat request; streams unless body says stream False. Returns (and logs) a record."""
    body = dict(body, model=MODEL)
    stream = body.setdefault("stream", True)
    if stream:
        body["stream_options"] = {"include_usage": True}
    rec = dict(meta, lane=lane, t0=time.time(), stream=stream, status=None)
    names = {t["function"]["name"] for t in body.get("tools") or []}
    if names:
        rec["tools_n"] = len(names)
    text, reas, calls = [], [], {}
    try:
        conn = http.client.HTTPConnection(U.hostname, U.port or 80, timeout=3600)
        conn.request("POST", U.path + "/chat/completions", json.dumps(body).encode(),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["status"] = resp.status
        if resp.status != 200:
            rec["error"] = resp.read()[:400].decode(errors="replace")
        elif not stream:
            r = json.loads(resp.read())
            m = r["choices"][0]["message"]
            text.append(m.get("content") or "")
            reas.append(m.get("reasoning") or m.get("reasoning_content") or "")
            for i, c in enumerate(m.get("tool_calls") or []):
                calls[i] = {"id": c["id"], "name": c["function"]["name"], "args": c["function"]["arguments"]}
            rec["finish"], rec["done"] = r["choices"][0]["finish_reason"], True
            _usage(rec, r.get("usage"))
        else:
            last, gap, tgap, hgap, n = None, 0.0, 0.0, 0.0, 0
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                now = time.time()
                if data == b"[DONE]":
                    rec["done"] = True
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    _usage(rec, ev["usage"])
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    got = d.get("content") or d.get("reasoning") or d.get("reasoning_content") or d.get("tool_calls")
                    if got:
                        if last is None:
                            rec["ttft"] = now - rec["t0"]
                        elif (d.get("tool_calls") and not (d.get("content") or d.get("reasoning") or
                                                           d.get("reasoning_content"))) or "<tool_call>" in (d.get("content") or ""):
                            # before a tool-call piece: a typed value written whole (the arguments so far end at a key)
                            # or any other piece (or a call cut by max_tokens, sent as text)
                            if any(calls.get(t.get("index", 0), {}).get("args", "").endswith(": ")
                                   for t in d.get("tool_calls") or []):
                                hgap = max(hgap, now - last)
                            else:
                                tgap = max(tgap, now - last)
                        else:
                            gap = max(gap, now - last)
                        last, n = now, n + 1
                    text.append(d.get("content") or "")
                    reas.append(d.get("reasoning") or d.get("reasoning_content") or "")
                    for tcd in d.get("tool_calls") or []:
                        c = calls.setdefault(tcd.get("index", 0), {"id": None, "name": "", "args": ""})
                        c["id"] = tcd.get("id") or c["id"]
                        f = tcd.get("function") or {}
                        c["name"] += f.get("name") or ""
                        c["args"] += f.get("arguments") or ""
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
                if cut and n >= cut:
                    rec["error"], rec["status"] = None, "closed"
                    break
            rec["max_gap"], rec["tool_gap"], rec["chunks"] = (gap if n > 1 else None), (tgap or None), n
            rec["typed_gap"] = hgap or None
        conn.close()
    except Exception as e:  # noqa: BLE001 — every failure is a record
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["dur"] = time.time() - rec["t0"]
    rec["text"] = "".join(text)
    rec["n_reason"] = len("".join(reas))
    cl = [calls[k] for k in sorted(calls)]
    rec["n_calls"] = len(cl)
    if names:
        bad = [c for c in cl if c["name"] not in names or not _json_obj(c["args"])]
        # a call the reply ended inside (its arguments never closed — vLLM streams it so too) and a name the request's
        # tools do not have (the model's choice; vLLM passes it on, the client answers it) are counted apart
        open_ = [c for c in bad if c["name"] in names and not c["args"].rstrip().endswith("}") and c is cl[-1]]
        unknown = [c for c in bad if c["name"] not in names and _json_obj(c["args"])]
        rec["malformed"], rec["unclosed"] = len(bad) - len(open_) - len(unknown), len(open_)
        rec["unknown_tool"] = [c["name"] for c in unknown] or None
        if bad:
            rec["bad_calls"] = [{"name": c["name"], "unclosed": c in open_ or c in unknown, "len": len(c["args"]),
                                 "head": c["args"][:300], "tail": c["args"][-300:]} for c in bad]
        rec["malformed"] += "<tool_call>" in rec["text"] and rec.get("finish") != "length"  # a cut call stays text
    out = dict(rec)
    out["calls"], out["reasoning"] = cl, "".join(reas)
    if lane not in ("long", "media", "probe", "aux"):
        rec["text"] = rec["text"][:200]
    with LOCK:
        LOG.write(json.dumps(rec, ensure_ascii=False) + "\n")
        LOG.flush()
    return out


def _usage(rec, u):
    if u:
        rec["prompt"], rec["completion"] = u.get("prompt_tokens"), u.get("completion_tokens")
        rec["cached"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def _json_obj(s):
    try:
        return isinstance(json.loads(s), dict)
    except ValueError:
        return False


def running():
    return time.time() < STOP_AT[0]


def sid(s):
    return hashlib.sha256(s["id"].encode()).hexdigest()[:8]


# ---- corpus -------------------------------------------------------------------------------------------------------
SESS = [json.loads(x) for x in open(a.corpus)]
SUBS = [s for s in SESS if s["source"] == "subagent" and s["tools"]]
MAINS = [s for s in SESS if s["source"] in ("telegram", "cli", "local") and s["tools"] and len(s["messages"]) > 60]
CRON = [s for s in SESS if s["source"] == "cron"]
KIDS = {}
for s in SESS:
    if s["parent"]:
        KIDS.setdefault(s["parent"], []).append(s)
CAP = int(CTX * 0.55)  # a step's prompt above this ends the session (room for the long reads)


def steps(ms):
    return [i for i, m in enumerate(ms) if m["role"] == "assistant" and i > 0 and ms[i - 1]["role"] in ("user", "tool")]


def chars(ms):
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "")) +
               len(m.get("reasoning_content") or "") for m in ms)


def trace(lane, s, idx, prev=None, **kw):
    ms = s["messages"]
    for n, i in enumerate(idx):
        if not running():
            return
        r = call(lane, {"messages": ms[:i], "tools": s["tools"], "max_tokens": a.max_tokens}, sid=sid(s), step=n,
                 prev_prompt=prev, ref_tool=bool(ms[i].get("tool_calls")), **kw)
        if r["status"] != 200 or not r.get("prompt"):
            return
        prev = r["prompt"]
        if prev > CAP:
            return


def title(s):
    u = next((m["content"] for m in s["messages"] if m["role"] == "user"), "")[:2000]
    call("aux", {"messages": [{"role": "system", "content": "Generate a short, descriptive title (3-7 words) for a "
                                                           "conversation that starts with the following exchange. "
                                                           "Return only the title text."},
                              {"role": "user", "content": u}],
                 "max_tokens": 64, "reasoning_effort": "none", "stream": False}, what="title")


def lane_trace(rng):
    pool = SUBS[:]
    while running():
        rng.shuffle(pool)
        for s in pool:
            if not running():
                return
            trace("trace", s, steps(s["messages"]))


def lane_live(rng):
    pool = SUBS[:]
    while running():
        rng.shuffle(pool)
        for s in pool:
            if not running():
                return
            ms = s["messages"]
            if not steps(ms):
                continue
            first = steps(ms)[0]
            conv = ms[:first]
            results = [m["content"] for m in ms[first:] if m["role"] == "tool"]
            prev, k = None, 0
            for n in range(30):
                if not running():
                    break
                r = call("live", {"messages": conv, "tools": s["tools"], "max_tokens": a.max_tokens}, sid=sid(s), step=n,
                         prev_prompt=prev)
                if r["status"] != 200 or not r.get("prompt") or r.get("finish") == "length" or not r["calls"]:
                    break
                prev = r["prompt"]
                if n == 0:
                    title(s)
                m = {"role": "assistant", "content": r["text"] or None,
                     "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                                    for c in r["calls"]]}
                if r["reasoning"]:
                    m["reasoning_content"] = r["reasoning"]
                conv = conv + [m]
                for c in r["calls"]:
                    if k >= len(results):
                        break
                    conv.append({"role": "tool", "tool_call_id": c["id"], "content": results[k]})
                    k += 1
                if k >= len(results) or prev > CAP:
                    break


def lane_window(rng):
    while running():
        s = rng.choice(MAINS)
        ms, idx = s["messages"], steps(s["messages"])
        ok = [j for j, i in enumerate(idx) if 30_000 <= chars(ms[:i]) <= 450_000]
        if not ok:
            continue
        j = rng.choice(ok)
        trace("window", s, idx[j:j + 20])


def lane_brief(rng):
    while running():
        for p in CRON:
            if not running():
                return
            kids = sorted(KIDS.get(p["id"], []), key=lambda s: s["started"])
            groups = []
            for k in kids:
                if groups and k["started"] - groups[-1][0]["started"] < 120:
                    groups[-1].append(k)
                else:
                    groups.append([k])
            ms = p["messages"]
            idx = steps(ms)
            todo = list(groups)

            def after(i, ms=ms, todo=todo):
                if any(c["function"]["name"] == "delegate_task" for c in ms[i].get("tool_calls") or []) and todo:
                    g = todo.pop(0)
                    ts = [threading.Thread(target=trace, args=("brief", k, steps(k["messages"])), kwargs={"sub": True})
                          for k in g]
                    for t in ts:
                        t.start()
                    for t in ts:
                        t.join()
            prev = None
            for n, i in enumerate(idx):
                if not running():
                    return
                r = call("brief", {"messages": ms[:i], "tools": p["tools"], "max_tokens": a.max_tokens}, sid=sid(p),
                         step=n, prev_prompt=prev, ref_tool=bool(ms[i].get("tool_calls")))
                if r["status"] != 200:
                    break
                prev = r.get("prompt")
                if n == 0:
                    title(p)
                after(i)


def lane_aux(rng):
    t_next_comp = time.time() + 120 * a.tick
    while running():
        _sleep(150 * a.tick)
        if not running():
            return
        title(rng.choice(SESS))
        if time.time() >= t_next_comp:
            t_next_comp = time.time() + 600 * a.tick
            s = rng.choice(MAINS)
            ms = s["messages"]
            i = rng.randrange(len(ms) // 3, len(ms))
            parts, n = [], 0
            for m in ms[1:i]:
                t = f"[{m['role']}] " + (m.get("content") or json.dumps(m.get("tool_calls") or "", ensure_ascii=False))
                parts.append(t[:6000])
                n += len(parts[-1])
                if n > 110_000:
                    break
            call("aux", {"messages": [{"role": "system", "content": "You compress conversation history. Summarize the "
                                       "conversation below so the assistant can continue the task: goals, decisions, "
                                       "files and facts found, open steps. Be concise and keep exact names and numbers."},
                                      {"role": "user", "content": "\n\n".join(parts)}],
                         "max_tokens": 2048, "reasoning_effort": "none", "stream": False}, what="compress")


def _sleep(sec):
    end = time.time() + sec
    while time.time() < end and running():
        time.sleep(min(5, end - time.time()))


# ---- media --------------------------------------------------------------------------------------------------------
FONT = {"0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
        "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
        "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
        "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
        "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100"}
COLOURS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (40, 70, 230), "yellow": (240, 210, 20)}


def picture(w, h, number, square, noise, rng):
    """(numbers without 0: this font's 0 has a slash and reads as 8 in a large noisy picture) RGB PNG: a soft gradient with fine noise (photo-like texture), the number in large dark digits at the upper
    left, a coloured square at the lower right."""
    tpl = []
    for band in range(8):
        base = [(150 + 60 * x // w + 10 * band, 160 - 30 * band // 8 + 40 * x // w, 140 + 50 * band // 8) for x in range(0, w)]
        for v in range(8):
            row = bytearray(3 * w)
            for x, (r, g, b) in enumerate(base):
                e = rng.randint(-noise, noise) if noise else 0
                row[3 * x:3 * x + 3] = bytes((min(255, max(0, r + e)), min(255, max(0, g + e)), min(255, max(0, b + e))))
            tpl.append(row)
    px = [bytearray(tpl[(y * 8 // h) * 8 + rng.randrange(8)]) for y in range(h)]
    s = max(4, h // 40)
    rects = []
    for k, ch in enumerate(number):
        for i, bit in enumerate(FONT[ch]):
            if bit == "1":
                r, q = divmod(i, 5)
                x0, y0 = w // 16 + k * 6 * s + q * s, h // 12 + r * s
                rects.append((x0, y0, x0 + s, y0 + s, (20, 20, 20)))
    q = min(w, h) // 4
    rects.append((w - q - w // 16, h - q - h // 12, w - w // 16, h - h // 12, COLOURS[square]))
    for x0, y0, x1, y1, c in rects:
        run = bytes(c) * (x1 - x0)
        for y in range(y0, y1):
            px[y][3 * x0:3 * x1] = run
    raw = b"".join(b"\0" + bytes(r) for r in px)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def part(data, kind="image", mime="image/png"):
    k = f"{kind}_url"
    return {"type": k, k: {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}}


ASK_PIC = "What number is written in the picture, and what colour is the square in the lower right? Answer in one short sentence."


def media_jobs(rng):
    md = a.media
    photos = [("photo12", 4000, 3000, n, c) for n, c in (("58213", "blue"), ("74916", "red"), ("31587", "green"))]
    small = [("small", 640, 480, n, c) for n, c in (("4721", "red"), ("9358", "blue"), ("2613", "yellow"))]
    jobs = []
    for i in range(3):
        name, w, h, n, c = photos[i]
        jobs.append((name, [part(picture(w, h, n, c, 6, rng)), {"type": "text", "text": ASK_PIC}], (n, c)))
        name, w, h, n, c = small[i]
        jobs.append((name, [part(picture(w, h, n, c, 0, rng)), {"type": "text", "text": ASK_PIC}], (n, c)))
        jobs.append(("chart", [part(open(os.path.join(md, "chart.png"), "rb").read()),
                               {"type": "text", "text": "What number is in the chart's title? Answer with the number."}],
                     ("4721", None)))
        jobs.append(("scene", [part(open(os.path.join(md, "scene.png"), "rb").read()),
                               {"type": "text", "text": "Describe this picture in two sentences."}], None))
        jobs.append(("video", [part(open(os.path.join(md, "long.mp4"), "rb").read(), "video", "video/mp4"),
                               {"type": "text", "text": "What happens in this video? Two sentences."}], None))
    return jobs


def media_req(name, content, want):
    r = call("media", {"messages": [{"role": "user", "content": content}], "max_tokens": 128, "temperature": 0,
                       "reasoning_effort": "none"}, what=name)
    if want and r["status"] == 200:
        t = r["text"].replace(",", "")
        ok = want[0] in t and (want[1] is None or want[1] in t.lower())
        with LOCK:
            LOG.write(json.dumps({"lane": "check", "what": name, "check": ok, "text": r["text"][:200],
                                  "t0": r["t0"], "status": 200}) + "\n")
            LOG.flush()
    return r


def lane_media(rng, jobs):
    k = 0
    while running():
        name, content, want = jobs[k % len(jobs)]
        media_req(name, content, want)
        k += 1
        _sleep(150 * a.tick)


# ---- long reads ---------------------------------------------------------------------------------------------------
def book(name):
    t = open(os.path.join(a.books, name + ".txt"), encoding="utf-8-sig").read().replace("\r\n", "\n")
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
    ps = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", t[s.end():e.start()])]
    return [p for p in ps if p]


def long_text(order, target, rng):
    paras = []
    for b in order:
        paras += book(b)
    recs = []
    for d in (0.2, 0.5, 0.85):
        rid, units = rng.randint(1000, 9999), rng.randint(10000, 99999)
        recs.append({"rid": rid, "units": units, "depth": d,
                     "text": f"Record {rid}: The customs house at Greywick logged {units} units of linen."})
    cum = [0]
    for p in paras:
        cum.append(cum[-1] + len(p) + 2)

    def build(k):
        body = list(paras[:k])
        for r in sorted(recs, key=lambda r: -r["depth"]):
            body.insert(bisect.bisect_left(cum, r["depth"] * cum[k], 0, k + 1), r["text"])
        return "Below is a long text. Answer the question that follows it.\n\n" + "\n\n".join(body)

    def q(rid):
        return f"\n\nQuestion: How many units did Record {rid} log? Answer with the number only."
    k, best = bisect.bisect_left(cum, target * 4.2), None
    for _ in range(12):
        k = max(1, min(k, len(paras)))
        n = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": build(k) + q(recs[0]["rid"])}]},
                 1200)["count"]
        if n <= target and (best is None or n > best[1]):
            best = (k, n)
        if target - 1024 <= n <= target:
            break
        nk = bisect.bisect_left(cum, cum[k] + (target - 512 - n) * cum[k] / n)
        k = nk if nk != k else (k - 1 if n > target else k + 1)
    return build(best[0]), best[1], recs, q


def long_req(what, text, rec, q):
    r = call("long", {"messages": [{"role": "user", "content": text + q(rec["rid"])}], "max_tokens": 64,
                      "temperature": 0, "reasoning_effort": "none"}, what=what)
    ok = r["status"] == 200 and str(rec["units"]) in r["text"].replace(",", "")
    with LOCK:
        LOG.write(json.dumps({"lane": "check", "what": what, "check": ok, "text": r["text"][:200],
                              "t0": r["t0"], "status": 200}) + "\n")
        LOG.flush()


def lane_long(L1, L2):
    t0 = time.time()
    _wait_until(t0 + 8 * 60 * a.tick)
    if not running():
        return
    text, n, recs, q = L1
    long_req(f"cold {n // 1000}K", text, recs[1], q)
    long_req(f"same text, other record {n // 1000}K", text, recs[2], q)
    _wait_until(t0 + a.minutes * 60 * 2 / 3)
    if not running():
        return
    text, n, recs, q = L2
    long_req(f"cold {n // 1000}K (other text)", text, recs[0], q)


def _wait_until(t):
    while time.time() < t and running():
        time.sleep(5)


# ---- cut, probe, metrics ------------------------------------------------------------------------------------------
def lane_cut(rng):
    while running():
        _sleep(360 * a.tick)
        if running():
            call("cut", {"messages": [{"role": "user", "content": "Explain step by step how a hash map handles "
                                                                  "collisions, with an example in Python."}],
                         "max_tokens": 4096}, cut=20, what="closed after 20 chunks")


PROBE = {"messages": [{"role": "user", "content": "Write a short paragraph about rivers."}], "temperature": 1.0,
         "seed": 4242, "max_tokens": 200, "reasoning_effort": "none", "stream": False}


def lane_probe():
    while running():
        _sleep(600 * a.tick)
        if running():
            call("probe", PROBE, what="during the load")


def metrics_loop(stop):
    f = open(os.path.join(a.out, "metrics.csv"), "a")
    while not stop.is_set():
        try:
            m = get("/metrics", 10)
            v = {k: float(x) for k, x in re.findall(r'^(vllm:\w+)\{[^}]*\} ([\d.e+]+)$', m, re.M)}
            f.write(f"{time.time():.1f},{v.get('vllm:num_requests_running', -1):.0f},"
                    f"{v.get('vllm:num_requests_waiting', -1):.0f},{v.get('vllm:prompt_tokens_total', 0):.0f},"
                    f"{v.get('vllm:prompt_tokens_cached_total', 0):.0f},{v.get('vllm:generation_tokens_total', 0):.0f}\n")
            f.flush()
        except Exception as e:  # noqa: BLE001
            f.write(f"{time.time():.1f},err,{type(e).__name__}\n")
        stop.wait(5)


def idle_state():
    m = get("/metrics", 10)
    v = {k: float(x) for k, x in re.findall(r'^(vllm:\w+)\{[^}]*\} ([\d.e+]+)$', m, re.M)}
    h = json.loads(get("/health", 10)).get("status")
    return {"running": int(v["vllm:num_requests_running"]), "waiting": int(v["vllm:num_requests_waiting"]), "health": h}


def phase(k):
    PH[k] = time.time()
    json.dump(PH, open(os.path.join(a.out, "phases.json"), "w"), indent=1)


# ---- run ----------------------------------------------------------------------------------------------------------
rng = random.Random(a.seed)
say("soak", f"{MODEL} · window {CTX:,} · {len(SUBS)} subagent / {len(MAINS)} main / {len(CRON)} cron sessions · "
            f"{a.minutes:g} min")
phase("setup")
t = time.time()
jobs = media_jobs(rng)
say("setup", f"media ready · {len(jobs)} requests in turn · 12 MP picture {len(jobs[0][1][0]['image_url']['url']) / 1e6:.1f} "
             f"MB base64 · {time.time() - t:.0f} s")
k1, k2 = (int(x) * 1000 for x in a.long.split(","))
k1 = min(k1, CTX - 2048)
k2 = min(k2, CTX - 2048)
t = time.time()
L1 = long_text(("pg2600", "pg1399"), k1, rng)
L2 = long_text(("pg1399", "pg2600"), k2, rng)
say("setup", f"long texts {L1[1]:,} and {L2[1]:,} tokens · {time.time() - t:.0f} s")
r = call("probe", PROBE, what="before the load")
say("probe", f"{r['status']} · {r['text'][:60]!r}")
for name, content, want in (jobs[0], jobs[4]):  # 12 MP picture and the video: the tower's memory is taken
    r = media_req(name, content, want)
    say("warm", f"{name}: {r['status']} · first token {r.get('ttft') or 0:.1f} s · {r['text'][:60]!r}")
stop = threading.Event()
mt = threading.Thread(target=metrics_loop, args=(stop,), daemon=True)
mt.start()
phase("base_start")
time.sleep(a.idle)
phase("base_end")

phase("load_start")
STOP_AT[0] = time.time() + a.minutes * 60
lanes = [("trace", lane_trace, (random.Random(a.seed + 1),)), ("live", lane_live, (random.Random(a.seed + 2),)),
         ("window", lane_window, (random.Random(a.seed + 3),)), ("brief", lane_brief, (random.Random(a.seed + 4),)),
         ("aux", lane_aux, (random.Random(a.seed + 5),)), ("media", lane_media, (random.Random(a.seed + 6), jobs[1:])),
         ("long", lane_long, (L1, L2)), ("cut", lane_cut, (random.Random(a.seed + 7),)), ("probe", lane_probe, ())]
ths = []
for name, fn, args in lanes:
    th = threading.Thread(target=fn, args=args, name=name, daemon=True)
    th.start()
    ths.append(th)


def progress():
    n0 = 0
    while any(t.is_alive() for t in ths):
        time.sleep(300)
        rs = [json.loads(x) for x in open(os.path.join(a.out, "requests.jsonl"))]
        new = rs[n0:]
        n0 = len(rs)
        err = sum(1 for r in new if r["lane"] != "cut" and (r["status"] != 200 or r.get("error")))
        g = [r["max_gap"] for r in new if r.get("max_gap")]
        bad = sum(1 for r in new if r.get("check") is False)
        say("load", f"{(time.time() - PH['load_start']) / 60:.0f} min · {len(new)} requests in 5 min · errors {err} · "
                    f"wrong answers {bad} · longest gap {max(g) if g else 0:.1f} s · "
                    f"lanes alive {sum(t.is_alive() for t in ths)}")


threading.Thread(target=progress, daemon=True).start()
for th in ths:
    th.join()
phase("load_end")
phase("drained")
time.sleep(10)
phase("after_start")
time.sleep(a.idle)
phase("after_end")
r = call("probe", PROBE, what="after the load")
say("probe", f"{r['status']} · {r['text'][:60]!r}")
PH["end_state"] = idle_state()
phase("done")
stop.set()
say("soak", f"load done · end state {PH['end_state']}")
