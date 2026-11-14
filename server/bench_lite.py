"""Bench-like measurement against the running server (stdlib only; from the Mac): python3 server/bench_lite.py [URL]
Mirrors the bench's settings (llm docs bench §3–§4): thinking on (reasoning_effort unset → xhigh), no sampling values
sent (server defaults), seed 1 (1..n at once), answers run to their natural end (max_tokens cap 8192).
  decode 1 / 4 at once — context ≈ --ctx tokens (a fixed document + one question), decode tok/s per stream = completion
                          tokens after the first one / time after the first token; rounds from /metrics
  read                  — cold prompts of --read tokens, max_tokens 1 → prompt tokens / time to first token
  next turn             — that conversation + ≈7K new tokens, max_tokens 1 → seconds to first token, reuse %
  together              — the bench's n at once (bench §4.1): n cold --read conversations read first (max_tokens 1), then
                          the same n written at once from the cache (seed 1..n); a reply that ends is replaced at once by
                          the next request (same prompt, seed + k·n) so n stay writing; window = the last stream's first
                          token + --window s, then the streams are closed (at that time, whether or not a line is
                          arriving); tokens = /metrics generation counter delta / window (plus the server's
                          per-batch-size round counters); then how long the server takes to stop (bench: within 5 s)
  agent                 — the bench's One agent order (bench §5): a cold --read prompt written to its natural end
                          (read = prompt tokens / time to first token, solo decode tok/s at that context), then right
                          after the answer the next turn (same conversation + ≈7K new tokens, max_tokens 1)
The document is the model's README + chat template repeated (text the model can talk about), with a nonce up front
so every cold prompt is new to the server."""
import argparse
import http.client
import json
import socket
import threading
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://localhost:8888/v1")
ap.add_argument("--doc", required=True, help="text file to build prompts from (e.g. the model's README + template)")
ap.add_argument("--ctx", type=int, default=8192, help="approx. prompt tokens for the decode cases")
ap.add_argument("--read", type=int, default=131072, help="approx. prompt tokens for the read / next-turn cases")
ap.add_argument("--cap", type=int, default=8192)
ap.add_argument("--only", default="decode,read", help="decode, read, agent, together (comma list)")
ap.add_argument("--n", type=int, default=4, help="together: conversations writing at once")
ap.add_argument("--window", type=float, default=90.0, help="together: seconds from the last stream's first token")
a = ap.parse_args()
BASE = a.url
MODEL = json.loads(urllib.request.urlopen(BASE + "/models", timeout=30).read())["data"][0]["id"]
DOC = open(a.doc).read()
CH_PER_TOK = 4.4  # measured on this document (≈4.4 characters per token), only to size the prompts


def say(tag, msg):
    print(f"\033[36m[lite {tag}]\033[0m \033[32m{msg}\033[0m", flush=True)


def warn(tag, msg):
    print(f"\033[36m[lite {tag}]\033[0m \033[33m{msg}\033[0m", flush=True)


def text(tokens, nonce):
    n = int(tokens * CH_PER_TOK)
    body = (DOC * (n // len(DOC) + 1))[:n]
    return f"[{nonce}]\n{body}"


def metrics():
    m = {}
    for line in urllib.request.urlopen(BASE.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode().splitlines():
        if line and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            m[k if k.startswith("q38_") else k.split("{")[0]] = float(v)
    return m


def rounds(d):
    """per (sequences, depth) from /metrics deltas: rounds, tokens per sequence-round, ms inside spec_round, ms between
    round starts (no prefill chunk in between)"""
    keys = sorted({k.split("{")[1] for k in d if k.startswith("q38_round_rounds")})
    out = []
    for lab in keys:
        g = lambda n: d.get(f"q38_round_{n}_total{{{lab}", 0)  # noqa: E731
        n = g("rounds")
        if n < 1:
            continue
        B = int(lab.split('"')[1])
        wall = g("wall_ms") / g("wall_rounds") if g("wall_rounds") else float("nan")
        out.append(f"B={B} k={lab.split(chr(34))[3]} {int(n)} rounds · {g('tokens') / n / B:.2f} tokens · engine "
                   f"{g('engine_ms') / n:.1f} ms · interval {wall:.1f} ms")
    return " | ".join(out)


def stream(messages, max_tokens, seed, out, i):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, usage, txt = time.time(), None, None, ""
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])
        usage = ch.get("usage") or usage
        for c in ch.get("choices", []):
            d = c.get("delta", {})
            piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
            if piece and first is None:
                first = time.time()
            txt += piece
    out[i] = {"t0": t0, "first": first, "end": time.time(), "usage": usage, "chars": len(txt)}


def run(n, prompt_tokens, max_tokens, nonce_each=True):
    q = "\n\nBased on the document above, explain in detail how its parts fit together and what a reader should check first."
    nonce = uuid.uuid4().hex[:12]
    msgs = [[{"role": "user", "content": text(prompt_tokens, uuid.uuid4().hex[:12] if nonce_each else nonce) + q}] for _ in range(n)]
    out = [None] * n
    m0 = metrics()
    th = [threading.Thread(target=stream, args=(msgs[i], max_tokens, i + 1, out, i)) for i in range(n)]
    [t.start() for t in th]
    [t.join() for t in th]
    m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    return out, d, msgs


if "decode" in a.only:
    for n in (1, 4):
        out, d, _ = run(n, a.ctx, a.cap)
        rates = [(o["usage"]["completion_tokens"] - 1) / (o["end"] - o["first"]) for o in out]
        drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
        acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        say(f"decode {n} at once", f"input {out[0]['usage']['prompt_tokens']:,} · output "
                                   f"{', '.join(str(o['usage']['completion_tokens']) for o in out)} · "
                                   f"per stream {', '.join(f'{x:.1f}' for x in rates)} tok/s · "
                                   f"total {sum(rates):.1f} tok/s · "
                                   f"per round {1 + acc / max(drafts, 1):.2f} tokens")
        say(f"rounds {n} at once", rounds(d))
if "read" in a.only:
    out, d, msgs = run(1, a.read, 1)
    o = out[0]
    pt = o["usage"]["prompt_tokens"]
    say("read", f"input {pt:,} · first token {o['first'] - o['t0']:.2f} s → {pt / (o['first'] - o['t0']):,.0f} tok/s")
    conv = msgs[0] + [{"role": "assistant", "content": "I read it."},
                      {"role": "user", "content": text(7000, "turn2") + "\n\nNow compare this new part with the first one."}]
    out2 = [None]
    stream(conv, 1, 1, out2, 0)
    o2 = out2[0]
    u = o2["usage"]
    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    say("next turn", f"input {u['prompt_tokens']:,} · reused {c:,} ({100 * c / u['prompt_tokens']:.0f}%) · new tokens "
                     f"{u['prompt_tokens'] - c:,} · first token {o2['first'] - o2['t0']:.2f} s")
if "agent" in a.only:
    out, d, msgs = run(1, a.read, a.cap)
    o = out[0]
    u = o["usage"]
    pt, ct = u["prompt_tokens"], u["completion_tokens"]
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    say("agent read", f"input {pt:,} · first token {o['first'] - o['t0']:.2f} s → "
                      f"{pt / (o['first'] - o['t0']):,.0f} tok/s")
    say("agent write", f"context {pt // 1024}K · output {ct:,} · {(ct - 1) / (o['end'] - o['first']):.1f} tok/s · "
                       f"per round {1 + acc / max(drafts, 1):.2f} tokens")
    conv = msgs[0] + [{"role": "assistant", "content": "I read it."},
                      {"role": "user", "content": text(7000, "turn2") + "\n\nNow compare this new part with the first one."}]
    out2 = [None]
    stream(conv, 1, 1, out2, 0)  # right after the answer, as the bench's next turn
    o2 = out2[0]
    u2 = o2["usage"]
    c = (u2.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    say("agent next turn", f"input {u2['prompt_tokens']:,} · reused {c:,} ({100 * c / u2['prompt_tokens']:.0f}%) · "
                           f"new tokens {u2['prompt_tokens'] - c:,} · first token {o2['first'] - o2['t0']:.2f} s")


def wait_idle(sec=1.0, limit=180):
    """until the server has had no running request for `sec` seconds (bench: before a new measurement)"""
    t0, quiet = time.time(), None
    while time.time() - t0 < limit:
        m = metrics()
        if m.get("vllm:num_requests_running", 0) + m.get("vllm:num_requests_waiting", 0) == 0:
            quiet = quiet or time.time()
            if time.time() - quiet >= sec:
                return
        else:
            quiet = None
        time.sleep(0.2)


if "together" in a.only:
    n = a.n
    q = "\n\nBased on the document above, explain in detail how its parts fit together and what a reader should check first."
    msgs = [[{"role": "user", "content": text(a.read, uuid.uuid4().hex[:12]) + q}] for _ in range(n)]
    out = [None] * n
    th = [threading.Thread(target=stream, args=(msgs[i], 1, i + 1, out, i)) for i in range(n)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    say(f"together read {n}", f"input {out[0]['usage']['prompt_tokens']:,} × {n} · latest first token "
                              f"{max(o['first'] for o in out) - t0:.1f} s")
    wait_idle()
    first = [None] * n
    state = {"stop": None, "closed": False}
    reqs = [0] * n
    live = [None] * n  # each writer's open response (the main thread shuts its socket at the window end)
    lock = threading.Lock()

    def writer(i):
        k = 0
        while True:
            body = {"model": MODEL, "messages": msgs[i], "max_tokens": a.cap, "seed": i + 1 + k * n, "stream": True}
            req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with lock:
                if state["closed"]:
                    return
            resp = urllib.request.urlopen(req, timeout=3600)
            with lock:
                live[i] = resp
                if state["closed"]:  # the window ended while this request was being sent
                    resp.close()
                    return
            reqs[i] += 1
            try:
                for line in resp:
                    if first[i] is None and line.startswith(b"data: {") and b'"content": ""' not in line[:200]:
                        first[i] = time.time()
            except (OSError, ValueError, http.client.HTTPException):  # shut by the main thread
                pass
            with lock:
                live[i] = None
            resp.close()
            if state["closed"]:
                return
            k += 1  # ended inside the window: the next request at once

    m0 = metrics()
    ws = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    [t.start() for t in ws]
    while any(f is None for f in first):
        time.sleep(0.05)
    start = max(first)
    state["stop"] = start + a.window
    time.sleep(max(0.0, state["stop"] - time.time()))
    with lock:  # close every stream now, also one whose server is holding its output back (no line to wake the reader)
        state["closed"] = True
        for r in live:
            if r is not None:
                try:  # unblocks the writer's read; the writer closes the response
                    r.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                except (AttributeError, OSError):
                    pass
    [t.join() for t in ws]
    stop = time.time()
    wait_idle(0.5)
    idle = time.time() - stop
    m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    gen = d.get("vllm:generation_tokens_total", 0)
    per_round = 1 + d.get("vllm:spec_decode_num_accepted_tokens_total", 0) / max(
        d.get("vllm:spec_decode_num_drafts_total", 1), 1)
    say(f"together write {n}", f"window {a.window:.0f} s (first-token spread {start - min(first):.1f} s) · generated "
                               f"{int(gen):,} tokens → {gen / a.window:.1f} tok/s total · {sum(reqs)} requests · "
                               f"per round {per_round:.2f} tokens")
    say(f"together rounds {n}", rounds(d))
    (say if idle - 0.5 <= 5 else warn)(f"together close {n}", f"server stops {max(idle - 0.5, 0):.1f} s after the "
                                                              f"close (bench: within 5 s)")
