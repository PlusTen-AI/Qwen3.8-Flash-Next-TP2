"""Checks against a running server — no reference data needed, stdlib only (Python >= 3.8), any machine that reaches
the server.  python3 checks/server_check.py [base URL] [--only smoke,exact,reuse,needle] [--needle 8192,32768,131072]

  smoke   /v1/models · chat (non-stream, greedy) · chat stream with thinking (reasoning / content split, usage chunk)
          · next turn reuses the cached prompt · four at once · a closed stream stops the request · same seed twice
  exact   four prompts with seeds: alone, all four at once, alone again → the same text every time (the engine is
          deterministic for a seed and batch-invariant)
  reuse   a five-turn conversation over a ~6K-token document: from the second turn on, ≥ 90% of each prompt is reused
  needle  a short fact hidden at 10% / 50% / 90% of a long filler text, for each --needle length (capped by the
          server's context): the answer must contain it
One line per case; exit code 1 if any case fails."""
import argparse
import json
import random
import sys
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8888/v1")
ap.add_argument("--only", default="smoke,exact,reuse,needle")
ap.add_argument("--needle", default="8192,32768,131072", help="prompt lengths in tokens (approximate)")
a = ap.parse_args()
BASE = a.url.rstrip("/")
bad = 0


def say(ok, tag, msg):
    global bad
    bad += not ok
    print(f"\033[36m[check {tag}]\033[0m \033[{32 if ok else 31}m{msg}\033[0m", flush=True)


def post(path, body, stream=False, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else json.loads(r.read())


def chat(messages, **kw):
    b = {"model": MODEL, "messages": messages, "reasoning_effort": "none"}
    b.update(kw)
    return post("/chat/completions", b)


m = json.loads(urllib.request.urlopen(BASE + "/models", timeout=30).read())["data"][0]
MODEL, CTX = m["id"], int(m.get("max_model_len") or 0)
only = set(a.only.split(","))


# ---- smoke --------------------------------------------------------------------------------------------------------
def smoke():
    say(True, "models", f"{MODEL} · context {CTX:,}")
    t0 = time.time()
    r = chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], max_tokens=32,
             temperature=0)
    c = r["choices"][0]
    say("paris" in (c["message"]["content"] or "").lower(), "chat",
        f"{c['message']['content']!r} · finish {c['finish_reason']} · usage {r['usage']['prompt_tokens']}+"
        f"{r['usage']['completion_tokens']} · {time.time() - t0:.2f} s")

    body = {"model": MODEL, "messages": [{"role": "user", "content": "Is 91 a prime number? Think briefly, then answer."}],
            "reasoning_effort": "low", "max_tokens": 512, "stream": True, "stream_options": {"include_usage": True},
            "seed": 1}
    t0, first, reasoning, content, usage = time.time(), None, "", "", None
    for line in post("/chat/completions", body, stream=True):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])
        usage = ch.get("usage") or usage
        for ci in ch.get("choices", []):
            d = ci.get("delta", {})
            if (d.get("reasoning_content") or d.get("content")) and first is None:
                first = time.time() - t0
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
    dt = time.time() - t0
    say(bool(reasoning) and bool(content) and usage is not None, "stream",
        f"thinking {len(reasoning)} chars · answer {content.strip()[:60]!r} · first piece {first or 0:.2f} s · "
        f"{usage and usage['completion_tokens']} tokens · "
        f"{(usage['completion_tokens'] if usage else 0) / max(dt - (first or 0), 1e-6):.1f} tok/s")

    turn2 = body["messages"] + [{"role": "assistant", "content": content, "reasoning_content": reasoning},
                                {"role": "user", "content": "And 97?"}]
    t0 = time.time()
    u = chat(turn2, reasoning_effort="low", max_tokens=1)["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    say(cached > 0, "next turn", f"prompt {u['prompt_tokens']} · reused {cached} ({100 * cached / u['prompt_tokens']:.0f}%)"
                                 f" · first token {time.time() - t0:.2f} s")

    res = [None] * 4

    def one(i):
        rr = chat([{"role": "user", "content": f"Write two sentences about the number {i + 7}."}], max_tokens=128,
                  seed=i + 1)
        res[i] = rr["usage"]["completion_tokens"]
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    say(all(res), "4 at once", f"{sum(x or 0 for x in res)} tokens · {time.time() - t0:.2f} s")

    def running():
        for line in urllib.request.urlopen(BASE.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode().splitlines():
            if line.startswith("vllm:num_requests_running"):
                return int(float(line.split()[-1]))
    r = post("/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "Count from 1 to 500."}],
                                   "reasoning_effort": "none", "max_tokens": 4000, "stream": True}, stream=True)
    for i, _ in enumerate(r):
        if i > 6:
            break
    r.close()
    t0, n = time.time(), None
    while time.time() - t0 < 5:
        n = running()
        if n == 0:
            break
        time.sleep(0.2)
    say(n == 0, "disconnect", f"stream closed · {n} request(s) running after {time.time() - t0:.1f} s")

    b = [{"role": "user", "content": "Name three colours."}]
    a1 = chat(b, max_tokens=48, seed=1)["choices"][0]["message"]["content"]
    a2 = chat(b, max_tokens=48, seed=1)["choices"][0]["message"]["content"]
    say(a1 == a2, "seed", f"same seed twice → {'same' if a1 == a2 else 'DIFFERENT'} text · {a1.strip()[:50]!r}")


# ---- exact --------------------------------------------------------------------------------------------------------
def exact():
    prompts = ["Describe a lighthouse at night in four sentences.", "List five uses of copper and explain one.",
               "Write a short dialogue between a cat and a clock.", "Explain why ice floats, step by step."]

    def run(i, out):
        out[i] = chat([{"role": "user", "content": prompts[i]}], max_tokens=160, seed=100 + i)["choices"][0]["message"]["content"]
    alone, together, again = [None] * 4, [None] * 4, [None] * 4
    for i in range(4):
        run(i, alone)
    th = [threading.Thread(target=run, args=(i, together)) for i in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    for i in range(4):
        run(i, again)
    same = [alone[i] == together[i] == again[i] for i in range(4)]
    say(all(same), "exact", f"4 prompts × (alone, at once, alone again): {sum(same)}/4 identical"
                            + ("" if all(same) else f" · differs: {[i for i in range(4) if not same[i]]}"))


# ---- filler text --------------------------------------------------------------------------------------------------
WORDS = ("river stone market window garden lantern harbor meadow copper violet thunder orchard canyon silver pocket "
         "ladder candle mirror forest bridge winter summer pepper cotton marble island valley rocket").split()


def filler(n_lines, seed):
    g = random.Random(seed)
    return "\n".join(f"Note {i + 1}: the {g.choice(WORDS)} near the {g.choice(WORDS)} was {g.choice(WORDS)} and "
                     f"{g.choice(WORDS)} on day {g.randint(1, 999)}." for i in range(n_lines))


def tokens_per_line():
    u = chat([{"role": "user", "content": filler(400, 7)}], max_tokens=1)["usage"]
    return u["prompt_tokens"] / 400


# ---- reuse --------------------------------------------------------------------------------------------------------
def reuse():
    tpl = tokens_per_line()
    msgs = [{"role": "system", "content": "Answer questions about these notes.\n\n" + filler(int(6000 / tpl), 11)}]
    worst = 1.0
    for k, q in enumerate(["Which word appears in note 3?", "What day is in note 10?", "Summarize note 25.",
                           "Which notes mention a bridge? Name two.", "What is in the last note?"]):
        msgs.append({"role": "user", "content": q})
        r = chat(msgs, max_tokens=64, seed=k + 1)
        u = r["usage"]
        frac = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) / u["prompt_tokens"]
        if k:
            worst = min(worst, frac)
        msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"] or ""})
    say(worst >= 0.9, "reuse", f"5 turns over ~6K tokens · least reused from turn 2 on: {100 * worst:.1f}% (needs ≥ 90%)")


# ---- needle -------------------------------------------------------------------------------------------------------
def needle():
    tpl = tokens_per_line()
    for n in [int(x) for x in a.needle.split(",") if x]:
        if CTX and n > CTX - 1024:
            say(True, "needle", f"{n:,} skipped (server context {CTX:,})")
            continue
        lines = filler(int((n - 200) / tpl), 23).split("\n")
        found = []
        for depth in (0.1, 0.5, 0.9):
            code = f"{random.Random(n + int(depth * 10)).randint(1000, 9999)}-QX"
            at = int(len(lines) * depth)
            text = "\n".join(lines[:at] + [f"Important: the access code for the blue vault is {code}."] + lines[at:])
            t0 = time.time()
            r = chat([{"role": "user", "content": text + "\n\nWhat is the access code for the blue vault? Reply with "
                                                         "the code only."}], max_tokens=32, temperature=0)
            ans = r["choices"][0]["message"]["content"] or ""
            found.append(code in ans)
            last = (r["usage"]["prompt_tokens"], time.time() - t0)
        say(all(found), "needle", f"{last[0]:,} tokens · depth 10/50/90%: {''.join('✓' if f else '✗' for f in found)} · "
                                  f"last request {last[1]:.1f} s")


for name, fn in (("smoke", smoke), ("exact", exact), ("reuse", reuse), ("needle", needle)):
    if name in only:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one failing case must not hide the others
            say(False, name, f"error: {e!r}"[:200])
sys.exit(1 if bad else 0)
