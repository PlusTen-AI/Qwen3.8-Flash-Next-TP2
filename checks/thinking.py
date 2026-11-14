"""Thinking controls against a running server (stdlib): the off forms, levels, bad effort, thinking budget, a loop
prompt; with a key argument the API-key checks instead.  python3 checks/thinking.py http://<head>:8888/v1 [key]"""
import json
import sys
import urllib.error
import urllib.request

URL = sys.argv[1].rstrip("/")
KEY = sys.argv[2] if len(sys.argv) > 2 else None
bad = 0


def say(ok, tag, msg):
    global bad
    bad += not ok
    print(f"[t {tag}] {'OK' if ok else 'FAIL'} · {msg}", flush=True)


def post(body, key=KEY):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(), h)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def chat(content, **kw):
    b = {"model": "x", "messages": [{"role": "user", "content": content}], "max_tokens": 2000, "seed": 7}
    b.update(kw)
    return post(b)


if KEY:
    c, j = post({"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}, key=None)
    say(c == 401, "key", f"no key → {c} {json.dumps(j)[:80]}")
    c, j = post({"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}, key="wrong")
    say(c == 401, "key", f"wrong key → {c}")
    c, j = post({"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    say(c == 200, "key", f"right key → {c}")
    h = urllib.request.urlopen(URL.rsplit("/v1", 1)[0] + "/health", timeout=5).status
    say(h == 200, "key", f"/health without key → {h}")
    sys.exit(1 if bad else 0)

for name, kw in (("reasoning.enabled false", {"reasoning": {"enabled": False}}), ("think false", {"think": False}),
                 ("kwargs effort none", {"chat_template_kwargs": {"reasoning_effort": "none"}}),
                 ("reasoning_effort none", {"reasoning_effort": "none"})):
    c, j = chat("What is 17 * 3? Answer with the number.", **kw)
    m = j["choices"][0]["message"] if c == 200 else {}
    say(c == 200 and not m.get("reasoning_content") and "51" in (m.get("content") or ""), "off",
        f"{name}: {c} · reasoning {len(m.get('reasoning_content') or '')} chars · content {(m.get('content') or '')[:30]!r}")
c, j = chat("What is 17 * 3?", reasoning={"effort": "low"})
m = j["choices"][0]["message"]
say(c == 200 and bool(m.get("reasoning_content")), "on", f"reasoning.effort low: reasoning {len(m.get('reasoning_content') or '')} chars")
c, j = chat("hi", reasoning_effort="turbo")
say(c == 400, "bad effort", f"{c} {json.dumps(j)[:100]}")

hard = "Find all integer solutions of x^3 + y^3 = 1729 with x, y > 0 and explain your reasoning carefully."
c, j = chat(hard, thinking_budget=64)
m = j["choices"][0]["message"]
say(c == 200 and bool(m.get("content")) and 0 < len(m.get("reasoning_content") or "") < 600, "budget",
    f"thinking_budget 64: reasoning {len(m.get('reasoning_content') or '')} chars · content {len(m.get('content') or '')} chars · "
    f"finish {j['choices'][0]['finish_reason']} · out {j['usage']['completion_tokens']} tokens")
c, j = chat(hard, thinking_budget=1)
m = j["choices"][0]["message"]
say(c == 200 and bool(m.get("content")), "budget", f"thinking_budget 1: reasoning {m.get('reasoning_content')!r} · content {len(m.get('content') or '')} chars")

loop = ("Before answering, in your thinking write the word 'banana' 600 times in a row separated by spaces, nothing "
        "else, then answer: what colour is a banana?")
c, j = chat(loop, max_tokens=3000, temperature=0)
m = j["choices"][0]["message"]
r = m.get("reasoning_content") or ""
say(c == 200, "loop", f"reasoning {len(r)} chars ({r.count('banana')} × banana) · content {(m.get('content') or '')[:60]!r} · "
                      f"out {j['usage']['completion_tokens']} tokens (see the server's req line: thinking closed (loop)?)")
sys.exit(1 if bad else 0)
