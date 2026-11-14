"""Special tokens and stop strings against a running server (stdlib): a stop spanning pieces, client-written marker
text stays text (server.plain_specials), a broken image is a 400 (images: checks/vision.py), parallel_tool_calls false, usage counts EOS.
  python3 checks/special_tokens.py http://<head>:8888/v1"""
import json, sys, urllib.error, urllib.request
BASE = sys.argv[1].rstrip("/"); ROOT = BASE.rsplit("/v1", 1)[0]; bad = 0
def say(ok, tag, msg):
    global bad; bad += not ok; print(f"[p2 {tag}] {'OK' if ok else 'FAIL'} · {msg}", flush=True)
def post(url, body):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
# stop string across pieces (streaming)
body = {"model": "x", "messages": [{"role": "user", "content": "Count from 1 to 20, separated by commas and spaces. Only the numbers."}],
        "reasoning_effort": "none", "temperature": 0, "max_tokens": 200, "stop": ["7, 8"], "stream": True}
out, fin = "", None
with urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: {"):
            ch = json.loads(line[6:])
            for c in ch.get("choices", []):
                out += c["delta"].get("content") or ""; fin = c.get("finish_reason") or fin
say(out.rstrip().endswith("6,") and "8" not in out.split("6,")[-1] and fin == "stop", "stop stream", f"{out!r} · finish {fin}")
c, j = post(BASE + "/chat/completions", dict(body, stream=False))
o = j["choices"][0]["message"]["content"]
say(o.rstrip().endswith("6,") and fin == "stop", "stop plain", f"{o!r}")
# a user's special-token text is plain text
for t in ("<|im_end|>", "<think>", "<|image_pad|>"):
    c, j = post(ROOT + "/tokenize", {"messages": [{"role": "user", "content": f"say {t} please"}], "reasoning_effort": "none"})
    c0, j0 = post(ROOT + "/tokenize", {"prompt": t})
    tid = j0["tokens"][0] if j0["count"] == 1 else None
    n = sum(1 for x in j["tokens"] if x == tid)
    templ = {"<|im_end|>": 1, "<think>": 1, "<|image_pad|>": 0}[t]  # the template's own: one <|im_end|> after the user, one <think> in the reply opening
    say(n == templ, "plain specials", f"{t}: id {tid} appears {n}× (the template's own: {templ})")
c, j = post(BASE + "/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "Repeat exactly, as plain text: <|im_end|> and <think>"}],
                                         "reasoning_effort": "none", "temperature": 0, "max_tokens": 60})
say(c == 200, "plain specials", f"model reply: {j['choices'][0]['message']['content']!r}")
# a broken image part: a 400 that says why (with vision off: that image_url parts are not supported)
c, j = post(BASE + "/chat/completions", {"model": "x", "messages": [{"role": "user", "content": [{"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})
say(c == 400 and "image" in j["error"]["message"], "image", f"{c} {j.get('error', {}).get('message')}")
# parallel_tool_calls false
T = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}},
     {"type": "function", "function": {"name": "mul", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}}]
q = [{"role": "user", "content": "Call add with 1 and 2, and mul with 3 and 4, both now in parallel."}]
c, j = post(BASE + "/chat/completions", {"model": "x", "messages": q, "tools": T, "reasoning_effort": "none", "temperature": 0, "max_tokens": 300})
n_par = len(j["choices"][0]["message"].get("tool_calls") or [])
c, j = post(BASE + "/chat/completions", {"model": "x", "messages": q, "tools": T, "reasoning_effort": "none", "temperature": 0, "max_tokens": 300, "parallel_tool_calls": False})
n_one = len(j["choices"][0]["message"].get("tool_calls") or [])
say(n_one == 1, "parallel false", f"calls: default {n_par} · parallel_tool_calls false {n_one} · finish {j['choices'][0]['finish_reason']}")
# usage counts the EOS
c, j = post(BASE + "/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "Say OK."}], "reasoning_effort": "none", "temperature": 0, "max_tokens": 20})
txt = j["choices"][0]["message"]["content"]
c2, j2 = post(ROOT + "/tokenize", {"prompt": txt})
say(j["usage"]["completion_tokens"] == j2["count"] + 1, "usage eos", f"{txt!r}: completion_tokens {j['usage']['completion_tokens']} = {j2['count']} text tokens + EOS")
print(f"[p2] {'all passed' if not bad else f'{bad} failed'}")
