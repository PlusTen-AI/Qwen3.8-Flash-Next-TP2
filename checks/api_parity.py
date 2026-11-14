"""vLLM 0.32 parity checks against a running server (stdlib; any machine): tool_choice none / auto / required / named
(thinking on and off), typed arguments, stream fields, include_reasoning, thinking_token_budget 0, vLLM error texts,
penalties / min_p, /tokenize /detokenize /version /v1/models/{id}.  python3 checks/api_parity.py http://<head>:8888/v1"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/")  # .../v1
ROOT = BASE.rsplit("/v1", 1)[0]
bad = 0
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "add", "description": "Add two integers", "parameters": {
             "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]


def say(ok, tag, msg):
    global bad
    bad += not ok
    print(f"[p1 {tag}] {'OK' if ok else 'FAIL'} · {msg}", flush=True)


def call(path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request((ROOT if path.startswith("/tok") or path.startswith("/det") or path == "/version" else BASE) + path,
                                 data, {"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def chat(msgs, **kw):
    b = {"model": "x", "messages": msgs, "max_tokens": 1500, "seed": 3}
    b.update(kw)
    return call("/chat/completions", b)


W = [{"role": "user", "content": "What's the weather in Paris for the next 3 days?"}]
for think in (True, False):
    eff = "low" if think else "none"
    c, j = chat(W, tools=TOOLS, tool_choice="required", reasoning_effort=eff)
    ch = j["choices"][0] if c == 200 else {}
    m = ch.get("message", {})
    tc = m.get("tool_calls") or []
    say(c == 200 and tc and ch["finish_reason"] == "tool_calls" and m.get("content") == "", "required",
        f"thinking {think}: {c} · calls {[t['function']['name'] + t['function']['arguments'] for t in tc]} · finish "
        f"{ch.get('finish_reason')} · content {m.get('content')!r}")
    c, j = chat([{"role": "user", "content": "Hello, how are you?"}], tools=TOOLS, reasoning_effort=eff,
                tool_choice={"type": "function", "function": {"name": "add"}})
    ch = j["choices"][0] if c == 200 else {}
    m = ch.get("message", {})
    tc = m.get("tool_calls") or []
    say(c == 200 and len(tc) == 1 and tc[0]["function"]["name"] == "add" and ch["finish_reason"] == "stop", "named",
        f"thinking {think}: {c} · calls {[t['function']['name'] + t['function']['arguments'] for t in tc]} · finish "
        f"{ch.get('finish_reason')}")
c, j = chat([{"role": "user", "content": "Add 2 and 40 using the tool."}], tools=TOOLS, reasoning_effort="low")
m = j["choices"][0]["message"]
args = [json.loads(t["function"]["arguments"]) for t in m.get("tool_calls") or []]
say(c == 200 and args and all(isinstance(v, int) for a in args for v in a.values()), "auto typed",
    f"{args} · finish {j['choices'][0]['finish_reason']}")
c, j = chat([{"role": "user", "content": "Add 2 and 40 using the tool."}], tools=TOOLS, tool_choice="none",
            reasoning_effort="none")
m = j["choices"][0]["message"]
say(c == 200 and not m.get("tool_calls") and "<tool_call>" not in (m.get("content") or ""), "none",
    f"tool_calls {m.get('tool_calls')} · content {(m.get('content') or '')[:60]!r}")
c, j = chat(W, tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}})
say(c == 400 and "does not match" in j["error"]["message"], "named bad", f"{c} {j.get('error', {}).get('message')}")

# streaming: reasoning field names, tool deltas, usage
body = {"model": "x", "messages": [{"role": "user", "content": "Add 5 and 6 with the tool."}], "tools": TOOLS,
        "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 1500}
req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
keys, fin, usage, tdel = set(), None, None, []
with urllib.request.urlopen(req, timeout=900) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])
        usage = ch.get("usage") or usage
        for c0 in ch.get("choices", []):
            keys |= set(c0["delta"])
            tdel += c0["delta"].get("tool_calls") or []
            fin = c0.get("finish_reason") or fin
say({"reasoning", "reasoning_content"} <= keys and tdel and fin == "tool_calls" and usage and
    usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) > 0, "stream",
    f"delta keys {sorted(keys)} · tool deltas {[(d.get('id', '')[:14], d['function'].get('name'), d['function'].get('arguments')) for d in tdel]} · "
    f"finish {fin} · usage {usage}")

# reasoning controls
c, j = chat([{"role": "user", "content": "What is 6*7?"}], reasoning_effort="low", include_reasoning=False)
m = j["choices"][0]["message"]
say(c == 200 and m.get("reasoning") is None and j["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0,
    "include_reasoning", f"reasoning {m.get('reasoning')!r} · reasoning_tokens {j['usage']['completion_tokens_details']}")
c, j = chat([{"role": "user", "content": "What is 6*7?"}], thinking_token_budget=0)
m = j["choices"][0]["message"]
say(c == 200 and not m.get("reasoning") and "42" in (m.get("content") or ""), "budget 0", f"content {m.get('content')!r}")

# errors
W = call("/models")[1]["data"][0]["max_model_len"]  # the server's window
c, j = chat([{"role": "user", "content": "word " * (W + 6000)}], max_tokens=10)
e = j.get("error", {})
say(c == 400 and f"maximum context length is {W} tokens" in e.get("message", "") and e.get("param") == "input_tokens",
    "too long", f"{c} · {e.get('message', '')[:150]} · type {e.get('type')}")
c, j = chat([{"role": "user", "content": "Say OK."}], max_tokens=W, reasoning_effort="none")
say(c == 200, "max clamp", f"max_tokens {W} → {c} · finish {j.get('choices', [{}])[0].get('finish_reason')}")
for kw, want in (({"max_tokens": 0}, "max_tokens must be at least 1, got 0."), ({"temperature": 3}, "temperature must be in [0, 2]"),
                 ({"min_p": 1.5}, "min_p must be in [0, 1]"), ({"presence_penalty": 3}, "presence_penalty must be in [-2, 2]"),
                 ({"repetition_penalty": 0}, "repetition_penalty must be greater than zero"),
                 ({"logit_bias": {"1": 5}}, "logit_bias is unavailable"), ({"response_format": {"type": "json_object"}}, "response_format"),
                 ({"reasoning_effort": "turbo"}, "reasoning effort must be one of")):
    c, j = chat([{"role": "user", "content": "hi"}], **kw)
    msg = j.get("error", {}).get("message", "")
    say(c == 400 and want in msg, "error", f"{kw} → {c} · {msg[:110]}")

# penalties + min_p accepted, and they change a greedy answer
base = [{"role": "user", "content": "List the numbers from 1 to 30 separated by commas."}]
c1, j1 = chat(base, temperature=0, reasoning_effort="none", max_tokens=120)
c2, j2 = chat(base, temperature=0, reasoning_effort="none", max_tokens=120, repetition_penalty=1.5, presence_penalty=1.0,
              frequency_penalty=1.0)
c3, j3 = chat(base, temperature=0.8, reasoning_effort="none", max_tokens=120, min_p=0.1, seed=5)
a1 = j1["choices"][0]["message"]["content"] if c1 == 200 else ""
a2 = j2["choices"][0]["message"]["content"] if c2 == 200 else ""
say(c1 == c2 == c3 == 200 and a1 != a2, "penalties", f"plain {a1[:40]!r} · penalised {a2[:40]!r} · min_p {c3}")

# endpoints
c, j = call("/tokenize", {"prompt": "Hello world"})
say(c == 200 and j["count"] == len(j["tokens"]) and j["max_model_len"] == W, "tokenize", f"{j}")
c2, j2 = call("/detokenize", {"tokens": j["tokens"]})
say(c2 == 200 and j2["prompt"] == "Hello world", "detokenize", f"{j2}")
c, j = call("/tokenize", {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"})
say(c == 200 and j["count"] > 5, "tokenize chat", f"count {j.get('count')}")
c, j = call("/version")
say(c == 200 and "version" in j, "version", f"{j}")
c, j = call("/models/Qwen/Qwen3.8-Flash-Next")
say(c == 200 and j.get("max_model_len") == W, "model card", f"{j}")
c, j = call("/models/other")
say(c == 404, "model 404", f"{c} {j.get('error', {}).get('message')}")
print(f"[p1] {'all passed' if not bad else f'{bad} failed'}")
sys.exit(1 if bad else 0)
