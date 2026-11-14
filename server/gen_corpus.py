"""Generate a token-frequency corpus from the running server (stdlib only; from the Mac), for the P006 draft vocabulary.
Prompts: server/corpus_prompts.json (Korean / English / code / math / tool calls / other languages). Serving conditions:
thinking on, server default sampling, seed per prompt, natural end up to --cap tokens, --par requests at once.
Output: one JSON line per prompt {i, split, reasoning, content, tool_calls, usage} → --out (appends; done ids skipped).
  python3 server/gen_corpus.py --out <file.jsonl> [URL]"""
import argparse
import json
import os
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://localhost:8888/v1")
ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--par", type=int, default=4)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "corpus_prompts.json")))
MODEL = json.loads(urllib.request.urlopen(a.url + "/models", timeout=30).read())["data"][0]["id"]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string",
          "enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return stdout",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
         {"type": "function", "function": {"name": "search_docs", "description": "Search the project documentation",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
          "required": ["query"]}}}]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["i"] for l in open(a.out) if l.strip()}
todo = [i for i in range(len(P)) if i not in done]
lock = threading.Lock()
t_start = time.time()


def one(i):
    p = P[i]
    body = {"model": MODEL, "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": a.cap, "seed": 7000 + i}
    if p.get("tools"):
        body["tools"] = TOOLS
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
    msg = r["choices"][0]["message"]
    rec = {"i": i, "split": "build" if i % 2 == 0 else "eval", "kind": p["kind"], "reasoning": msg.get("reasoning_content") or "",
           "content": msg.get("content") or "", "tool_calls": msg.get("tool_calls") or [], "usage": r.get("usage")}
    with lock:
        with open(a.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = rec["usage"]["completion_tokens"] if rec["usage"] else 0
        print(f"\033[36m[corpus]\033[0m \033[32m#{i} {p['kind']} · output {n} · {time.time() - t_start:.0f} s\033[0m",
              flush=True)


def worker(q):
    while True:
        with lock:
            if not q:
                return
            i = q.pop(0)
        try:
            one(i)
        except Exception as e:  # noqa: BLE001 — keep going; the line says which prompt failed
            print(f"\033[36m[corpus]\033[0m \033[31m#{i} failed: {e}\033[0m", flush=True)


th = [threading.Thread(target=worker, args=(todo,)) for _ in range(a.par)]
[t.start() for t in th]
[t.join() for t in th]
print(f"\033[36m[corpus]\033[0m \033[32mdone · {time.time() - t_start:.0f} s · {a.out}\033[0m", flush=True)
