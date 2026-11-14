"""Same-seed output check against the running server (stdlib only; from the Mac): run it before and after a server or
scheduler change and compare the two files — the outputs must be identical (the engine is deterministic for a seed).
Cases: a ~20K-character document turn, the follow-up turn (prefix reuse through the slot snapshot), a cold Korean prompt.
  python3 server/det_check.py <out.json> --doc <text file> [URL]     then:  cmp before.json after.json"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--doc", required=True, help="any long text (e.g. the model README + chat template, as bench_lite)")
ap.add_argument("url", nargs="?", default="http://localhost:8888/v1")
a = ap.parse_args()
doc = open(a.doc).read()[:20000]


def ask(msgs, seed, mt=120):
    body = {"model": "x", "messages": msgs, "max_tokens": mt, "seed": seed}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "||" + (m.get("content") or "")


c1 = [{"role": "user", "content": doc + "\n\nSummarize this in three sentences."}]
out = {"t1": ask(c1, 11)}
out["t2"] = ask(c1 + [{"role": "assistant", "content": "It is a model card."}, {"role": "user", "content": "Which formats are used?"}], 12)
out["cold"] = ask([{"role": "user", "content": "작은 바닷가 마을에 대한 짧은 시를 써 줘."}], 13)
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"\033[36m[det]\033[0m \033[32m{', '.join(f'{k} {len(v)} chars' for k, v in out.items())} → {a.out}\033[0m")
