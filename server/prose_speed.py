"""Solo decode speed by kind of text (stdlib only; from the Mac): one request at a time against the running server,
English / Korean prose prompts, thinking on (server default) and off (reasoning_effort "none"), server default sampling,
max_tokens 1500. Per request tok/s after the first token and tokens per round (/metrics deltas); then averages per
kind. The round time is ~the same for every kind; the speed differences are the drafts' acceptance.
  python3 server/prose_speed.py [URL]"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8888/v1"
MODEL = json.loads(urllib.request.urlopen(BASE + "/models", timeout=30).read())["data"][0]["id"]
P = [
    ("en", "Write a reflective essay of about 800 words on what it means to grow up in a small coastal town."),
    ("en", "Write a short story of about 800 words about an old clockmaker who repairs a stranger's watch."),
    ("en", "Write an 800-word opinion piece on why public libraries still matter in the age of the internet."),
    ("ko", "작은 바닷가 마을에서 자란다는 것이 어떤 의미인지에 대해 800단어 정도의 수필을 써 줘."),
    ("ko", "낡은 시계방 주인이 낯선 손님의 시계를 고쳐 주는 이야기를 800단어 정도의 단편 소설로 써 줘."),
    ("ko", "인터넷 시대에도 공공 도서관이 여전히 중요한 이유에 대해 800단어 정도의 칼럼을 써 줘."),
]


def metrics():
    m = {}
    for line in urllib.request.urlopen(BASE.rsplit("/v1", 1)[0] + "/metrics", timeout=10).read().decode().splitlines():
        if line and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            m[k.split("{")[0]] = float(v)
    return m


def run(prompt, think, seed):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}}
    if not think:
        body["reasoning_effort"] = "none"
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    m0 = metrics()
    t0, first, usage, nr, nc = time.time(), None, None, 0, 0
    for line in urllib.request.urlopen(req, timeout=3600):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])
        usage = ch.get("usage") or usage
        for c in ch.get("choices", []):
            d = c.get("delta", {})
            r, ct = d.get("reasoning_content") or "", d.get("content") or ""
            if (r or ct) and first is None:
                first = time.time()
            nr += len(r)
            nc += len(ct)
    end = time.time()
    m1 = metrics()
    dr = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
    da = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
    n = usage["completion_tokens"]
    return (n - 1) / (end - first), 1 + da / max(dr, 1), n, nr, nc


res = []
for think in (True, False):
    for i, (lang, p) in enumerate(P):
        tps, tpr, n, nr, nc = run(p, think, 100 + i)
        res.append((think, lang, tps, tpr, n))
        print(f"{'thinking on' if think else 'thinking off'} {lang} #{i}: {tps:5.1f} tok/s · per round {tpr:.2f} · "
              f"output {n} (thinking {nr} chars · answer {nc} chars)", flush=True)
for think in (True, False):
    for lang in ("en", "ko"):
        r = [x for x in res if x[0] == think and x[1] == lang]
        tot_n = sum(x[4] for x in r)
        print(f"== {'thinking on' if think else 'thinking off'} {lang}: mean {sum(x[2] for x in r) / len(r):.1f} "
              f"tok/s · per round {sum(x[3] * x[4] for x in r) / tot_n:.2f} (token-weighted)", flush=True)
sys.stdout.flush()
