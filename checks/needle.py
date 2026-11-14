"""Long-context recall against a running server — stdlib only (Python >= 3.8), any machine that reaches the server.

  python3 checks/needle.py [base URL] [--lens 128,512,1024] [--limit N] [--books DIR] [--out FILE]

The text is War and Peace followed by Anna Karenina (Project Gutenberg pg2600 + pg1399, header and footer cut,
paragraphs, whitespace folded; downloaded once into --books and checked by sha256). Per length, 30 records ("Record
4821: The grain depot at Dunmere logged 53172 units of rye flour.") go in at 3–97% depth (seeded by the length's
label, so every run and every server gets the same records), and the first k paragraphs are fitted so the prompt, chat
template included, is target−512 … target tokens (the server's /tokenize). target = min(K·1024, window − 320).
Each record is asked once, in depth order, one request at a time, thinking off, temperature 0, max_tokens 64: the first
question of a length is a cold read, the rest share the text and differ only in the question at the end (they show how
much of the prompt the server reuses). A record counts as found when the answer holds its number and no other
record's. One line per length; exit code 1 if any record is missed.
The records, fitting and grading follow a long-context test used to compare servers on these machines, so the results
line up question by question."""
import argparse
import bisect
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

N = 30
FIT_BAND = 512
MAX_TOKENS = 64
RESERVE = 256
BOOKS = (("pg2600", "2d5bb2ad5f422765e714617e21fa31bbaf8958aa79682c86fca6660fcc5d1b2b",
          "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"),          # War and Peace
         ("pg1399", "a3e29e08c15c63bb22069f4256316c2ebebd422ecb5c33462ddc00ae00b27571",
          "https://www.gutenberg.org/cache/epub/1399/pg1399.txt"))          # Anna Karenina
OFFICES = ["harbor office", "grain depot", "customs house", "river station", "salt warehouse", "mill yard",
           "post office", "timber yard", "wool exchange", "coal wharf"]
PLACES = ["Avonlea", "Brindlemoor", "Castorford", "Dunmere", "Elsworth", "Fallowby", "Greywick", "Holloway",
          "Ingleton", "Juniper Cross", "Kettering Vale", "Larkspur"]
ITEMS = ["barley", "lamp oil", "copper wire", "linen", "rye flour", "tallow candles", "iron nails", "indigo dye",
         "rope", "dried cod"]
QUESTION = "Question: How many units did Record {rid} log? Answer with the number only."
INTRO = "Below is a long text. Answer the question that follows it.\n\n"

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8888/v1")
ap.add_argument("--lens", default="128,512,1024", help="lengths in K tokens (1024 = 1M), capped by the server's window")
ap.add_argument("--limit", type=int, default=N, help="records per length (the first N by depth)")
ap.add_argument("--books", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".work",
                                                "needle"), help="folder with pg2600.txt and pg1399.txt (downloaded "
                                                                 "there if missing)")
ap.add_argument("--out", help="write every answer here (JSON lines)")
ap.add_argument("--key", default=os.environ.get("Q38_SERVER_API_KEY", ""), help="API key, if the server wants one")
a = ap.parse_args()
BASE = a.url.rstrip("/")
ROOT = BASE.rsplit("/v1", 1)[0]
HDR = {"Content-Type": "application/json", **({"Authorization": f"Bearer {a.key}"} if a.key else {})}


def say(ok, tag, msg):
    print(f"\033[36m[needle {tag}]\033[0m \033[{32 if ok else 31}m{msg}\033[0m", flush=True)


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), HDR)
    return urllib.request.urlopen(req, timeout=timeout)


# ---- the text and the records ------------------------------------------------------------------------------------
def book(name, sha, url):
    path = os.path.join(a.books, name + ".txt")
    if not os.path.exists(path):
        os.makedirs(a.books, exist_ok=True)
        say(True, "books", f"downloading {url}")
        data = urllib.request.urlopen(url, timeout=300).read()
        if hashlib.sha256(data).hexdigest() != sha:
            sys.exit(f"{url}: sha256 differs from the pinned one — not used")
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != sha:
        sys.exit(f"{path}: sha256 differs from the pinned one ({sha[:12]}…)")
    t = open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
    ps = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", t[s.end():e.start()])]
    return [p for p in ps if p]


def records(label, taken):
    rng = random.Random(f"niah-ref-{label}")
    used, recs = set(), []
    for i in range(N):
        while True:
            rid = rng.randint(1000, 9999)
            units = rng.randint(10000, 99999)
            if rid not in used and str(units) not in taken and units not in [r["units"] for r in recs]:
                break
        used.add(rid)
        recs.append({"rid": rid, "units": units, "depth": 0.03 + i * 0.94 / 29,
                     "text": f"Record {rid}: The {rng.choice(OFFICES)} at {rng.choice(PLACES)} logged {units} units "
                             f"of {rng.choice(ITEMS)}."})
    return recs


def text_for(paras, cum, recs, k):
    body, total = list(paras[:k]), cum[k]
    for r in sorted(recs, key=lambda r: -r["depth"]):
        idx = bisect.bisect_left(cum, r["depth"] * total, 0, k + 1)
        body.insert(min(idx, k), r["text"])
    return "\n\n".join(body)


def msgs(text, rid):
    return [{"role": "user", "content": INTRO + text + "\n\n" + QUESTION.format(rid=rid)}]


def count(text, rid):
    r = post(ROOT + "/tokenize", {"model": MODEL, "messages": msgs(text, rid), "add_generation_prompt": True}, 600)
    return json.loads(r.read())["count"]


def fit(paras, cum, recs, target):
    """The first k paragraphs (records in) so the prompt is target−512 … target tokens → (text, tokens) or None."""
    rid0, k, best = recs[0]["rid"], bisect.bisect_left(cum, target * 4.2), None
    for _ in range(12):
        k = max(1, min(k, len(paras)))
        n = count(text_for(paras, cum, recs, k), rid0)
        if n <= target and (best is None or n > best[1]):
            best = (k, n)
        if target - FIT_BAND <= n <= target:
            break
        if k == len(paras) and n < target - FIT_BAND:
            return None
        want = cum[k] + (target - 256 - n) / (n / cum[k])
        nk = bisect.bisect_left(cum, want)
        k = nk if nk != k else (k - 1 if n > target else k + 1)
    if best is None:
        return None
    return text_for(paras, cum, recs, best[0]), best[1]


def grade(content, rec, recs):
    nums = {x.replace(",", "") for x in re.findall(r"\d[\d,]*", content)}
    ok = str(rec["units"]) in nums
    decoy = any(str(r["units"]) in nums for r in recs if r is not rec)
    return "hit" if ok and not decoy else "mixed" if ok else "decoy" if decoy else "other" if nums else "none"


def ask(text, rid):
    """→ (content, seconds to the first piece, total seconds, usage)."""
    t0, first, out, usage = time.time(), None, [], {}
    body = {"model": MODEL, "messages": msgs(text, rid), "reasoning_effort": "none", "temperature": 0, "top_p": 1,
            "max_tokens": MAX_TOKENS, "stream": True, "stream_options": {"include_usage": True}}
    with post(BASE + "/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    first = first or time.time() - t0
                    out.append(piece)
    return "".join(out), first or time.time() - t0, time.time() - t0, usage


# ---- run -----------------------------------------------------------------------------------------------------------
m = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/models", headers=HDR), timeout=30).read())["data"][0]
MODEL, WINDOW = m["id"], int(m.get("max_model_len") or 0)
books = [book(*b) for b in BOOKS]
paras = books[0] + books[1]
cum = [0]
for p in paras:
    cum.append(cum[-1] + len(p) + 2)
taken = set(re.findall(r"\d+", "\n".join(paras)))
say(True, "server", f"{MODEL} · window {WINDOW:,} · text {len(paras):,} paragraphs")
out = open(a.out, "a") if a.out else None
bad = 0
for k in [float(x.rstrip("kKmM")) * (1024 if x.lower().endswith("m") else 1) for x in a.lens.split(",") if x]:
    nominal = int(round(k * 1024))
    kk = int(round(nominal / 1024))
    label = f"{kk // 1024}M" if kk >= 1024 and kk % 1024 == 0 else f"{kk}K"
    if nominal > WINDOW:
        say(True, label, f"skipped — the server's window is {WINDOW:,}")
        continue
    recs = records(label, taken)
    target = min(nominal, WINDOW - MAX_TOKENS - RESERVE)
    t0 = time.time()
    got = fit(paras, cum, recs, target)
    if got is None:
        say(False, label, f"the text is too short for {target:,} tokens")
        bad += 1
        continue
    text, ntok = got
    say(True, label, f"prompt {ntok:,} tokens (target {target:,}) · fitted in {time.time() - t0:.0f} s · asking "
                     f"{min(a.limit, N)} records")
    res, firsts = [], []
    for i, rec in enumerate(recs[:a.limit]):
        content, first, total, usage = ask(text, rec["rid"])
        g = grade(content, rec, recs)
        res.append((rec, g))
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if i:
            firsts.append(first)
        if out:
            out.write(json.dumps({"len": label, "rid": rec["rid"], "depth": round(rec["depth"], 4), "grade": g,
                                  "content": content, "first_s": round(first, 2), "total_s": round(total, 2),
                                  "prompt_tokens": usage.get("prompt_tokens"), "cached": cached}) + "\n")
            out.flush()
        if i == 0:
            say(True, label, f"cold read {first:.1f} s ({ntok / first:,.0f} tokens/s) · first answer {g}")
    hits = sum(g == "hit" for _, g in res)
    miss = [f"{r['depth'] * 100:.0f}% {g}" for r, g in res if g != "hit"]
    fs = sorted(firsts)
    shared = f" · later questions: first token {fs[0]:.1f}–{fs[-1]:.1f} s (median {fs[len(fs) // 2]:.1f})" if fs else ""
    say(hits == len(res), label, f"{hits}/{len(res)} found{shared}" + (f" · missed: {', '.join(miss)}" if miss else ""))
    bad += hits != len(res)
sys.exit(1 if bad else 0)
