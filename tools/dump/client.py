"""Dump client: fixed prompts → token ids + greedy outputs with top-20 logprobs. Stdlib only.

Prompts (fixed, from read-only files so they never drift):
  short  — one chat question
  long   — model README + chat template + a question (> indexer budget, so QSA selection is exercised)
  short2 — `short` again: determinism check (same tokens and logprobs expected)
Vision (with --media DIR, files from media.py): chat requests with images and videos as data URLs through the
server's own chat path (its template, processor and placeholder expansion); prompt token ids as the server built them.
Writes OUT/tokens.json. Usage: client.py URL OUT MODEL_DIR [--tokenize-only] [--media DIR]
"""
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qlog  # noqa: E402

URL, OUT, MD = sys.argv[1:4]
ONLY_TOKENIZE = "--tokenize-only" in sys.argv
MEDIA = sys.argv[sys.argv.index("--media") + 1] if "--media" in sys.argv else None
MODEL = "qdump"


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        qlog.fail("dump request", f"{path} HTTP {e.code} · {e.read().decode()[:200]}")


_TOK = None


def chat_ids(text):
    """Chat template applied locally (the model's own tokenizer + template, effort low) → token ids."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MD)
    txt = _TOK.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
                                   reasoning_effort="low")
    return list(_TOK(txt, add_special_tokens=False)["input_ids"])


def gen(ids, n):
    r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": n, "temperature": 0.0,
                                 "logprobs": 20, "return_tokens_as_token_ids": True, "seed": 1})
    lp = r["choices"][0]["logprobs"]
    toks = [int(t.split(":", 1)[1]) for t in lp["tokens"]]
    top = [{k.split(":", 1)[1]: v for k, v in d.items()} for d in lp["top_logprobs"]]
    return toks, lp["token_logprobs"], top, r["choices"][0]["text"]


short = "Explain in three sentences why the sky is blue, and give one everyday example of the same effect."
readme = open(os.path.join(MD, "README.md")).read()
tmpl = open(os.path.join(MD, "chat_template.jinja")).read()
long = (readme + "\n\n---\n\n" + tmpl + "\n\n---\n\nUsing only the document above: which components are stored in "
        "NVFP4, which in MXFP8, and what precision do the hyperconnection norms use? Answer briefly.")
src = {"short": short, "long": long}
res = {"source_sha256": {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in src.items()}}
if ONLY_TOKENIZE:
    for name, text in src.items():
        ids = chat_ids(text)
        qlog.ok(f"dump {name}", f"tokens {len(ids):,} · first {ids[:4]} · end {_TOK.decode(ids[-6:])!r}")
    sys.exit(0)
for name, text, n in (("short", short, 32), ("long", long, 16), ("short2", short, 32)):
    t0 = time.time()
    ids = chat_ids(text)
    toks, lps, top, txt = gen(ids, n)
    res[name] = {"prompt_ids": ids, "output_ids": toks, "logprobs": lps, "top_logprobs": top, "text": txt}
    qlog.ok(f"dump {name}", f"input {len(ids):,} · output {len(toks)} · {time.time() - t0:.1f} s · {txt[:40]!r}")

same = res["short"]["output_ids"] == res["short2"]["output_ids"]
dl = max(abs(a - b) for a, b in zip(res["short"]["logprobs"], res["short2"]["logprobs"]))
res["determinism"] = {"same_tokens": same, "max_logprob_diff": dl}
json.dump(res, open(os.path.join(OUT, "tokens.json"), "w"))
(qlog.ok if same and dl == 0 else qlog.warn)("dump repeat", f"short twice · tokens {'same' if same else 'differ'} · "
                                                         f"max logprob diff {dl:.2e}")


# ---- vision -------------------------------------------------------------------------------------------------------
def part(name):
    """Content part for a media file: a data URL (no network, no local paths)."""
    raw = open(os.path.join(MEDIA, name), "rb").read()
    if name.endswith(".png"):
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}
    return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(raw).decode()}}


def text(s):
    return {"type": "text", "text": s}


def chat(messages, n):
    r = post("/v1/chat/completions", {
        "model": MODEL, "messages": messages, "max_tokens": n, "temperature": 0.0, "seed": 1,
        "logprobs": True, "top_logprobs": 20, "return_tokens_as_token_ids": True, "return_token_ids": True,
        "skip_special_tokens": False, "chat_template_kwargs": {"reasoning_effort": "low"}})
    c = r["choices"][0]
    lp = c["logprobs"]["content"]
    toks = [int(x["token"].split(":", 1)[1]) for x in lp]
    top = [{t["token"].split(":", 1)[1]: t["logprob"] for t in x["top_logprobs"]} for x in lp]
    return {"prompt_ids": r.get("prompt_token_ids"), "output_ids": toks, "token_ids": c.get("token_ids"),
            "logprobs": [x["logprob"] for x in lp], "top_logprobs": top, "text": c["message"].get("content")}


if MEDIA:
    sums = json.load(open(os.path.join(MEDIA, "sha256.json")))
    user = lambda *p: {"role": "user", "content": list(p)}  # noqa: E731
    cases = [
        ("chart", [user(part("chart.png"), text("What number is written in the title, and which bar is the tallest? "
                                                 "Answer briefly."))], 24, ["chart.png"]),
        ("two", [user(text("First picture:"), part("scene.png"), text("Second picture:"), part("tiny.png"),
                      text("What is written in each picture?"))], 24, ["scene.png", "tiny.png"]),
        ("rgba", [user(part("rgba.png"), text("Describe the shape and its color in one sentence."))], 16, ["rgba.png"]),
        ("turns", [user(part("chart.png"), text("How many bars are there?")),
                   {"role": "assistant", "content": "There are four bars: A, B, C and D."},
                   user(text("Which one is the shortest, and what is its value?"))], 16, ["chart.png"]),
        ("clip", [user(part("clip.mp4"), text("Which direction does the red square move?"))], 24, ["clip.mp4"]),
        ("long", [user(part("long.mp4"), text("What does the text in the top-left corner show at the end?"))], 24,
         ["long.mp4"]),
        ("chart2", [user(part("chart.png"), text("What number is written in the title, and which bar is the tallest? "
                                                  "Answer briefly."))], 24, ["chart.png"]),
    ]
    vis = {"media_sha256": sums, "cases": {}}
    for name, msgs, n, files in cases:
        t0 = time.time()
        out = chat(msgs, n)
        strip = json.loads(json.dumps(msgs))  # the record keeps the request without the data URLs
        for m in strip:
            for p in m["content"] if isinstance(m["content"], list) else []:
                for k in ("image_url", "video_url"):
                    if k in p:
                        p[k]["url"] = p[k]["url"][:30] + "…"
        out.update({"messages": strip, "files": files, "max_tokens": n})
        vis["cases"][name] = out
        np_ = len(out["prompt_ids"] or [])
        nimg = sum(1 for t in out["prompt_ids"] or [] if t in (248056, 248057))
        qlog.ok(f"dump {name}", f"input {np_:,} (media slots {nimg:,}) · output {len(out['output_ids'])} · "
                                f"{time.time() - t0:.1f} s · {(out['text'] or '')[:40]!r}")
    a, b = vis["cases"]["chart"], vis["cases"]["chart2"]
    same = a["output_ids"] == b["output_ids"] and a["prompt_ids"] == b["prompt_ids"]
    dl = max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"]))
    vis["determinism"] = {"same_tokens": same, "max_logprob_diff": dl}
    res["vision"] = vis
    json.dump(res, open(os.path.join(OUT, "tokens.json"), "w"))
    (qlog.ok if same and dl == 0 else qlog.warn)("dump repeat", f"chart twice · tokens {'same' if same else 'differ'}"
                                                                f" · max logprob diff {dl:.2e}")
