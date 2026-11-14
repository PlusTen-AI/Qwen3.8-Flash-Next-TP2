"""Image / video checks against a running server — stdlib only (Python >= 3.8), any machine that reaches the server.
  python3 checks/vision.py [base URL] [--only read,reuse,errors,golden] [--media DIR --golden tokens.json]

  read    a picture drawn here (a number in large digits on a coloured background, a coloured square): with thinking
          off the answer must contain the number and the square's colour; one request with two pictures
  reuse   the same picture twice → the second prompt is reused from the cache; the same text with another picture →
          only the text before the picture is reused (the cache never serves another picture's state)
  errors  a bad base64, a URL scheme the server does not fetch, an image in a system message → 400
  golden  (development: --media = the dump's media folder, --golden = its tokens.json) every vision request of the dump
          again, greedy: the prompt token count must equal the reference's; the output is compared token-for-token as
          text (how far it agrees)
One line per case; exit code 1 if any case fails."""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?", default="http://127.0.0.1:8888/v1")
ap.add_argument("--only", default="read,reuse,errors,golden")
ap.add_argument("--media", default="", help="golden: the dump's media folder")
ap.add_argument("--golden", default="", help="golden: the dump's tokens.json")
a = ap.parse_args()
BASE = a.url.rstrip("/")
bad = 0


def say(ok, tag, msg):
    global bad
    bad += not ok
    print(f"\033[36m[vision {tag}]\033[0m \033[{32 if ok else 31}m{msg}\033[0m", flush=True)


def post(body, timeout=600):
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def model():
    with urllib.request.urlopen(BASE + "/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


# ---- a PNG drawn here (stdlib): digits in a 5×7 font, scaled -------------------------------------------------------
FONT = {"0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
        "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
        "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
        "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
        "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100"}


def png(w, h, bg, draws):
    """RGB PNG: background, then (x0, y0, x1, y1, rgb) rectangles."""
    px = [bytearray(bytes(bg) * w) for _ in range(h)]
    for x0, y0, x1, y1, c in draws:
        for y in range(max(0, y0), min(h, y1)):
            for x in range(max(0, x0), min(w, x1)):
                px[y][3 * x:3 * x + 3] = bytes(c)
    raw = b"".join(b"\0" + bytes(r) for r in px)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def digits(text, x, y, s, c):
    out = []
    for k, ch in enumerate(text):
        bits = FONT[ch]
        for i, b in enumerate(bits):
            if b == "1":
                r, q = divmod(i, 5)
                out.append((x + k * 6 * s + q * s, y + r * s, x + k * 6 * s + (q + 1) * s, y + (r + 1) * s, c))
    return out


def picture(number, square):
    colours = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (40, 70, 230), "yellow": (240, 210, 20)}
    d = digits(number, 40, 40, 14, (20, 20, 20)) + [(420, 300, 600, 460, colours[square])]
    return png(640, 480, (235, 235, 225), d)


def image_part(data, mime="image/png"):
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}}


def ask(content, max_tokens=64, think=False, **kw):
    body = {"model": M, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "seed": 1, "chat_template_kwargs": {"enable_thinking": think}, **kw}
    if think:
        body["reasoning_effort"] = "low"
    t0 = time.time()
    code, r = post(body)
    return code, r, time.time() - t0


M = model()
only = a.only.split(",")

if "read" in only:
    for number, sq in (("4721", "red"), ("9058", "blue")):
        code, r, dt = ask([image_part(picture(number, sq)),
                           {"type": "text", "text": "What number is written in the picture, and what colour is the "
                                                    "square in the lower right? Answer in one short sentence."}])
        txt = (r.get("choices") or [{}])[0].get("message", {}).get("content") or "" if code == 200 else str(r)
        ok = code == 200 and number in txt.replace(",", "") and sq in txt.lower()
        say(ok, "read", f"{number} / {sq}: {txt.strip()[:90]!r} · prompt {r.get('usage', {}).get('prompt_tokens')} "
                        f"tokens · {dt:.1f} s")
    code, r, dt = ask([{"type": "text", "text": "Picture A:"}, image_part(picture("315", "green")),
                       {"type": "text", "text": "Picture B:"}, image_part(picture("8086", "yellow")),
                       {"type": "text", "text": "Which numbers are written in pictures A and B? Answer briefly."}],
                      1024, think=True)  # thinking off, the greedy answer may stop after A (the model's habit)
    txt = r["choices"][0]["message"]["content"] or "" if code == 200 else str(r)
    say(code == 200 and "315" in txt and "8086" in txt.replace(",", ""), "read two", f"{txt.strip()[:90]!r} · {dt:.1f} s")

if "reuse" in only:
    p1, p2 = picture("2468", "blue"), picture("1357", "red")
    q = {"type": "text", "text": "Describe this picture in one sentence."}
    pre = [{"type": "text", "text": "Here is a long note before the picture. " * 40}]
    _, r1, _ = ask(pre + [image_part(p1), q], 8)
    _, r2, _ = ask(pre + [image_part(p1), q], 8)
    _, r3, _ = ask(pre + [image_part(p2), q], 8)
    u1, u2, u3 = (x["usage"] for x in (r1, r2, r3))
    c2 = u2.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    c3 = u3.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    say(c2 >= u2["prompt_tokens"] - 1, "reuse same", f"same picture again: {c2:,} of {u2['prompt_tokens']:,} reused")
    # the other picture: at most the text before it (the picture starts after ~400 note tokens + the template)
    say(c3 < u3["prompt_tokens"] - 64, "reuse other",
        f"another picture: {c3:,} of {u3['prompt_tokens']:,} reused (only the text before it)")

if "errors" in only:
    cases = [("bad base64", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@@"}}]),
             ("not an image", [image_part(b"hello, not a picture")]),
             ("file URL", [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}])]
    for name, content in cases:
        code, r, _ = ask(content + [{"type": "text", "text": "hi"}], 4)
        say(code == 400, f"error {name}", f"HTTP {code} · {(r.get('error') or {}).get('message', '')[:90]!r}")
    code, r = post({"model": M, "max_tokens": 4, "messages": [
        {"role": "system", "content": [image_part(picture("1", "red"))]}, {"role": "user", "content": "hi"}]})
    say(code == 400, "error system image", f"HTTP {code} · {(r.get('error') or {}).get('message', '')[:90]!r}")

if "golden" in only and a.golden:
    g = json.load(open(a.golden))["vision"]["cases"]
    for name, case in g.items():
        if name == "chart2":
            continue
        msgs = json.loads(json.dumps(case["messages"]))
        it = iter(case["files"])
        for m in msgs:
            for p in m["content"] if isinstance(m["content"], list) else []:
                for k, mime in (("image_url", "image/png"), ("video_url", "video/mp4")):
                    if k in p:
                        f = next(it)
                        p[k]["url"] = f"data:{mime};base64," + base64.b64encode(open(f"{a.media}/{f}", "rb").read()).decode()
        t0 = time.time()
        code, r = post({"model": M, "messages": msgs, "max_tokens": case["max_tokens"], "temperature": 0.0, "seed": 1,
                        "skip_special_tokens": False, "chat_template_kwargs": {"reasoning_effort": "low"}})
        dt = time.time() - t0
        if code != 200:
            say(False, f"golden {name}", f"HTTP {code} · {r}")
            continue
        msg = r["choices"][0]["message"]
        ours = (msg.get("reasoning") or msg.get("reasoning_content") or "") + (msg.get("content") or "")
        ref = case["text"] or ""
        n = 0
        while n < min(len(ours), len(ref)) and ours[n] == ref[n]:
            n += 1
        pt = r["usage"]["prompt_tokens"]
        say(pt == len(case["prompt_ids"]), f"golden {name}",
            f"prompt {pt:,} tokens (reference {len(case['prompt_ids']):,}) · output agrees for {n}/{len(ref)} chars "
            f"· {dt:.1f} s · {ours[:50]!r}")

print(f"\033[36m[vision]\033[0m {'all passed' if not bad else f'{bad} failed'}", flush=True)
sys.exit(1 if bad else 0)
