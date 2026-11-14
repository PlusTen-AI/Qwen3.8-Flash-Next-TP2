"""Chat-side checks (no GPU): server/chat.py against the golden dump and the Qwen3 parser rules.
  docker run … qwen38-engine/dev:<tag> python3 /work/src/server/check_chat.py
1. Template + tokeniser: the dump client's prompts (tools/dump/client.py: transformers' apply_chat_template with the
   model's template, reasoning_effort "low") rendered by Chat — both chat_template.jinja and the served
   chat_template.local.jinja (what the reference deployment served — the model folder's, else the repo copy in
   tools/vllm_render/, kept when that recipe was removed) — must give exactly the golden prompt_ids.
2. Detokeniser: incremental deltas join to the one-shot decode (golden outputs, token by token).
3. Parser: reasoning / content / tool calls on fixed strings fed in every split."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "tools")]
import qlog  # noqa: E402

from chat import Chat, Detok, Parser  # noqa: E402

MD = "/models/Qwen3.8-Flash-Next-FP4-Local"
gold = json.load(open("/work/results/raw/dump/20261111-1632/tokens.json"))
short = "Explain in three sentences why the sky is blue, and give one everyday example of the same effect."
long = (open(os.path.join(MD, "README.md")).read() + "\n\n---\n\n" + open(os.path.join(MD, "chat_template.jinja")).read() +
        "\n\n---\n\nUsing only the document above: which components are stored in NVFP4, which in MXFP8, and what "
        "precision do the hyperconnection norms use? Answer briefly.")
bad = 0
LOCAL = os.path.join(MD, "chat_template.local.jinja")
if not os.path.exists(LOCAL):
    LOCAL = os.path.join(os.path.dirname(HERE), "tools", "vllm_render", "chat_template.local.jinja")
for tmpl in ("chat_template.jinja", LOCAL):
    c = Chat(MD, tmpl)
    res = []
    for name, text in (("short", short), ("long", long)):
        ids = c.encode(c.render([{"role": "user", "content": text}], reasoning_effort="low"))
        g = gold[name]["prompt_ids"]
        same = ids == g
        bad += not same
        diff_at = next((i for i, (x, y) in enumerate(zip(ids, g)) if x != y), min(len(ids), len(g)))
        verdict = "same" if same else f"differ (golden {len(g):,}, first difference {diff_at})"
        res.append(f"{name} {len(ids):,} tokens {verdict}")
    (qlog.ok if all(r.endswith(" same") for r in res) else qlog.warn)(f"chat template {os.path.basename(tmpl)}",
                                                                      " · ".join(res))

c = Chat(MD)
ok = 0
for name in ("short", "long"):
    out = gold[name]["output_ids"]
    d = Detok(c)
    txt = "".join(d.add([t]) for t in out) + d.flush()
    ok += txt == c.decode(out)
bad += ok != 2
(qlog.ok if ok == 2 else qlog.warn)("chat detok", f"text joined token by token = one-shot decode {ok}/2")

# 2b. encode cache: Chat.encode (pieces between added tokens, LRU) = the tokenizer's own encode, cold and warm, on the
# golden prompts, a long multi-turn conversation (the next turn shares its prefix) and edge strings around added tokens
import time  # noqa: E402

doc = long * 6
conv = [{"role": "user", "content": doc + "\n\nSummarize."}, {"role": "assistant", "content": "It is a model card."},
        {"role": "user", "content": "Now list the formats.\n" + doc[:30000]}]
texts = [c.render(conv[:1]), c.render(conv), c.render(conv, reasoning_effort="none"),
         "a<|im_end|>b<|im_start|><|im_start|>x<think></think><|im_end", "<|im_end|>" * 3 + " " * 300 + "<tool_call>\n{\"a\": 1}",
         "한국어 문장입니다.<|endoftext|>" + "é" * 400 + "\u0301<think>\n" + "코드 " * 200, ""]
for text in (short, long):
    texts.append(c.render([{"role": "user", "content": text}], reasoning_effort="low"))
eq = sum(c.encode(t) == c.tok.encode(t, add_special_tokens=False).ids for t in texts for _ in range(2))
c2 = Chat(MD)
t0 = time.perf_counter()
c2.encode(texts[0])
t1 = time.perf_counter()
c2.encode(texts[1])
t2 = time.perf_counter()
c2.tok.encode(texts[1], add_special_tokens=False)
t3 = time.perf_counter()
bad += eq != 2 * len(texts)
(qlog.ok if eq == 2 * len(texts) else qlog.warn)(
    "chat encode cache", f"{eq}/{2 * len(texts)} same (cold·cached) · next turn {len(c2.encode(texts[1])):,} tokens "
                         f"{1e3 * (t2 - t1):.0f} ms (uncached {1e3 * (t3 - t2):.0f} ms, first turn "
                         f"{1e3 * (t1 - t0):.0f} ms)")

cases = [  # (thinking, text, expected reasoning, content, tool names)
    (True, "let me think\n</think>\n\nThe answer is 4.", "let me think\n", "\n\nThe answer is 4.", []),
    (True, "hmm</think>ok</think> fine", "hmm", "ok fine", []),
    (False, "plain answer <b>bold</b>", "", "plain answer <b>bold</b>", []),
    (True, "plan\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>",
     "plan\n", "", ["get_weather"]),  # whitespace-only content before a call is dropped (vLLM's parser engine)
    (True, "x<tool_call>\n<function=a>\n<parameter=p>\n1\n</parameter>\n</function>\n</tool_call>", "x", "", ["a"]),
    (False, "Sure.\n<function=b>\n<parameter=q>\nhi\n</parameter>\n</function>", "", "Sure.\n", ["b"]),
]
q = "plan: quote <tool_call>\n</think>\n\nThe template wraps calls in <tool_call>\n<function=f> tags; see </tool_call>."
cases_off = [  # no tools offered: the tags are text (vLLM parses tool calls only when the request has tools)
    (True, q, "plan: quote <tool_call>\n", "\n\nThe template wraps calls in <tool_call>\n<function=f> tags; see </tool_call>.", []),
    (False, "Sure.\n<function=b>\n<parameter=q>\nhi\n</parameter>\n</function>", "",
     "Sure.\n<function=b>\n<parameter=q>\nhi\n</parameter>\n</function>", []),
]
pf = 0
for tools, th, text, er, ec, et in [(True, *c) for c in cases] + [(False, *c) for c in cases_off]:
    good = True
    for split in range(1, len(text) + 1):
        p = Parser(th, tools)
        evs = []
        for i in range(0, len(text), split):
            evs += p.feed(text[i:i + split])
        evs += p.feed("", final=True)
        r = "".join(v for k, v in evs if k == "reasoning")
        cn = "".join(v for k, v in evs if k == "content")
        tn = [v["function"]["name"] for k, v in evs if k == "tool"]
        good &= (r, cn, tn) == (er, ec, et)
    pf += good
args = json.loads([v for k, v in Parser(False).feed(cases[3][1].split("</think>")[1], final=True) if k == "tool"][0]["function"]["arguments"])
good_args = args == {"city": "Seoul"}
nc = len(cases) + len(cases_off)
bad += pf != nc or not good_args
(qlog.ok if pf == nc and good_args else qlog.warn)(
    "chat parser", f"{pf}/{nc} cases ({len(cases_off)} without tools in the request) · same result at every split "
                   f"size · arguments {'JSON equal' if good_args else args}")
sys.exit(1 if bad else 0)
