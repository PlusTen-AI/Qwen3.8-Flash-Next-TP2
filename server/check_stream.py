"""Streamed tool calls against vLLM 0.32 (no GPU, no model; plain Python): the step texts of every case in
tools/vllm_render/stream_vllm.json (vLLM's own streaming path in the reference image: vllm_stream.py) go through our
Parser(stream=True) one step at a time, and the deltas are compared with vLLM's:
  per step   each call's name and argument pieces (what goes out at which step)
  joined     reasoning, content, each call's arguments, finish_reason (app.py's rule)
  final      the ("tool", call) event's arguments = the joined pieces (the non-streamed reply)
Known differences (ours kept on purpose, listed, not failed): no tools in the request (ours: the markup stays text;
vLLM drops it), a <tool_call> block in JSON form (ours: a call; vLLM drops it), parallel_tool_calls false (ours: the
reply ends after the first call; vLLM sends empty chunks for the second).
Run: python3 server/check_stream.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "tools")]
from chat import Parser  # noqa: E402

G = json.load(open(os.path.join(ROOT, "tools", "vllm_render", "stream_vllm.json")))
CASES = {c["name"]: c for c in json.load(open(os.path.join(ROOT, "tools", "vllm_render", "stream_cases.json")))}
KNOWN = {"no_tools_in_request": "no tools: ours keeps the markup as text",
         "tool_call_without_function": "JSON-form block: ours makes a call",
         "two_calls_parallel_false": "parallel_tool_calls false: ours ends after the first call"}


def theirs(res):
    """per step: {"r": reasoning, "c": content, "calls": {index: [name or None, args]}} and the joined summary"""
    per = {}
    for ch in res["chunks"]:
        d, st = ch["delta"], per.setdefault(ch["step"], {"r": "", "c": "", "calls": {}})
        st["r"] += d.get("reasoning") or ""
        st["c"] += d.get("content") or ""
        for tc in d.get("tool_calls") or []:
            x = st["calls"].setdefault(tc["index"], ["", ""])
            f = tc.get("function") or {}
            x[0] += f.get("name") or ""
            x[1] += f.get("arguments") or ""
    return per


def ours(case, res):
    tools = case.get("tools") or []
    tc = case.get("tool_choice")
    tcn = "named" if isinstance(tc, dict) else (tc or "auto")
    schemas = {t["function"]["name"]: (t["function"].get("parameters") or {}).get("properties") or {} for t in tools}
    par = Parser(case.get("thinking", True), bool(tools), tcn == "none", schemas, stream=True)
    per, calls, finals, ntool = {}, {}, [], 0
    steps = res["steps"]
    end = "length" if res["end"] == "length" else "stop"
    for k, txt in enumerate(steps):
        st = per.setdefault(k, {"r": "", "c": "", "calls": {}})
        for kind, v in par.feed(txt, final=end if k == len(steps) - 1 else False):
            if kind == "reasoning":
                st["r"] += v
            elif kind == "content":
                st["c"] += v
            elif kind == "tool_delta":
                x = st["calls"].setdefault(v["index"], ["", ""])
                f = v["function"]
                x[0] += f.get("name") or ""
                x[1] += f.get("arguments") or ""
                if "id" in v:
                    ntool += 1
                    calls[v["index"]] = [v["id"], ""]
                calls[v["index"]][1] += f.get("arguments") or ""
            elif kind == "tool":
                finals.append(v)
    fin = "tool_calls" if ntool and end == "stop" and tcn != "named" else end
    return per, calls, finals, fin


def joined(per):
    r = "".join(p["r"] for _, p in sorted(per.items()))
    c = "".join(p["c"] for _, p in sorted(per.items()))
    calls = {}
    for _, p in sorted(per.items()):
        for i, (n, a) in p["calls"].items():
            x = calls.setdefault(int(i), ["", ""])
            x[0] += n
            x[1] += a
    return r, c, calls


bad = known = 0
for res in G["results"]:
    case = CASES[res["name"]]
    tag = f"{res['name']} ×{res['chunk']}"
    t_per = theirs(res)
    o_per, o_calls, o_finals, o_fin = ours(case, res)
    tr, tcn_, tcalls = joined(t_per)
    orr, oc, ocalls = joined(o_per)
    probs = []
    if orr != tr:
        probs.append(f"reasoning {orr[-40:]!r} vs {tr[-40:]!r}")
    if oc != tcn_:
        probs.append(f"content {oc!r} vs {tcn_!r}")
    if {k: v for k, v in ocalls.items()} != tcalls:
        probs.append(f"calls {ocalls} vs {tcalls}")
    t_fin = res["summary"]["finish_reason"]
    if o_fin != t_fin:
        probs.append(f"finish {o_fin} vs {t_fin}")
    # per step: tool names and argument pieces at the same steps
    steps_t = {k: {int(i): v for i, v in p["calls"].items()} for k, p in t_per.items() if p["calls"]}
    steps_o = {k: p["calls"] for k, p in o_per.items() if p["calls"]}
    if not probs and steps_o != steps_t:
        diff = sorted(set(steps_o) ^ set(steps_t)) or [k for k in steps_t if steps_t[k] != steps_o.get(k)]
        probs.append(f"tool pieces at other steps: first {diff[:3]} — ours {[steps_o.get(k) for k in diff[:2]]} vs "
                     f"vLLM {[steps_t.get(k) for k in diff[:2]]}")
    # the whole call (non-streamed reply) = the joined pieces
    for f in o_finals:
        i = f["index"]
        if o_calls.get(i, ["", None])[1] != f["function"]["arguments"] or o_calls[i][0] != f["id"]:
            probs.append(f"call {i}: whole {f['function']['arguments']!r} ≠ joined {o_calls.get(i)}")
    if probs and res["name"] in KNOWN:
        known += 1
        print(f"known {tag}: {KNOWN[res['name']]} ({probs[0][:120]})")
    elif probs:
        bad += 1
        print(f"FAIL  {tag}: " + " · ".join(probs)[:600])
    else:
        print(f"ok    {tag}")
n = len(G["results"])
print(f"{n - bad - known}/{n} equal, {known} known differences, {bad} failed")
sys.exit(1 if bad else 0)
