"""Our prompt rendering against vLLM 0.32's (the reference deployment), offline: render_cases.json through server/chat.py
(as the server does: normalize, thinking_mode, render, encode — without plain_specials, which vLLM does not have)
compared with render_vllm.json (vLLM's own render path in the reference image, saved by vllm_side.py before that image
was removed). Equal text and token ids, or both refused. Runs in our image:
  docker run --rm --network none … -v <work>:/work -v <models>:/models:ro qwen38-plus:<tag> \
      python3 /work/src/tools/vllm_render/check_render.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "server"), os.path.join(ROOT, "tools")]
from chat import Chat, normalize_messages, normalize_tools, thinking_mode  # noqa: E402

C = Chat(sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.8-Flash-Next-FP4-Local")
cases = json.load(open(os.path.join(HERE, "render_cases.json")))
ref = json.load(open(os.path.join(HERE, "render_vllm.json")))["results"]
KNOWN = {"args object (vLLM refuses)": "arguments sent as an object: vLLM refuses the request, we accept it"}
same = refused = known = bad = 0
for case, r in zip(cases, ref):
    b = {k: v for k, v in case.items() if k != "name"}
    try:
        th, lv = thinking_mode(b)
        kw = dict(b.get("chat_template_kwargs") or {}, enable_thinking=th)
        if th:
            kw["reasoning_effort"] = lv
        t = C.render(normalize_messages(b["messages"]), tools=normalize_tools(b.get("tools")),
                     reasoning_effort=lv or "none", kwargs=kw)
        ours = {"text": t, "ids": C.encode(t)}
    except Exception as e:  # noqa: BLE001
        ours = {"error": f"{type(e).__name__}: {e}"[:200]}
    if "error" in r and "error" in ours:
        refused += 1
        print(f"ok    {case['name']}: refused both ways ({ours['error'][:70]})")
    elif "error" not in r and "error" not in ours and r["ids"] == ours["ids"] and r["text"] == ours["text"]:
        same += 1
        print(f"ok    {case['name']}: {len(r['ids']):,} tokens equal")
    elif case["name"] in KNOWN:
        known += 1
        print(f"known {case['name']}: {KNOWN[case['name']]}")
    else:
        bad += 1
        print(f"FAIL  {case['name']}: vLLM {r.get('error') or str(len(r['ids'])) + ' tokens'} · ours "
              f"{ours.get('error') or str(len(ours['ids'])) + ' tokens'}")
print(f"{same} equal · {refused} refused both ways · {known} known · {bad} differ — of {len(cases)}")
sys.exit(1 if bad else 0)
