import json, sys
sys.path[:0] = ["/work/src/server", "/work/src/tools"]
from chat import Chat, normalize_messages, normalize_tools, thinking_mode
C = Chat("/models/Qwen3.8-Flash-Next-FP4-Local")
out = []
for b in json.load(sys.stdin):
    b = {k: v for k, v in b.items() if k != "name"}  # the case's label
    try:
        th, lv = thinking_mode(b)
        kw = dict(b.get("chat_template_kwargs") or {}, enable_thinking=th)
        if th: kw["reasoning_effort"] = lv
        t = C.render(normalize_messages(b["messages"]), tools=normalize_tools(b.get("tools")), reasoning_effort=lv or "none", kwargs=kw)
        out.append({"text": t, "ids": C.encode(t)})
    except Exception as e:
        out.append({"error": f"{type(e).__name__}: {e}"[:200]})
print(json.dumps(out))
