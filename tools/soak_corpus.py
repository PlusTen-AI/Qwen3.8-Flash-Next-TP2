"""(development, this site) Conversations for tools/soak.py from a Hermes state database — read only.

  python3 tools/soak_corpus.py <state.db> <out.jsonl>

One JSON line per stored session (its active transcript, in order): {id, source, parent, started, tools, messages}.
messages are OpenAI chat messages as Hermes sends them: the session's system prompt first; assistant turns with
content, reasoning_content and tool_calls ({id, type, function{name, arguments}}); tool results with tool_call_id;
user turns with the text Hermes sent (api_content when it differs from the stored text). tools: the session's tool
definitions (Hermes stores them by hash next to the system prompts); a session without them (older subagents) gets the
newest subagent tool set, so its recorded tool calls render against real definitions. The output holds private
conversation text: keep it outside the repo and delete it after the run."""
import json
import sqlite3
import sys

src, out = sys.argv[1], sys.argv[2]
db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
blobs = dict(db.execute("select hash, prompt from system_prompts"))


def tools_of(h):
    try:
        return json.loads(blobs[h])["tools"] if h in blobs else None
    except (ValueError, KeyError):
        return None


rows = db.execute("select id, source, parent_session_id, started_at, system_prompt_hash, tool_names from sessions "
                  "order by started_at").fetchall()
sub_tools = None
for sid, source, parent, started, sph, tn in rows:
    if source == "subagent" and tools_of(tn):
        sub_tools = tools_of(tn)
n_out = 0
with open(out, "w") as f:
    for sid, source, parent, started, sph, tn in rows:
        sysp = blobs.get(sph)
        tools = tools_of(tn) or (sub_tools if source == "subagent" else None)
        msgs = [{"role": "system", "content": sysp}] if sysp else []
        for role, content, api, calls, rc, r, tcid in db.execute(
                "select role, content, api_content, tool_calls, reasoning_content, reasoning, tool_call_id from messages "
                "where session_id=? and active=1 order by id", (sid,)):
            if role == "assistant":
                m = {"role": "assistant", "content": api if api is not None else (content or "")}
                if rc or r:
                    m["reasoning_content"] = rc or r
                if calls:
                    m["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["function"]["name"],
                                                     "arguments": c["function"]["arguments"]}}
                                       for c in json.loads(calls)]
                    m["content"] = m["content"] or None
            elif role == "tool":
                m = {"role": "tool", "content": content or "", "tool_call_id": tcid}
            elif role == "user":
                m = {"role": "user", "content": api if api is not None else (content or "")}
            else:
                continue
            msgs.append(m)
        if not any(m["role"] == "assistant" for m in msgs):
            continue
        f.write(json.dumps({"id": sid, "source": source, "parent": parent, "started": started, "tools": tools,
                            "messages": msgs}, ensure_ascii=False) + "\n")
        n_out += 1
print(f"{n_out} sessions → {out}")
