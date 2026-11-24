vllm_render — compare our prompt rendering with vLLM 0.32's own render path (the reference deployment's vLLM image).
The reference recipe and its image were removed on 2026-11-14; vLLM's side was saved first, so the comparisons run offline:
  check_render.py   render_cases.json (27 cases) through our chat.py vs render_vllm.json (vLLM's render, saved by
                    vllm_side.py): 24 equal, 2 refused both ways, 1 known difference (arguments sent as an object)
  server/check_stream.py   streamed tool calls vs stream_vllm.json (saved by vllm_stream.py)
  chat_template.local.jinja   the template the reference served (a copy from the model folder; server/check_chat.py
                    uses it when the model folder no longer has it)
vllm_side.py / vllm_stream.py need the reference image (kept as the record of how the saved files were made).
vllm_side.py runs inside the reference image (CPU, no network): request bodies (JSON list on stdin) → vLLM's
server-path render (ChatCompletionRequest → parse_chat_messages → safe_apply_chat_template) → text and token ids.
our_side.py runs in our image: the same bodies through server/chat.py (normalize, thinking_mode, render, encode).
Example (on C; cases.json = a JSON list of chat request bodies):
  M=<models>
  cat cases.json | docker run -i --rm --network none --label project=qwen38-engine --user 1000:1000 -e HOME=/tmp \
    -v $M:/models:ro -v $PWD/vllm_side.py:/tmp/v.py:ro --entrypoint python3 <reference vLLM image> /tmp/v.py > vllm.json
  cat cases.json | docker run -i --rm --network none --label project=qwen38-engine --user 1000:1000 -e HOME=/tmp \
    -v <work>:/work -v $M:/models:ro qwen38-plus:<tag> python3 /work/src/tools/vllm_render/our_side.py > ours.json
2026-11-13: 23/24 cases identical (the 24th: arguments sent as an object — vLLM refuses it, we accept).
our_side.py renders without plain_specials (vLLM has none).

Streamed tool calls (vllm_stream.py, subagent 2026-11-13): vLLM 0.32's real chat-completion streaming path in the
reference image (OutputProcessor + chat_completion_stream_generator + the qwen3_xml / qwen3 parser engine) on the cases
in stream_cases.json, 1 / 3 / 5 / 13 tokens per step → stream_vllm.json (every SSE chunk with the step it came from).
  docker run -i --rm --network none --label project=qwen38-engine --user 1000:1000 -e HOME=/tmp -v $M:/models:ro \
    -v $PWD:/w --entrypoint python3 <reference vLLM image> /w/vllm_stream.py < stream_cases.json > stream_vllm.json
server/check_stream.py replays the same step texts through our Parser(stream=True): 88/100 equal per step (names and
argument pieces at the same steps), 12 known differences (no tools in the request, a JSON-form block,
parallel_tool_calls false — ours kept on purpose).
