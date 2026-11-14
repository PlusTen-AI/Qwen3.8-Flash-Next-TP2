"""Golden for streamed tool calls: vLLM 0.32's real chat-completion streaming path in the reference image (CPU, no
network). Runs as the reference server is configured (recipe.yaml: --enable-auto-tool-choice
--tool-call-parser qwen3_xml --reasoning-parser qwen3, chat-template.local): stdin = JSON list of cases, stdout = JSON.
A case: {"name", "tools", "tool_choice"?, "thinking": bool, "text": raw model output after the prompt,
"chunk": int or list (tokens per engine step, default 1), "end": "eos" (default: <|im_end|> appended, STOP) or
"length" (max_tokens cut, no eos), "extra"?: more request fields}.
Path: text → checkpoint tokenizer → EngineCoreOutput per step → v1 OutputProcessor (real incremental detokenizer,
RequestOutputCollector, DELTA) → OpenAIServingChat.chat_completion_stream_generator (real; it builds the
DelegatingParser per request and calls parser.parse_delta per step) → SSE chunks. Request as the server sees it:
ChatCompletionRequest + chat_template_kwargs + the parser's adjust_request (as renderers/online_renderer.py does).
Out per case/chunk: every SSE chunk the server yields (role chunk first, no [DONE]), each with the engine step it
came from; tool-call ids replaced by <id0>, <id1>… (format in meta); a summary joined from the deltas."""
import asyncio, json, logging, re, sys, time, types, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import vllm
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.parser import ParserManager
from vllm.renderers.chat_utils import load_chat_template, parse_chat_messages
from vllm.renderers.hf import resolve_chat_template_content_format, safe_apply_chat_template
from vllm.renderers.params import merge_kwargs
from vllm.tokenizers import get_tokenizer
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector

MODEL = "/models/Qwen3.8-Flash-Next-FP4-Local"
SERVED = "Qwen/Qwen3.8-Flash-Next"
TEMPLATE = load_chat_template(MODEL + "/chat_template.local.jinja")
TOK = get_tokenizer(MODEL)
CFG = json.load(open(MODEL + "/config.json"))
MC_RENDER = types.SimpleNamespace(model=MODEL, tokenizer=MODEL, revision=None, tokenizer_revision=None, code_revision=None,
    trust_remote_code=False, enable_prompt_embeds=False, multimodal_config=None, hf_config=None,
    allowed_local_media_path="", allowed_media_domains=None)
# only get_tool_call_id_type reads it (model_type → "random" ids unless Kimi)
MC_SERVE = types.SimpleNamespace(model=MODEL, hf_overrides=None,
    hf_config=types.SimpleNamespace(model_type=CFG.get("model_type")),
    hf_text_config=types.SimpleNamespace(model_type=(CFG.get("text_config") or {}).get("model_type")))
PARSER_CLS = ParserManager.get_parser(tool_parser_name="qwen3_xml", reasoning_parser_name="qwen3", enable_auto_tools=True,
    model_name=MODEL, is_harmony=False, tool_strict_level="auto", tokenizer=TOK)
SRV = object.__new__(OpenAIServingChat)  # only the attributes the stream generator reads
SRV.__dict__.update(parser_cls=PARSER_CLS, model_config=MC_SERVE, response_role="assistant", chat_template=TEMPLATE,
    chat_template_content_format="auto", default_chat_template_kwargs={}, enable_auto_tools=True,
    enable_force_include_usage=False, _include_reasoning_tokens_details=True, enable_log_outputs=False,
    enable_log_deltas=True, request_logger=None, system_fingerprint=None)
EOS = TOK.eos_token_id
ID_RE = re.compile(r"chatcmpl-tool-[0-9a-f]+")


def prompt_ids(req):
    tool_dicts = None if req.tools is None else [t.model_dump() for t in req.tools]
    params = req.build_chat_params(TEMPLATE, "auto").with_defaults(merge_kwargs({}, dict(tools=tool_dicts, tokenize=False)))
    kw = params.get_apply_chat_template_kwargs()
    fmt = resolve_chat_template_content_format(chat_template=params.chat_template, tools=params.chat_template_kwargs.get("tools"),
        given_format=params.chat_template_content_format, tokenizer=TOK, model_config=MC_RENDER)
    conv, _, _ = parse_chat_messages(list(req.messages), MC_RENDER, content_format=fmt)
    text = safe_apply_chat_template(MC_RENDER, TOK, conv, **kw)
    return text, TOK.encode(text, add_special_tokens=False)


async def run(case, n):
    body = {"model": SERVED, "stream": True, "max_tokens": 100000,
            "messages": [{"role": "user", "content": case.get("user", "Go.")}],
            "chat_template_kwargs": {"enable_thinking": case.get("thinking", True)}}
    for k in ("tools", "tool_choice"):
        if k in case: body[k] = case[k]
    body.update(case.get("extra") or {})
    req = ChatCompletionRequest(**body)
    ptext, pids = prompt_ids(req)
    ctk = SRV._effective_chat_template_kwargs(req)
    if PARSER_CLS.always_adjust_request or PARSER_CLS.reasoning_parser_cls is not None or req.tool_choice != "none":
        req = PARSER_CLS(TOK, req.tools, model_config=MC_SERVE, chat_template_kwargs=ctk).adjust_request(request=req)
    sp = req.to_sampling_params(100000, {})
    end = case.get("end", "eos")
    ids = TOK.encode(case["text"], add_special_tokens=False) + ([EOS] if end == "eos" else [])
    steps = [ids[i:i + n] for i in range(0, len(ids), n)]
    ecr = EngineCoreRequest(request_id="r0", prompt_token_ids=pids, mm_features=None, sampling_params=sp, pooling_params=None,
        arrival_time=time.time(), lora_request=None, cache_salt=None, data_parallel_rank=None, external_req_id="chatcmpl-x")
    cur = {"step": 0}
    texts = []

    async def engine():  # AsyncLLM.generate as it drains the collector (v1/engine/async_llm.py)
        op = OutputProcessor(TOK, log_stats=False, stream_interval=1)
        q = RequestOutputCollector(sp.output_kind, "r0")
        op.add_request(ecr, ptext, None, 0, q)
        for k, s in enumerate(steps):
            fr = None if k < len(steps) - 1 else (FinishReason.STOP if end == "eos" else FinishReason.LENGTH)
            op.process_outputs([EngineCoreOutput(request_id="r0", new_token_ids=list(s), finish_reason=fr)])
            out = q.get_nowait()
            if out is None: continue
            cur["step"] = k
            texts.append(out.outputs[0].text)
            yield out
            if out.finished: break

    chunks, err = [], None
    async for sse in SRV.chat_completion_stream_generator(req, engine(), "chatcmpl-x", SERVED, [], TOK,
            RequestResponseMetadata(request_id="chatcmpl-x"), chat_template_kwargs=ctk, mm_token_counts=None):
        data = sse[len("data: "):].strip()
        if data == "[DONE]": continue
        d = json.loads(data)
        if "error" in d or not d.get("choices"):
            err = d; continue
        c = d["choices"][0]
        chunks.append({"step": cur["step"], "delta": c["delta"], "finish_reason": c.get("finish_reason"),
                       **({"stop_reason": c["stop_reason"]} if c.get("finish_reason") else {})})
    return {"ids": ids, "steps": texts, "chunks": chunks, "error": err}


def normalize(res):
    seen, sample = {}, None
    for c in res["chunks"]:
        for tc in c["delta"].get("tool_calls") or []:
            if tc.get("id"):
                sample = sample or tc["id"]
                tc["id"] = seen.setdefault(tc["id"], f"<id{len(seen)}>")
    return sample


def summary(res):
    reasoning, content, calls = "", "", {}
    for c in res["chunks"]:
        d = c["delta"]
        reasoning += d.get("reasoning") or ""
        content += d.get("content") or ""
        for tc in d.get("tool_calls") or []:
            s = calls.setdefault(tc["index"], {"index": tc["index"], "id": None, "type": None, "name": "", "arguments": ""})
            s["id"] = tc.get("id") or s["id"]; s["type"] = tc.get("type") or s["type"]
            f = tc.get("function") or {}
            s["name"] += f.get("name") or ""; s["arguments"] += f.get("arguments") or ""
    for s in calls.values():
        try: json.loads(s["arguments"]); s["json_ok"] = True
        except Exception: s["json_ok"] = False
    fin = [c["finish_reason"] for c in res["chunks"] if c["finish_reason"]]
    return {"reasoning": reasoning or None, "content": content or None, "tool_calls": list(calls.values()),
            "finish_reason": fin[-1] if fin else None}


out, id_sample = [], None
for case in json.load(sys.stdin):
    ch = case.get("chunk", 1)
    for n in (ch if isinstance(ch, list) else [ch]):
        r = {"name": case["name"], "chunk": n, "end": case.get("end", "eos")}
        try:
            res = asyncio.run(run(case, n))
            id_sample = normalize(res) or id_sample
            r.update(n_tokens=len(res["ids"]), steps=res["steps"], chunks=res["chunks"], summary=summary(res))
            if res["error"]: r["error"] = res["error"]
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:300]
        out.append(r)
meta = {"vllm": vllm.__version__, "parser": f"{PARSER_CLS.reasoning_parser_cls.__name__} + {PARSER_CLS.tool_parser_cls.__name__}",
        "eos": [EOS, TOK.decode([EOS])], "id_sample": id_sample and ID_RE.sub(lambda m: "chatcmpl-tool-" + "h" * (len(m.group()) - 14), id_sample)}
print(json.dumps({"meta": meta, "results": out}, ensure_ascii=False, indent=1))
