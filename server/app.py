"""OpenAI-compatible HTTP front (rank 0), shaped like vLLM's: /v1/models, /v1/chat/completions, /v1/completions,
/health. Sampling: request value → generation_config default → OpenAI default (vLLM's priority). Streaming is SSE with
reasoning_content / content / tool_calls deltas; `stream_options.include_usage` adds a usage chunk; usage carries
prompt_tokens_details.cached_tokens (prefix reuse). A dropped client cancels its request. Parameters the engine does
not implement (penalties, logprobs, n > 1, guided decoding) are refused with 400 rather than silently ignored."""
import asyncio
import hmac
import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

import settings
from chat import Detok, Parser, TemplateError, normalize_messages, normalize_tools, quote_tools, thinking_mode
from scheduler import Req

class EngineGone(Exception):
    """The engine stopped under a request (the other rank is gone)."""


class Front:
    """ASGI entry. Before the engine is ready and while stopping it answers by itself: /health → 503 {"status":
    "loading" | "stopping" | "stopped" | "failed"}, any other path → 503. When ready everything goes to the app; requests
    already inside it (running streams) go on while stopping."""

    def __init__(self, api_key=""):
        self.app, self.state = None, "loading"
        self.key = ("Bearer " + api_key).encode() if api_key else None

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.key and scope["path"].startswith("/v1"):
            auth = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(auth, self.key):
                await self._answer(receive, send, 401, {"error": {"message": "invalid or missing API key", "type":
                                                                  "authentication_error", "code": 401}})
                return
        if scope["type"] == "http" and (self.state != "ready" or self.app is None):
            st = self.state
            await self._answer(receive, send, 503, {"status": st} if scope["path"] == "/health" else
                               {"error": {"message": f"server is {st}", "type": "ServiceUnavailableError", "param": None,
                                          "code": 503}})
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _answer(receive, send, status, obj):
        while True:  # read the request body first: closing with unread data would reset the connection
            m = await receive()
            if m["type"] != "http.request" or not m.get("more_body"):
                break
        body = json.dumps(obj).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


class BadRequest(ValueError):
    """A 400 with vLLM's wording; param names the field (vLLM appends " (parameter=…, value=…)" for these)."""

    def __init__(self, msg, param=None, value=None):
        super().__init__(msg + (f" (parameter={param}, value={value})" if param is not None else ""))
        self.param = param


# accepted only at their neutral value: (neutral, message) — vLLM 0.32's text where it has one
UNSUPPORTED = {
    "logit_bias": (None, "logit_bias is unavailable on this server"),
    "logprobs": (False, "logprobs are unavailable on this server"),
    "top_logprobs": (None, "top_logprobs are unavailable on this server"),
    "prompt_logprobs": (None, "prompt_logprobs are unavailable on this server"),
    "n": (1, "n > 1 is unavailable on this server"),
    "best_of": (None, "best_of is unavailable on this server"),
    "guided_json": (None, "guided decoding is unavailable on this server"),
    # Hermes drops response_format and retries when the text has "response_format" with "unavailable" / "json_schema"
    "response_format": (None, "response_format json_object / json_schema is unavailable on this server (no guided "
                              "decoding)"),
}
ERR_TYPE = {400: "BadRequestError", 401: "AuthenticationError", 404: "NotFoundError", 499: "BadRequestError",
            500: "InternalServerError", 503: "ServiceUnavailableError"}


def build(chat, sched, served, ctx, depth, think_cfg=None, plain_specials=True, media=None):
    think_cfg = think_cfg or {}
    quote = chat.quote if plain_specials else None
    enc = chat.encode_shielded if plain_specials else chat.encode
    app = FastAPI()

    def err(code, msg, param=None):
        """vLLM's error body: {"error": {message, type, param, code}}."""
        return JSONResponse({"error": {"message": msg, "type": ERR_TYPE.get(code, "BadRequestError"), "param": param,
                                       "code": code}}, status_code=code)

    def sampling(b):
        for k, (v, msg) in UNSUPPORTED.items():
            if b.get(k) is not None and b[k] != v and not (k == "response_format" and b[k] == {"type": "text"}):
                raise BadRequest(msg, k, b[k])
        d = chat.defaults
        temp = float(b["temperature"]) if b.get("temperature") is not None else float(d["temperature"])
        top_p = float(b["top_p"]) if b.get("top_p") is not None else float(d["top_p"])
        top_k = int(b["top_k"]) if b.get("top_k") is not None else int(d["top_k"])
        seed = int(b["seed"]) if b.get("seed") is not None else uuid.uuid4().int & ((1 << 63) - 1)
        if not 0 <= temp <= 2:
            raise BadRequest(f"temperature must be in [0, 2], got {temp}.", "temperature", temp)
        if not 0 < top_p <= 1:
            raise BadRequest(f"top_p must be in (0, 1], got {top_p}.", "top_p", top_p)
        if top_k < -1:
            raise BadRequest(f"top_k must be 0 (disable), or at least 1, got {top_k}.", "top_k", top_k)
        min_p = float(b["min_p"]) if b.get("min_p") is not None else float(d.get("min_p", 0.0))
        rep = float(b["repetition_penalty"]) if b.get("repetition_penalty") is not None else 1.0
        freq = float(b["frequency_penalty"]) if b.get("frequency_penalty") is not None else 0.0
        pres = float(b["presence_penalty"]) if b.get("presence_penalty") is not None else 0.0
        if not 0 <= min_p <= 1:
            raise BadRequest(f"min_p must be in [0, 1], got {min_p}.", "min_p", min_p)
        if not rep > 0:
            raise BadRequest(f"repetition_penalty must be greater than zero, got {rep}.", "repetition_penalty", rep)
        for k, v in (("presence_penalty", pres), ("frequency_penalty", freq)):
            if not -2 <= v <= 2:
                raise BadRequest(f"{k} must be in [-2, 2], got {v}.", k, v)
        if temp < 1e-5:  # greedy (vLLM: below 1e-5; truncation off, penalties still apply)
            temp, min_p = 0.0, 0.0
        elif temp < 0.01:  # vLLM raises tiny temperatures to 0.01
            temp = 0.01
        return temp, (top_k if top_k > 0 else 0), top_p, seed, min_p, (rep, freq, pres)

    def limits(b, n_prompt):
        """vLLM's rules: a prompt with no room left is an error (the window and the token count in the text — clients
        such as Hermes read them to adapt); max_tokens beyond the room is cut to it; omitted = all the room. The room
        keeps 2·depth positions for the last speculative round."""
        room = ctx - n_prompt - 2 * depth  # the last round writes cache positions up to prompt + max_tokens + 2·depth − 3
        if room < 1:
            raise BadRequest(f"This model's maximum context length is {ctx} tokens. However, your prompt contains "
                             f"{n_prompt} input tokens, which exceeds this limit even without any output tokens. "
                             "Please reduce the length of the input prompt.", "input_tokens", n_prompt)
        mt = b.get("max_completion_tokens") if b.get("max_completion_tokens") is not None else b.get("max_tokens")
        if mt is None:
            return room
        mt = int(mt)
        if mt < 1:
            raise BadRequest(f"max_tokens must be at least 1, got {mt}.", "max_tokens", mt)
        if mt > ctx:
            raise BadRequest(f"max_tokens={mt} cannot be greater than max_model_len={ctx}. Please request fewer "
                             "output tokens.", "max_tokens", mt)
        return min(mt, room)

    def tool_choice(b, tl):
        """→ ("none" | "auto" | "required" | "named", function name or None), with vLLM's defaults and errors."""
        tc = b.get("tool_choice")
        if tc is None:
            return ("auto" if tl else "none"), None
        if isinstance(tc, dict):
            name = (tc.get("function") or {}).get("name")
            if not tl:
                raise BadRequest("When using `tool_choice`, `tools` must be set.", "tool_choice")
            if name not in {t["function"]["name"] for t in tl}:
                raise BadRequest("The tool specified in `tool_choice` does not match any of the specified `tools`",
                                 "tool_choice")
            return "named", name
        if tc not in ("none", "auto", "required"):
            raise BadRequest(f"Invalid value for `tool_choice`: {tc!r}", "tool_choice")
        if tc == "required" and not tl:
            raise BadRequest("When using `tool_choice`, `tools` must be set.", "tool_choice")
        return tc, None

    def submit(prompt, b, label, thinking=False, budget=None, **attrs):
        temp, top_k, top_p, seed, min_p, pen = sampling(b)
        mt = limits(b, len(prompt))
        loop, q = asyncio.get_running_loop(), asyncio.Queue()
        r = Req(prompt, mt, temp, top_k, top_p, seed, lambda t, f: loop.call_soon_threadsafe(q.put_nowait, (t, f)), label)
        r.min_p, r.pen = min_p, pen
        r.thinking, r.budget = thinking, budget  # the scheduler's thinking budget / loop guard (it sees token ids)
        for k, v in attrs.items():  # everything the scheduler reads is set before it can see the request
            setattr(r, k, v)
        sched.submit(r)
        return r, q

    async def tokens(r, q, req):
        """(new token ids, finish) until finished. The request is cancelled when the client goes away: checked about
        once a second even while tokens flow, and whenever the response is torn down early (generator closed)."""
        done, last = False, time.time()
        try:
            while True:
                try:
                    t, f = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    t, f = None, None
                if time.time() - last > 1.0 or t is None:
                    last = time.time()
                    if await req.is_disconnected():
                        return
                if t is None:
                    continue
                if f == "error":
                    raise EngineGone(r.error or sched.error or "the engine stopped")
                yield t, f
                if f:
                    done = True
                    return
        finally:
            if not done:
                r.cancelled = True

    tlog = open("/work/results/serve/timing.log", "a")

    def timing(r):
        """one line per request at its first streamed text: ms between the marks"""
        th, d = getattr(r, "th", None), lambda a, b: f"{1e3 * (b - a):.0f}" if a and b else "-"
        if not th:
            return
        tlog.write(f"#{r.id} prompt {len(r.prompt):,} (reused {r.cached:,}) · body {d(th[0], th[1])} · parse {d(th[1], th[2])} · "
                   f"render {d(th[2], th[3])} · tokenize {d(th[3], th[4])} · queued→placed {d(r.t0, r.t_place)} · "
                   f"placed→prefill done {d(r.t_place, r.t_pf)} · prefill done→first token {d(r.t_pf, r.t_first)} · "
                   f"first token→SSE {d(r.t_first, r.t_sse)} · total {d(th[0], r.t_sse)} ms\n")
        tlog.flush()

    def usage(r, n_out):
        np = getattr(r, "n_prompt", len(r.prompt))  # without tokens forced onto the prompt's end (tool_choice)
        n_out = max(n_out, r.n_out)  # the scheduler's count includes the EOS token (as vLLM's)
        return {"prompt_tokens": np, "completion_tokens": n_out, "total_tokens": np + n_out,
                "prompt_tokens_details": {"cached_tokens": r.cached},
                "completion_tokens_details": {"reasoning_tokens": min(r.n_think, n_out)}}

    @app.get("/health")
    async def health():
        return {"status": "ok", "busy": sched.busy()}

    @app.get("/metrics")
    async def metrics():
        """Prometheus text with vLLM's names for what the bench reads: requests running / waiting (disconnect check)
        and the speculative-decoding counters (tokens per round = 1 + accepted / drafts)."""
        run, wait = sched.counts()
        st, lab = sched.stats, f'{{model_name="{served}"}}'
        rows = [("gauge", "vllm:num_requests_running", run), ("gauge", "vllm:num_requests_waiting", wait),
                ("counter", "vllm:spec_decode_num_drafts_total", st["drafts"]),
                ("counter", "vllm:spec_decode_num_draft_tokens_total", st["draft_tokens"]),
                ("counter", "vllm:spec_decode_num_accepted_tokens_total", st["accepted"]),
                ("counter", "vllm:prompt_tokens_total", st["prompt"]),
                ("counter", "vllm:prompt_tokens_cached_total", st["cached"]),
                ("counter", "vllm:generation_tokens_total", st["out"]),
                ("counter", "vllm:request_success_total", st["requests"])]
        txt = "".join(f"# TYPE {n.removesuffix('_total')} {t}\n{n}{lab} {v}\n" for t, n, v in rows)
        # rounds by (sequences, depth): count, tokens, ms inside spec_round, ms between round starts (and its count)
        for (B, k), v in sorted(dict(sched.rstats).items()):
            for name, x in zip(("rounds", "tokens", "engine_ms", "wall_ms", "wall_rounds"), v):
                txt += f'q38_round_{name}_total{{b="{B}",k="{k}"}} {x}\n'
        # prefix reuse: each slot's history and the shared pool checkpoints it holds (tokens, seconds since last use)
        slots, free, evicted = sched.ck_status()
        txt += f"q38_checkpoints_free {free}\nq38_checkpoints_evicted_total {evicted}\n"
        for i, busy, n, ck in slots:
            txt += f'q38_slot_tokens{{slot="{i}",busy="{int(busy)}"}} {n}\n'
            for c, m, age in ck:
                txt += (f'q38_checkpoint_tokens{{slot="{i}",buffer="{c}"}} {m}\n'
                        f'q38_checkpoint_idle_seconds{{slot="{i}",buffer="{c}"}} {age:.1f}\n')
        return PlainTextResponse(txt)

    def card():
        return {"id": served, "object": "model", "created": int(time.time()), "owned_by": "qwen38-engine",
                "root": served, "parent": None, "max_model_len": ctx}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [card()]}

    @app.get("/v1/models/{name:path}")
    async def model(name: str):
        return card() if name == served else err(404, f"The model `{name}` does not exist.", "model")

    @app.get("/version")
    async def version():
        return {"version": settings.version(), "engine": "plus"}

    @app.post("/tokenize")
    async def tokenize(req: Request):
        """vLLM's /tokenize: {prompt} or {messages, tools, chat_template_kwargs, add_generation_prompt} →
        {count, max_model_len, tokens, token_strs}."""
        try:
            b = await req.json()
            if "messages" in b:
                thinking, level = thinking_mode(b)
                kw = dict(b.get("chat_template_kwargs") or {}, enable_thinking=thinking)
                if thinking:
                    kw["reasoning_effort"] = level
                items = [] if media is not None else None
                text = chat.render(normalize_messages(b["messages"], quote, items),
                                   tools=quote_tools(normalize_tools(b.get("tools")), quote) or normalize_tools(b.get("tools")),
                                   reasoning_effort=level or "none", kwargs=kw,
                                   add_generation_prompt=b.get("add_generation_prompt", True))
                ids = enc(text)
                if items:  # the prompt as the model gets it: each placeholder expanded (no vision tower)
                    ids = await media.expand_only(items, ids)
            else:
                ids = chat.encode(str(b["prompt"]))
        except (TemplateError, ValueError, KeyError, TypeError) as e:
            return err(400, str(e), getattr(e, "param", None))
        strs = [chat.tok.id_to_token(i) for i in ids] if b.get("return_token_strs") else None
        return {"count": len(ids), "max_model_len": ctx, "tokens": ids, "token_strs": strs}

    @app.post("/detokenize")
    async def detokenize(req: Request):
        try:
            b = await req.json()
            return {"prompt": chat.tok.decode([int(x) for x in b["tokens"]], skip_special_tokens=False)}
        except (ValueError, KeyError, TypeError) as e:
            return err(400, str(e))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        th = [time.time()]  # timing marks (results/serve/timing.log): handler start, body read, parsed, rendered, encoded
        raw = await req.body()
        th.append(time.time())
        try:
            b = json.loads(raw)
        except ValueError as e:
            return err(400, f"invalid JSON body: {e}")
        th.append(time.time())
        try:
            if "messages" not in b:
                raise BadRequest("Field required: messages", "messages")
            thinking, level = thinking_mode(b)
            kw = dict(b.get("chat_template_kwargs") or {}, enable_thinking=thinking)
            if thinking:
                kw["reasoning_effort"] = level  # the decided level, whatever form it came in
            tl = normalize_tools(b.get("tools"))
            tc, named = tool_choice(b, tl)
            items = [] if media is not None else None  # image_url / video_url parts, in order
            text = chat.render(normalize_messages(b["messages"], quote, items), tools=quote_tools(tl, quote) or tl,
                               reasoning_effort=level or "none",
                               kwargs=kw)
            th.append(time.time())
            prompt = enc(text)
            mm = key = None
            if items:  # pixels, the vision tower, then each placeholder expanded and the mRoPE positions
                prompt, key, mm, mstat = await media.prepare(items, prompt)
            n_prompt = len(prompt)
            th.append(time.time())
            # tool_choice required / named: the reply starts with a call (vLLM: a grammar from the first token, or right
            # after </think> when thinking) — here the call's opening tokens are forced there
            # required: only the <tool_call> token (forcing "\n<function=" too splits the text where the model would
            # not, and the names it then writes drift); named: the call's opening with the name
            force = (chat.encode(f"<tool_call>\n<function={named}>\n") if tc == "named" else
                     chat.encode("<tool_call>") if tc == "required" else None)
            if force and not thinking:
                prompt = prompt + force
                key = key + force if key is not None else None
            # thinking budget: vLLM's thinking_token_budget (also thinking_budget); 0 = answer at once; else the setting
            budget = next((b[k] for k in ("thinking_token_budget", "thinking_budget") if b.get(k) is not None), None)
            if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 0):
                raise BadRequest(f"thinking_token_budget must be a non-negative integer, got {budget}.",
                                 "thinking_token_budget", budget)
            if budget is None:
                budget = think_cfg.get(f"budget_{level}", 0) or None if thinking else None
            extra = {"th": th, "n_prompt": n_prompt}
            if mm is not None:
                extra["media"], extra["key"] = mm, key
            if force:
                extra["force"], extra["prefix_out"] = (None, force) if not thinking else (force, None)
            if tl and tc != "none" and (tc == "named" or b.get("parallel_tool_calls") is False):
                extra["stop_tok"] = chat._sp_id.get("</tool_call>")  # one call, then the reply ends
            r, q = submit(prompt, b, "chat", thinking, budget, **extra)
        except (TemplateError, ValueError, KeyError, TypeError) as e:
            return err(400, str(e), getattr(e, "param", None))
        show_reasoning = b.get("include_reasoning", True) is not False  # vLLM: still generated, returned as null
        schemas = {t["function"]["name"]: (t["function"].get("parameters") or {}).get("properties") or {}
                   for t in tl or []}  # tool calls parsed when tools are offered; dropped for tool_choice "none"
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        stops = [b["stop"]] if isinstance(b.get("stop"), str) else list(b.get("stop") or [])

        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": served,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        async def run():
            """→ events (kind, value) and the final finish reason / token count."""
            det, n, held = Detok(chat), 0, ""
            par = Parser(thinking, bool(tl), tc == "none", schemas, stream=bool(b.get("stream")))
            hold = max(len(s) for s in stops) - 1 if stops else 0
            async for t, f in tokens(r, q, req):
                n += len(t)
                txt = det.add(t) if t else ""
                if f:
                    txt += det.flush()
                if stops:  # stop strings end the answer (not returned); the last len−1 chars wait — a stop may span pieces
                    held += txt
                    hits = [held.find(s) for s in stops if s and s in held]
                    if hits:
                        txt, held, f = held[:min(hits)], "", "stop"
                        r.cancelled = True
                    elif f:
                        txt, held = held, ""
                    else:
                        cut = max(0, len(held) - hold)
                        txt, held = held[:cut], held[cut:]
                for ev in par.feed(txt, final=f):
                    yield ev
                if f:
                    yield ("finish", ("stop" if f == "abort" else f, n))
                    return

        if b.get("stream"):
            inc = (b.get("stream_options") or {}).get("include_usage", False)

            async def sse():
                yield f"data: {json.dumps(chunk({'role': 'assistant', 'content': ''}))}\n\n"
                try:
                    async for x in sse_events():
                        yield x
                except EngineGone as e:
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': ERR_TYPE[503], 'param': None, 'code': 503}})}\n\n"
                yield "data: [DONE]\n\n"

            async def sse_events():
                ntool = 0
                async for kind, v in run():
                    if kind == "reasoning":
                        if r.t_sse is None:
                            r.t_sse = time.time()
                            timing(r)
                        if show_reasoning:  # vLLM 0.32 names it `reasoning`; older clients read `reasoning_content`
                            yield f"data: {json.dumps(chunk({'reasoning': v, 'reasoning_content': v}), ensure_ascii=False)}\n\n"
                    elif kind == "content":
                        if r.t_sse is None:
                            r.t_sse = time.time()
                            timing(r)
                        yield f"data: {json.dumps(chunk({'content': v}), ensure_ascii=False)}\n\n"
                    elif kind == "tool_delta":  # vLLM's streaming: the name first, then the arguments in pieces
                        ntool += "id" in v
                        yield f"data: {json.dumps(chunk({'tool_calls': [v]}), ensure_ascii=False)}\n\n"
                    elif kind == "tool":
                        pass  # the whole call: already streamed as its deltas
                    else:
                        fin, n = v
                        tcf = 'tool_calls' if ntool and fin == 'stop' and tc != "named" else fin  # vLLM's rule
                        yield f"data: {json.dumps(chunk({}, tcf))}\n\n"
                        if inc:
                            u = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": served,
                                 "choices": [], "usage": usage(r, n)}
                            yield f"data: {json.dumps(u)}\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        reasoning, content, calls, fin, n = "", "", [], "stop", 0
        try:
            async for kind, v in run():
                if kind == "reasoning":
                    reasoning += v
                elif kind == "content":
                    content += v
                elif kind == "tool":
                    calls.append({k: v[k] for k in ("id", "type", "function")})
                else:
                    fin, n = v
        except EngineGone as e:
            return err(503, str(e))
        if r.cancelled and fin != "stop":
            return err(499, "client closed request")
        rs = (reasoning or None) if show_reasoning else None
        msg = {"role": "assistant", "content": content or (None if calls else ""), "reasoning": rs, "reasoning_content": rs}
        if calls:
            msg["tool_calls"], fin = calls, ("tool_calls" if fin == "stop" and tc != "named" else fin)
        if tc in ("required", "named"):
            msg["content"] = content  # vLLM: "" rather than null
        return {"id": cid, "object": "chat.completion", "created": created, "model": served,
                "choices": [{"index": 0, "message": msg, "finish_reason": fin}], "usage": usage(r, n)}

    @app.post("/v1/completions")
    async def completions(req: Request):
        b = await req.json()
        try:
            p = b["prompt"]
            prompt = list(p) if isinstance(p, list) and all(isinstance(x, int) for x in p) else chat.encode(str(p))
            r, q = submit(prompt, b, "completion")
        except (ValueError, KeyError, TypeError) as e:
            return err(400, str(e), getattr(e, "param", None))
        cid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(text, finish=None):
            return {"id": cid, "object": "text_completion", "created": created, "model": served,
                    "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]}

        if b.get("stream"):
            async def sse():
                det, n = Detok(chat), 0
                try:
                    async for t, f in tokens(r, q, req):
                        n += len(t)
                        txt = (det.add(t) if t else "") + (det.flush() if f else "")
                        if txt or f:
                            yield f"data: {json.dumps(chunk(txt, None if not f else ('stop' if f == 'abort' else f)), ensure_ascii=False)}\n\n"
                except EngineGone as e:
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': ERR_TYPE[503], 'param': None, 'code': 503}})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")
        ids, fin = [], "stop"
        try:
            async for t, f in tokens(r, q, req):
                ids += t
                if f:
                    fin = "stop" if f == "abort" else f
        except EngineGone as e:
            return err(503, str(e))
        out = chunk(chat.decode(ids), fin)
        out["usage"] = usage(r, len(ids))
        return out

    return app
