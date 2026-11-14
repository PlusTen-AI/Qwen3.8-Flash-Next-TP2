"""Chat side of the server: template rendering, tokenisation, detokenisation and the Qwen3 output parser — the same
behaviour as the reference serving stack (vLLM 0.32: transformers' apply_chat_template, `reasoning-parser qwen3`,
its chat_template.local.jinja = the model's chat_template.jinja + reasoning-effort names — done here in render(), so
the checkpoint's own template is enough). Pure Python on the head node; the engine never sees text.

Template: transformers renders with an ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks, loopcontrols), its own
`tojson` (ensure_ascii False) and the globals raise_exception / strftime_now; vLLM passes reasoning_effort and, when the
request has one and the client did not set it, enable_thinking = (reasoning_effort != "none").
Parser (vllm/parser/qwen3.py): output starts in reasoning when thinking is on (the prompt ends with "<think>\\n"),
"</think>" switches to content (a later "</think>" is dropped), "<tool_call>" — or a bare "<function=" in content —
starts a tool call; parameters are "<parameter=name>\\nvalue\\n</parameter>", values kept as strings, arguments = JSON.
"""
import array
import collections
import json
import math
import os
import re
import uuid
from datetime import datetime

import jinja2
import jinja2.ext
import jinja2.sandbox
from tokenizers import Tokenizer

try:
    import qlog  # tools/qlog.py (the server puts tools/ on the path)
except ImportError:  # pragma: no cover — chat.py used on its own
    qlog = None


EFFORT_ALIAS = {"minimal": "low", "high": "xhigh", "max": "xhigh"}  # unset → "xhigh" (the template's top level)
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def thinking_mode(body):
    """Thinking on/off and level from any of the request forms clients use, read in one place so the template and
    the parser see the same decision: reasoning_effort (OpenAI; as vLLM it overrides chat_template_kwargs'),
    chat_template_kwargs {enable_thinking (wins for on/off), reasoning_effort}, reasoning {enabled, effort}
    (OpenRouter style), think (Ollama style: bool or a level).
    Effort "none" turns thinking off wherever it comes from. Levels: minimal → low, high / max → xhigh, unset → xhigh.
    Returns (enabled, level or None)."""
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise ValueError("chat_template_kwargs must be an object")
    r = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    think = body.get("think")
    # top-level reasoning_effort overrides chat_template_kwargs' (vLLM 0.32 renderers/params.py), then the others
    eff = next((v for v in (body.get("reasoning_effort"), kw.get("reasoning_effort"), r.get("effort"),
                            think if isinstance(think, str) else None) if v is not None), None)
    if eff is not None and (not isinstance(eff, str) or eff.lower() not in EFFORTS):
        raise ValueError(f"reasoning effort must be one of {', '.join(EFFORTS)}, got {eff!r}")
    eff = eff.lower() if eff is not None else None
    if "enable_thinking" in kw:
        en = bool(kw["enable_thinking"])
    elif r.get("enabled") is not None:
        en = bool(r["enabled"])
    elif isinstance(think, bool):
        en = think
    else:
        en = True
    if eff == "none":
        en = False
    return en, ((EFFORT_ALIAS.get(eff, eff) or "xhigh") if en else None)


def _media_part(p):
    """("image" | "video", url) of an OpenAI image_url / video_url content part, else None."""
    for kind, key in (("image", "image_url"), ("video", "video_url")):
        if p.get("type") == key:
            v = p.get(key)
            url = v.get("url") if isinstance(v, dict) else v
            if not isinstance(url, str) or not url:
                raise ValueError(f"a {key} part needs {key}.url")
            return kind, url
    return None


def normalize_messages(messages, quote=None, media=None):
    """The messages as vLLM 0.32 hands them to the template (renderers/chat_utils.py), so prompts match its tokens:
    string content → [{"type": "text", "text": s}], null → []; a tool message's text parts joined with "\n"; an
    assistant's `reasoning` (wins) / `reasoning_content` set as both; tool_calls arguments (JSON string) → object
    (None / "" / invalid / not an object → {}), an empty tool_calls list dropped.
    media (a list, when the server has vision): image_url / video_url parts become the template's {"type": "image"} /
    {"type": "video"} items and their (kind, url) are appended in order; without it such parts are a 400 (the model
    would otherwise answer as if there were no image).
    quote (Chat.quote, server.plain_specials): the special-token texts a client wrote become ordinary text — in every
    message's content (system included), reasoning and tool-call arguments (tool definitions: quote_tools)."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise ValueError("each message needs a role")
        m = dict(m)
        c = m.get("content")
        parts = [{"type": "text", "text": c}] if isinstance(c, str) else [] if c is None else list(c)
        for i, p in enumerate(parts):
            mp = _media_part(p) if isinstance(p, dict) and media is not None else None
            if mp:
                media.append(mp)
                parts[i] = {"type": mp[0]}
            elif not isinstance(p, dict) or p.get("type") != "text":
                kind = p.get("type") if isinstance(p, dict) else type(p).__name__
                raise ValueError(f"content part type '{kind}' is not supported by this server"
                                 f"{'' if media is not None else ' (vision is off)'}")
        if quote:  # every role: a system prompt is client text too (agents put files in it)
            parts = [dict(p, text=quote(p.get("text", ""))) if p.get("type") == "text" else p for p in parts]
        if m["role"] == "tool" and all(isinstance(p, dict) and p.get("type") == "text" for p in parts):
            m["content"] = "\n".join(p.get("text", "") for p in parts)
        else:
            m["content"] = parts
        if m["role"] == "assistant":
            rs = m.get("reasoning") if m.get("reasoning") is not None else m.get("reasoning_content")
            if rs is not None:
                if quote and isinstance(rs, str):
                    rs = quote(rs)
                m["reasoning"] = m["reasoning_content"] = rs
            tc = m.get("tool_calls")
            if not tc:
                m.pop("tool_calls", None)
            else:
                calls = []
                for t in tc:
                    if (t or {}).get("type", "function") != "function":
                        raise ValueError("chat completions only support assistant tool_calls of type 'function'.")
                    f = dict(t.get("function") or {})
                    a = f.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a) if a.strip() else {}
                        except ValueError:
                            a = {}
                    f["arguments"] = _quote_values(a, quote) if isinstance(a, dict) and quote else \
                        a if isinstance(a, dict) else {}
                    calls.append({**t, "function": f})
                m["tool_calls"] = calls
        out.append(m)
    return out


def _quote_values(x, quote):
    if isinstance(x, str):
        return quote(x)
    if isinstance(x, dict):
        return {k: _quote_values(v, quote) for k, v in x.items()}
    if isinstance(x, list):
        return [_quote_values(v, quote) for v in x]
    return x


_SENT = re.compile("\ue000(\\d+)\ue001")  # a quoted marker in rendered text: \ue000<index>\ue001


def _punct(ch):
    return not (ch.isalpha() or ch.isnumeric() or ch.isspace())


def quote_tools(tools, quote):
    """The tool definitions as rendered, with every string (names, descriptions, schema text) quoted."""
    return None if tools is None or not quote else _quote_values(tools, quote)


def normalize_tools(tools):
    """Tools as vLLM's request model dumps them: {"type": "function", "function": {name, description, parameters}}
    (description / parameters null when absent; strict, defer_loading only when set; other keys dropped)."""
    if tools is None:
        return None
    if not isinstance(tools, list) or not tools:
        raise ValueError("`tools` must not be an empty array. Either provide at least one tool or omit the field "
                         "entirely.")
    out = []
    for t in tools:
        f = (t or {}).get("function") or {}
        if not isinstance(f.get("name"), str):
            raise ValueError("each tool needs function.name")
        g = {"name": f["name"], "description": f.get("description"), "parameters": f.get("parameters")}
        for k in ("strict", "defer_loading"):
            if f.get(k) is not None:
                g[k] = f[k]
        out.append({"type": "function", "function": g})
    return out


class TemplateError(ValueError):
    pass


def _raise(msg):
    raise TemplateError(msg)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class Chat:
    def __init__(self, model_dir, template="chat_template.jinja"):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        # encode cache: the tokenizer splits added tokens out first (all flags off: no strip, not normalized) and
        # encodes the text between them independently, so those pieces can be cached — a follow-up turn re-encodes
        # only its new pieces. Longest match first at each position, as the tokenizer's added-token matcher.
        added = json.load(open(os.path.join(model_dir, "tokenizer.json")))["added_tokens"]
        self._sp = re.compile("|".join(re.escape(x["content"]) for x in sorted(added, key=lambda x: -len(x["content"]))))
        self._sp_id = {x["content"]: x["id"] for x in added}
        self.think_end = self._sp_id.get("</think>")
        self.im_start = self._sp_id.get("<|im_start|>")
        # plain_specials: the same tokenizer without its added tokens encodes a quoted marker as ordinary text
        # (identical ids for any text without markers); the sentinel's own lead character is quoted as well
        tj = json.load(open(os.path.join(model_dir, "tokenizer.json")))
        self._plain = Tokenizer.from_str(json.dumps(dict(tj, added_tokens=[])))
        self.markers = sorted((x["content"] for x in added), key=len, reverse=True) + ["\ue000"]
        self._mk_re = re.compile("|".join(re.escape(x) for x in self.markers))
        self._mk_id = {x: i for i, x in enumerate(self.markers)}
        self._cache, self._cache_chars = collections.OrderedDict(), 0
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda f: datetime.now().strftime(f)
        self.tmpl = env.from_string(open(os.path.join(model_dir, template)).read())
        gc = json.load(open(os.path.join(model_dir, "generation_config.json")))
        eos = gc.get("eos_token_id", [])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        # server defaults = generation_config (vLLM: request → these → OpenAI defaults)
        self.defaults = {"temperature": gc.get("temperature", 1.0), "top_k": gc.get("top_k", -1),
                         "top_p": gc.get("top_p", 1.0)}

    def render(self, messages, tools=None, reasoning_effort=None, kwargs=None, add_generation_prompt=True):
        kw = dict(kwargs or {})
        extra = {"add_generation_prompt": add_generation_prompt, "reasoning_effort": reasoning_effort}
        if reasoning_effort is not None and "enable_thinking" not in kw:
            extra["enable_thinking"] = reasoning_effort != "none"
        kw = {**extra, **kw}  # the client's chat_template_kwargs win (vLLM merge order)
        v = kw.get("reasoning_effort")  # effort names (as the served template's prelude did): after the merge
        kw["reasoning_effort"] = "xhigh" if v is None else EFFORT_ALIAS.get(v, v) if isinstance(v, str) else v
        if tools:
            kw["tools"] = tools
        return self.tmpl.render(messages=messages, bos_token="", eos_token="<|im_end|>", **kw)

    def thinking(self, reasoning_effort=None, kwargs=None):
        """Does generation start inside <think> (the parser's initial state)?"""
        kw = kwargs or {}
        if "enable_thinking" in kw:
            return bool(kw["enable_thinking"])
        return reasoning_effort != "none"

    def quote(self, s):
        """The markers in s (every added token's text) as sentinels, for encode_shielded."""
        if not isinstance(s, str) or not self._mk_re.search(s):
            return s
        return self._mk_re.sub(lambda m: f"\ue000{self._mk_id[m.group(0)]}\ue001", s)

    def encode_shielded(self, text):
        """encode(), except quoted markers (Chat.quote) are ordinary text: each quoted marker, with the punctuation run
        around it (and one space before / the newlines after, as the pre-tokenizer groups them), is encoded by the
        tokenizer without added tokens — never across a real (template) marker. Text without quotes: encode()."""
        if "\ue000" not in text:
            return self.encode(text)
        segs, pos = [], 0  # [plain?, text]
        for m in _SENT.finditer(text):
            segs.append([False, text[pos:m.start()]])
            segs.append([True, self.markers[int(m.group(1))]])
            pos = m.end()
        segs.append([False, text[pos:]])
        real = self.markers[:-1]
        for i, (pl, s) in enumerate(segs):
            if not pl:
                continue
            if i > 0 and not segs[i - 1][0]:  # punctuation (and one space) before it, not out of a real marker
                prev = segs[i - 1][1]
                k = len(prev)
                while k > 0 and _punct(prev[k - 1]) and prev[k - 1] not in "\r\n" and \
                        not any(prev[:k].endswith(r) for r in real):
                    k -= 1
                if k > 0 and prev[k - 1] == " " and not any(prev[:k - 1].endswith(r) for r in real):
                    k -= 1
                segs[i - 1][1], segs[i][1] = prev[:k], prev[k:] + segs[i][1]
            if i + 1 < len(segs) and not segs[i + 1][0]:  # punctuation and newlines after it
                nxt, k = segs[i + 1][1], 0
                while k < len(nxt) and _punct(nxt[k]) and nxt[k] not in "\r\n" and \
                        not any(nxt.startswith(r, k) for r in real):
                    k += 1
                while k < len(nxt) and nxt[k] in "\r\n":
                    k += 1
                segs[i][1], segs[i + 1][1] = segs[i][1] + nxt[:k], nxt[k:]
        ids, plain = [], ""
        for pl, s in segs:  # adjacent quoted runs (only empty text between them) are one plain piece
            if pl:
                plain += s
            elif s:
                if plain:
                    ids += self._plain.encode(plain, add_special_tokens=False).ids
                    plain = ""
                ids += self.encode(s)
        if plain:
            ids += self._plain.encode(plain, add_special_tokens=False).ids
        return ids

    def encode(self, text):
        """= tok.encode(text, add_special_tokens=False).ids, with the pieces between added tokens cached (LRU)."""
        ids, pos = [], 0
        for m in self._sp.finditer(text):
            if m.start() > pos:
                ids += self._piece(text[pos:m.start()])
            ids.append(self._sp_id[m.group()])
            pos = m.end()
        if pos < len(text):
            ids += self._piece(text[pos:])
        return ids

    def _piece(self, s, min_chars=256, max_chars=64_000_000):
        if len(s) < min_chars:
            return self.tok.encode(s, add_special_tokens=False).ids
        hit = self._cache.get(s)
        if hit is not None:
            self._cache.move_to_end(s)
            return hit
        ids = array.array("i", self.tok.encode(s, add_special_tokens=False).ids)  # 4 B per id
        self._cache[s] = ids
        self._cache_chars += len(s)
        while self._cache_chars > max_chars and len(self._cache) > 1:
            k, _ = self._cache.popitem(last=False)
            self._cache_chars -= len(k)
        return ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


class Detok:
    """Incremental detokeniser (vLLM's prefix/read offsets): each step decodes only a short window, so a long answer
    costs O(n); an unfinished UTF-8 sequence (replacement character at the end) is held back until it completes."""

    def __init__(self, chat):
        self.chat, self.ids, self.pre, self.read = chat, [], 0, 0

    def add(self, ids):
        self.ids += ids
        head = self.chat.decode(self.ids[self.pre:self.read])
        txt = self.chat.decode(self.ids[self.pre:])
        if len(txt) <= len(head) or txt.endswith("�"):
            return ""
        self.pre, self.read = self.read, len(self.ids)
        return txt[len(head):]

    def flush(self):
        head = self.chat.decode(self.ids[self.pre:self.read])
        txt = self.chat.decode(self.ids[self.pre:])
        self.pre = self.read = len(self.ids)
        return txt[len(head):]


_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>\n]*)>?", re.DOTALL)


def _trim_nl(v):
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _tool_call(body, idx, schemas=None):
    """A call from a <tool_call> block's body: Qwen XML (<function=NAME> + <parameter=K>), else the JSON form
    {"name", "arguments"} (Hermes / Qwen2.5 style). None if neither gives a name — the block then stays text."""
    m = _FUNC_RE.search(body)
    name = m.group(1).strip() if m else ""
    if not name:
        try:
            j = json.loads(body.strip())
        except ValueError:
            return None
        if not isinstance(j, dict) or not isinstance(j.get("name"), str) or not j["name"].strip():
            return None
        a = j.get("arguments", j.get("parameters", {}))
        if isinstance(a, str):
            try:
                a = json.loads(a) if a.strip() else {}
            except ValueError:
                return None
        if not isinstance(a, dict):
            return None
        return {"index": idx, "id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}", "type": "function",
                "function": {"name": j["name"].strip(), "arguments": json.dumps(a, ensure_ascii=False)}}
    args = {p.group(1): _trim_nl(p.group(2)) for p in _PARAM_RE.finditer(body)}
    props = ((schemas or {}).get(name) or {})
    if props:  # vLLM parser_engine._fix_arg_types: values to the tool schema's types; no schema → strings
        args = {k: (_coerce(v, _schema_types(props[k])) if isinstance(props.get(k), dict) else v)
                for k, v in args.items()}
    return {"index": idx, "id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


_TYPE_ALIAS = {"int": "integer", "long": "integer", "float": "number", "double": "number", "bool": "boolean",
               "dict": "object", "list": "array", "str": "string", "text": "string", "enum": "string"}
_PY_TYPE = {type(None): "null", bool: "boolean", int: "integer", float: "number", str: "string", list: "array",
            dict: "object"}
_ORDER = ("null", "integer", "number", "boolean", "object", "array", "string")


def _schema_types(s):
    """vLLM tool_parsers/utils.extract_types_from_schema: type (string or list, aliases mapped), enum value types,
    anyOf / oneOf / allOf unioned; nothing found → string."""
    if not isinstance(s, dict):
        return ["string"]
    out, t = [], s.get("type")
    for x in (t if isinstance(t, list) else [t]):
        if isinstance(x, str):
            out.append(_TYPE_ALIAS.get(x, x))
    for v in s.get("enum") or []:
        out.append(_PY_TYPE.get(type(v), "string"))
    for k in ("anyOf", "oneOf", "allOf"):
        for sub in s.get(k) or []:
            out += _schema_types(sub)
    return list(dict.fromkeys(out)) or ["string"]


def _finite(x):
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, list):
        return all(_finite(y) for y in x)
    if isinstance(x, dict):
        return all(_finite(y) for y in x.values())
    return True


def _coerce(v, types):
    """vLLM coerce_to_schema_type: the allowed types in the order null > integer > number > boolean > object > array >
    string; nothing fits → json.loads(v) if that works (finite), else the raw string."""
    for t in (x for x in _ORDER if x in types):
        if t == "null" and v.lower() == "null":
            return None
        if t == "integer":
            try:
                return int(v)
            except ValueError:
                pass
        if t == "number":
            try:
                f = float(v)
                if math.isfinite(f):
                    return int(f) if f.is_integer() else f
            except ValueError:
                pass
        if t == "boolean" and v.lower() in ("true", "1", "false", "0"):
            return v.lower() in ("true", "1")
        if t in ("object", "array"):
            try:
                x = json.loads(v)
                if _finite(x):
                    return x
            except ValueError:
                pass
        if t == "string":
            return v
    try:
        x = json.loads(v)
        return x if _finite(x) else v
    except ValueError:
        return v


_HEAD_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>")
_PEND_RE = re.compile(r"<\s*/\s*parameter\s*>")
_VALUE_TAGS = ("</parameter>", "<parameter=", "</function>", "</tool_call>")


def _tag_tail(v):
    """length of the longest suffix of v that is a proper prefix of a tag that can end a value"""
    for n in range(min(len(v), max(len(t) for t in _VALUE_TAGS) - 1), 0, -1):
        if any(t.startswith(v[-n:]) for t in _VALUE_TAGS):
            return n
    return 0


def _args_prefix(body, props):
    """The arguments JSON of a call's body so far (the text after <function=NAME>), cut to what cannot change — as vLLM
    0.32's parser engine streams it (tools/vllm_render/stream_vllm.json): "{" and the first key at its parameter's
    ">"; a string value (schema type string, or no schema for the tool) as it comes, JSON-escaped, with one leading and
    one trailing newline trimmed (a trailing newline and a possible tag start wait); its closing quote with the next
    separator; any other value (typed by the schema, or a key the schema does not have) whole, once the next parameter
    or </function> arrives; "}" at </function>. The same values as _tool_call's."""
    k = body.find("</function>")
    done = k >= 0
    if done:
        body = body[:k]
    heads = list(_HEAD_RE.finditer(body))
    if not heads:
        return "{}" if done else ""
    parts = []
    for j, h in enumerate(heads):
        key = h.group(1)
        raw = body[h.end():heads[j + 1].start() if j + 1 < len(heads) else len(body)]
        e = _PEND_RE.search(raw)
        if e:
            raw = raw[:e.start()]
        p = props.get(key) if props else None
        types = _schema_types(p) if isinstance(p, dict) else None
        text = not props or types == ["string"]
        kj = json.dumps(key, ensure_ascii=False) + ": "
        if j + 1 < len(heads) or done:  # its value is final
            v = _trim_nl(raw)
            parts.append(kj + json.dumps(_coerce(v, types) if types else v, ensure_ascii=False))
        elif not text:  # typed / not in the schema: the value waits until it is final
            parts.append(kj)
        else:
            if e:
                v = _trim_nl(raw)
            else:
                v = raw[1:] if raw.startswith("\n") else raw
                v = v[:len(v) - _tag_tail(v)]
                if v.endswith("\n"):  # may be the trimmed one
                    v = v[:-1]
            parts.append(kj + json.dumps(v, ensure_ascii=False)[:-1])
    return "{" + ", ".join(parts) + ("}" if done else "")


class Parser:
    """Streaming Qwen3 parser. feed(text) → list of ("reasoning" | "content", str) and ("tool", call) events; text
    that may be the start of a tag is held back until it is decided. Tool calls are parsed only when the request offers
    tools, as vLLM does: without them a <tool_call> or <function= the model writes — e.g. quoting a chat template — is
    plain text, not a call that swallows (and holds back) the rest of the answer. With tool_choice "none" (suppress)
    a call block is dropped entirely (vLLM: neither content nor tool_calls). schemas: tool name → its parameters'
    properties, for typing the arguments.
    Content that is only whitespace so far is held: dropped once a call is made, else sent with the next text (vLLM's
    parser engine: no "\n\n" content in front of a call).
    stream=True: calls also go out as they are written, as vLLM 0.32 streams them — ("tool_delta", delta) events: first
    {id, type, index, function: {name}} with the first text after <function=NAME>, then {index, function: {arguments:
    piece}} (_args_prefix). A call cut off (max_tokens, or the reply ended inside it) is not closed. The ("tool", call)
    event at </tool_call> still carries the whole call (same id; the pieces join to its arguments)."""
    def __init__(self, thinking, tools=True, suppress=False, schemas=None, stream=False):
        self.state = "reasoning" if thinking else "content"
        self.buf, self.tool, self.ntools = "", "", 0
        self.suppress, self.schemas, self.open, self.stream = suppress, schemas, "", stream
        self.ws, self.said, self.called = "", False, False  # held whitespace content, real content sent, a call made
        self.cs = None  # the open call's stream: [id, arguments sent]
        self.tags = {"reasoning": ("</think>", "<think>", "<tool_call>") if tools else ("</think>", "<think>"),
                     "content": ("</think>", "<tool_call>", "<function=") if tools else ("</think>",)}

    def _hold(self, s):  # length of the longest suffix of s that is a proper prefix of a tag
        tags = self.tags[self.state]
        for n in range(min(len(s), max(len(t) for t in tags) - 1), 0, -1):
            if any(t.startswith(s[-n:]) for t in tags):
                return n
        return 0

    def _emit(self, ev, kind, s):
        if not s:
            return
        if kind != "content" or self.said:
            ev.append((kind, s))
        elif not s.strip():
            self.ws += s
        else:
            ev.append(("content", ("" if self.called else self.ws) + s))
            self.ws, self.said = "", True

    def _stream_call(self, body):
        """the open call's deltas from its body so far (after <tool_call>, or from a bare <function=)"""
        m = _FUNC_RE.search(body)
        if not m or not body[m.end():] or not body[m.start():m.end()].endswith(">"):
            return []
        name, ev = m.group(1).strip(), []
        if not name:
            return []
        if self.cs is None:
            self.cs, self.called = [f"chatcmpl-tool-{uuid.uuid4().hex[:16]}", ""], True
            ev.append(("tool_delta", {"id": self.cs[0], "type": "function", "index": self.ntools,
                                      "function": {"name": name}}))
        target = _args_prefix(body[m.end():], (self.schemas or {}).get(name))
        if len(target) > len(self.cs[1]) and target.startswith(self.cs[1]):
            ev.append(("tool_delta", {"index": self.ntools, "function": {"arguments": target[len(self.cs[1]):]}}))
            self.cs[1] = target
        return ev

    def _close(self, ev, call):
        """a whole call: the ("tool", call) event; streaming, the arguments not yet sent (none when the pieces were
        complete) or the whole call in one delta (a JSON-form block)"""
        if self.stream:
            if self.cs is None:
                ev.append(("tool_delta", dict(call)))
            else:
                call["id"], a = self.cs[0], call["function"]["arguments"]
                if a.startswith(self.cs[1]) and len(a) > len(self.cs[1]):
                    ev.append(("tool_delta", {"index": call["index"], "function": {"arguments": a[len(self.cs[1]):]}}))
                elif not a.startswith(self.cs[1]) and qlog is not None:  # never expected: the pieces sent are not a
                    # prefix of the whole call (the client's arguments differ from the non-streamed ones)
                    qlog.warn("tool call", f"streamed arguments differ from the call {call['function']['name']}: sent "
                                           f"{self.cs[1][-80:]!r} · whole {a[-80:]!r}")
            self.cs = None
        self.called = True
        ev.append(("tool", call))
        self.ntools += 1

    def feed(self, text, final=False):
        self.buf += text
        ev = []
        while True:
            b = self.buf
            if self.state == "tool":  # look for the end tag in everything collected (it may span pieces)
                s = self.tool + b
                j = s.find("</tool_call>")
                if self.stream and not self.suppress:
                    ev += self._stream_call(s if j < 0 else s[:j])
                if j < 0:
                    self.tool, self.buf = s, ""
                    if final:  # unclosed at the end
                        if self.stream:  # vLLM: a call cut off stays as streamed (not closed, nothing added)
                            if self.cs is not None:
                                self.cs, self.ntools = None, self.ntools + 1
                            elif final != "length" and not self.suppress:  # a JSON-form block ended by the reply
                                call = _tool_call(self.tool, self.ntools, self.schemas)
                                if call is not None:
                                    self._close(ev, call)
                        else:  # a call if the reply ended by itself, text if max_tokens cut it
                            call = None if final == "length" else _tool_call(self.tool, self.ntools, self.schemas)
                            if call is None:
                                self._emit(ev, "content", self.open + self.tool)
                            elif not self.suppress:
                                self._close(ev, call)
                        self._flush(ev)
                    return ev
                call = _tool_call(s[:j], self.ntools, self.schemas)
                if call is None:  # no function name: the block stays text
                    self.cs = None
                    self._emit(ev, "content", self.open + s[:j + len("</tool_call>")])
                elif not self.suppress:
                    self._close(ev, call)
                self.tool, self.buf, self.state = "", s[j + len("</tool_call>"):], "content"
                continue
            tags = self.tags[self.state]
            hits = [(b.find(t), t) for t in tags if b.find(t) >= 0]
            if not hits:
                n = 0 if final else self._hold(b)
                self._emit(ev, self.state, b[:len(b) - n])
                self.buf = b[len(b) - n:]
                if final:
                    self._flush(ev)
                return ev
            j, t = min(hits)
            self._emit(ev, self.state, b[:j])
            rest = b[j + len(t):]
            if t == "</think>":
                self.state = "content"
            elif t == "<think>":
                pass
            elif t == "<tool_call>":
                self.state, self.tool, self.open = "tool", "", "<tool_call>"
            else:  # bare <function=
                self.state, self.tool, self.open = "tool", "<function=", ""
            self.buf = rest

    def _flush(self, ev):
        """the end: whitespace-only content goes out if no call was made"""
        if self.ws and not self.said and not self.called:
            ev.append(("content", self.ws))
        self.ws = ""
