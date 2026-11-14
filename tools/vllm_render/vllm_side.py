import json, logging, sys, types, warnings
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.chat_utils import load_chat_template, parse_chat_messages
from vllm.renderers.hf import resolve_chat_template_content_format, safe_apply_chat_template
from vllm.renderers.params import merge_kwargs
from vllm.tokenizers import get_tokenizer
MODEL = "/models/Qwen3.8-Flash-Next-FP4-Local"
TEMPLATE = load_chat_template(MODEL + "/chat_template.local.jinja")
TOK = get_tokenizer(MODEL)
MC = types.SimpleNamespace(model=MODEL, tokenizer=MODEL, revision=None, tokenizer_revision=None, code_revision=None,
    trust_remote_code=False, enable_prompt_embeds=False, multimodal_config=None, hf_config=None,
    allowed_local_media_path="", allowed_media_domains=None)
def render(body):
    req = ChatCompletionRequest(**dict(body, model="x"))
    tool_dicts = None if req.tools is None else [t.model_dump() for t in req.tools]
    defaults = merge_kwargs({}, dict(tools=tool_dicts, tokenize=False))
    params = req.build_chat_params(TEMPLATE, "auto").with_defaults(defaults)
    kwargs = params.get_apply_chat_template_kwargs()
    fmt = resolve_chat_template_content_format(chat_template=params.chat_template,
        tools=params.chat_template_kwargs.get("tools"), given_format=params.chat_template_content_format, tokenizer=TOK, model_config=MC)
    conv, _, _ = parse_chat_messages(list(req.messages), MC, content_format=fmt)
    return safe_apply_chat_template(MC, TOK, conv, **kwargs)
out = []
for b in json.load(sys.stdin):
    b = {k: v for k, v in b.items() if k != "name"}  # the case's label
    try:
        t = render(b)
        out.append({"text": t, "ids": TOK.encode(t, add_special_tokens=False)})
    except Exception as e:
        out.append({"error": f"{type(e).__name__}: {e}"[:200]})
print(json.dumps(out))
