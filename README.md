# Qwen3.8-Flash-Next-engine

A dedicated C++/CUDA inference engine and OpenAI-compatible server for **Qwen3.8-Flash-Next** in NVIDIA's
`Qwen3.8-Flash-Next-FP4-Local` checkpoint, on **two GB10 machines** (DGX Spark class) joined by a direct ConnectX-7
cable. The model is split across both machines (tensor parallel, TP = 2); one machine (the head) serves HTTP.

- **Lossless**: the checkpoint's own precision everywhere (NVFP4 / MXFP8 weights as stored, bf16 activations and KV
  cache, fp32 recurrent state and logits). Deterministic: the same seed gives the same text, alone or in a batch.
- **Fast**: MTP speculative decoding (exact acceptance — the output distribution is unchanged), RDMA all-reduce over
  both CX7 rails, prefix reuse across turns and across prompts that share a text.
- **Long**: 640K tokens per request by default (YaRN ×4; up to 1,048,576 by setting), the caches of all requests in
  one memory pool that requests are admitted by — a full machine never stops the server. A long prompt being read
  shares the prefill with the others: a short prompt cuts in and gets its first token within one chunk's time.
- **OpenAI-compatible**: `/v1/chat/completions` (streaming, thinking, tool calls, images and videos),
  `/v1/completions`, `/v1/models`, `/health`, `/metrics`.

Measured on one pair (128K-token prompts, thinking on, default sampling):

| | |
|---|---:|
| reading a 128K prompt | ~5,000 tok/s |
| one writer at 128K context | 80–90 tok/s |
| four writers at once (total) | ~180–195 tok/s |
| next turn of a 130K conversation (+7K new tokens) | 1.53 s to the first token |
| reading a 1M prompt | 243 s (4,300 tok/s) |
| long-context recall (`checks/needle.py`, 30 records) | 30/30 up to 640K · 29/30 at 768K and 896K · 24/30 at 1M |

## Requirements

- Two GB10 machines with a **direct CX7 cable**, IPv4 addresses on the CX7 interfaces (one subnet per cable),
  RoCE v2 (the default), MTU 9000 recommended. Using both halves of the CX7 port (two rails) is automatic when both
  have an address.
- **Docker** usable by your user on both machines, with the **NVIDIA container toolkit** (`--gpus all`).
- **ssh with a key** from the head to the worker (no password prompt).
- **Python ≥ 3.11** on both hosts (standard library only — nothing is installed on the hosts).
- About **100 GiB free memory** per machine while running (the caches take what the model leaves, less
  `server.memory_free_gib`), **~100 GB disk** for the checkpoint and **~25 GB** for the image on each machine.

## Quick start (on the head machine)

```bash
git clone <this repo> qwen38-engine && cd qwen38-engine
cp serve.local.toml.example serve.local.toml
$EDITOR serve.local.toml          # the worker's ssh name, its CX7 address, where the checkpoint is
./start.sh download               # only if you do not have the checkpoint yet (~96 GiB, then copied to the worker)
./start.sh check                  # everything start needs, one line each
./start.sh                        # first start also builds the image on both machines (10–30 min), later ~2 min
curl http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": "Hello!"}]}'
./stop.sh
```

`serve.local.toml` needs three things:

```toml
[cluster]
worker = "user@192.168.1.12"     # how the head reaches the worker by ssh
worker_addr = ["10.0.0.2"]       # the worker's address on the CX7 cable
[paths]
weights = "/home/user/models/Qwen3.8-Flash-Next-FP4-Local"   # same path on both machines
```

Nothing about your network is stored: at every start the head reads the route to `worker_addr` (interface and its
own address), the RDMA device on that interface, its RoCE v2 GID entry and the other half of the same CX7 port as a
second rail; the worker does the same toward the head's addresses. The start line shows what was found:

```
[start] link · 2 rails · roceP2p1s0f0 10.0.0.1 ↔ roceP2p1s0f1 10.0.0.2 · rocep1s0f0 10.0.1.1 ↔ rocep1s0f1 10.0.1.2 · GID 3/3 · MTU 9000
```

## Commands

| command | what it does |
|---|---|
| `./start.sh` | copy the repo to the worker, read the link, build the image and the tensor table if missing, start both ranks, wait until `/health` is 200 |
| `./start.sh restart` | stop, then start |
| `./stop.sh` | stop both ranks gracefully (below); logs are kept in `<work>/results/serve/` |
| `./start.sh status` | containers and `/health` |
| `./start.sh logs [N]` | the last N screen lines of the head |
| `./start.sh check` | settings, ssh, link (ping at full MTU on every rail), Docker and the NVIDIA toolkit, image, GPUs free, memory, ports, the checkpoint the same on both machines |
| `./start.sh download` | the checkpoint from Hugging Face into `paths.weights`, then copied to the worker (set `HF_TOKEN` if needed); `--only config.json` tries the whole path with one small file |

Any setting can be given for one run: `./start.sh --set server.slots=2 --set server.context=65536`, or as
`Q38_SERVER_SLOTS=2 ./start.sh`.

**Start**: `/health` answers 503 `{"status": "loading"}` until both ranks are ready, then 200. If one rank fails, the
other is stopped too and the reason is printed. If the server is already running, start says so and stops.

**Stop**: new requests get 503; running requests get `server.drain_seconds` (30) to finish and are then ended with a
normal ending (`finish_reason` "stop", and `[DONE]` for streams); then both ranks release the GPU, NCCL and RDMA
resources and exit. Stopping twice is harmless; stopping during start works too. If one rank is killed or exits, the
other notices at once, answers running requests with an error (503, or an error event in a stream) and exits instead of
waiting. If a whole machine or the cable goes away, TCP keepalive on the link ends it within about 20 s (this case is
not tested yet).

## Settings

All settings and their defaults are in [`serve.toml`](serve.toml) (with comments). Put only what differs into
`serve.local.toml`. Precedence: `--set` > `Q38_<SECTION>_<KEY>` > `serve.local.toml` > `serve.toml`. Unknown keys
are errors. Both ranks compare their settings and sources when they connect and refuse to run if they differ.

| setting | default | |
|---|---|---|
| `cluster.worker` | — | ssh name of the worker (required) |
| `cluster.worker_addr` | host of `worker` | the worker's CX7 address(es); one per rail to choose rails yourself |
| `cluster.rails` | 0 | 0 = all found, 1 = one rail |
| `cluster.gid_index` | -1 | RoCE v2 GID index; -1 = read from the device |
| `cluster.nccl_port`, `link_port` | 29611 (+1, +2), 29650 | TCP ports on the CX7 addresses (+2: the vision tower's NCCL) |
| `paths.weights` | — | the checkpoint folder, same path on both machines (required) |
| `paths.work` | `.work` | build cache, logs, tensor table |
| `paths.worker_repo` | same as the head | where the repo is copied on the worker |
| `server.port` | 8888 | HTTP port (`server.host` 0.0.0.0) |
| `server.served_name` | `Qwen/Qwen3.8-Flash-Next` | model name in the API |
| `server.slots` | 8 | requests served at the same time |
| `server.context` | 655360 | tokens per request (prompt + output); up to 1,048,576 (needs `model.yarn_factor` 4) |
| `server.kv_pool` | true | all requests' caches in one pool taken at start; false = `context` tokens per slot up front |
| `server.memory_free_gib` | 15 | memory left free on each machine after the pool (less slowed prefill by ~5% here) |
| `server.kv_pool_gib` | 0 | cap on the pool (0 = all the room) |
| `server.output_reserve` | 32768 | output tokens a request is admitted with when `max_tokens` is larger or absent; a longer reply takes more as it goes and ends (`length`) only when the pool is full |
| `server.checkpoints` | 128 | most prefix checkpoints shared by all slots (≈ 55 MiB each, in free pool memory): the start of a prompt's last message, 1,024 tokens before its end, its last chunk boundary; kept by recency like a prefix cache (when room is needed, the least recently used checkpoint or idle slot goes, busy slot or not) |
| `model.yarn_factor` | 4.0 | YaRN factor of the attention layers' RoPE (≤ 1 = off, the checkpoint's own 262,144 positions); `server.context` ≤ 262,144 × factor |
| `model.yarn_ramp` | 262144 | length the YaRN correction range is computed from (262,144 as HF transformers; vLLM's M-RoPE uses the window — in our needle runs that missed 2 of 30 at 512K) |
| `server.draft_depth` | 4 | MTP drafts per round |
| `server.drain_seconds` | 30 | time running requests get on stop |
| `server.api_key` | — | optional; see API |
| `server.plain_specials` | true | special-token text a client wrote stays text (see API) |
| `thinking.budget_low` / `_medium` / `_xhigh` | 0 (none) | thinking-token budget per effort level |
| `thinking.loop_guard` | true | close a looping thinking once per reply |
| `vision.enabled` | true | image / video input (the vision tower, half of every layer on each machine) |
| `vision.fetch_urls` | false | also fetch `http(s)://` media URLs (else `data:` URLs only) |
| `vision.max_concurrent` | 1 | requests whose media are encoded at the same time; the others wait their turn |
| `vision.max_items` / `max_media_tokens` / `max_item_mb` | 32 / 65536 / 64 | per request: images + videos, their placeholder tokens, one file's size |
| `vision.cache_mb` | 256 | embeddings of recent files: a picture sent again (every turn resends it) skips the tower |
| `patches.<name>` | see `engine/src/model/patches.def` | engine variants (all verified bit-exact against the base) |

## API

OpenAI-style, shaped like vLLM 0.32's serving of this model (the reference deployment):

- `POST /v1/chat/completions` — `messages`, `tools`, `tool_choice` (`auto`, `none`, `required`, a named function),
  `parallel_tool_calls`, `stream` (+ `stream_options.include_usage`), `max_tokens` / `max_completion_tokens`,
  `temperature`, `top_p`, `top_k`, `seed`, `stop`. Messages and tools are normalised as vLLM does before the chat
  template (content parts, tool-call arguments, reasoning on past turns), so prompts get the same tokens.
- Answers carry `reasoning` (also as `reasoning_content`) and `content`; tool calls come as `tool_calls` with arguments
  typed by the tool's JSON schema. Streamed, a call goes out as it is written, as vLLM streams it: its name first,
  then the arguments in pieces (a string value as it comes, a typed value once complete). `usage` has `prompt_tokens_details.cached_tokens` (prefix reuse) and
  `completion_tokens_details.reasoning_tokens`. `include_reasoning: false` returns the reasoning as null.
- `tool_choice: "required"` / a named function: the reply starts with a tool call (right after the thinking when
  thinking is on); a named function gives exactly one call. `"none"`: tools stay in the prompt, calls are dropped.
- Thinking: on by default (effort `xhigh`). Any of these forms is understood, and the template and the output parser
  always see the same decision: `reasoning_effort` (wins over `chat_template_kwargs.reasoning_effort`, as in vLLM),
  `chat_template_kwargs: {"enable_thinking", "reasoning_effort"}`, `reasoning: {"enabled", "effort"}`,
  `think: false`. Effort `none` turns thinking off; `minimal`→`low`, `medium`, `high`/`max`→`xhigh`.
- Thinking budget: `thinking_token_budget` (tokens; 0 = answer at once) per request, or `thinking.budget_<level>` in the
  settings (default: none). When thinking reaches it, the server closes it (`</think>`) and the model answers. A
  thinking loop (the last 256 thinking tokens repeating a pattern of ≤ 8 tokens) is closed the same way
  (`thinking.loop_guard`, on by default).
- Sampling: `temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`
  (vLLM's formulas and order; the penalties and min_p act on the target model's rows only, so speculative decoding
  stays exact). Defaults come from the checkpoint's `generation_config.json` (T 1.0, top_k 20, top_p 0.95).
- Errors use vLLM's body `{"error": {"message", "type", "param", "code"}}` and texts. A prompt with no room left in the
  window is a 400 that states the window and the prompt's token count; a `max_tokens` larger than the room left is cut
  to it (as vLLM). Refused with 400 rather than ignored: `logprobs`, `n` > 1, `logit_bias`, `response_format` /
  guided decoding.
- Images and videos: content parts `{"type": "image_url", "image_url": {"url": ...}}` and
  `{"type": "video_url", "video_url": {"url": ...}}` in user (and tool) messages, as `data:<type>;base64,...` URLs
  (`http(s)://` when `vision.fetch_urls` is on). The pixels, video frames, timestamps, placeholder tokens and 3-D
  (mRoPE) positions are those of the reference deployment, bit for bit: images resized to multiples of 32 pixels within
  65,536 … 16,777,216 pixels (one token per 32×32), videos sampled at 2 frames per second (4 … 768 frames, one token
  group per two frames with its time stamp). The vision tower is split over both machines (each runs half of every
  layer; the halves' fp32 partial sums are added before the one rounding) and its peak memory is taken at start with
  the largest picture and video, before the KV pool. It runs at the checkpoint's precision (weights decoded
  exactly, bf16 activations, fp32 accumulation). `usage.prompt_tokens` and `/tokenize` count the expanded
  prompt; prefix reuse covers prompts with media (keyed by the files' contents). Time to the first token: an 800×600
  picture ~0.4 s, a 1920×1080 one ~1.4 s, a 12 s video ~0.9 s; a picture sent again ~0.1–0.2 s. The largest picture
  (4096×4096, 16,404 tokens) takes ~12 s (tower 7.6 s over both machines, then its 16K-token prompt); the tower runs in
  slices between the other requests' rounds, so their replies slow down but keep flowing while it is encoded.
- Special-token text a client writes — any of the tokenizer's added tokens: `<|im_end|>`, `<|im_start|>`, `<think>`,
  `<tool_call>`, image markers … in any message (system included), reasoning, tool-call arguments or tool
  definitions — is encoded as ordinary text, so only the chat template's own
  markers are control tokens (`server.plain_specials`; vLLM and HF would turn written text into the real token).
  Prompts without such text get the same tokens either way.
- Stop strings may span several tokens; `parallel_tool_calls: false` ends the reply after the first call; a tool call
  cut by `max_tokens` stays text (finish_reason `length`) — streamed, the pieces already sent stay, unclosed (as vLLM); `usage.completion_tokens` counts the end-of-sequence token
  (as vLLM).
- API key (optional): set `server.api_key` in `serve.local.toml` or `Q38_SERVER_API_KEY` in the environment of
  `./start.sh`; clients then send `Authorization: Bearer <key>` for `/v1/*` (`/health` and `/metrics` stay open).
- Also: `POST /v1/completions` (text or token ids), `GET /v1/models` (with `max_model_len`), `GET /v1/models/{id}`,
  `POST /tokenize`, `POST /detokenize`, `GET /version`, `GET /health` (200 only when ready), `GET /metrics`
  (Prometheus, vLLM names for requests and speculative decoding).

## Checking a running server

```bash
python3 checks/server_check.py http://<head>:8888/v1                 # smoke, exact, reuse, needle
python3 checks/server_check.py http://<head>:8888/v1 --only needle --needle 32768,131072
python3 checks/needle.py http://<head>:8888/v1 --lens 128,512           # long-context recall, 30 records per length
python3 checks/vision.py http://<head>:8888/v1                          # images: reading, reuse, errors
```

`smoke` covers the API paths; `exact` runs four seeded prompts alone, together and again and expects identical text;
`reuse` expects ≥ 90% of each later turn's prompt to come from the cache; `needle` hides a fact at 10/50/90% of long
prompts and expects it back. `checks/needle.py` hides 30 records in two public-domain novels (downloaded once from
Project Gutenberg, checked by sha256) and asks each once; the first question of a length reads the whole text, the
others reuse it. `checks/vision.py` draws its own pictures (a number and a coloured square), expects the answer to
name both, a picture sent again to be reused from the cache and another picture not, and broken input to get a 400.

## Layout

- `engine/src` — the engine (C++/CUDA): `core` (weights, communication), `model` (the step, MTP, patches),
  `kernels/<family>` (linear, MoE, HC, GDN, QSA attention, sampling, RDMA all-reduce). Built inside the image at the
  first start of each machine (cache in `<work>/cache`).
- `server` — the HTTP front, chat template and parser, scheduler, the command link between the ranks, settings, and
  `control.py` (what `start.sh` / `stop.sh` run on the head); `vision.py` (image / video decoding, the prompt's media
  tokens and positions), `vision_tower.py` (the vision tower in PyTorch), `media.py` (the requests' media).
- `docker/dev.Dockerfile` — the image (NGC PyTorch base + the HTTP packages + Pillow / OpenCV for media), built by
  `./start.sh` on each machine.
- `tools/tensor_table.py` — the checkpoint's tensor table (made at the first start).

## License and credits

The model weights are under their own licenses: Qwen3.8-Flash-Next (Qwen) and the NVIDIA Open Model Agreement for the
`Qwen3.8-Flash-Next-FP4-Local` checkpoint — read and accept them on Hugging Face. The serving behaviour (chat
template handling, reasoning and tool-call parsing, API shape) follows vLLM; the configuration style follows existing
two-node DGX Spark recipes.
