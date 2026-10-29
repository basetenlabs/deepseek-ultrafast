# OpenAI-compatible server (`serve`)

`crates/server` wraps the engine in an HTTP server that speaks the OpenAI API.

**Required:** Redis (`DSUF_REDIS_URL`, default `redis://127.0.0.1:6379`; run it without
persistence: `redis-server --save '' --appendonly no`), an active RDMA port (every ACTIVE
port is used; `UCX_NET_DEVICES` restricts them), and an address peers can reach for KV transfer:
`DSUF_KV_HOST` (default: the machine's first network address, as `hostname -I` lists it) and
`DSUF_KV_PORT` (default 19500). Replicas share KV only with an identical model, tokenizer, `--max-seq`, GPU count
and decode numerics.
Generation requests accept the conversation ID as `X-Session-ID`, `Session-ID`, or the
`X-Claude-Code-Session-ID` / `X-DeepSeek-Harness-Session-ID` that Claude Code and DeepSeek Harness send. Multi-turn requests without one return HTTP 400; a headerless single-turn request gets a generated ID. Blank or
ambiguous IDs also return HTTP 400. JSON/SSE responses return the selected ID and its
source. `/health` and `/readyz` return 503 until the engine has loaded and from the start of
shutdown.
Redis holds the KV ownership directory: every lookup, claim and check is one atomic Lua
script, and ownership changes are published on a pub/sub channel. Replicas sharing a cache use
the same Redis. A Redis restart
loses cache hints, not model state: its new epoch fences every old ownership revision.
[deploy/README.md](../deploy/README.md) covers Docker, Kubernetes and the configuration.

```
cargo build --release -p deepseek-ultrafast-server
target/release/serve --ckpt /path/to/models --gpus 0,1,2,3,4,5,6,7 --port 8000 [--max-seq 1048576] [--chunk 8192] [--model-name deepseek-v4.1-flash]
```

The server is ready when `/readyz` returns 200 and the JSON `health_ready` event is
emitted. Weights load from safetensors through the default-on parallel loader; startup
uses bounded pinned staging with a pageable fallback.

All four generation APIs accept `deepseek-ai/DeepSeek-V4.1-Flash` and
`deepseek-ai/DeepSeek-V4.1-Flash-Fast` as names for the same loaded model.
`/v1/models` lists both aliases plus the configured `--model-name`, without
duplicates. Responses retain the configured model name (default
`deepseek-v4.1-flash`); switching request aliases changes neither tokenization
nor session/KV identity, weights, or generation behavior.

The default capacity is 1,048,576 tokens (prompt + output, less speculative headroom).
Use `--max-seq 131072` for a smaller allocation. Decode top-k keeps its 128K-sized
shared-memory cache and streams larger live rows, so a 1M allocation does not select
a slower kernel at 100K context. All four generation endpoints accept JSON bodies up to **64 MiB**, including escaped text,
conversation/tool history, and token-ID arrays. Axum's default 2 MiB limit is too small
for ordinary 800K-token requests. This wire-size limit is independent of `--max-seq`:
oversized bodies return 413; prompts/outputs exceeding the token capacity still return 400.

CED bounded SWA replay prefill is always enabled, with no runtime feature flag. It skips
full-prompt decoder work and reconstructs the decoder's 128-row window. Short prefix hits
reuse retained encoder outputs where available; decoder replay KV is never prefix state.

## Endpoints
- **`GET /health`, `GET /readyz`, `GET /v1/models`**
- **`POST /v1/responses`**: stateless JSON / named SSE via deepseek-recipe, including plaintext
  reasoning, function/namespace/custom tool calls and full-history continuations. Use
  `max_output_tokens`, `reasoning.effort`, and `store: false`; no response storage or
  `previous_response_id`.
- **`POST /v1/messages`**: stateless Anthropic-style JSON / named SSE, including thinking,
  client tools/results, images and full-history continuations. Requires `max_tokens`;
  no stored conversations or server tool execution. `context_management` is accepted only
  with edits that clear nothing (`clear_thinking_*` with `keep: "all"`, as Claude Code sends).
- **`POST /v1/chat/completions`**
  - Request conversion/validation, V4.1 prompt rendering, and streamed/full response parsing use [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe), pinned to `8cadfede7063c896b944e7bae05daa3549ae97ea` (the same revision as DeepSeek-V4.1-Flash). `src/chat.rs` is a compatibility adapter, not a second template/parser.
  - Thinking is on by default for ordinary requests. Native `thinking: {"type": "enabled" | "disabled"}` (or a boolean) takes precedence over `reasoning.enabled`, then `chat_template_args` / `chat_template_kwargs`, then effort/defaults. Template flags accept `enable_thinking` before `thinking`, including boolean, integer and common string forms such as `"false"` / `"off"`.
  - `reasoning_effort`: 1–100 or recipe's named levels (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; default high = 75). `none` disables thinking unless a higher-priority thinking control overrides it. Numeric effort remains a DeepSeek Ultrafast compatibility extension.
  - Output is split into `reasoning_content`, `content` and `tool_calls`. Recipe parses DSML tool calls into JSON-string arguments; streaming emits incremental tool-argument deltas, not whole buffered tool calls. Full responses accumulate the same deltas.
  - `tools` with `tool_choice` auto / none / required / a named function. Required/named choices use recipe's normal tool-call parsing.
  - Literal `<｜image｜>` and `<｜deepseek_image｜>` strings in text, reasoning, tool results, schemas and historical JSON arguments are preserved as ordinary text tokens, never image slots. Real image blocks are supported through recipe's resolver and the checkpoint vision tower; image limits follow the recipe and checkpoint vision configuration.
- **`POST /v1/completions`**: `prompt` as a string or token ids, streamed or not. `max_tokens` defaults to 16 here, as in OpenAI.
- **Common parameters:**
  - `temperature` and `top_p` are fixed at 1; other values are rejected. `seed` sets the
    random stream.
  - `stream` on every endpoint; `stream_options.include_usage` on Chat/Completions only.
  - `max_tokens` / `max_completion_tokens`, `stop` on Chat/Completions; Responses uses `max_output_tokens`.
    Messages requires native `max_tokens` and uses `stop_sequences`.
    Chat and Responses default to **256,000 output tokens**. Every output budget, default or explicit,
    is capped at 256,000 and at what the context has left (`--max-seq` minus the prompt and 64 tokens
    of speculative headroom), so any prompt that fits the context gets an answer; generation that
    reaches the cap stops with the `length` finish reason.
    Legacy Completions still defaults to 16.
- **Chat/Completions usage:** reports `prompt_tokens_details.cached_tokens`, the prefix that was reused, and the engine's generated token counts (including EOS). Chat streaming preserves the opt-in `stream_options.include_usage` usage-only event before `[DONE]`.
- **Rejected with 400 on every generation API, before prompt tokenization/session selection/enqueue:** a multi-turn request without a session ID header, enabled `logprobs`, non-null `top_logprobs`/`prompt_logprobs`, logprobs via `include`, structured-output controls, and tool-definition `strict: true`. Errors remain JSON even when `stream: true`. Omitted/null controls, `logprobs: false`, plain-text formats and ordinary tools remain supported. Applications must authorize tool execution and handle errors/incomplete calls. Validation inspects controls, not option names quoted in prompts, metadata or property names.
- **Also rejected with 400:** `n` other than 1, unsupported media/files, and prompts or `max_tokens` that exceed the context.

## Thinking template aliases

**Chat Completions, Responses and Messages** all accept either flag, independently
or together:

```json
"chat_template_args": {"thinking": false, "enable_thinking": false}
```

The same keys work in `chat_template_kwargs`; `true` enables thinking. Existing
loose boolean compatibility (integers and strings such as `"false"`/`"off"`) is
preserved. Within an object, `enable_thinking` wins over `thinking`; args take
precedence over kwargs. An invalid present `enable_thinking` does not fall back to
its sibling `thinking`.

Chat and Messages prioritize non-null native `thinking`, then `reasoning.enabled`, then
these aliases, then effort/defaults. Responses prioritizes an explicit non-null
native `reasoning.effort`; otherwise the aliases map to `high`/`none`. Both alias
containers are applied consistently across APIs so changed controls select the
intended rendered history. Raw `/v1/completions` does not apply a chat template:
its caller-supplied prompt/control tokens remain unchanged.
