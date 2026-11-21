# Using the gateway from Home Site

How Home Site can discover models and use them through the gateway, instead of
reaching llama.cpp on port 8080 itself. Nothing here changes Home Site: these
are notes for the change, which in Home Site needs an ADR replacing
[ADR 0024](../../HomeSite/docs/decisions/0024-ai-through-opencode.md) and its
own tests.

What was verified on 2026-11-17, with this gateway and real models on the
owner's Mac, is marked **verified**. Everything else is a reading of Home
Site's code, not a run.

## The address and the token

| | |
|---|---|
| Base URL, from a container | `http://host.docker.internal:8090/v1` (**verified** from an OrbStack container, while the gateway listens on `127.0.0.1` only; the Mac's LAN address is refused) |
| Base URL, on the Mac | `http://127.0.0.1:8090/v1` |
| Authentication | `Authorization: Bearer <token>`. The token is `~/.config/luaLLM-MCP/gateway.token` |

`host.docker.internal` is "local" under ADR 0024's rule for telling local from
cloud, so the gateway's models would count as local without any change there.

The token is a secret of the same kind as a provider key: it can start models
on the Mac. Keep it where Home Site keeps the engine's provider key today
(`engine.env`), never in a site, a log, or a link. To replace it, delete
`gateway.token`, run `make gateway-token`, and restart the gateway.

## Discovering models: `GET /v1/models`

```json
{
  "object": "list",
  "data": [{
    "id": "qwen",
    "object": "model",
    "owned_by": "luallm",
    "description": "General assistant",
    "default": true,
    "loaded": true,
    "capabilities": {
      "images": true,
      "context_tokens": 262144,
      "max_output_tokens": 4096,
      "reasoning": {
        "disabled_by_profile": false,
        "effort_levels": ["low", "medium", "xhigh"],
        "enable_thinking": true,
        "preserve_reasoning": true
      }
    },
    "capabilities_verified_at": "2026-11-17T16:52:05+00:00"
  }]
}
```

`id` is what Home Site stores as the model and sends as `model`. The owner
chooses the names in `gateway.json`; Home Site never sees model files, presets,
or flags.

`default` marks the one `default_model` in `gateway.json` names, at most one.
Home Site's applications choose a model for each request, from the list and
each one's `description`, and use this one when they don't. With no default, a
request that doesn't choose is refused rather than sent to another.

`capabilities` is what that model's own llama-server reported the last time it
loaded (**verified** for the owner's Qwen3.8 and Maginum). It is `null` until
the model has loaded once under its current settings. Home Site's model
metadata maps onto it:

| Home Site today (from OpenCode's `opencode.json`) | From the gateway |
|---|---|
| `modalities.input` includes `image` | `capabilities.images` is `true` |
| `limit.context` | `capabilities.context_tokens` |
| `limit.output` | `capabilities.max_output_tokens` (may be `null`: no cap but the context) |
| A thinking variant, sent as `reasoningEffort` | A value in `reasoning.effort_levels` |
| The `none` variant, `chat_template_kwargs.enable_thinking: false` | Allowed when `reasoning.enable_thinking` is `true`, or `disabled_by_profile` is `true` (already off) |

Unknown is `null`, not `false`. Today Home Site refuses a capability that isn't
stated (ADR 0024: OpenCode silently dropped images); the same rule would treat
`null` as "not available yet". The gateway enforces it either way: it refuses
images or reasoning controls it hasn't verified.

`loaded` says whether a request now would find the model warm. A cold model
can take tens of seconds to load (Maginum-24B, 18 GB: 7–8 s from cold here,
1 s with the file cached).

## Sending requests: `POST /v1/chat/completions`

The body the relay sends for a conversation today (`relay.py`, `converse`) is
made of fields the gateway accepts: `model`, `messages` with string content,
`stream: true`, `stream_options.include_usage`, `max_completion_tokens`, and
either `reasoning_effort` or `chat_template_kwargs.enable_thinking`. Images,
for one-shot generation, go as `image_url` parts with base64 `data:` URIs,
which the relay already checks. The full list, and what's refused, is in the
README under "What a request may contain".

Differences from talking to llama.cpp directly:

- **`max_completion_tokens` must fit.** More than `max_output_tokens`, or than
  the context, is `400`, not truncated. Take the limit from `/v1/models`.
- **Reasoning levels are checked before the model sees them.** The owner's
  Qwen accepts `low`, `medium`, `xhigh` and raises an error for `high`
  (Home Site's ADR 0028). The gateway answers `high` with `400
  capability_missing` and a message listing the levels (**verified**), rather
  than llama.cpp's template error. Its message mentions "reasoning_effort", so
  `direct.mjs`'s check for /reasoning|thinking/ still classifies it.
- **`enable_thinking: false` works** where the template reads it: the
  **verified** reply had no `reasoning_content`, and the same prompt with
  `reasoning_effort: "low"` had some.
- **Images:** **verified**: a red PNG to Qwen3.8 answered "red".
- **`model` in replies** is the gateway's name, not the model file's path.

### Cold starts while streaming

A streaming request to a model that isn't loaded gets `200` at once and SSE
comment lines, `: loading model`, every 10 seconds until the model is ready
(**verified** with a real cold start). The relay reads only lines starting with
`data:` (`Conversation._lines`), so it skips them, and `direct.mjs` passes the
bytes through. If loading fails after that, the stream ends with one
`data: {"error": {...}}` event and no `[DONE]`; the relay would need to
recognise that event as an error.

A non-streaming request waits for the load, so Home Site's timeouts should
allow for a first load.

### Cancelling

Closing the connection stops the generation and frees the model: **verified**
with real llama.cpp, whose `/slots` showed it idle 0.1 s after a stream was
closed, and 0.5 s after a non-streaming request was. `direct.mjs` already
aborts its upstream request when the relay's connection closes, and the relay
closes its connection when a conversation is stopped, so this works through
both.

### Errors

All errors use OpenAI's shape: `{"error": {"message", "type", "code", "param"}}`.

| Status, code | Meaning | For Home Site |
|---|---|---|
| `400` `invalid_request`, `unsupported_parameter`, `capability_missing` | The request asked for something not allowed or not available | A bug, or a capability the model lacks |
| `401` `invalid_api_key` | Token missing or wrong | Configuration |
| `404` `model_not_found` | Not a name in `gateway.json` | Like `ai_model_unavailable` today |
| `503` `model_busy`, with `Retry-After: 30` | The model is in use under another name's profile | Try later; there's no queue |
| `503` `model_running_externally`, `model_unavailable`, `luallm_unavailable` | The owner's setup needs attention | Like `model_unreachable` |
| `502` `model_start_failed`, `model_crashed`, `upstream_unreachable` | The model couldn't start or answer | Like `model_unreachable` |
| `504` `model_load_timeout` | Loading took longer than the gateway waits | Try later |
| others, `upstream_error` | llama-server's own error, with its status | As today |

The gateway's own messages don't quote requests. An `upstream_error` carries
llama-server's message, as a direct connection would; `direct.mjs` already
keeps provider words from reaching applications.

## Two ways to adopt it

**Keep OpenCode, point it at the gateway.** In the engine's `opencode.json`,
set the `llama.cpp` provider's `baseURL` to the gateway and give it the token as
its `apiKey`, with model keys matching the gateway's names. This is a
configuration change only. **Not verified:** OpenCode's own one-shot requests
might include fields the gateway refuses (it refuses anything it doesn't
know), so run Home Site's `make test-ai-engine` against the gateway first. The
`variants` in `opencode.json` would still need to match `effort_levels` by
hand.

**Drop OpenCode.** The relay already sends conversations as plain Chat
Completions through `direct.mjs`; one-shot generation could go the same way,
straight to the gateway, and the model list would come from `/v1/models`. That
removes what ADR 0024 contains OpenCode for: the deny-all ruleset, the tool-call
checks, polling, session cleanup, and the engine password that's worth a shell.
It's the larger change, and the one recommended in the discussion that led to
the gateway.

Either way, the model stays the administrator's choice in Home Site: the
gateway only decides which names exist.
