# luaLLM-MCP

MCP Server for luaLLM - safe model management via leases.

## Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the MCP server
make install
```

## Configuration

Runtime state is stored in OS-appropriate application data directories:

- **Linux/macOS**: `~/.config/luaLLM-MCP/`
- **Windows**: `%LOCALAPPDATA%\luaLLM-MCP\`

Override with:
```bash
export LUALLM_MCP_STATE_DIR=/custom/path
export LUALLM_MCP_LEASE_TTL_SECONDS=7200      # 2 hours (default)
export LUALLM_MCP_READY_TIMEOUT_SECONDS=600   # acquire waits this long for model load
export LUALLM_MCP_IDLE_GRACE_SECONDS=300      # keep a released model warm this long (0 = stop at once)
export LUALLM_MCP_MEMORY_BUDGET_GB=96         # memory models may use (default: 75% of RAM)
export LUALLM_MCP_MEMORY_OVERHEAD=1.2         # loaded footprint ≈ GGUF size × this
```

### Idle grace period and memory pressure

Releasing a model's last lease does not stop it right away. It stays
running, idle, for `LUALLM_MCP_IDLE_GRACE_SECONDS` (5 minutes by default),
so a workflow that releases and re-acquires the same model gets the warm
server back instead of a stop plus a cold load. A background sweep stops
models whose grace period has lapsed; leases dropped by the TTL are treated
the same way. Idle models are also stopped when the MCP server shuts down.

Before starting a model, the MCP estimates whether it fits:
GGUF size × `LUALLM_MCP_MEMORY_OVERHEAD`, against the memory budget minus
every running server (leased, idle, or external). If it doesn't fit, idle
models are stopped longest-idle first until it does; the stopped models are
listed in the acquire response as `evicted_idle`. Leased and externally
started servers are never evicted. If the model still won't fit after every
idle model is gone, the start goes ahead anyway with a warning in the log,
because the estimate is a heuristic.

## Usage

```bash
# Run the MCP server
make run

# Or directly with uv
uv run python mcp_server.py
```

The server uses stdio transport for MCP communication. It will print startup info and then wait for requests from an MCP client.

## Profiles

Profiles are operator-curated launch configurations: named bundles of extra
llama.cpp flags (e.g. image projection, a larger context window, a generation
cap). MCP clients can only apply a profile **by name** — they never supply
raw flags — so everything that reaches llama-server is something you vetted
in advance.

### Setting up profiles

**1. Create the file.** Profiles live in `profiles.json` in the MCP state
dir, next to the lease state file. Create the file if it does not exist:

- **Linux/macOS**: `~/.config/luaLLM-MCP/profiles.json`
- **Windows**: `%LOCALAPPDATA%\luaLLM-MCP\profiles.json`
- If you run the server with `LUALLM_MCP_STATE_DIR` set, profiles.json lives
  in that directory instead.

**2. Define your profiles.** Keys are the profile names your MCP clients
will pass; each entry holds the flags that get appended to
`luallm start <model>`:

```json
{
  "profiles": {
    "vision": {
      "description": "Enable image projection for multimodal input",
      "args": ["--mmproj", "~/dev/localGGUF/Qwen3.8-27B-mmproj-BF16.gguf"]
    },
    "big-context": {
      "description": "32k context window on top of the throughput preset",
      "preset": "throughput",
      "args": ["-c", "32768"]
    },
    "short-gen": {
      "description": "Cap generation at 512 tokens",
      "args": "-n 512"
    }
  }
}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `args` | yes | Extra llama.cpp flags: a list of strings, or one string that is split like a shell line (`"-c 32768"`). |
| `preset` | no | A raw luallm preset (`throughput`, `cold-start`) the profile starts from. Profile `args` are appended last and win flag conflicts (last flag wins in llama.cpp). |
| `description` | no | Shown to MCP clients by `list_profiles`; describe *what* the profile does. |

Flag syntax is whatever your llama.cpp build supports (`llama-server
--help`). Common ones: `-c N` context, `-n N` max predicted tokens,
`--mmproj <file.gguf>` image projection, `-fa` flash attention.

**3. Verify.** The file hot-reloads on every profile tool call — no server
restart needed. With the server running, check what clients will see:

```bash
uv run python -c "
from mcp_server import _handle_list_profiles
import json; print(json.dumps(_handle_list_profiles(), indent=2))"
```

You should see your profiles under `data.profiles`. If the file has
problems (bad JSON, misspelled keys, invalid values) they are returned in
`data.warnings` and the broken entries are skipped — nothing fails
silently. `--preset` and `--json` inside `args` are rejected outright;
both are already managed by the server.

### How clients use profiles

- `list_profiles` — enumerate available profiles (name, description, flags)
- `acquire_model(model_name, profile="big-context", ...)` — starts the
  model with the profile's preset and flags, or reuses an instance that was
  started with the *same* profile. The response includes `profile` and
  `args` so the caller can see which configuration a lease is attached to.

```json
{ "success": true, "data": {
    "lease_id": "…", "instance_id": "…", "port": 8082,
    "model": "Qwen3.6-27B-Q6_K", "preset": null,
    "profile": "small-ctx", "args": ["-c", "4096"],
    "reused": false, "ready_after_s": 11.0 } }
```

### Rules and errors

luallm runs at most one server per model, so:

- **Switching configuration requires a stop.** Acquiring a preset/profile
  that differs from the running instance fails with
  `MODEL_RUNNING_DIFFERENT_CONFIG` while the instance holds leases. Release
  them or `force_stop_model`, then acquire again. If the instance is only
  idle (no leases, inside its grace period), it is stopped and restarted
  under the requested configuration automatically.
- **Profile/preset cannot be applied to a server started outside the MCP**
  (e.g. from a terminal) — that fails with `MODEL_RUNNING_EXTERNALLY`.
- If the model **crashes during startup** (bad model file, invalid flag —
  a mmproj-only CLIP file used as a main model, for example) the acquire
  fails fast with `SERVER_EXITED` instead of waiting out the load timeout,
  and points at `luallm logs <model>`.

The plain `preset` argument still works for raw luallm presets, but a caller
must give at most one of `preset` / `profile` at a time
(`PRESET_AND_PROFILE_EXCLUSIVE` otherwise).

> **Note on image projection:** `--mmproj` is supplemental — it must point
> at a CLIP/mmproj GGUF file and be paired with a real text model. A
> mmproj-only file cannot itself be started as a model, and only pairs with
> architectures whose projection tensors it was exported for.

## OpenAI gateway

`gateway.py` puts one OpenAI-compatible address in front of your models, for
programs that want to *use* a model rather than manage one: Home Site, Pi,
curl, any OpenAI SDK. It runs the same lifecycle as the MCP tools, sharing
their state, leases, profiles, and idle grace period.

```
GET  /health                no token; says only that the gateway is up
GET  /v1/models             the names you expose, with verified capabilities
POST /v1/chat/completions   one request, on one lease
```

A chat request names one of *your* names as `model`. The gateway takes a
lease, starts the model if it isn't running (or reuses it), waits until it's
loaded, sends the checked request to that model's llama-server, and streams
the reply back. The lease is released when the reply ends, fails, or the
client disconnects. Nothing else is offered over HTTP: no start, stop, logs,
or profiles.

### Setting it up

**1. Choose what to expose.** Create `gateway.json` in the state dir, next to
`profiles.json`:

```json
{
  "listen": "127.0.0.1:8090",
  "models": {
    "qwen":   { "model": "Qwen3.8-27B-Uncensored-Q6_K", "profile": "director",
                "max_output_tokens": 4096, "description": "General assistant" },
    "writer": { "model": "Skyfall-31B-v4.2-Q8_0", "profile": "writer" }
  }
}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `models.<name>` | yes | The name clients use: letters, digits, `. _ : -`. Only these are served |
| `model` | yes | A luallm model name (`luallm list`). Clients never see it |
| `profile` | no | A profile from `profiles.json`. Its flags are applied when the gateway starts the model |
| `max_output_tokens` | no | Longest reply allowed, and the default when a request doesn't say |
| `description` | no | Shown in `/v1/models` |
| `listen` | no | `host:port`, default `127.0.0.1:8090`. Loopback only unless you also set `"allow_non_loopback": true` |
| `max_request_mb` | no | Largest request body, default 32 |

The same model can appear under several names with different profiles. luallm
runs one server per model, so only one of them can be loaded at a time (see
below).

**2. Make a token.** `make gateway-token` writes a random token to
`gateway.token` in the state dir, readable only by you. Or set
`LUALLM_GATEWAY_TOKEN` (32 characters or more). Clients send it as
`Authorization: Bearer <token>`.

**3. Check, then run.**

```bash
make gateway-check   # parses gateway.json and the token, lists the names
make gateway         # serves until Ctrl-C
```

Stopping it (Ctrl-C or SIGTERM) ends requests in flight, releases their
leases, and stops idle models, as the MCP server does when it exits.

**Restart running MCP clients after upgrading.** Every process that shares
the state file must be on this version: older ones don't take the locks
described under [Architecture](#cross-process-safety).

### Using it

```bash
TOKEN=$(cat ~/.config/luaLLM-MCP/gateway.token)
curl -s localhost:8090/v1/models -H "Authorization: Bearer $TOKEN"
curl -s localhost:8090/v1/chat/completions -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

From a container on this Mac (OrbStack or Docker Desktop), the address is
`http://host.docker.internal:8090/v1`. Verified with OrbStack: the gateway
listening on `127.0.0.1` is reachable that way, and not from the Mac's LAN
address. Any container on the Mac can reach it, so the token is what keeps
others out.

### What a request may contain

The gateway builds the request it sends llama-server from these fields only,
each checked. Anything else is refused with `400` and names the field; it is
never forwarded or silently dropped.

| Field | Notes |
|-------|-------|
| `model`, `messages` | Roles `system`, `user`, `assistant`. Content is a string or `text` / `image_url` parts |
| `image_url` | Only base64 `data:image/...` URIs, never web addresses (llama-server would fetch them), and only for a model verified to take images |
| `stream`, `stream_options.include_usage` | |
| `max_tokens` / `max_completion_tokens` | No more than `max_output_tokens` or the model's context |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `seed`, `stop`, `n` (1 only) | Range-checked |
| `response_format` | `text`, `json_object`, or `json_schema` with a `schema` |
| `reasoning_effort` | Only a level the model's chat template accepts |
| `chat_template_kwargs` | Only `enable_thinking`, and only where the template reads it |
| `reasoning_content` on assistant messages | Only where the template preserves reasoning |

Refused: tools and tool messages, `logprobs`, `logit_bias`, `user`, and
llama.cpp's own per-request controls (`id_slot`, `cache_prompt`, `lora`,
`grammar`, `samplers`, `n_probs`, ...), which would let a client reach past
the gateway: other requests' slots, adapters, or saved state.

### Capabilities are verified, not assumed

After a model loads, the gateway reads its llama-server's `/props` and takes
from it:

- **images:** `modalities.vision`;
- **context:** `n_ctx`;
- **reasoning:** the template's `reasoning_effort_levels`, whether it reads
  `enable_thinking`, and whether it supports preserved reasoning.

A profile whose flags turn thinking off (`--reasoning off`,
`--reasoning-budget 0`) makes reasoning controls unavailable, whatever the
template says. These are shown in `/v1/models` once verified, with
`capabilities_verified_at`; `null` means not yet. Before a model's first load
nothing is verified, so a request needing images or reasoning controls waits
for the load and is then accepted or refused; plain requests always go
through. Verified results are kept per name and forgotten when its model or
profile flags change.

What the gateway *can't* see: flags luallm adds on its own (`model_overrides`,
`thinking` budgets in luallm's config). They take effect, and anything they
change that `/props` reports (vision from an `--mmproj` override, context) is
seen; anything it doesn't report (a thinking budget) isn't.

### Behaviour worth knowing

- **Cold starts.** A streaming request whose model is still loading gets its
  `200` and SSE comment lines (`: loading model`) every 10 s
  (`LUALLM_GATEWAY_KEEPALIVE_SECONDS`) until the model is ready, so clients and
  proxies don't time out. SSE parsers ignore comment lines. An error after that
  point arrives as a final `data: {"error": ...}` event instead of a status
  code. A non-streaming request just waits. Either way, the wait is bounded by
  `LUALLM_MCP_READY_TIMEOUT_SECONDS`.
- **Disconnects.** A client that goes away cancels its request: the gateway
  closes its connection to llama-server, which stops generating, and releases
  the lease. Measured with a real model: generation stopped 0.1 s after the
  client closed a stream, and 0.5 s after it closed a non-streaming request.
  A disconnect while the model is loading releases the lease and leaves the
  model loading, idle, for the next request.
- **Another configuration is running.** If the model is in use under another
  name's profile, the request gets `503 model_busy` with `Retry-After: 30`. No
  queue. If it's only idle, it's restarted under the requested profile.
- **Editing configuration.** `gateway.json` and `profiles.json` are read again
  for each request; a request already running keeps the model and flags it
  started with, and nothing is stopped because of an edit. If a profile's flags
  change while its model is in use, new requests get `503 model_busy` until
  it's released, then it restarts with the new flags. `listen` is read only at
  startup.
- **Servers started outside luallm-MCP** (for example `luallm run` in a
  terminal) are used, never stopped, by a name *without* a profile; a name with
  a profile gets `503 model_running_externally`.
- **A crashed gateway** leaves its leases behind; they're tied to its process
  and dropped by the next sweep of any luaLLM-MCP process, or when the gateway
  starts again.
- **Errors** use OpenAI's shape: `{"error": {"message", "type", "code", "param"}}`.
  Lifecycle failures are `503 luallm_unavailable` / `model_unavailable` /
  `model_busy` / `model_running_externally`, `502 model_start_failed` /
  `model_crashed` / `upstream_unreachable`, and `504 model_load_timeout`, with
  generic messages; details that name files or luallm's output stay in the
  gateway's log. An error from llama-server itself is passed on with its status
  and message.
- **Logs** have one line per request: an ID, the name, streaming, cold start,
  time to ready, status, and duration. Never prompts, replies, images, or the
  token.

For Home Site specifically, see [docs/homesite.md](docs/homesite.md).

## Manual Testing

Test that the server initializes correctly:

```bash
uv run python -c "
import asyncio
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp import ClientSession

async def test():
    params = StdioServerParameters(command='uv', args=['run', 'python', 'mcp_server.py'])
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f'Found {len(tools.tools)} tools:')
            for t in tools.tools:
                print(f'  - {t.name}')

asyncio.run(test())
"
```

Expected output:
```
Found 11 tools:
  - list_models
  - get_model_info
  - get_server_status
  - get_capabilities
  - acquire_model
  - release_model
  - force_stop_model
  - get_model_logs
  - health_check
  - list_leases
  - list_profiles
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `list_models()` | List all available models |
| `get_model_info(model_name)` | Get model metadata |
| `get_server_status()` | Get running servers |
| `get_capabilities()` | Server capabilities plus the full LLM usage guide |
| `acquire_model(model_name, preset?, profile?, client_id?, purpose?)` | Acquire lease on model; optional curated profile or raw luallm preset |
| `release_model(lease_id)` | Release lease |
| `force_stop_model(model_name)` | Stop a model regardless of leases; escape hatch for orphaned models |
| `get_model_logs(instance_id, lines=100)` | Get model logs |
| `health_check()` | Check luaLLM availability |
| `list_leases()` | List current leases |
| `list_profiles()` | List curated launch profiles |

## How LLMs learn to use the server

MCP clients only see what the server sends them: the tool list and the
handshake instructions. This server puts its whole usage contract (workflow,
lease rules, error recovery) in that handshake, so connected LLMs never need
to read the source:

- **Handshake instructions** — `SERVER_INSTRUCTIONS` in `mcp_server.py`,
  sent in the initialize result and picked up by clients like Claude
  Desktop/Code. Most clients feed this to the model at session start.
- **`get_capabilities`** — echoes the same `SERVER_INSTRUCTIONS` text as a
  `usage` field, the in-band fallback for clients that don't surface
  handshake instructions.

Keep `SERVER_INSTRUCTIONS` current when tools or semantics change; it is the
only documentation a connected LLM is guaranteed to see.

## Safety Features

- **Allowlist only**: Only approved tools exposed
- **Lease-based lifecycle**: Models only stopped when all leases released (after an idle grace period)
- **Instance isolation**: Different presets and profiles = different instances
- **Vetted flags only**: MCP clients never pass raw flags; only operator-defined profiles
- **Atomic state writes**: State file updated safely
- **Cross-process locking**: MCP processes and the gateway can't lose each
  other's leases, start a model twice, or stop one another process has leased
- **Gateway allowlist**: only names in `gateway.json`, behind a bearer token,
  with requests rebuilt from checked fields
- **Structured errors**: All responses are JSON with success/error fields

## Development

```bash
make check   # Verify imports work
make test    # Run tests
make clean   # Clean build artifacts
```

Tests that need real processes use `tests/fakes/`: a fake `luallm` executable
that starts real fake llama-servers on free ports, and records every start and
stop. `test_concurrency.py` runs several processes against one state dir;
`test_gateway.py` runs the gateway under uvicorn and disconnects on real
sockets. None of them touch your models or your state dir.

Each safety control was removed once to watch its tests fail: the state lock
(150 leases from 6 processes became 29), the lifecycle lock (5 concurrent
acquires started the model 5 times; a release stopped a leased server),
cancelling on disconnect, releasing leases, releasing a lease taken by a start
that was cancelled, the token check, rebuilding the upstream request, and the
orphaned-lease sweeps. Two have a second line of defence, so removing one alone
doesn't fail a test: closing the upstream response (httpx also closes a
connection whose request is cancelled), and the gateway's startup sweep (its
reaper's first pass also runs it). Removing both sweeps does fail.

## Architecture

### Lease Model

Models are "leased" when started via MCP. Workflows acquire leases via `acquire_model()` and release them with `release_model(lease_id)`. A model is only stopped once the last lease is released and its idle grace period has lapsed, or earlier if another model needs its memory.

### State Schema

```json
{
  "schema_version": "1.0",
  "instances": [
    {
      "instance_id": "uuid",
      "model": "mistral-7b",
      "preset": "throughput",
      "profile": "big-context",
      "managed_by_mcp": true,
      "started_at": "2024-01-01T10:00:00Z",
      "port": 8080,
      "idle_since": null,
      "args": ["-c", "32768"]
    }
  ],
  "leases": [
    {
      "lease_id": "uuid",
      "instance_id": "uuid",
      "client_id": "claude-3-5-sonnet",
      "purpose": "chat-completion",
      "acquired_at": "2024-01-01T10:01:00Z",
      "renewed_at": "2024-01-01T10:30:00Z",
      "owner_pid": null
    }
  ]
}
```

`args` are the flags a managed server was started with (`null` when unknown,
for adopted servers and older records). `owner_pid` ties a lease to a live
process: the gateway sets it for each request's lease, and a sweep drops it if
that process is gone or the lease goes unrenewed past the TTL. MCP leases
leave it `null` and outlive the stdio process that took them, as before.
Fields this version doesn't know are ignored when reading, so an older and a
newer version can share the file.

### Cross-process safety

Each stdio MCP client runs its own server process, and the gateway is another,
all sharing `state.json`. Two `flock` locks in the state dir coordinate them:

- `state.lock`: each read-modify-write of `state.json` reloads it under the
  lock, so no process saves over another's change.
- `lifecycle.lock`: deciding whether to start, reuse, swap, or stop a model,
  and doing it, happens under this lock: acquire's decision, the release that
  stops a model, eviction, idle reaping, and force stop. It's held across
  `luallm start` and `luallm stop`, which return quickly, but not while a model
  loads: once its record and lease exist, other acquirers reuse it and wait too.

The logic itself is in `core.py`, shared by `mcp_server.py` and `gateway.py`.

### Instance Matching

`acquire_model()` only reuses an existing instance when model, preset, and
profile all match exactly. Different presets or profiles create different
instances.