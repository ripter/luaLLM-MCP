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
- **Structured errors**: All responses are JSON with success/error fields

## Development

```bash
make check   # Verify imports work
make test    # Run tests
make clean   # Clean build artifacts
```

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
      "idle_since": null
    }
  ],
  "leases": [
    {
      "lease_id": "uuid",
      "instance_id": "uuid",
      "client_id": "claude-3-5-sonnet",
      "purpose": "chat-completion",
      "acquired_at": "2024-01-01T10:01:00Z",
      "renewed_at": "2024-01-01T10:30:00Z"
    }
  ]
}
```

### Instance Matching

`acquire_model()` only reuses an existing instance when model, preset, and
profile all match exactly. Different presets or profiles create different
instances.