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
```

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

Profiles live in `profiles.json` in the MCP state dir (same place as lease
state, honors `LUALLM_MCP_STATE_DIR`):

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

- `args` is a list of flag strings (or one string, split like a shell line).
- `preset` (optional) is a raw luallm preset (`throughput`, `cold-start`) the
  profile starts from; profile args are appended last and win flag conflicts.
- `description` is optional and shown by `list_profiles`.

The file hot-reloads on every `list_profiles` / `acquire_model` call, so
editing it does not require restarting the server. Broken entries are
reported back via the tools instead of silently ignored; `--preset` and
`--json` inside `args` are rejected (both are already managed by the server).

The MCP tool surface:

- `list_profiles` — enumerate available profiles (name, description, flags)
- `acquire_model` — gains a `profile` argument; on start it applies the
  profile's preset and flags and records the profile in the instance state,
  on reuse it will only attach to an instance started with the same profile

Rules that follow from luallm running at most one server per model:

- Switching configuration requires a stop: acquiring a preset/profile that
  differs from the running instance fails with `MODEL_RUNNING_DIFFERENT_CONFIG`.
  Release its leases (stops the model) or `force_stop_model`, then acquire.
- A profile/preset cannot be applied to a server started outside the MCP;
  that fails with `MODEL_RUNNING_EXTERNALLY` (same as preset today).

The plain `preset` argument still works for raw luallm presets, but a caller
must give at most one of `preset` / `profile` at a time.

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
| `get_capabilities()` | List MCP capabilities |
| `acquire_model(model_name, preset?, client_id?, purpose?)` | Acquire lease on model |
| `release_model(lease_id)` | Release lease |
| `get_model_logs(instance_id, lines=100)` | Get model logs |
| `health_check()` | Check luaLLM availability |
| `list_leases()` | List current leases |

## Safety Features

- **Allowlist only**: Only approved tools exposed
- **Lease-based lifecycle**: Models only stopped when all leases released
- **Instance isolation**: Different presets = different instances
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

Models are "leased" when started via MCP. Workflows acquire leases via `acquire_model()` and release them with `release_model(lease_id)`. A model is only stopped when the last lease is released.

### State Schema

```json
{
  "schema_version": "1.0",
  "instances": [
    {
      "instance_id": "uuid",
      "model": "mistral-7b",
      "preset": "throughput",
      "managed_by_mcp": true,
      "started_at": "2024-01-01T10:00:00Z",
      "port": 8080
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

`acquire_model()` only reuses an existing instance when both model and preset match exactly. Different presets create different instances.