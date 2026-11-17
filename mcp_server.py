"""MCP Server for luaLLM - safe model management via leases."""

import asyncio
import json
import logging
import os
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability

import core
from lua_client import LuaLLMClient
from lease_manager import LeaseManager
from profiles import ProfileRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("luaLLM-MCP")

# Shown to every MCP client at initialize time and echoed by
# get_capabilities. This is the only docs a connected LLM is guaranteed to
# see, so it must carry the whole usage contract — keep it current when
# tools or semantics change.
SERVER_INSTRUCTIONS = """\
luaLLM-MCP starts and stops local llama.cpp model servers via the `luallm` \
CLI, with lease-based lifecycle control. It does not proxy requests; the \
tools manage servers, and you then call the model over HTTP.

## Workflow

1. `list_models` — discover available model names.
2. `acquire_model(model_name, client_id?, purpose?)` — start (or reuse) a \
server and lease it. Blocks until the model is loaded and serving. Returns \
`lease_id`, `instance_id`, and `port` — keep the lease_id.
3. Use the model over HTTP with the agent's own web/fetch capability: \
`http://127.0.0.1:{port}` (llama.cpp OpenAI-compatible HTTP API, e.g. \
POST /v1/chat/completions). No MCP tool does inference for you.
4. `release_model(lease_id)` when done work.

## Rules

- Every tool response is JSON `{"success": bool, "data" or "error"}`; check \
`success` before touching `data`.
- Leases are refcounted: concurrent acquires share one instance. Releasing \
the last lease keeps the model warm for an idle grace period (default 5 min) \
so a quick re-acquire reuses it; after that it is stopped. Idle models are \
also stopped early, longest-idle first, when another model needs the memory. \
Unrenewed leases expire (default TTL 2 h).
- To start with tuned flags, `list_profiles` and pass a profile by name to \
`acquire_model` — never invent llama.cpp flags. `preset` and `profile` are \
mutually exclusive.
- luallm runs at most one server per model. Acquiring a different \
preset/profile than the running one fails with `MODEL_RUNNING_DIFFERENT_CONFIG`; \
release its leases (or `force_stop_model`), then re-acquire. A model that \
is only idle (no leases) is restarted under the new configuration for you.

## Error recovery

- `MODEL_NOT_READY` — still loading; model keeps loading in background. \
Acquire again shortly.
- `SERVER_EXITED` — model died during startup; `get_model_logs(instance_id)` \
for the cause (bad file, invalid flag).
- `MODEL_RUNNING_EXTERNALLY` — server was started outside the MCP; acquire \
without preset/profile to adopt it (release will not stop it).
- `force_stop_model` is the escape hatch for stuck/orphaned models.
- `list_leases` and `get_server_status` are read-only diagnostics."""

mcp_server = Server("luaLLM", instructions=SERVER_INSTRUCTIONS)
client = LuaLLMClient()
lease_manager = LeaseManager()
profile_registry = ProfileRegistry()



# The lifecycle lives in core.py, shared with the OpenAI gateway. These
# wrappers keep this module's names, and build the context from its globals
# at call time, so patching client, lease_manager, profile_registry,
# _probe_health, or _process_alive here still takes effect.

_error_detail = core.error_detail
_ready_timeout = core.ready_timeout
_parse_size = core.parse_size
_total_memory_bytes = core.total_memory_bytes
_memory_budget_bytes = core.memory_budget_bytes
_footprint = core.footprint
_probe_health = core.probe_health
_process_alive = core.process_alive


def _ctx() -> core.Ctx:
    return core.Ctx(client=client, leases=lease_manager, profiles=profile_registry,
                    probe=lambda port: _probe_health(port),
                    alive=lambda pid: _process_alive(pid))


async def _wait_until_ready(port: int | None, timeout_s: float,
                            interval_s: float = 1.0,
                            pid: int | None = None) -> tuple[str, float]:
    return await core.wait_until_ready(
        port, timeout_s, interval_s, pid,
        probe=lambda p: _probe_health(p), alive=lambda p: _process_alive(p))


def _find_running_server(model_name: str) -> dict | None:
    return core.find_running_server(_ctx(), model_name)


def _stop_idle_instance(instance) -> bool:
    return core.stop_idle_instance(_ctx(), instance)


def _make_room_for(model_name: str, model_sizes: dict[str, int | None]) -> list[str]:
    return core.make_room_for(_ctx(), model_name, model_sizes)


def _reap_idle() -> list[str]:
    return core.reap_idle(_ctx())


async def _idle_reaper_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(_reap_idle)
        except Exception:
            logger.exception("Idle reaper sweep failed")
        grace = lease_manager.idle_grace_seconds()
        await asyncio.sleep(min(15.0, max(1.0, grace)))


def _stop_all_idle() -> None:
    core.stop_all_idle(_ctx())


async def _handle_acquire(model_name: str | None, preset: str | None,
                          client_id: str | None, purpose: str | None,
                          profile_name: str | None = None) -> dict:
    return await core.acquire(_ctx(), model_name, preset, client_id, purpose,
                              profile_name=profile_name)


def _handle_release(lease_id: str | None) -> dict:
    return core.release(_ctx(), lease_id)


def _handle_force_stop(model_name: str | None) -> dict:
    return core.force_stop(_ctx(), model_name)


def _handle_list_profiles() -> dict:
    return core.list_profiles(_ctx())



@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_models",
            description="List all available models in the models directory. Call this first to discover model names.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_model_info",
            description="Get cached metadata for a model",
            inputSchema={
                "type": "object",
                "properties": {"model_name": {"type": "string", "description": "Model name"}},
                "required": ["model_name"],
            },
        ),
        Tool(
            name="get_server_status",
            description="Get running and stopped server instances",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_capabilities",
            description="Return server capabilities, the full usage guide (workflow, rules, error recovery), and the supported operations",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="acquire_model",
            description=(
                "Acquire a lease on a model instance. Starts the model if not running, "
                "blocks until it is READY to serve. Returns lease_id, instance_id, and "
                "the live OpenAI-compatible HTTP port — call the model at "
                "http://127.0.0.1:{port}. Same-model concurrent acquires share one "
                "instance; different presets/profiles cannot coexist for one model. "
                "Optional 'profile' selects an operator-curated launch configuration "
                "(see list_profiles); optional 'preset' is a raw luallm preset. "
                "Give at most one of the two."),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model name"},
                    "preset": {"type": "string", "description": "Raw luallm preset (throughput, cold-start)"},
                    "profile": {
                        "type": "string",
                        "description": "Named curated profile with preconfigured flags (use list_profiles)",
                    },
                    "client_id": {"type": "string", "description": "Client identifier for lease ownership"},
                    "purpose": {"type": "string", "description": "Purpose of the lease"},
                },
                "required": ["model_name"],
            },
        ),
        Tool(
            name="release_model",
            description=(
                "Release a lease by lease_id (from acquire_model). When this was "
                "its last lease the model stays warm for an idle grace period "
                "(default 5 min) so a re-acquire is instant, then stops. Running "
                "workflows must release each lease they acquire."),
            inputSchema={
                "type": "object",
                "properties": {"lease_id": {"type": "string", "description": "Lease ID to release"}},
                "required": ["lease_id"],
            },
        ),
        Tool(
            name="force_stop_model",
            description="Stop a model regardless of leases and purge its lease records. Escape hatch for orphaned models.",
            inputSchema={
                "type": "object",
                "properties": {"model_name": {"type": "string", "description": "Model name"}},
                "required": ["model_name"],
            },
        ),
        Tool(
            name="get_model_logs",
            description="Get the last N lines of a model instance's log file (instance_id from acquire_model). Use this when acquire fails with SERVER_EXITED or a request times out, to see recent llama.cpp activity (timing lines, errors, crashes).",
            inputSchema={
                "type": "object",
                "properties": {
                    "instance_id": {"type": "string", "description": "Instance ID"},
                    "lines": {"type": "integer", "default": 100, "description": "Max lines to return"},
                },
                "required": ["instance_id"],
            },
        ),
        Tool(
            name="health_check",
            description="Check if luaLLM is available and working",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="list_leases",
            description="List all current leases (read-only diagnostic)",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="list_profiles",
            description=(
                "List curated model launch profiles. Pass one by name as the "
                "'profile' argument of acquire_model to start a model with its "
                "preconfigured settings (e.g. image projection, context size)."),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    result: dict = {"success": True}
    
    if name == "list_models":
        resp = client.list_models()
        if resp["success"]:
            result["data"] = resp.get("data", [])
        else:
            result["success"] = False
            result["error"] = _error_detail(resp)
    
    elif name == "get_model_info":
        model_name = arguments.get("model_name")
        if not model_name:
            result["success"] = False
            result["error"] = "MISSING_MODEL_NAME"
        else:
            resp = client.get_model_info(model_name)
            if resp["success"]:
                result["data"] = resp.get("data", {})
            else:
                result["success"] = False
                result["error"] = _error_detail(resp)
    
    elif name == "get_server_status":
        resp = client.get_server_status()
        if resp["success"]:
            result["data"] = resp.get("data", {})
        else:
            result["success"] = False
            result["error"] = _error_detail(resp)
    
    elif name == "get_capabilities":
        result["data"] = {
            "tools": ["list_models", "get_model_info", "get_server_status",
                     "get_capabilities", "acquire_model", "release_model",
                     "force_stop_model", "get_model_logs", "health_check",
                     "list_leases", "list_profiles"],
            "version": "1.0",
            # Same guide sent in the initialize result — in-band fallback for
            # clients that do not surface server instructions to the model.
            "usage": SERVER_INSTRUCTIONS,
        }
    
    elif name == "acquire_model":
        result = await _handle_acquire(
            arguments.get("model_name"),
            arguments.get("preset"),
            arguments.get("client_id"),
            arguments.get("purpose"),
            profile_name=arguments.get("profile"),
        )
    
    elif name == "release_model":
        result = _handle_release(arguments.get("lease_id"))

    elif name == "force_stop_model":
        result = _handle_force_stop(arguments.get("model_name"))
    
    elif name == "get_model_logs":
        instance_id = arguments.get("instance_id")
        lines = arguments.get("lines", 100)
        
        if not instance_id:
            result["success"] = False
            result["error"] = "MISSING_INSTANCE_ID"
        else:
            state = lease_manager.load()
            instance = lease_manager.get_instance(instance_id)
            if instance:
                resp = client.get_logs(instance.model)
                if resp["success"]:
                    log_file = resp.get("data", {}).get("log_file", "")
                    content = ""
                    if log_file:
                        try:
                            # Read only the last 512KB so a long-lived log
                            # cannot bloat the response.
                            with open(log_file, "rb") as f:
                                f.seek(0, os.SEEK_END)
                                size = f.tell()
                                f.seek(max(0, size - 512 * 1024))
                                content = f.read().decode("utf-8", errors="replace")
                        except OSError as e:
                            result["success"] = False
                            result["error"] = f"LOG_READ_ERROR: {e}"
                    if result.get("success", True):
                        if lines:
                            content = "\n".join(content.splitlines()[-lines:])
                        result["data"] = {"logs": content, "log_file": log_file}
                else:
                    result["success"] = False
                    result["error"] = _error_detail(resp)
            else:
                result["success"] = False
                result["error"] = "INSTANCE_NOT_FOUND"
    
    elif name == "health_check":
        resp = client.health_check()
        result["data"] = {"healthy": resp["success"]}
    
    elif name == "list_leases":
        result["data"] = lease_manager.list_leases()

    elif name == "list_profiles":
        result = _handle_list_profiles()

    else:
        result["success"] = False
        result["error"] = f"UNKNOWN_TOOL: {name}"
    
    return [TextContent(type="text", text=json.dumps(result))]


async def main():
    state = lease_manager.load()
    
    expired = lease_manager.expire_stale_leases()
    if expired:
        logger.info(f"Expired {len(expired)} stale leases")
    
    status_resp = client.get_server_status()
    if status_resp["success"]:
        servers = status_resp.get("data", {}).get("servers", [])
        for srv in servers:
            if srv.get("state") == "running":
                model = srv.get("model")
                existing = lease_manager.find_instance(model)
                if existing and not existing.managed_by_mcp:
                    logger.warning(f"Externally managed server: {model}")
    
    logger.info("luaLLM-MCP server ready (stdio transport)")
    reaper = asyncio.create_task(_idle_reaper_loop())
    try:
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="luaLLM-MCP",
                    server_version="0.1.0",
                    instructions=SERVER_INSTRUCTIONS,
                    capabilities=ServerCapabilities(tools=ToolsCapability()),
                ),
            )
    finally:
        reaper.cancel()
        _stop_all_idle()


if __name__ == "__main__":
    asyncio.run(main())