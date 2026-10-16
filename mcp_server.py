"""MCP Server for luaLLM - safe model management via leases."""

import asyncio
import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability

from lua_client import LuaLLMClient
from lease_manager import LeaseManager
from profiles import ProfileRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("luaLLM-MCP")

mcp_server = Server("luaLLM")
client = LuaLLMClient()
lease_manager = LeaseManager()
profile_registry = ProfileRegistry()


def _error_detail(resp: dict) -> str:
    """Best-available error text from a lua_client response.

    luallm reports errors inconsistently (stderr, non-JSON stdout, or JSON
    on stdout with a non-zero exit), so combine everything present instead
    of returning an empty string.
    """
    data = resp.get("data")
    if isinstance(data, dict) and data.get("message"):
        # luallm --json failures are structured: {"error": code, "message": text}
        code = data.get("error")
        message = str(data["message"])
        return f"{code}: {message}" if code and code != message else message
    parts = []
    error = resp.get("error")
    if error:
        parts.append(str(error))
    stderr = (resp.get("stderr") or "").strip()
    if stderr:
        parts.append(f"stderr: {stderr[:500]}")
    stdout = (resp.get("stdout") or "").strip()
    if stdout:
        parts.append(f"stdout: {stdout[:500]}")
    if not parts and data:
        parts.append(data if isinstance(data, str) else json.dumps(data))
    if not parts:
        returncode = resp.get("returncode")
        parts.append(
            f"luallm exited with code {returncode} and no output"
            if returncode is not None
            else "UNKNOWN_ERROR"
        )
    return " | ".join(parts)


def _ready_timeout() -> float:
    """How long acquire_model waits for the model to finish loading."""
    return float(os.environ.get("LUALLM_MCP_READY_TIMEOUT_SECONDS", "600"))


def _process_alive(pid: int | None) -> bool:
    """True if a process with this pid is running (best-effort, no pid = yes)."""
    if pid is None:
        return True
    try:
        result = subprocess.run(["ps", "-p", str(pid)],
                                capture_output=True, text=True,
                                check=False)
        return result.returncode == 0
    except (OSError, ValueError):
        return True


def _probe_health(port: int) -> int:
    """Health state of the model server on port: 2 ready, 1 loading, 0 refused.

    llama.cpp answers 200 on /health once loaded and 503 while loading
    (urllib surfaces that as HTTPError — still proof the server is up).
    A refusal means the port is closed: normal for the first moment after
    launch, fatal once the server has ever accepted a connection, and
    inconclusive otherwise (pid-liveness resolves it via _wait_until_ready).
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            return 2 if resp.status == 200 else 1
    except urllib.error.HTTPError:
        return 1
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0


async def _wait_until_ready(port: int | None, timeout_s: float,
                            interval_s: float = 1.0,
                            pid: int | None = None) -> tuple[str, float]:
    """Poll the model server until it is ready to serve requests.

    A lease means "ready to use": llama.cpp accepts connections while the
    model is still loading, so acquire must wait for /health to go green.
    Returns ("ready" | "timeout" | "crashed", seconds waited).

    "crashed" fires when the server is provably gone: a port that had
    accepted connections starts refusing them, or a known-daemon pid
    disappears while the port keeps refusing. Either means the process
    died during load and will never become ready — most often a broken
    model file or an invalid flag from a profile; waiting out the full
    timeout would just stall the caller. "timeout" keeps the instance
    record because a slow load is still in progress; "crashed" callers
    must drop it. No port means nothing to probe.
    """
    if port is None:
        return "ready", 0.0
    loop = asyncio.get_event_loop()
    started = loop.time()
    deadline = started + timeout_s
    ever_accepted = False
    while True:
        state = _probe_health(port)
        if state == 2:
            return "ready", loop.time() - started
        if state == 1:
            ever_accepted = True
        else:
            if ever_accepted:
                logger.error(f"Model server on port {port} stopped accepting "
                             f"connections — the process died during startup.")
                return "crashed", loop.time() - started
            if pid is not None and not _process_alive(pid):
                logger.error(f"Model server process {pid} is gone while port "
                             f"{port} refuses connections — died before the "
                             f"listener opened.")
                return "crashed", loop.time() - started
        if loop.time() >= deadline:
            return "timeout", loop.time() - started
        await asyncio.sleep(interval_s)


def _find_running_server(model_name: str) -> dict | None:
    """Look up a running luallm server for the model, or None."""
    status_resp = client.get_server_status()
    if not status_resp["success"]:
        return None
    for srv in status_resp.get("data", {}).get("servers", []):
        if srv.get("model") == model_name and srv.get("state") == "running":
            return srv
    return None


async def _handle_acquire(model_name: str | None, preset: str | None,
                          client_id: str | None, purpose: str | None,
                          profile_name: str | None = None) -> dict:
    """acquire_model: reuse an MCP-managed instance, adopt an externally
    started one (never stopped on release), or start the model — then wait
    until the model is actually ready to serve before returning the lease.

    A profile (profiles.json) bundles curated extra flag arguments; clients
    can only select them by name, and a profile may itself reference a
    luallm preset. preset and profile are exclusive on the tool contract.
    """
    profile = None
    if profile_name:
        if preset:
            return {
                "success": False,
                "error": ("PRESET_AND_PROFILE_EXCLUSIVE: give preset or profile, "
                          "not both — a profile already specifies its own luallm "
                          "preset via profiles.json."),
            }
        profile = profile_registry.get(profile_name)
        if profile is None:
            profiles, errors = profile_registry.load()
            msg = (f"PROFILE_NOT_FOUND: '{profile_name}' is not a defined "
                   f"profile (from {profile_registry.path}).")
            if errors:
                msg += " Profiles file problems: " + "; ".join(errors)
            elif profiles:
                msg += " Available profiles: " + ", ".join(p.name for p in profiles)
            else:
                msg += " No profiles are defined yet — create them in " \
                       f"{profile_registry.path} (see README)."
            return {"success": False, "error": msg}

    if not model_name:
        return {"success": False, "error": "MISSING_MODEL_NAME"}

    models_resp = client.list_models()
    if not models_resp["success"]:
        return {"success": False, "error": "CANNOT_LIST_MODELS"}
    if model_name not in [m.get("name") for m in models_resp.get("data", [])]:
        return {"success": False, "error": "MODEL_NOT_FOUND"}

    # The luallm preset actually applied: the caller's, or the one the
    # profile bundles with its extra flags.
    effective_preset = profile.preset if profile else preset

    instance = lease_manager.find_instance(model_name, effective_preset, profile_name)
    running = _find_running_server(model_name)
    lease = None
    reused = True
    pid = None

    if instance and running is None:
        # Stale record: the server behind it is gone (stopped or crashed
        # outside MCP). Drop it and start fresh below.
        logger.info(f"Dropping stale instance record for {model_name}")
        lease_manager.remove_instance(instance.instance_id)
        instance = None

    if instance:
        # Trust live status over the record (server may have moved ports).
        port = running.get("port")
        instance.port = int(port) if port is not None else instance.port
        instance, lease = lease_manager.acquire_lease(
            model_name, effective_preset, client_id, purpose, profile=profile_name)
    elif running:
        other = lease_manager.find_instance_for_model(model_name)
        if other:
            # One server per model (luallm refuses a second start), so this
            # server belongs to the other record's configuration. Adopting it
            # under the requested config would claim flags it doesn't carry.
            cfg = (f"profile '{other.profile}'" if other.profile
                   else f"preset '{other.preset}'" if other.preset
                   else "the default configuration")
            return {
                "success": False,
                "error": (f"MODEL_RUNNING_DIFFERENT_CONFIG: {model_name} is already "
                          f"running with {cfg}; luallm cannot start a second server "
                          f"for the same model. Release its leases or force_stop_model, "
                          f"then acquire again with the requested configuration."),
            }
        # Started outside the MCP (e.g. `luallm start` in a terminal).
        # Adopt it so workflows can use it; release will not stop it.
        if preset or profile:
            wanted = f"profile '{profile_name}'" if profile else f"preset '{preset}'"
            return {
                "success": False,
                "error": (f"MODEL_RUNNING_EXTERNALLY: {model_name} is already running "
                          f"outside MCP management, so {wanted} cannot be applied. "
                          f"Acquire without a preset/profile to use it, or stop it first."),
            }
        port = running.get("port")
        instance, lease = lease_manager.acquire_lease(
            model_name, effective_preset, client_id, purpose,
            port=int(port) if port is not None else None,
            managed_by_mcp=False)
        logger.info(f"Adopted externally started instance of {model_name}")
    else:
        start_resp = client.start_model(
            model_name, effective_preset,
            extra_args=list(profile.args) if profile else None)
        if not start_resp["success"]:
            return {"success": False, "error": _error_detail(start_resp)}
        start_data = start_resp.get("data", {})
        port = start_data.get("port")
        pid = start_data.get("pid")
        reused = False
        instance, lease = lease_manager.acquire_lease(
            model_name, effective_preset, client_id, purpose,
            port=int(port) if port is not None else None, profile=profile_name)

    # A lease means "ready to use": block until the server can actually
    # serve (llama.cpp answers 503 on /health while loading weights).
    timeout = _ready_timeout()
    ready_state, waited = await _wait_until_ready(instance.port, timeout, pid=pid)
    if ready_state != "ready":
        # Take the lease back so failed acquires don't accumulate
        # refcounts. A slow load keeps the instance record (the model is
        # still loading); a crashed server's record is garbage and goes.
        lease_manager.release_lease(
            lease.lease_id, keep_instance=(ready_state == "timeout"))
    if ready_state == "crashed":
        return {
            "success": False,
            "error": (f"SERVER_EXITED: {model_name} died during startup (port "
                      f"{instance.port} stopped accepting connections). Check "
                      f"`luallm logs {model_name}` for the actual error — a bad "
                      f"model file or an invalid profile flag are common causes."),
        }
    if ready_state == "timeout":
        return {
            "success": False,
            "error": (f"MODEL_NOT_READY: {model_name} did not become ready within "
                      f"{int(timeout)}s (still loading). It continues loading in the "
                      f"background — acquire again shortly, or force_stop_model to abort."),
        }

    data = {
        "instance_id": instance.instance_id,
        "lease_id": lease.lease_id,
        "model": model_name,
        "preset": effective_preset,
        "port": instance.port,
        "reused": reused,
        "external": not instance.managed_by_mcp,
        "ready_after_s": round(waited, 1),
    }
    if profile is not None:
        data["profile"] = profile.name
        data["args"] = profile.args
    if pid is not None:
        data["pid"] = pid
    return {"success": True, "data": data}


def _handle_release(lease_id: str | None) -> dict:
    """release_model: drop the lease; stop the model when this was the last
    lease on an MCP-managed instance. A failed stop is an error, not a
    warning — otherwise models silently pile up."""
    if not lease_id:
        return {"success": False, "error": "MISSING_LEASE_ID"}
    release_result = lease_manager.release_lease(lease_id)
    if not release_result["success"]:
        return {"success": False, "error": release_result.get("error", "RELEASE_FAILED")}

    stopped = False
    if release_result.get("instance_stopped"):
        stop_resp = client.stop_model(release_result["model"])
        if stop_resp["success"]:
            stopped = True
        else:
            return {
                "success": False,
                "error": (f"LEASE_RELEASED_BUT_STOP_FAILED: the lease was released but "
                          f"{release_result['model']} could not be stopped: {_error_detail(stop_resp)}"),
                "data": {**release_result, "stopped": False},
            }
    return {"success": True, "data": {**release_result, "stopped": stopped}}


def _handle_force_stop(model_name: str | None) -> dict:
    """force_stop_model: stop a model regardless of leases and purge its
    instance/lease records. The escape hatch for orphaned models."""
    if not model_name:
        return {"success": False, "error": "MISSING_MODEL_NAME"}
    removed = lease_manager.remove_model_records(model_name)
    stop_resp = client.stop_model(model_name)
    if not stop_resp["success"]:
        return {
            "success": False,
            "error": f"STOP_FAILED: {_error_detail(stop_resp)}",
            "data": {"model": model_name, "stopped": False, **removed},
        }
    logger.info(f"Force-stopped {model_name} "
                f"(removed {removed['removed_leases']} lease(s))")
    return {"success": True, "data": {"model": model_name, "stopped": True, **removed}}


def _handle_list_profiles() -> dict:
    """list_profiles: discover curated launch profiles and the file to
    edit to define more. Subprocess-free, so it doubles as a config
    sanity check."""
    profiles, errors = profile_registry.load()
    data: dict = {"profiles": [p.to_dict() for p in profiles], "file": str(profile_registry.path)}
    if errors:
        data["warnings"] = errors
    return {"success": True, "data": data}


@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_models",
            description="List all available models in the models directory",
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
            description="List MCP server capabilities and supported operations",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="acquire_model",
            description=(
                "Acquire a lease on a model instance. Starts the model if not running. "
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
            description="Release a lease. Stops the model if this is the last lease.",
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
            description="Get logs for a model instance",
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
                    logs = resp.get("data", {}).get("log_file", "")
                    if lines and logs:
                        log_lines = logs.split("\n")[:lines]
                        logs = "\n".join(log_lines)
                    result["data"] = {"logs": logs}
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
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="luaLLM-MCP",
                server_version="0.1.0",
                capabilities=ServerCapabilities(tools=ToolsCapability()),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())