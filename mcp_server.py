"""MCP Server for luaLLM - safe model management via leases."""

import asyncio
import json
import logging
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability

from lua_client import LuaLLMClient
from lease_manager import LeaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("luaLLM-MCP")

mcp_server = Server("luaLLM")
client = LuaLLMClient()
lease_manager = LeaseManager()


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


def _find_running_server(model_name: str) -> dict | None:
    """Look up a running luallm server for the model, or None."""
    status_resp = client.get_server_status()
    if not status_resp["success"]:
        return None
    for srv in status_resp.get("data", {}).get("servers", []):
        if srv.get("model") == model_name and srv.get("state") == "running":
            return srv
    return None


def _handle_acquire(model_name: str | None, preset: str | None,
                    client_id: str | None, purpose: str | None) -> dict:
    """acquire_model: reuse an MCP-managed instance, adopt an externally
    started one (never stopped on release), or start the model."""
    if not model_name:
        return {"success": False, "error": "MISSING_MODEL_NAME"}

    models_resp = client.list_models()
    if not models_resp["success"]:
        return {"success": False, "error": "CANNOT_LIST_MODELS"}
    if model_name not in [m.get("name") for m in models_resp.get("data", [])]:
        return {"success": False, "error": "MODEL_NOT_FOUND"}

    def lease_data(instance, lease, reused: bool, pid=None) -> dict:
        data = {
            "instance_id": instance.instance_id,
            "lease_id": lease.lease_id,
            "model": model_name,
            "preset": preset,
            "port": instance.port,
            "reused": reused,
            "external": not instance.managed_by_mcp,
        }
        if pid is not None:
            data["pid"] = pid
        return data

    instance = lease_manager.find_instance(model_name, preset)
    running = _find_running_server(model_name)

    if instance:
        if running is None:
            # Stale record: the server behind it is gone (stopped or
            # crashed outside MCP). Drop it and start fresh below.
            logger.info(f"Dropping stale instance record for {model_name}")
            lease_manager.remove_instance(instance.instance_id)
            instance = None
        else:
            # Trust live status over the record (server may have moved ports).
            port = running.get("port")
            instance.port = int(port) if port is not None else instance.port
            instance, lease = lease_manager.acquire_lease(model_name, preset, client_id, purpose)
            return {"success": True, "data": lease_data(instance, lease, reused=True)}

    if running:
        # Started outside the MCP (e.g. `luallm start` in a terminal).
        # Adopt it so workflows can use it; release will not stop it.
        if preset:
            return {
                "success": False,
                "error": (f"MODEL_RUNNING_EXTERNALLY: {model_name} is already running outside "
                          f"MCP management, so the '{preset}' preset cannot be applied. "
                          f"Acquire without a preset to use it, or stop it first."),
            }
        port = running.get("port")
        instance, lease = lease_manager.acquire_lease(
            model_name, preset, client_id, purpose,
            port=int(port) if port is not None else None,
            managed_by_mcp=False)
        logger.info(f"Adopted externally started instance of {model_name}")
        return {"success": True, "data": lease_data(instance, lease, reused=True)}

    start_resp = client.start_model(model_name, preset)
    if not start_resp["success"]:
        return {"success": False, "error": _error_detail(start_resp)}
    data = start_resp.get("data", {})
    port = data.get("port")
    instance, lease = lease_manager.acquire_lease(
        model_name, preset, client_id, purpose,
        port=int(port) if port is not None else None)
    return {"success": True, "data": lease_data(instance, lease, reused=False, pid=data.get("pid"))}


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
            description="Acquire a lease on a model instance. Starts the model if not running.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model name"},
                    "preset": {"type": "string", "description": "Optional preset (throughput, cold-start)"},
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
                     "list_leases"],
            "version": "1.0",
        }
    
    elif name == "acquire_model":
        result = _handle_acquire(
            arguments.get("model_name"),
            arguments.get("preset"),
            arguments.get("client_id"),
            arguments.get("purpose"),
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