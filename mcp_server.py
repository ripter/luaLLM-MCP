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
            result["error"] = resp.get("stderr", "Unknown error")
    
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
                result["error"] = resp.get("stderr", "Unknown error")
    
    elif name == "get_server_status":
        resp = client.get_server_status()
        if resp["success"]:
            result["data"] = resp.get("data", {})
        else:
            result["success"] = False
            result["error"] = resp.get("stderr", "Unknown error")
    
    elif name == "get_capabilities":
        result["data"] = {
            "tools": ["list_models", "get_model_info", "get_server_status",
                     "get_capabilities", "acquire_model", "release_model",
                     "get_model_logs", "health_check", "list_leases"],
            "version": "1.0",
        }
    
    elif name == "acquire_model":
        model_name = arguments.get("model_name")
        preset = arguments.get("preset")
        client_id = arguments.get("client_id")
        purpose = arguments.get("purpose")
        
        if not model_name:
            result["success"] = False
            result["error"] = "MISSING_MODEL_NAME"
        else:
            models_resp = client.list_models()
            if not models_resp["success"]:
                result["success"] = False
                result["error"] = "CANNOT_LIST_MODELS"
            elif model_name not in [m.get("name") for m in models_resp.get("data", [])]:
                result["success"] = False
                result["error"] = "MODEL_NOT_FOUND"
            else:
                instance = lease_manager.find_instance(model_name, preset)
                if instance:
                    _, lease = lease_manager.acquire_lease(model_name, preset, client_id, purpose)
                    result["data"] = {
                        "instance_id": instance.instance_id,
                        "lease_id": lease.lease_id,
                        "model": model_name,
                        "preset": preset,
                        "reused": True,
                    }
                else:
                    start_resp = client.start_model(model_name, preset)
                    if start_resp["success"]:
                        data = start_resp.get("data", {})
                        _, lease = lease_manager.acquire_lease(model_name, preset, client_id, purpose)
                        result["data"] = {
                            "instance_id": lease.instance_id,
                            "lease_id": lease.lease_id,
                            "model": model_name,
                            "preset": preset,
                            "pid": data.get("pid"),
                            "port": data.get("port"),
                            "reused": False,
                        }
                    else:
                        result["success"] = False
                        result["error"] = start_resp.get("stderr", "START_FAILED")
    
    elif name == "release_model":
        lease_id = arguments.get("lease_id")
        if not lease_id:
            result["success"] = False
            result["error"] = "MISSING_LEASE_ID"
        else:
            release_result = lease_manager.release_lease(lease_id)
            if release_result["success"]:
                if release_result.get("instance_stopped"):
                    stop_resp = client.stop_model(release_result["model"])
                    if not stop_resp["success"]:
                        logger.warning(f"Failed to stop {release_result['model']}")
                result["data"] = release_result
            else:
                result["success"] = False
                result["error"] = release_result.get("error", "RELEASE_FAILED")
    
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
                    result["error"] = resp.get("stderr", "LOGS_FAILED")
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