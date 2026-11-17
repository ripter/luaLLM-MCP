"""An OpenAI-compatible gateway to local llama.cpp models, managed by luallm.

    GET  /health                no token; says only that the gateway is up
    GET  /v1/models             the names in gateway.json, with verified capabilities
    POST /v1/chat/completions   one request, on one lease

Each chat request takes a lease on its model (starting it if needed, through
the same lifecycle as the MCP tools), waits until the model is ready, sends
the checked request to the model's llama-server, and streams the reply back.
The lease is released when the reply ends, fails, or the client goes away;
a client going away also closes the upstream connection, which makes
llama.cpp stop generating.

It's a plain ASGI app rather than a framework route so that disconnect
handling and cleanup are explicit here, not a side effect of a framework
version.

Usage:
    python gateway.py              serve
    python gateway.py --check      validate configuration and token, then exit
    python gateway.py --init-token create gateway.token if missing
"""

import argparse
import asyncio
import hmac
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Awaitable, Callable

import httpx

import core
from gateway_config import (CapabilityCache, ConfigError, Route, capabilities_from_props,
                            create_token, load_config, load_token)
from gateway_validate import RequestError, apply_capabilities, check_request
from lease_manager import LeaseManager, get_state_dir
from lua_client import LuaLLMClient
from profiles import ProfileRegistry

logger = logging.getLogger("luallm-gateway")

KEEPALIVE_S = float(os.environ.get("LUALLM_GATEWAY_KEEPALIVE_SECONDS", "10"))
RENEW_S = float(os.environ.get("LUALLM_GATEWAY_RENEW_SECONDS", "60"))
CLIENT_ID = "luallm-gateway"

# What a client sees for each lifecycle failure. Messages are generic on
# purpose: the details name model files and luallm output, which stay in
# this gateway's log.
_LIFECYCLE_ERRORS = {
    "CANNOT_LIST_MODELS": (503, "luallm_unavailable", "luallm couldn't list its models"),
    "MODEL_NOT_FOUND": (503, "model_unavailable", "this model's file isn't available to luallm"),
    "MODEL_RUNNING_DIFFERENT_CONFIG": (503, "model_busy",
                                       "the model is in use under another configuration; "
                                       "try again when it's released"),
    "MODEL_RUNNING_EXTERNALLY": (503, "model_running_externally",
                                 "the model is running outside luallm's management, under "
                                 "unknown settings"),
    "STOP_FAILED": (503, "model_busy", "the model couldn't be restarted for this configuration"),
    "START_FAILED": (502, "model_start_failed", "the model failed to start"),
    "SERVER_EXITED": (502, "model_crashed", "the model's server exited while loading"),
    "MODEL_NOT_READY": (504, "model_load_timeout", "the model didn't finish loading in time"),
}

Send = Callable[[dict], Awaitable[None]]
Receive = Callable[[], Awaitable[dict]]


def _error_body(message: str, code: str, param: str | None = None,
                kind: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": kind, "code": code, "param": param}}


async def _send_json(send: Send, status: int, body: dict,
                     headers: list[tuple[bytes, bytes]] | None = None) -> None:
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(data)).encode()),
                            *(headers or [])]})
    await send({"type": "http.response.body", "body": data, "more_body": False})


class Failure(Exception):
    """A request failed in a way the client should be told about."""

    def __init__(self, status: int, code: str, message: str, param: str | None = None,
                 retry_after: int | None = None, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.param, self.retry_after, self.kind = param, retry_after, kind

    def body(self) -> dict:
        return _error_body(self.message, self.code, self.param, self.kind)


class Gateway:
    def __init__(self, state_dir: str, client: Any, token: str,
                 upstream_host: str = "127.0.0.1"):
        self.state_dir = state_dir
        self.leases = LeaseManager(state_dir=state_dir)
        self.profiles = ProfileRegistry(state_dir=state_dir)
        self.ctx = core.Ctx(client=client, leases=self.leases, profiles=self.profiles)
        self.caps_cache = CapabilityCache(state_dir)
        self._instance_caps: dict[tuple[str, str], dict | None] = {}
        self._token = token.encode()
        self.upstream_host = upstream_host
        self.pid = os.getpid()
        self.http: httpx.AsyncClient | None = None
        self._background: set[asyncio.Task] = set()
        self._reaper: asyncio.Task | None = None

    # -- ASGI --

    async def __call__(self, scope: dict, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        if path == "/health":
            if method != "GET":
                await _send_json(send, 405, _error_body("use GET", "method_not_allowed"))
                return
            await _send_json(send, 200, {"status": "ok"})
            return
        if path not in ("/v1/models", "/v1/chat/completions"):
            await _send_json(send, 404, _error_body("not found", "not_found"))
            return
        if not self._authorized(scope):
            await _send_json(send, 401, _error_body("missing or wrong bearer token",
                                                    "invalid_api_key", kind="authentication_error"),
                             headers=[(b"www-authenticate", b"Bearer")])
            return
        if path == "/v1/models":
            if method != "GET":
                await _send_json(send, 405, _error_body("use GET", "method_not_allowed"))
                return
            await self._models(send)
            return
        if method != "POST":
            await _send_json(send, 405, _error_body("use POST", "method_not_allowed"))
            return
        await self._chat_asgi(scope, receive, send)

    def _authorized(self, scope: dict) -> bool:
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                scheme, _, presented = value.partition(b" ")
                return scheme.lower() == b"bearer" and hmac.compare_digest(
                    presented.strip(), self._token)
        return False

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await self.startup()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def startup(self) -> None:
        # Never route model traffic through a proxy from the environment.
        self.http = httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(connect=5.0, read=None, write=60.0, pool=None))
        dropped = await asyncio.to_thread(self.leases.drop_orphaned_leases)
        if dropped:
            logger.info("dropped %d lease(s) left by a process that's gone", len(dropped))
        self._reaper = asyncio.create_task(self._reap_loop())

    async def shutdown(self) -> None:
        if self._reaper:
            self._reaper.cancel()
        if self._background:
            await asyncio.wait(self._background, timeout=10)
        # Anything still held by this process is released now rather than
        # left for the next sweep.
        for lease_id in await asyncio.to_thread(self.leases.leases_owned_by, self.pid):
            await asyncio.to_thread(core.release, self.ctx, lease_id)
        await asyncio.to_thread(core.stop_all_idle, self.ctx)
        if self.http:
            await self.http.aclose()

    async def _reap_loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(core.reap_idle, self.ctx)
            except Exception:
                logger.exception("idle reaper sweep failed")
            await asyncio.sleep(min(15.0, max(1.0, self.leases.idle_grace_seconds())))

    # -- /v1/models --

    def _config(self):
        try:
            return load_config(self.state_dir, self.profiles)
        except ConfigError as e:
            logger.error("configuration: %s", e)
            raise Failure(503, "gateway_misconfigured",
                          "the gateway's configuration is broken; see its log",
                          kind="server_error")

    async def _models(self, send: Send) -> None:
        try:
            config = self._config()
        except Failure as f:
            await _send_json(send, f.status, f.body())
            return
        status = await asyncio.to_thread(self.ctx.client.get_server_status)
        running = {s.get("model") for s in (status.get("data") or {}).get("servers", [])
                   if s.get("state") == "running"} if status.get("success") else set()
        state = await asyncio.to_thread(self.leases.load)
        data = []
        for route in config.routes.values():
            entry = self.caps_cache.get(route)
            data.append({
                "id": route.name,
                "object": "model",
                "created": 0,
                "owned_by": "luallm",
                "description": route.description,
                "loaded": self._loaded(route, running, state),
                "capabilities": entry["capabilities"] if entry else None,
                "capabilities_verified_at": entry["verified_at"] if entry else None,
            })
        await _send_json(send, 200, {"object": "list", "data": data})

    @staticmethod
    def _loaded(route: Route, running: set, state) -> bool:
        """Whether a request now would find this route's server warm."""
        if route.model not in running:
            return False
        preset = route.profile.preset if route.profile else None
        profile = route.profile.name if route.profile else None
        records = [i for i in state.instances if i.model == route.model]
        match = next((i for i in records if i.preset == preset and i.profile == profile), None)
        if match:
            return not match.managed_by_mcp or match.args == route.start_args
        return not records and route.profile is None  # an external server it would adopt

    # -- /v1/chat/completions --

    async def _chat_asgi(self, scope: dict, receive: Receive, send: Send) -> None:
        rid = uuid.uuid4().hex[:12]
        try:
            config = self._config()
            body = await self._read_body(scope, receive, config.max_request_bytes)
        except Failure as f:
            await _send_json(send, f.status, f.body())
            return

        # The handler runs as its own task, watched for the client going
        # away. A disconnect cancels it; its finally blocks close the
        # upstream connection and release the lease.
        handler = asyncio.create_task(self._chat(rid, config, body, send))
        watcher = asyncio.create_task(self._wait_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait({handler, watcher},
                                         return_when=asyncio.FIRST_COMPLETED)
            if handler not in done:
                logger.info("request %s: client disconnected", rid)
                handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
        finally:
            watcher.cancel()
            if not handler.done():
                handler.cancel()
                await asyncio.gather(handler, return_exceptions=True)

    @staticmethod
    async def _wait_for_disconnect(receive: Receive) -> None:
        while (await receive())["type"] != "http.disconnect":
            pass

    @staticmethod
    async def _read_body(scope: dict, receive: Receive, limit: int) -> Any:
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > limit:
                        raise Failure(413, "request_too_large",
                                      f"the request is larger than {limit} bytes")
                except ValueError:
                    raise Failure(400, "invalid_request", "bad Content-Length")
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise Failure(400, "client_disconnected", "the client went away")
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                raise Failure(413, "request_too_large",
                              f"the request is larger than {limit} bytes")
            chunks.append(chunk)
            if not message.get("more_body"):
                break
        try:
            return json.loads(b"".join(chunks))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise Failure(400, "invalid_json", "the request body isn't valid JSON")

    async def _chat(self, rid: str, config, body: Any, send: Send) -> None:
        started = time.monotonic()
        outcome = {"status": None, "route": None, "cold": False, "ready_s": 0.0,
                   "stream": False}
        committed = False  # response headers sent; errors must go in-stream

        async def fail(f: Failure) -> None:
            outcome["status"] = f"{f.status} {f.code}"
            if committed:
                await self._send_sse(send, b"data: " + json.dumps(f.body()).encode() + b"\n\n")
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            else:
                headers = [(b"retry-after", str(f.retry_after).encode())] if f.retry_after else []
                await _send_json(send, f.status, f.body(), headers)

        lease_id = None
        renewer = None
        try:
            try:
                checked = check_request(body)
            except RequestError as e:
                raise Failure(e.status, e.code, e.message, e.param)
            outcome["stream"] = checked["stream"]
            route = config.routes.get(body["model"])
            if route is None:
                raise Failure(404, "model_not_found",
                              f"'{body['model']}' isn't a model this gateway serves", "model")
            outcome["route"] = route.name

            # Refuse early from capabilities verified before, so a request
            # that can't run never starts a model; checked again below
            # against this request's own server.
            cached = self.caps_cache.get(route)
            if cached:
                self._apply(checked, route, cached["capabilities"])

            begun = await self._acquire(rid, route)
            if isinstance(begun, dict):
                logger.warning("request %s: %s", rid, begun["error"])
                status, code, message = _LIFECYCLE_ERRORS.get(
                    begun.get("code"), (502, "model_unavailable", "the model couldn't be acquired"))
                raise Failure(status, code, message, kind="server_error",
                              retry_after=30 if status == 503 else None)
            lease_id = begun.lease.lease_id
            renewer = asyncio.create_task(self._renew(lease_id))
            outcome["cold"] = not begun.reused
            port = begun.instance.port

            ready = await asyncio.to_thread(core.probe_health, port) == 2 if port else True
            if not ready:
                if checked["stream"]:
                    # Commit to a stream now and send comments while the model
                    # loads, so clients and proxies don't time out. SSE parsers
                    # ignore lines starting with ':'.
                    await self._start_stream(send)
                    committed = True
                state, waited = await self._wait_ready(begun, send if committed else None)
                outcome["ready_s"] = round(waited, 1)
                if state != "ready":
                    result = await asyncio.to_thread(
                        core.acquire_failed, self.ctx, begun, state, core.ready_timeout())
                    lease_id = None  # acquire_failed took it back
                    logger.warning("request %s: %s", rid, result["error"])
                    status, code, message = _LIFECYCLE_ERRORS[result["code"]]
                    raise Failure(status, code, message, kind="server_error")

            caps = await self._capabilities(route, begun.instance)
            upstream_body = self._apply(checked, route, caps)
            await self._proxy(route, port, upstream_body, send, committed, outcome)
        except Failure as f:
            await fail(f)
        except asyncio.CancelledError:
            outcome["status"] = "client_disconnected"
            raise
        except Exception:
            logger.exception("request %s: gateway error", rid)
            try:
                await fail(Failure(500, "gateway_error", "the gateway failed; see its log",
                                   kind="server_error"))
            except Exception:
                pass
        finally:
            if renewer:
                renewer.cancel()
            if lease_id:
                self._release_soon(rid, lease_id)
            logger.info("request %s: model=%s stream=%s cold=%s ready_s=%s status=%s total_s=%.2f",
                        rid, outcome["route"], outcome["stream"], outcome["cold"],
                        outcome["ready_s"], outcome["status"], time.monotonic() - started)

    @staticmethod
    def _apply(checked: dict, route: Route, caps: dict | None) -> dict:
        try:
            return apply_capabilities(checked, route, caps)
        except RequestError as e:
            raise Failure(e.status, e.code, e.message, e.param)

    async def _acquire(self, rid: str, route: Route) -> core.Begun | dict:
        """acquire_begin in a thread. The thread can't be interrupted, so if
        the request is cancelled meanwhile, the lease it takes is released
        as soon as it exists rather than left to the reaper."""
        begin = asyncio.ensure_future(asyncio.to_thread(
            core.acquire_begin, self.ctx, route.model, None, CLIENT_ID,
            f"gateway request {rid}", profile=route.profile, owner_pid=self.pid,
            expected_args=route.start_args))
        try:
            return await asyncio.shield(begin)
        except asyncio.CancelledError:
            async def release_when_done() -> None:
                try:
                    begun = await begin
                except Exception:
                    return
                if isinstance(begun, core.Begun):
                    self._release_soon(rid, begun.lease.lease_id)
            task = asyncio.create_task(release_when_done())
            self._background.add(task)
            task.add_done_callback(self._background.discard)
            raise

    def _release_soon(self, rid: str, lease_id: str) -> None:
        """Release in a task of its own: it must finish even though the
        request's task may be being cancelled."""
        async def release() -> None:
            try:
                result = await asyncio.to_thread(core.release, self.ctx, lease_id)
                if not result["success"]:
                    logger.warning("request %s: release: %s", rid, result["error"])
            except Exception:
                logger.exception("request %s: release failed; the reaper will drop the lease",
                                 rid)
        task = asyncio.create_task(release())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _renew(self, lease_id: str) -> None:
        while True:
            await asyncio.sleep(RENEW_S)
            await asyncio.to_thread(self.leases.renew_lease, lease_id)

    async def _wait_ready(self, begun: core.Begun, stream_send: Send | None) -> tuple[str, float]:
        wait = asyncio.create_task(core.wait_until_ready(
            begun.instance.port, core.ready_timeout(), pid=begun.pid, in_thread=True))
        try:
            while True:
                done, _ = await asyncio.wait({wait}, timeout=KEEPALIVE_S)
                if done:
                    return wait.result()
                if stream_send:
                    await self._send_sse(stream_send, b": loading model\n\n")
        finally:
            wait.cancel()

    async def _capabilities(self, route: Route, instance) -> dict | None:
        """Capabilities of this request's own server, from its /props, once
        per server. None if they can't be read."""
        key = (instance.instance_id, route.fingerprint)
        if key in self._instance_caps:
            return self._instance_caps[key]
        caps = None
        try:
            r = await self.http.get(f"http://{self.upstream_host}:{instance.port}/props",
                                    timeout=10)
            if r.status_code == 200:
                caps = capabilities_from_props(r.json(), route)
                await asyncio.to_thread(self.caps_cache.put, route, caps)
        except (httpx.HTTPError, ValueError):
            logger.warning("couldn't read /props for %s; treating capabilities as unverified",
                           route.name)
        if caps is not None:  # a failed read is tried again next request
            self._instance_caps[key] = caps
        return caps

    @staticmethod
    async def _start_stream(send: Send) -> None:
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream"),
                                (b"cache-control", b"no-cache")]})

    @staticmethod
    async def _send_sse(send: Send, data: bytes) -> None:
        await send({"type": "http.response.body", "body": data, "more_body": True})

    async def _proxy(self, route: Route, port: int, upstream_body: dict, send: Send,
                     committed: bool, outcome: dict) -> None:
        url = f"http://{self.upstream_host}:{port}/v1/chat/completions"
        request = self.http.build_request("POST", url, json=upstream_body)
        try:
            response = await self.http.send(request, stream=True)
        except httpx.HTTPError as e:
            logger.warning("upstream %s unreachable: %s", route.name, type(e).__name__)
            raise Failure(502, "upstream_unreachable", "the model's server didn't answer",
                          kind="server_error")
        try:
            if response.status_code != 200:
                raw = await self._read_capped(response, 64 * 1024)
                try:
                    err = json.loads(raw)
                    message = (err.get("error") or {}).get("message") or "upstream error"
                except (ValueError, AttributeError):
                    message = "upstream error"
                raise Failure(response.status_code if 400 <= response.status_code < 600 else 502,
                              "upstream_error", str(message)[:2000],
                              kind="invalid_request_error" if response.status_code < 500
                              else "server_error")

            if not upstream_body["stream"]:
                raw = await response.aread()
                try:
                    payload = json.loads(raw)
                except ValueError:
                    raise Failure(502, "upstream_error", "the model's server sent bad JSON",
                                  kind="server_error")
                if isinstance(payload, dict) and "model" in payload:
                    payload["model"] = route.name
                outcome["status"] = "200"
                await _send_json(send, 200, payload)
                return

            if not committed:
                await self._start_stream(send)
            async for line in response.aiter_lines():
                await self._send_sse(send, self._rewrite_sse_line(line, route.name) + b"\n")
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            outcome["status"] = "200"
        finally:
            # Closing the connection is what tells llama.cpp to stop. On a
            # cancelled task this await still runs: only one cancellation
            # is delivered.
            await response.aclose()

    @staticmethod
    async def _read_capped(response: httpx.Response, cap: int) -> bytes:
        out = b""
        async for chunk in response.aiter_bytes():
            out += chunk
            if len(out) >= cap:
                break
        return out[:cap]

    @staticmethod
    def _rewrite_sse_line(line: str, name: str) -> bytes:
        """Show the exposed name, not the model file's path, in each event."""
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                event = json.loads(line[6:])
            except ValueError:
                return line.encode()
            if isinstance(event, dict) and "model" in event:
                event["model"] = name
                return b"data: " + json.dumps(event).encode()
        return line.encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="validate configuration and token, then exit")
    parser.add_argument("--init-token", action="store_true",
                        help="create gateway.token in the state dir if it's missing")
    parser.add_argument("--luallm", default=os.environ.get("LUALLM_PATH", "luallm"),
                        help=argparse.SUPPRESS)
    opts = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    state_dir = get_state_dir()

    if opts.init_token:
        path, created = create_token(state_dir)
        print(f"{'Created' if created else 'Kept the existing'} token file {path} (mode 600).")
        return 0

    try:
        config = load_config(state_dir, ProfileRegistry(state_dir=state_dir))
        token = load_token(state_dir)
    except ConfigError as e:
        print(f"luallm-gateway: {e}", file=sys.stderr)
        return 2
    for w in config.warnings:
        logger.warning("gateway.json: %s", w)

    if opts.check:
        print(f"listen: {config.host}:{config.port}")
        for r in config.routes.values():
            print(f"  {r.name}: {r.model}" + (f" (profile {r.profile.name})" if r.profile else ""))
        if not config.routes:
            print("  no models are exposed")
        return 0

    import uvicorn
    gateway = Gateway(state_dir, LuaLLMClient(luallm_path=opts.luallm), token)
    logger.info("serving %d model name(s) on http://%s:%d", len(config.routes),
                config.host, config.port)
    uvicorn.run(gateway, host=config.host, port=config.port, lifespan="on",
                access_log=False, log_level="warning", timeout_graceful_shutdown=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
