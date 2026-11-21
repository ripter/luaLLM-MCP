"""The gateway as a real process: uvicorn serving gateway.py, a fake luallm
starting real fake llama-servers, and clients on real sockets. Disconnects
here are real TCP closes, not mocked cancellation."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-token-" + "x" * 40


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(predicate, timeout=10.0, interval=0.05, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}")


class GatewayProc:
    def __init__(self, tmp_path: Path, fake, models: dict, profiles: dict | None = None,
                 **env):
        self.state = tmp_path / "state"
        self.state.mkdir(exist_ok=True)
        self.fake = fake
        self.port = _free_port()
        self.write_config(models)
        self.write_profiles(profiles or {})
        self.env = fake.env(
            LUALLM_MCP_STATE_DIR=self.state, LUALLM_GATEWAY_TOKEN=TOKEN,
            LUALLM_MCP_IDLE_GRACE_SECONDS="300", LUALLM_MCP_MEMORY_BUDGET_GB="1000",
            LUALLM_GATEWAY_KEEPALIVE_SECONDS="0.3", **env)
        self.log = tmp_path / "gateway.log"
        self.proc = None

    def write_config(self, models: dict, **extra) -> None:
        (self.state / "gateway.json").write_text(json.dumps(
            {"listen": f"127.0.0.1:{self.port}", "models": models, **extra}))

    def write_profiles(self, profiles: dict) -> None:
        (self.state / "profiles.json").write_text(json.dumps({"profiles": profiles}))

    def start(self) -> "GatewayProc":
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "gateway.py"), "--luallm", str(self.fake.path)],
            env=self.env, stdout=open(self.log, "ab"), stderr=subprocess.STDOUT, cwd=ROOT)
        wait_for(self._healthy, what="gateway /health")
        return self

    def _healthy(self) -> bool:
        if self.proc.poll() is not None:
            raise AssertionError("gateway exited:\n" + self.log.read_text())
        try:
            return httpx.get(self.url("/health"), timeout=1).status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self, sig=signal.SIGTERM) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def post(self, body: dict, token: str = TOKEN, timeout=30) -> httpx.Response:
        return httpx.post(self.url("/v1/chat/completions"), json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {token}"})

    def leases(self) -> list:
        p = self.state / "state.json"
        return json.loads(p.read_text())["leases"] if p.exists() else []

    def instances(self) -> list:
        p = self.state / "state.json"
        return json.loads(p.read_text())["instances"] if p.exists() else []

    def raw(self, body: dict) -> socket.socket:
        """Send a request on a raw socket, for tests that close it themselves."""
        data = json.dumps(body).encode()
        s = socket.create_connection(("127.0.0.1", self.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                  b"Authorization: Bearer " + TOKEN.encode() + b"\r\n"
                  b"Content-Type: application/json\r\nContent-Length: "
                  + str(len(data)).encode() + b"\r\n\r\n" + data)
        return s


def read_until(sock: socket.socket, needle: bytes, count: int = 1, timeout=15) -> bytes:
    sock.settimeout(timeout)
    buf = b""
    while buf.count(needle) < count:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf


def chat(model="tiny", **extra):
    return {"model": model, "messages": [{"role": "user", "content": "hello"}], **extra}


@pytest.fixture
def gateway(tmp_path, fake_luallm):
    started = []

    def make(models=None, profiles=None, **env):
        g = GatewayProc(tmp_path, fake_luallm, models or {"tiny": {"model": "tiny"}},
                        profiles, **env)
        started.append(g)
        return g.start()
    yield make
    for g in started:
        g.stop()


# -- authentication and routing --

def test_health_needs_no_token_but_everything_else_does(gateway):
    g = gateway()
    assert httpx.get(g.url("/health")).json() == {"status": "ok"}
    assert httpx.get(g.url("/v1/models")).status_code == 401
    r = httpx.get(g.url("/v1/models"), headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert g.post(chat(), token="x" * 51).status_code == 401
    assert httpx.get(g.url("/v1/models"), headers={"Authorization": f"Bearer {TOKEN}"}
                     ).status_code == 200
    assert httpx.get(g.url("/v1/embeddings"), headers={"Authorization": f"Bearer {TOKEN}"}
                     ).status_code == 404


def test_unlisted_models_and_bad_requests_never_start_anything(gateway, fake_luallm):
    g = gateway()
    r = g.post(chat(model="something-else"))
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = g.post(chat(id_slot=1))
    assert r.status_code == 400 and r.json()["error"]["param"] == "id_slot"
    r = httpx.post(g.url("/v1/chat/completions"), content=b"{nope",
                   headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.json()["error"]["code"] == "invalid_json"
    assert fake_luallm.lines("starts.log") == []


def test_request_size_is_limited(gateway, tmp_path):
    g = gateway()
    cfg = json.loads((g.state / "gateway.json").read_text())
    cfg["max_request_mb"] = 0.001
    (g.state / "gateway.json").write_text(json.dumps(cfg))
    r = g.post(chat(messages=[{"role": "user", "content": "x" * 5000}]))
    assert r.status_code == 413


# -- serving --

def test_cold_start_then_reply_then_lease_released(gateway, fake_luallm):
    g = gateway(FAKE_LLAMA_LOAD_DELAY="1")
    r = g.post(chat())
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "tiny"  # the exposed name, not the model file's path
    assert r.json()["choices"][0]["message"]["content"] == "t0 t1 t2 t3 t4"
    wait_for(lambda: g.leases() == [], what="lease released")
    assert len(g.instances()) == 1 and g.instances()[0]["idle_since"]
    assert len(fake_luallm.lines("starts.log")) == 1

    assert g.post(chat()).status_code == 200  # warm: no second start
    assert len(fake_luallm.lines("starts.log")) == 1


def test_streaming_cold_start_sends_keepalives_then_the_stream(gateway):
    g = gateway(FAKE_LLAMA_LOAD_DELAY="1.5")
    s = g.raw(chat(stream=True))
    data = read_until(s, b"data: [DONE]")
    s.close()
    head, _, stream = data.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200")
    assert b": loading model" in stream
    assert stream.index(b": loading model") < stream.index(b"data: {")
    events = [json.loads(l[6:]) for l in stream.decode().splitlines()
              if l.startswith("data: {")]
    assert all(e["model"] == "tiny" for e in events)
    wait_for(lambda: g.leases() == [], what="lease released")


def test_upstream_request_contains_only_checked_fields(gateway, fake_luallm):
    g = gateway({"tiny": {"model": "tiny", "max_output_tokens": 64}})
    assert g.post(chat(temperature=0.5, seed=7)).status_code == 200
    sent = [e["body"] for e in fake_luallm.events("tiny") if e["event"] == "request"]
    assert sent[-1] == {"messages": [{"role": "user", "content": "hello"}], "stream": False,
                        "temperature": 0.5, "seed": 7, "max_tokens": 64, "model": "tiny"}


def test_models_lists_names_and_verified_capabilities(gateway):
    g = gateway({"tiny": {"model": "tiny", "description": "small"},
                 "eyes": {"model": "tiny", "profile": "vision"}},
                {"vision": {"args": ["--mmproj", "/m/proj.gguf", "-c", "8192"]}})
    auth = {"Authorization": f"Bearer {TOKEN}"}
    before = {m["id"]: m for m in httpx.get(g.url("/v1/models"), headers=auth).json()["data"]}
    assert set(before) == {"tiny", "eyes"}
    assert before["eyes"]["capabilities"] is None and before["eyes"]["loaded"] is False
    assert "proj.gguf" not in json.dumps(before)

    assert g.post(chat("eyes")).status_code == 200
    after = {m["id"]: m for m in httpx.get(g.url("/v1/models"), headers=auth).json()["data"]}
    caps = after["eyes"]["capabilities"]
    assert caps["images"] is True and caps["context_tokens"] == 8192
    assert caps["reasoning"]["effort_levels"] == ["low", "medium", "xhigh"]
    assert after["eyes"]["loaded"] is True
    assert after["tiny"]["loaded"] is False  # same model, other configuration
    assert not any(m["default"] for m in after.values())


def test_models_marks_the_default_and_reads_it_again(gateway):
    g = gateway({"tiny": {"model": "tiny"}, "other": {"model": "tiny"}})
    auth = {"Authorization": f"Bearer {TOKEN}"}
    g.write_config({"tiny": {"model": "tiny"}, "other": {"model": "tiny"}}, default_model="other")
    listed = {m["id"]: m["default"] for m in httpx.get(g.url("/v1/models"), headers=auth).json()["data"]}
    assert listed == {"tiny": False, "other": True}


# -- capabilities enforced against the model's own server --

IMAGE = {"role": "user", "content": [
    {"type": "text", "text": "what colour"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}


def test_capabilities_are_enforced(gateway, fake_luallm):
    g = gateway({"text": {"model": "tiny"},
                 "eyes": {"model": "eyes", "profile": "vision"},
                 "quiet": {"model": "quiet", "profile": "nothink"}},
                {"vision": {"args": ["--mmproj", "/m/p.gguf"]},
                 "nothink": {"args": ["--reasoning", "off"]}},
                FAKE_LUALLM_MODELS="tiny,eyes,quiet")

    r = g.post(chat("text", messages=[IMAGE]))
    assert r.status_code == 400 and r.json()["error"]["code"] == "capability_missing"
    assert g.post(chat("eyes", messages=[IMAGE])).status_code == 200

    r = g.post(chat("text", reasoning_effort="high"))
    assert r.status_code == 400 and "low, medium, xhigh" in r.json()["error"]["message"]
    assert g.post(chat("text", reasoning_effort="xhigh")).status_code == 200
    sent = [e["body"] for e in fake_luallm.events("tiny") if e["event"] == "request"]
    assert sent[-1]["reasoning_effort"] == "xhigh"

    r = g.post(chat("quiet", reasoning_effort="low"))
    assert r.status_code == 400 and "turned off by its profile" in r.json()["error"]["message"]
    # Refused before reaching the model: it never got a reasoning request.
    assert all("reasoning_effort" not in e["body"] for e in fake_luallm.events("quiet")
               if e["event"] == "request")


def test_upstream_errors_are_passed_on(gateway):
    """A model that refuses (here: an effort its template rejects, reached
    because capabilities couldn't be known in advance) answers with its
    own error, not a hang or a 200."""
    g = gateway(FAKE_LLAMA_REASONING_LEVELS="")
    assert g.post(chat()).status_code == 200
    r = g.post(chat(reasoning_effort="low"))
    assert r.status_code == 400 and "doesn't accept reasoning_effort" in r.json()["error"]["message"]


# -- cancellation: real disconnects --

def test_disconnect_mid_stream_stops_generation_and_releases_the_lease(gateway, fake_luallm):
    g = gateway(FAKE_LLAMA_TOKENS="400", FAKE_LLAMA_TOKEN_DELAY="0.02")
    assert g.post(chat(max_tokens=5)).status_code == 200  # warm it up
    s = g.raw(chat(stream=True))
    read_until(s, b"data: {", count=3)
    assert len(g.leases()) == 1
    s.close()

    gone = wait_for(lambda: [e for e in fake_luallm.events("tiny")
                             if e["event"] == "disconnected"], timeout=5,
                    what="upstream to see the disconnect")
    assert gone[0]["after"] < 100  # stopped early, not after all 400 tokens
    wait_for(lambda: g.leases() == [], timeout=5, what="lease released")
    assert not [e for e in fake_luallm.events("tiny") if e["event"] == "completed"
                and e["tokens"] == 400]


def test_disconnect_during_a_non_streaming_reply_stops_generation(gateway, fake_luallm):
    g = gateway(FAKE_LLAMA_TOKENS="400", FAKE_LLAMA_TOKEN_DELAY="0.02")
    assert g.post(chat(max_tokens=5)).status_code == 200
    s = g.raw(chat())
    wait_for(lambda: len([e for e in fake_luallm.events("tiny") if e["event"] == "request"]) == 2,
             what="request to reach upstream")
    time.sleep(0.3)
    s.close()
    wait_for(lambda: [e for e in fake_luallm.events("tiny") if e["event"] == "disconnected"],
             timeout=5, what="upstream to see the disconnect")
    wait_for(lambda: g.leases() == [], timeout=5, what="lease released")


@pytest.mark.parametrize("stream", [True, False])
def test_disconnect_while_the_model_loads_releases_the_lease(gateway, fake_luallm, stream):
    g = gateway(FAKE_LLAMA_LOAD_DELAY="3")
    s = g.raw(chat(stream=stream))
    wait_for(lambda: len(g.leases()) == 1, what="lease taken")
    s.close()
    wait_for(lambda: g.leases() == [], timeout=5, what="lease released")
    # The model keeps loading, idle, for the next request; it never got this one.
    assert len(g.instances()) == 1
    assert [e for e in fake_luallm.events("tiny") if e["event"] == "request"] == []


def test_disconnect_while_the_model_is_being_started_releases_the_lease(gateway, fake_luallm):
    """The start decision runs in a thread that can't be interrupted; the
    lease it takes after the client left must still be released."""
    g = gateway(FAKE_LUALLM_START_DELAY="1.5")
    s = g.raw(chat())
    time.sleep(0.5)  # inside luallm start
    assert g.leases() == []
    s.close()
    wait_for(lambda: len(fake_luallm.lines("starts.log")) == 1, what="start to finish")
    time.sleep(0.5)
    wait_for(lambda: g.leases() == [], timeout=5, what="lease released")
    assert len(g.instances()) == 1


def test_many_disconnects_leak_no_leases(gateway, fake_luallm):
    g = gateway(FAKE_LLAMA_TOKENS="200", FAKE_LLAMA_TOKEN_DELAY="0.01")
    assert g.post(chat(max_tokens=5)).status_code == 200
    socks = [g.raw(chat(stream=bool(i % 2))) for i in range(12)]
    time.sleep(0.3)
    for s in socks:
        s.close()
    wait_for(lambda: g.leases() == [], timeout=10, what="every lease released")


def test_crash_while_loading_is_reported_and_cleaned_up(gateway):
    g = gateway(FAKE_LLAMA_LOAD_DELAY="30", FAKE_LLAMA_CRASH_AFTER="0.5")
    r = g.post(chat())
    assert r.status_code == 502 and r.json()["error"]["code"] == "model_crashed"
    assert g.leases() == [] and g.instances() == []


def test_a_killed_gateway_leaves_no_lease_behind(tmp_path, fake_luallm):
    g = GatewayProc(tmp_path, fake_luallm, {"tiny": {"model": "tiny"}},
                    FAKE_LLAMA_TOKENS="400", FAKE_LLAMA_TOKEN_DELAY="0.05").start()
    try:
        s = g.raw(chat(stream=True))
        read_until(s, b"data: {")
        assert len(g.leases()) == 1
        g.stop(signal.SIGKILL)
        s.close()
        assert len(g.leases()) == 1  # nobody released it...
        g.start()                    # ...so the next gateway drops it on startup
        wait_for(lambda: g.leases() == [], what="orphaned lease dropped")
    finally:
        g.stop()


def test_shutdown_releases_leases_of_requests_in_flight(gateway):
    g = gateway(FAKE_LLAMA_TOKENS="400", FAKE_LLAMA_TOKEN_DELAY="0.05")
    s = g.raw(chat(stream=True))
    read_until(s, b"data: {")
    assert len(g.leases()) == 1
    g.stop(signal.SIGTERM)
    s.close()
    assert g.proc.returncode is not None
    assert g.leases() == []


# -- configurations and hot reload --

def test_profile_conflict_is_a_503_while_leased_and_a_swap_once_idle(gateway, fake_luallm):
    g = gateway({"a": {"model": "tiny", "profile": "small"},
                 "b": {"model": "tiny", "profile": "big"}},
                {"small": {"args": ["-c", "4096"]}, "big": {"args": ["-c", "8192"]}},
                FAKE_LLAMA_TOKENS="40", FAKE_LLAMA_TOKEN_DELAY="0.05")
    s = g.raw(chat("a", stream=True))
    read_until(s, b"data: {")
    r = g.post(chat("b"))
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_busy"
    assert r.headers["retry-after"] == "30"
    read_until(s, b"data: [DONE]")
    s.close()
    wait_for(lambda: g.leases() == [], what="a released")
    assert g.post(chat("b")).status_code == 200
    assert len(fake_luallm.lines("starts.log")) == 2


def test_editing_config_never_disturbs_a_request_in_flight(gateway, fake_luallm):
    g = gateway({"a": {"model": "tiny", "profile": "p"}}, {"p": {"args": ["-c", "4096"]}},
                FAKE_LLAMA_TOKENS="40", FAKE_LLAMA_TOKEN_DELAY="0.05")
    s = g.raw(chat("a", stream=True))
    first = read_until(s, b"data: {")

    g.write_profiles({"p": {"args": ["-c", "8192"]}})
    r = g.post(chat("a"))  # new flags, but the old server is in use: refused, not restarted
    assert r.status_code == 503
    g.write_config({})     # and the name is gone altogether
    assert g.post(chat("a")).status_code == 404

    body = first + read_until(s, b"data: [DONE]")
    s.close()
    assert body.count(b'"content": "t') == 40  # the stream ran to the end
    assert len(fake_luallm.lines("starts.log")) == 1

    g.write_config({"a": {"model": "tiny", "profile": "p"}})
    wait_for(lambda: g.leases() == [], what="released")
    assert g.post(chat("a")).status_code == 200  # idle now: restarted with the new flags
    servers = fake_luallm.servers()
    assert [s["args"] for s in servers if s["state"] == "running"] == [["-c", "8192"]]


def test_mcp_and_gateway_share_one_server(gateway, fake_luallm):
    """An MCP client's lease and the gateway's requests use the same
    instance; a gateway request finishing doesn't stop the MCP's model."""
    import core
    from lease_manager import LeaseManager
    from lua_client import LuaLLMClient
    from profiles import ProfileRegistry

    g = gateway()
    ctx = core.Ctx(LuaLLMClient(luallm_path=str(fake_luallm.path)),
                   LeaseManager(state_dir=str(g.state)), ProfileRegistry(state_dir=str(g.state)))
    import asyncio
    held = asyncio.run(core.acquire(ctx, "tiny", None, "agent", "test"))
    assert held["success"], held
    assert g.post(chat()).status_code == 200
    assert len(fake_luallm.lines("starts.log")) == 1
    wait_for(lambda: [l["lease_id"] for l in g.leases()] == [held["data"]["lease_id"]],
             what="only the MCP lease left")
    assert fake_luallm.lines("stops.log") == []
    assert core.release(ctx, held["data"]["lease_id"])["success"]


# -- logging --

def test_logs_never_hold_prompts_replies_or_the_token(gateway, tmp_path):
    g = gateway(FAKE_LLAMA_WORD="REPLYCANARY")
    body = chat(messages=[{"role": "user", "content": "PROMPTCANARY"}])
    assert g.post(body).status_code == 200
    g.post(dict(body, stream=True))
    g.post(dict(body, id_slot=3))
    g.post(dict(body, model="nope"))
    g.stop()
    log = g.log.read_text()
    assert "request" in log  # it does log requests...
    for secret in ("PROMPTCANARY", "REPLYCANARY", TOKEN):
        assert secret not in log  # ...but never these
