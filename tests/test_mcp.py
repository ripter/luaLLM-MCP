"""Tests for luaLLM-MCP."""

import json

import pytest


def test_pyproject_exists():
    """Test that pyproject.toml exists and is valid."""
    import tomllib
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["name"] == "luaLLM-MCP"
    assert any("mcp" in dep for dep in data["project"]["dependencies"])


def test_lua_client_imports():
    """Test that lua_client module can be imported."""
    import lua_client
    client = lua_client.LuaLLMClient()
    assert client.luallm_path == "luallm"


def test_run_command_parses_json_after_banner_lines(monkeypatch):
    """luallm --json prefixes JSON with human-readable banner lines; the
    client must still find the payload (observed with `luallm start`)."""
    import subprocess
    import lua_client

    stdout = (
        "Starting daemon: Maginum-24B\n"
        '{"pid":81836,"model":"Maginum-24B","port":8080,"status":"started"}'
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    resp = lua_client.LuaLLMClient()._run_command(["start", "Maginum-24B", "--json"])

    assert resp["success"] is True
    assert resp["data"]["pid"] == 81836
    assert resp["data"]["port"] == 8080


def test_run_command_reports_non_json_error_output(monkeypatch):
    """Pure human-readable error output must surface stdout, not an empty
    error (observed when the model is already running outside MCP)."""
    import subprocess
    import lua_client

    stdout = (
        "Error: Maginum-24B is already running (PID 80181.0) on port 8080\n"
        "Use 'luallm stop Maginum-24B' first."
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=1, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    resp = lua_client.LuaLLMClient()._run_command(["start", "Maginum-24B", "--json"])

    assert resp["success"] is False
    assert resp["stdout"] == stdout


def test_error_detail_never_empty():
    """The MCP envelope error must be non-empty for every failure shape."""
    import mcp_server

    assert mcp_server._error_detail({"stderr": "boom"}) == "stderr: boom"
    assert "already running" in mcp_server._error_detail(
        {"stdout": "Error: already running", "stderr": ""}
    )
    assert mcp_server._error_detail({"returncode": 7}) == "luallm exited with code 7 and no output"
    assert mcp_server._error_detail({}) == "UNKNOWN_ERROR"
    # Structured luallm --json error objects win over raw stream dumps
    assert (
        mcp_server._error_detail(
            {"data": {"error": "start_failed", "message": "already running"}, "stdout": "raw"}
        )
        == "start_failed: already running"
    )


def test_server_ships_usage_instructions():
    """Handshake instructions are the only docs a connected LLM is
    guaranteed to see; they must carry the contract, not tool plumbing."""
    import mcp_server

    instructions = mcp_server.mcp_server.instructions
    assert instructions
    for fragment in (
        "acquire_model", "release_model", "lease_id", "instance_id",
        "127.0.0.1:{port}", "MODEL_RUNNING_DIFFERENT_CONFIG",
        "MODEL_NOT_READY", "SERVER_EXITED", "MODEL_RUNNING_EXTERNALLY",
    ):
        assert fragment in instructions, f"missing: {fragment}"


def test_get_capabilities_echoes_usage_guide():
    """In-band fallback for clients that don't surface server instructions:
    get_capabilities must return the same usage guide."""
    import asyncio

    import mcp_server

    result, = [json.loads(c.text) for c in asyncio.run(
        mcp_server.handle_call_tool("get_capabilities", {}))]
    assert result["success"] is True
    assert result["data"]["usage"] == mcp_server.SERVER_INSTRUCTIONS


def _fake_client(models, running=None, start=None, stop_fails=False):
    servers = list(running or [])

    class FakeClient:
        def __init__(self):
            self.stopped = []
            self.started = []

        def list_models(self):
            # models: list of names, or {name: luallm size string}
            if isinstance(models, dict):
                return {"success": True,
                        "data": [{"name": m, "size": sz} for m, sz in models.items()]}
            return {"success": True, "data": [{"name": m} for m in models]}

        def get_server_status(self):
            return {"success": True, "data": {"servers": servers}}

        def start_model(self, model, preset=None, extra_args=None):
            self.started.append(
                {"model": model, "preset": preset, "extra_args": extra_args})
            resp = start or {"success": False, "stderr": "start_model should not be called"}
            if resp.get("success"):
                # Mirror reality: a started model shows up in status.
                servers.append({
                    "model": model,
                    "state": "running",
                    "port": resp.get("data", {}).get("port"),
                })
            return resp

        def stop_model(self, model):
            if stop_fails:
                return {"success": False, "stderr": "stop exploded"}
            self.stopped.append(model)
            servers[:] = [s for s in servers if s.get("model") != model]
            return {"success": True, "data": {"stopped": True}}

    return FakeClient()


def _isolated(monkeypatch, tmp_path, client, ready=2, grace="0", budget_gb="1000"):
    """grace defaults to 0 (stop on last release) so lifecycle tests stay
    deterministic; idle-grace tests opt in. The memory budget is pinned so
    results don't depend on the test machine's RAM."""
    import mcp_server
    monkeypatch.setenv("LUALLM_MCP_IDLE_GRACE_SECONDS", grace)
    monkeypatch.setenv("LUALLM_MCP_MEMORY_BUDGET_GB", budget_gb)
    monkeypatch.delenv("LUALLM_MCP_MEMORY_OVERHEAD", raising=False)
    import lease_manager
    import profiles
    monkeypatch.setattr(mcp_server, "lease_manager",
                        lease_manager.LeaseManager(state_dir=str(tmp_path)))
    monkeypatch.setattr(mcp_server, "client", client)
    monkeypatch.setattr(mcp_server, "profile_registry",
                        profiles.ProfileRegistry(state_dir=str(tmp_path)))
    # Don't touch real ports in tests; readiness is simulated.
    # ready is the tri-state probe value: 0 refused, 1 loading, 2 ready.
    monkeypatch.setattr(mcp_server, "_probe_health", lambda port: ready)
    return mcp_server


def _write_profiles(tmp_path, entries: dict):
    import json
    (tmp_path / "profiles.json").write_text(json.dumps({"profiles": entries}))


def _acquire(mcp_server, *args, profile_name=None):
    import asyncio
    return asyncio.run(mcp_server._handle_acquire(*args, profile_name=profile_name))


def test_acquire_adopts_external_instance(monkeypatch, tmp_path):
    """A model running outside MCP management is adopted: lease + port are
    returned, and releasing never stops the external server."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["ornith"], running=[{"model": "ornith", "state": "running", "port": 8080}]))

    result = _acquire(mcp_server, "ornith", None, "test", "adopt")
    assert result["success"] is True
    assert result["data"]["external"] is True
    assert result["data"]["reused"] is True
    assert result["data"]["port"] == 8080

    release = mcp_server.lease_manager.release_lease(result["data"]["lease_id"])
    assert release["success"] is True
    assert release["instance_stopped"] is False


def test_acquire_external_with_preset_is_refused(monkeypatch, tmp_path):
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["ornith"], running=[{"model": "ornith", "state": "running", "port": 8080}]))
    result = _acquire(mcp_server, "ornith", "throughput", "test", "t")
    assert result["success"] is False
    assert "MODEL_RUNNING_EXTERNALLY" in result["error"]


def test_acquire_reuse_reports_port(monkeypatch, tmp_path):
    """Second acquire on an MCP-started model must include the port so
    clients can build the endpoint."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}}))

    first = _acquire(mcp_server, "tiny", None, "test", "t")
    assert first["success"] is True
    assert first["data"]["reused"] is False
    assert first["data"]["port"] == 8081

    second = _acquire(mcp_server, "tiny", None, "test", "t")
    assert second["data"]["reused"] is True
    assert second["data"]["external"] is False
    assert second["data"]["port"] == 8081


def test_acquire_drops_stale_instance_and_restarts(monkeypatch, tmp_path):
    """A record whose server is no longer running (stopped outside MCP)
    must be discarded — with its leases — and the model started fresh,
    instead of handing out a lease on a dead endpoint."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["skyfall"], start={"success": True, "data": {"pid": 2, "port": 8083}}))
    # Stale record with an old port, plus an orphaned lease on it.
    stale_instance, stale_lease = mcp_server.lease_manager.acquire_lease(
        "skyfall", None, "old", "old", port=8082)

    result = _acquire(mcp_server, "skyfall", None, "test", "t")
    assert result["success"] is True
    assert result["data"]["reused"] is False
    assert result["data"]["port"] == 8083
    assert result["data"]["instance_id"] != stale_instance.instance_id
    assert mcp_server.lease_manager.find_lease(stale_lease.lease_id) is None


def test_acquire_syncs_port_with_live_status(monkeypatch, tmp_path):
    """If the server moved ports since the record was written, the live
    port from get_server_status wins."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], running=[{"model": "tiny", "state": "running", "port": 9091}]))
    mcp_server.lease_manager.acquire_lease("tiny", None, "old", "old", port=8080)

    result = _acquire(mcp_server, "tiny", None, "test", "t")
    assert result["data"]["reused"] is True
    assert result["data"]["port"] == 9091


def test_release_stops_last_lease_and_reports_it(monkeypatch, tmp_path):
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client)

    first = _acquire(mcp_server, "tiny", None, "a", "t")
    second = _acquire(mcp_server, "tiny", None, "b", "t")

    rel1 = mcp_server._handle_release(second["data"]["lease_id"])
    assert rel1["success"] is True
    assert rel1["data"]["stopped"] is False
    assert rel1["data"]["remaining_leases"] == 1

    rel2 = mcp_server._handle_release(first["data"]["lease_id"])
    assert rel2["success"] is True
    assert rel2["data"]["stopped"] is True
    assert client.stopped == ["tiny"]


def test_release_surfaces_stop_failures(monkeypatch, tmp_path):
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}}, stop_fails=True))
    acq = _acquire(mcp_server, "tiny", None, "a", "t")
    rel = mcp_server._handle_release(acq["data"]["lease_id"])
    assert rel["success"] is False
    assert "STOP_FAILED" in rel["error"]
    assert "stop exploded" in rel["error"]


def test_release_never_stops_external_instances(monkeypatch, tmp_path):
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["ornith"], running=[{"model": "ornith", "state": "running", "port": 8080}]))
    acq = _acquire(mcp_server, "ornith", None, "a", "t")
    assert acq["data"]["external"] is True
    rel = mcp_server._handle_release(acq["data"]["lease_id"])
    assert rel["success"] is True
    assert rel["data"]["stopped"] is False
    assert rel["data"]["external"] is True


def test_force_stop_purges_records_and_stops(monkeypatch, tmp_path):
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client)
    # Two leases, would normally block stopping.
    _acquire(mcp_server, "tiny", None, "a", "t")
    _acquire(mcp_server, "tiny", None, "b", "t")

    result = mcp_server._handle_force_stop("tiny")
    assert result["success"] is True
    assert result["data"]["stopped"] is True
    assert result["data"]["removed_leases"] == 2
    assert client.stopped == ["tiny"]
    assert mcp_server.lease_manager.list_leases() == []


def test_acquire_waits_until_the_model_is_ready():
    """_wait_until_ready polls until the health probe goes green."""
    import asyncio
    import mcp_server

    probes = iter([0, 0, 1, 2])
    original = mcp_server._probe_health
    mcp_server._probe_health = lambda port: next(probes)
    try:
        state, waited = asyncio.run(
            mcp_server._wait_until_ready(8080, timeout_s=5, interval_s=0.01))
    finally:
        mcp_server._probe_health = original
    assert state == "ready"
    assert waited >= 0.02


def test_wait_until_ready_never_ready_times_out():
    import asyncio
    import mcp_server

    original = mcp_server._probe_health
    mcp_server._probe_health = lambda port: 1  # loading forever
    try:
        state, _ = asyncio.run(
            mcp_server._wait_until_ready(8080, timeout_s=0, interval_s=0.01))
    finally:
        mcp_server._probe_health = original
    assert state == "timeout"


def test_wait_until_ready_detects_crash_after_accepting_connections():
    """A port that answered /health then starts refusing means the server
    died during load; waiting out the timeout would just stall callers."""
    import asyncio
    import mcp_server

    probes = iter([1, 0])
    original = mcp_server._probe_health
    mcp_server._probe_health = lambda port: next(probes)
    try:
        state, _ = asyncio.run(
            mcp_server._wait_until_ready(8080, timeout_s=30, interval_s=0.01))
    finally:
        mcp_server._probe_health = original
    assert state == "crashed"


def test_wait_until_ready_pure_refusals_are_not_a_crash_yet():
    """Refusals before the listener ever accepted are inconclusive: the
    port binds a moment after launch. Must keep waiting, not 'crash'."""
    import asyncio
    import mcp_server

    original = mcp_server._probe_health
    mcp_server._probe_health = lambda port: 0
    try:
        state, _ = asyncio.run(
            mcp_server._wait_until_ready(8080, timeout_s=0, interval_s=0.01))
    finally:
        mcp_server._probe_health = original
    assert state == "timeout"


def test_acquire_fails_and_returns_lease_when_model_never_ready(monkeypatch, tmp_path):
    """If loading exceeds the timeout: error out, take the lease back, but
    keep the instance record — the model is still loading and a later
    acquire should reuse it."""
    monkeypatch.setenv("LUALLM_MCP_READY_TIMEOUT_SECONDS", "0")
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}}), ready=1)

    result = _acquire(mcp_server, "tiny", None, "test", "t")
    assert result["success"] is False
    assert "MODEL_NOT_READY" in result["error"]
    assert mcp_server.lease_manager.list_leases() == []
    assert mcp_server.lease_manager.find_instance("tiny") is not None


def test_acquire_reports_crashed_server_and_drops_record(monkeypatch, tmp_path):
    """A server that died during startup gets an immediate actionable
    error, its lease returned, and the instance record dropped — so the
    next acquire starts fresh instead of reusing a dead endpoint."""
    crashes = iter([1, 0])
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}}))
    monkeypatch.setattr(mcp_server, "_probe_health", lambda port: next(crashes))

    result = _acquire(mcp_server, "tiny", None, "test", "t")

    assert result["success"] is False
    assert "SERVER_EXITED" in result["error"]
    assert "luallm logs tiny" in result["error"]
    assert mcp_server.lease_manager.list_leases() == []
    assert mcp_server.lease_manager.find_instance("tiny") is None


def test_acquire_detects_dead_pid_before_listener_opens(monkeypatch, tmp_path):
    """A crashed model that never bound its port (pure refusals) still
    fails fast: pid from luallm start is gone while the port refuses."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], start={"success": True, "data": {"pid": 4711, "port": 8081}}),
        ready=0)
    import mcp_server as m
    monkeypatch.setattr(m, "_process_alive", lambda pid: pid != 4711)

    result = _acquire(mcp_server, "tiny", None, "test", "t")

    assert result["success"] is False
    assert "SERVER_EXITED" in result["error"]
    assert "luallm logs tiny" in result["error"]
    assert mcp_server.lease_manager.list_leases() == []
    assert mcp_server.lease_manager.find_instance("tiny") is None


def test_acquire_treats_dead_pid_alongside_running_listener_as_waiting():
    """_process_alive(pid)=None means 'unknown', treated as alive."""
    import mcp_server
    assert mcp_server._process_alive(None) is True


def test_acquire_reports_ready_wait_duration(monkeypatch, tmp_path):
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}}))
    result = _acquire(mcp_server, "tiny", None, "test", "t")
    assert result["success"] is True
    assert result["data"]["ready_after_s"] >= 0


def test_acquire_backfills_port_from_status(monkeypatch, tmp_path):
    """Instance records created before port tracking get their port from
    get_server_status on the next acquire."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], running=[{"model": "tiny", "state": "running", "port": 9090}]))
    # Simulate a pre-existing record without a port.
    mcp_server.lease_manager.acquire_lease("tiny", None, "old", "old")
    assert mcp_server.lease_manager.find_instance("tiny").port is None

    result = _acquire(mcp_server, "tiny", None, "test", "t")
    assert result["data"]["reused"] is True
    assert result["data"]["port"] == 9090


def test_acquire_refuses_when_model_runs_under_other_config(monkeypatch, tmp_path):
    """luallm can run a model only once, so if a running server belongs to
    a recorded configuration, adopting it under the requested one would
    claim flags it doesn't carry — refuse with what's running instead."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], running=[{"model": "tiny", "state": "running", "port": 8085}]))
    recorded, _ = mcp_server.lease_manager.acquire_lease(
        "tiny", None, "old", "old", port=8085, profile="vision")

    result = _acquire(mcp_server, "tiny", None, "test", "t")
    assert result["success"] is False
    assert "MODEL_RUNNING_DIFFERENT_CONFIG" in result["error"]
    assert "vision" in result["error"]
    # The other record and its server are untouched by the refusal.
    assert (mcp_server.lease_manager.find_instance_for_model("tiny").instance_id
            == recorded.instance_id)


def test_acquire_with_profile_applies_preset_and_args(monkeypatch, tmp_path):
    """acquire_model applies the profile's preset/flags on start, records
    the profile on the instance, and reports what was applied."""
    _write_profiles(tmp_path, {
        "vision": {
            "description": "Image projection enabled",
            "preset": "cold-start",
            "args": ["--mmproj", "/models/mmproj.gguf"],
        },
    })
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client)

    result = _acquire(mcp_server, "tiny", None, "test", "vision", profile_name="vision")

    assert result["success"] is True
    assert result["data"]["reused"] is False
    assert result["data"]["profile"] == "vision"
    assert result["data"]["preset"] == "cold-start"
    assert result["data"]["args"] == ["--mmproj", "/models/mmproj.gguf"]
    assert client.started == [{
        "model": "tiny", "preset": "cold-start",
        "extra_args": ["--mmproj", "/models/mmproj.gguf"]}]
    recorded = mcp_server.lease_manager.find_instance_for_model("tiny")
    assert recorded.profile == "vision"
    assert recorded.preset == "cold-start"

    # Same profile again reuses the recorded instance; a profile-less
    # acquire is a different configuration and must not attach to it.
    reuse = _acquire(mcp_server, "tiny", None, "test", "vision", profile_name="vision")
    assert reuse["success"] is True
    assert reuse["data"]["reused"] is True
    assert reuse["data"]["instance_id"] == result["data"]["instance_id"]
    other = _acquire(mcp_server, "tiny", None, "test", "t")
    assert other["success"] is False
    assert "MODEL_RUNNING_DIFFERENT_CONFIG" in other["error"]
    assert "vision" in other["error"]


def test_acquire_with_unknown_profile_lists_available(monkeypatch, tmp_path):
    _write_profiles(tmp_path, {"big-context": {"args": ["-c", "32768"]}})
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(["tiny"]))

    result = _acquire(mcp_server, "tiny", None, "test", "t", profile_name="bigcontext")

    assert result["success"] is False
    assert "PROFILE_NOT_FOUND" in result["error"]
    assert "big-context" in result["error"]
    assert "profiles.json" in result["error"]


def test_acquire_with_no_profiles_defined_points_at_file(monkeypatch, tmp_path):
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(["tiny"]))

    result = _acquire(mcp_server, "tiny", None, "test", "t", profile_name="vision")

    assert result["success"] is False
    assert "PROFILE_NOT_FOUND" in result["error"]
    assert str(tmp_path / "profiles.json") in result["error"]


def test_acquire_preset_and_profile_are_exclusive(monkeypatch, tmp_path):
    _write_profiles(tmp_path, {"vision": {"args": ["--mmproj", "/models/mmproj.gguf"]}})
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(["tiny"]))

    result = _acquire(mcp_server, "tiny", "throughput", "test", "t", profile_name="vision")

    assert result["success"] is False
    assert "PRESET_AND_PROFILE_EXCLUSIVE" in result["error"]
    # Refused before any subprocess work or lease booking happened.
    assert mcp_server.lease_manager.list_leases() == []


def test_acquire_reports_profile_in_reused_response(monkeypatch, tmp_path):
    """Reuse of a profile-started instance still carries the profile so
    clients can see which configuration the lease is attached to."""
    _write_profiles(tmp_path, {"short-gen": {"args": ["-n", "512"]}})
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], running=[{"model": "tiny", "state": "running", "port": 9092}]))
    mcp_server.lease_manager.acquire_lease(
        "tiny", None, "old", "old", port=9092, profile="short-gen")

    result = _acquire(mcp_server, "tiny", None, "test", "t", profile_name="short-gen")

    assert result["success"] is True
    assert result["data"]["profile"] == "short-gen"
    assert result["data"]["preset"] is None
    assert result["data"]["reused"] is True


def test_release_stops_profile_started_instance(monkeypatch, tmp_path):
    _write_profiles(tmp_path, {"short-gen": {"args": ["-n", "512"]}})
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client)

    acq = _acquire(mcp_server, "tiny", None, "a", "t", profile_name="short-gen")
    rel = mcp_server._handle_release(acq["data"]["lease_id"])

    assert rel["success"] is True
    assert rel["data"]["stopped"] is True
    assert client.stopped == ["tiny"]


def test_list_profiles_returns_entries_and_file_path(monkeypatch, tmp_path):
    _write_profiles(tmp_path, {
        "vision": {"description": "Image projection", "args": ["--mmproj", "/m.gguf"]},
        "big-context": {"args": "-c 32768"},
    })
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(["tiny"]))

    import asyncio
    result, = [json.loads(c.text) for c in asyncio.run(
        mcp_server.handle_call_tool("list_profiles", {}))]

    assert result["success"] is True
    assert result["data"]["profiles"] == [
        {"name": "big-context", "args": ["-c", "32768"]},
        {"name": "vision", "args": ["--mmproj", "/m.gguf"], "description": "Image projection"},
    ]
    assert result["data"]["file"] == str(tmp_path / "profiles.json")
    assert "warnings" not in result["data"]


def test_list_profiles_surfaces_broken_file(monkeypatch, tmp_path):
    (tmp_path / "profiles.json").write_text("{not json")
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(["tiny"]))

    import asyncio
    result, = [json.loads(c.text) for c in asyncio.run(
        mcp_server.handle_call_tool("list_profiles", {}))]

    assert result["success"] is True
    assert result["data"]["profiles"] == []
    assert any("could not parse" in w for w in result["data"]["warnings"])


def test_lease_manager_imports():
    """Test that lease_manager module can be imported."""
    import lease_manager
    manager = lease_manager.LeaseManager()
    state = manager.load()
    assert state.schema_version == "1.0"


def test_lease_manager_state_dir():
    """Test that state directory is correctly determined."""
    import lease_manager

    state_dir = lease_manager.get_state_dir()

    # Should use XDG_CONFIG_HOME or ~/.config
    assert "luaLLM-MCP" in state_dir


def test_lease_manager_acquire_release(monkeypatch):
    """Test lease acquire and release cycle (grace 0: stop immediately)."""
    import tempfile
    import lease_manager
    monkeypatch.setenv("LUALLM_MCP_IDLE_GRACE_SECONDS", "0")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = lease_manager.LeaseManager(state_dir=tmpdir)
        manager.load()
        
        instance, lease = manager.acquire_lease(
            model="test-model",
            preset=None,
            client_id="test-client",
            purpose="test"
        )
        
        assert instance.model == "test-model"
        assert lease.lease_id is not None
        assert lease.client_id == "test-client"
        
        # Release should stop the instance
        result = manager.release_lease(lease.lease_id)
        assert result["success"] is True
        assert result["instance_stopped"] is True


def test_lease_manager_presets():
    """Test that different presets create separate instances."""
    import tempfile
    import lease_manager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = lease_manager.LeaseManager(state_dir=tmpdir)
        manager.load()
        
        # Acquire with preset A
        inst_a, lease_a = manager.acquire_lease("model", "preset-a")
        
        # Acquire with preset B - should create new instance
        inst_b, lease_b = manager.acquire_lease("model", "preset-b")
        
        assert inst_a.instance_id != inst_b.instance_id
        assert inst_a.preset == "preset-a"
        assert inst_b.preset == "preset-b"


def test_lease_manager_profiles():
    """Test that different profiles create separate instances, including
    profiles that bundle the same underlying luallm preset."""
    import tempfile
    import lease_manager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = lease_manager.LeaseManager(state_dir=tmpdir)
        manager.load()

        inst_a, _ = manager.acquire_lease("model", "throughput", profile="vision")
        inst_b, _ = manager.acquire_lease("model", "throughput", profile="big-context")
        inst_none = manager.find_instance("model", "throughput")
        preset_only, _ = manager.acquire_lease("model", "throughput")

        assert inst_a.instance_id != inst_b.instance_id
        assert inst_a.profile == "vision"
        assert inst_b.profile == "big-context"
        assert inst_none is None
        assert preset_only.profile is None
        assert preset_only.preset == "throughput"


def test_lease_manager_renewal():
    """Test lease renewal."""
    import tempfile
    import lease_manager
    from datetime import datetime, timezone
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = lease_manager.LeaseManager(state_dir=tmpdir)
        manager.load()
        
        _, lease = manager.acquire_lease("model", None)
        old_renewed = lease.renewed_at
        
        # Renew the lease
        result = manager.renew_lease(lease.lease_id)
        assert result is True
        
        # Check renewed_at changed
        renewed_lease = manager.find_lease(lease.lease_id)
        assert renewed_lease.renewed_at > old_renewed


def test_lease_manager_expire():
    """Test lease expiration."""
    import tempfile
    import lease_manager
    from datetime import datetime, timezone, timedelta
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = lease_manager.LeaseManager(state_dir=tmpdir)
        manager.load()
        
        # Create an expired lease
        lease = lease_manager.Lease(
            lease_id="test-id",
            instance_id="inst-id",
            client_id="client",
            purpose="test",
            acquired_at=(datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
            renewed_at=(datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        )
        manager._state.leases.append(lease)
        manager.save()
        
        # Set short TTL for testing
        import os
        os.environ["LUALLM_MCP_LEASE_TTL_SECONDS"] = "3600"
        
        expired = manager.expire_stale_leases()
        assert "test-id" in expired

def _registry(tmp_path):
    import profiles
    return profiles.ProfileRegistry(state_dir=str(tmp_path))


def test_profile_registry_missing_file_means_no_profiles(tmp_path):
    registry = _registry(tmp_path)
    profiles_, errors = registry.load()
    assert profiles_ == []
    assert errors == []
    assert registry.get("anything") is None


def test_profile_registry_parses_full_entries(tmp_path):
    _write_profiles(tmp_path, {
        "vision": {
            "description": "Image projection enabled",
            "preset": "cold-start",
            "args": ["--mmproj", "/models/mmproj.gguf"],
        },
        "big-context": {"args": "-c 32768 -fa"},
    })
    by_name = {p.name: p for p in _registry(tmp_path).load()[0]}

    vision = by_name["vision"]
    assert vision.preset == "cold-start"
    assert vision.description == "Image projection enabled"
    assert vision.args == ["--mmproj", "/models/mmproj.gguf"]
    # String args split like a shell line.
    assert by_name["big-context"].args == ["-c", "32768", "-fa"]
    assert by_name["big-context"].preset is None
    assert _registry(tmp_path).get("nope") is None


def test_profile_registry_rejects_bad_values_and_reports(tmp_path):
    _write_profiles(tmp_path, {
        "t": "not an object",
        "u": {"unknown_key": 1},
        "d": {"description": 5},
        "p": {"preset": 7},
        "a": {"args": {"not": "a list"}},
        "lied": {"args": ["--preset", "hijack"]},
        "ok": {"args": ["-c", "4096"]},
    })
    profiles_, errors = _registry(tmp_path).load()

    names = [p.name for p in profiles_]
    assert names == ["ok"]
    # One error per broken entry — the operator can fix them all at once.
    assert len(errors) == 6
    assert any("profile 't'" in e for e in errors)


def test_profile_registry_hot_reloads(tmp_path):
    registry = _registry(tmp_path)
    assert registry.get("vision") is None
    _write_profiles(tmp_path, {"vision": {"args": ["--mmproj", "/m.gguf"]}})
    assert registry.get("vision") is not None


# --- idle grace period and memory-pressure eviction -------------------------

def _age_idle(mcp_server, model, seconds):
    """Backdate a model's idle_since so the grace period looks elapsed."""
    from datetime import datetime, timedelta, timezone
    inst = mcp_server.lease_manager.find_instance_for_model(model)
    inst.idle_since = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    mcp_server.lease_manager.save()


def test_default_idle_grace_is_five_minutes(monkeypatch, tmp_path):
    import lease_manager
    monkeypatch.delenv("LUALLM_MCP_IDLE_GRACE_SECONDS", raising=False)
    assert lease_manager.LeaseManager(state_dir=str(tmp_path)).idle_grace_seconds() == 300


def test_release_then_reacquire_reuses_warm_model(monkeypatch, tmp_path):
    """The point of the grace period: release → re-acquire must not stop and
    cold-start the same model."""
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300")

    first = _acquire(mcp_server, "tiny", None, "a", "t")
    rel = mcp_server._handle_release(first["data"]["lease_id"])
    assert rel["success"] is True
    assert rel["data"]["stopped"] is False
    assert rel["data"]["idle"] is True
    assert rel["data"]["idle_grace_s"] == 300
    assert client.stopped == []

    second = _acquire(mcp_server, "tiny", None, "a", "t")
    assert second["success"] is True
    assert second["data"]["reused"] is True
    assert second["data"]["instance_id"] == first["data"]["instance_id"]
    assert len(client.started) == 1
    assert mcp_server.lease_manager.find_instance_for_model("tiny").idle_since is None


def test_reaper_waits_out_the_grace_period_then_stops(monkeypatch, tmp_path):
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300")
    acq = _acquire(mcp_server, "tiny", None, "a", "t")
    mcp_server._handle_release(acq["data"]["lease_id"])

    assert mcp_server._reap_idle() == []
    _age_idle(mcp_server, "tiny", 299)
    assert mcp_server._reap_idle() == []
    _age_idle(mcp_server, "tiny", 301)
    assert mcp_server._reap_idle() == ["tiny"]
    assert client.stopped == ["tiny"]
    assert mcp_server.lease_manager.find_instance_for_model("tiny") is None


def test_reaper_never_stops_leased_or_external_models(monkeypatch, tmp_path):
    client = _fake_client(
        ["tiny", "ornith"], start={"success": True, "data": {"pid": 1, "port": 8081}},
        running=[{"model": "ornith", "state": "running", "port": 8080}])
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="1")
    _acquire(mcp_server, "tiny", None, "a", "t")
    ext = _acquire(mcp_server, "ornith", None, "a", "t")
    mcp_server._handle_release(ext["data"]["lease_id"])
    _age_idle(mcp_server, "tiny", 3600)
    _age_idle(mcp_server, "ornith", 3600)

    assert mcp_server._reap_idle() == []
    assert client.stopped == []


def test_reaper_keeps_record_when_stop_fails(monkeypatch, tmp_path):
    """A failed stop on a still-running server is retried next sweep."""
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}},
                          stop_fails=True)
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300")
    acq = _acquire(mcp_server, "tiny", None, "a", "t")
    mcp_server._handle_release(acq["data"]["lease_id"])
    _age_idle(mcp_server, "tiny", 600)

    assert mcp_server._reap_idle() == []
    assert mcp_server.lease_manager.find_instance_for_model("tiny") is not None


def test_expired_leases_leave_instance_idle_for_reaping(monkeypatch, tmp_path):
    """Leases dropped by TTL never went through release; the instance must
    still be treated as idle (stamped on first notice), not leak forever."""
    import lease_manager
    monkeypatch.setenv("LUALLM_MCP_IDLE_GRACE_SECONDS", "300")
    manager = lease_manager.LeaseManager(state_dir=str(tmp_path))
    instance, lease = manager.acquire_lease("tiny", None)
    manager._state.leases = []  # as if expire_stale_leases dropped it
    manager.save()
    assert instance.idle_since is None

    idle = manager.idle_instances()
    assert [i.instance_id for i in idle] == [instance.instance_id]
    assert idle[0].idle_since is not None
    assert manager.find_instance("tiny").idle_since == idle[0].idle_since
    assert manager.expired_idle_instances() == []


def test_crashed_start_is_not_kept_idle(monkeypatch, tmp_path):
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300")
    states = iter([1, 0])
    monkeypatch.setattr(mcp_server, "_probe_health", lambda port: next(states, 0))
    result = _acquire(mcp_server, "tiny", None, "a", "t")
    assert "SERVER_EXITED" in result["error"]
    assert mcp_server.lease_manager.find_instance_for_model("tiny") is None


def test_idle_model_under_other_config_is_swapped(monkeypatch, tmp_path):
    """With the grace period an idle model would otherwise block a new
    configuration with MODEL_RUNNING_DIFFERENT_CONFIG; nobody holds it, so
    it is stopped and restarted under the requested config."""
    _write_profiles(tmp_path, {"short-gen": {"args": ["-n", "512"]}})
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300")
    first = _acquire(mcp_server, "tiny", None, "a", "t")
    mcp_server._handle_release(first["data"]["lease_id"])

    second = _acquire(mcp_server, "tiny", None, "a", "t", profile_name="short-gen")
    assert second["success"] is True
    assert second["data"]["reused"] is False
    assert second["data"]["profile"] == "short-gen"
    assert second["data"]["evicted_idle"] == ["tiny"]
    assert client.stopped == ["tiny"]


def test_leased_model_under_other_config_still_refused(monkeypatch, tmp_path):
    _write_profiles(tmp_path, {"short-gen": {"args": ["-n", "512"]}})
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300")
    _acquire(mcp_server, "tiny", None, "a", "t")
    result = _acquire(mcp_server, "tiny", None, "a", "t", profile_name="short-gen")
    assert "MODEL_RUNNING_DIFFERENT_CONFIG" in result["error"]
    assert client.stopped == []


def _idle_pair(monkeypatch, tmp_path, budget_gb):
    """old (idle longest) and new (idle more recently), 20 GB each → 24 GB
    footprint with the default 1.2 overhead."""
    sizes = {"old": "20GB", "new": "20GB", "big": "20GB", "huge": "60GB"}
    client = _fake_client(sizes, start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300", budget_gb=budget_gb)
    a = _acquire(mcp_server, "old", None, "a", "t")
    b = _acquire(mcp_server, "new", None, "a", "t")
    mcp_server._handle_release(a["data"]["lease_id"])
    mcp_server._handle_release(b["data"]["lease_id"])
    return mcp_server, client


def test_memory_pressure_evicts_longest_idle_first(monkeypatch, tmp_path):
    # 48 GB used + 24 GB needed > 50 GB; evicting "old" alone is enough.
    mcp_server, client = _idle_pair(monkeypatch, tmp_path, budget_gb="50")
    result = _acquire(mcp_server, "big", None, "a", "t")
    assert result["success"] is True
    assert result["data"]["evicted_idle"] == ["old"]
    assert client.stopped == ["old"]
    assert mcp_server.lease_manager.find_instance_for_model("new") is not None


def test_memory_pressure_evicts_until_it_fits(monkeypatch, tmp_path):
    # huge needs 72 GB of an 80 GB budget: both idle models must go.
    mcp_server, client = _idle_pair(monkeypatch, tmp_path, budget_gb="80")
    result = _acquire(mcp_server, "huge", None, "a", "t")
    assert result["data"]["evicted_idle"] == ["old", "new"]


def test_no_eviction_when_memory_fits(monkeypatch, tmp_path):
    mcp_server, client = _idle_pair(monkeypatch, tmp_path, budget_gb="100")
    result = _acquire(mcp_server, "big", None, "a", "t")
    assert "evicted_idle" not in result["data"]
    assert client.stopped == []


def test_memory_pressure_never_evicts_leased_models(monkeypatch, tmp_path):
    """Only idle models are freed; a start that still won't fit proceeds
    (the estimate is a heuristic, not an admission limit)."""
    sizes = {"held": "20GB", "big": "20GB"}
    client = _fake_client(sizes, start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client, grace="300", budget_gb="30")
    _acquire(mcp_server, "held", None, "a", "t")
    result = _acquire(mcp_server, "big", None, "a", "t")
    assert result["success"] is True
    assert client.stopped == []


def test_parse_size():
    import mcp_server
    assert mcp_server._parse_size("26.6GB") == int(26.6 * 1024**3)
    assert mcp_server._parse_size("512MB") == 512 * 1024**2
    assert mcp_server._parse_size("1.5 TiB") == int(1.5 * 1024**4)
    assert mcp_server._parse_size(1234) == 1234
    assert mcp_server._parse_size("unknown") is None
    assert mcp_server._parse_size(None) is None
