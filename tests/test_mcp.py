"""Tests for luaLLM-MCP."""

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


def _fake_client(models, running=None, start=None, stop_fails=False):
    servers = list(running or [])

    class FakeClient:
        stopped = []

        def list_models(self):
            return {"success": True, "data": [{"name": m} for m in models]}

        def get_server_status(self):
            return {"success": True, "data": {"servers": servers}}

        def start_model(self, model, preset=None):
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


def _isolated(monkeypatch, tmp_path, client):
    import mcp_server
    import lease_manager
    monkeypatch.setattr(mcp_server, "lease_manager",
                        lease_manager.LeaseManager(state_dir=str(tmp_path)))
    monkeypatch.setattr(mcp_server, "client", client)
    return mcp_server


def test_acquire_adopts_external_instance(monkeypatch, tmp_path):
    """A model running outside MCP management is adopted: lease + port are
    returned, and releasing never stops the external server."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["ornith"], running=[{"model": "ornith", "state": "running", "port": 8080}]))

    result = mcp_server._handle_acquire("ornith", None, "test", "adopt")
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
    result = mcp_server._handle_acquire("ornith", "throughput", "test", "t")
    assert result["success"] is False
    assert "MODEL_RUNNING_EXTERNALLY" in result["error"]


def test_acquire_reuse_reports_port(monkeypatch, tmp_path):
    """Second acquire on an MCP-started model must include the port so
    clients can build the endpoint."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}}))

    first = mcp_server._handle_acquire("tiny", None, "test", "t")
    assert first["success"] is True
    assert first["data"]["reused"] is False
    assert first["data"]["port"] == 8081

    second = mcp_server._handle_acquire("tiny", None, "test", "t")
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

    result = mcp_server._handle_acquire("skyfall", None, "test", "t")
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

    result = mcp_server._handle_acquire("tiny", None, "test", "t")
    assert result["data"]["reused"] is True
    assert result["data"]["port"] == 9091


def test_release_stops_last_lease_and_reports_it(monkeypatch, tmp_path):
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client)

    first = mcp_server._handle_acquire("tiny", None, "a", "t")
    second = mcp_server._handle_acquire("tiny", None, "b", "t")

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
    acq = mcp_server._handle_acquire("tiny", None, "a", "t")
    rel = mcp_server._handle_release(acq["data"]["lease_id"])
    assert rel["success"] is False
    assert "STOP_FAILED" in rel["error"]
    assert "stop exploded" in rel["error"]


def test_release_never_stops_external_instances(monkeypatch, tmp_path):
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["ornith"], running=[{"model": "ornith", "state": "running", "port": 8080}]))
    acq = mcp_server._handle_acquire("ornith", None, "a", "t")
    assert acq["data"]["external"] is True
    rel = mcp_server._handle_release(acq["data"]["lease_id"])
    assert rel["success"] is True
    assert rel["data"]["stopped"] is False
    assert rel["data"]["external"] is True


def test_force_stop_purges_records_and_stops(monkeypatch, tmp_path):
    client = _fake_client(["tiny"], start={"success": True, "data": {"pid": 1, "port": 8081}})
    mcp_server = _isolated(monkeypatch, tmp_path, client)
    # Two leases, would normally block stopping.
    mcp_server._handle_acquire("tiny", None, "a", "t")
    mcp_server._handle_acquire("tiny", None, "b", "t")

    result = mcp_server._handle_force_stop("tiny")
    assert result["success"] is True
    assert result["data"]["stopped"] is True
    assert result["data"]["removed_leases"] == 2
    assert client.stopped == ["tiny"]
    assert mcp_server.lease_manager.list_leases() == []


def test_acquire_backfills_port_from_status(monkeypatch, tmp_path):
    """Instance records created before port tracking get their port from
    get_server_status on the next acquire."""
    mcp_server = _isolated(monkeypatch, tmp_path, _fake_client(
        ["tiny"], running=[{"model": "tiny", "state": "running", "port": 9090}]))
    # Simulate a pre-existing record without a port.
    mcp_server.lease_manager.acquire_lease("tiny", None, "old", "old")
    assert mcp_server.lease_manager.find_instance("tiny").port is None

    result = mcp_server._handle_acquire("tiny", None, "test", "t")
    assert result["data"]["reused"] is True
    assert result["data"]["port"] == 9090


def test_lease_manager_imports():
    """Test that lease_manager module can be imported."""
    import lease_manager
    manager = lease_manager.LeaseManager()
    state = manager.load()
    assert state.schema_version == "1.0"


def test_lease_manager_state_dir():
    """Test that state directory is correctly determined."""
    import lease_manager
    import os
    
    manager = lease_manager.LeaseManager()
    state_dir = manager._get_state_dir()
    
    # Should use XDG_CONFIG_HOME or ~/.config
    assert "luaLLM-MCP" in state_dir


def test_lease_manager_acquire_release():
    """Test lease acquire and release cycle."""
    import tempfile
    import lease_manager
    
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