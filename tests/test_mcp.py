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