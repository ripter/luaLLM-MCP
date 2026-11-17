import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import lease_manager


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_leases_of_a_dead_process_are_dropped(tmp_path):
    m = lease_manager.LeaseManager(state_dir=str(tmp_path))
    _, mcp_lease = m.acquire_lease("tiny", None)
    _, dead = m.acquire_lease("tiny", None, owner_pid=_dead_pid())
    import os
    _, live = m.acquire_lease("tiny", None, owner_pid=os.getpid())

    assert m.drop_orphaned_leases() == [dead.lease_id]
    remaining = {l["lease_id"] for l in m.list_leases()}
    assert remaining == {mcp_lease.lease_id, live.lease_id}


def test_unrenewed_owned_leases_expire_but_mcp_leases_are_left_alone(tmp_path, monkeypatch):
    """A recycled pid could keep a dead gateway's lease alive; the TTL still
    catches it. MCP leases keep their old behaviour: only expire_stale_leases
    (at MCP startup) touches them."""
    import os
    monkeypatch.setenv("LUALLM_MCP_LEASE_TTL_SECONDS", "60")
    m = lease_manager.LeaseManager(state_dir=str(tmp_path))
    _, mcp_lease = m.acquire_lease("tiny", None)
    _, owned = m.acquire_lease("tiny", None, owner_pid=os.getpid())
    old = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    state = m.load()
    for l in state.leases:
        l.renewed_at = old
    m.save()

    assert m.drop_orphaned_leases() == [owned.lease_id]
    assert [l["lease_id"] for l in m.list_leases()] == [mcp_lease.lease_id]


def test_state_with_unknown_fields_still_loads(tmp_path):
    """A newer version may add fields; an older reader must not discard the
    whole file (it would then overwrite every lease)."""
    (tmp_path / "state.json").write_text(json.dumps({
        "schema_version": "1.0",
        "instances": [{"instance_id": "i1", "model": "tiny", "preset": None,
                       "managed_by_mcp": True, "started_at": "2026-01-01T00:00:00+00:00",
                       "from_the_future": 1}],
        "leases": [{"lease_id": "l1", "instance_id": "i1", "client_id": None,
                    "purpose": None, "acquired_at": "2026-01-01T00:00:00+00:00",
                    "renewed_at": "2026-01-01T00:00:00+00:00", "also_new": True}],
    }))
    m = lease_manager.LeaseManager(state_dir=str(tmp_path))
    assert m.find_lease("l1") is not None
    assert m.get_instance("i1").model == "tiny"


def test_instance_records_its_start_flags(tmp_path):
    m = lease_manager.LeaseManager(state_dir=str(tmp_path))
    inst, _ = m.acquire_lease("tiny", None, profile="big", args=["-c", "8192"])
    assert m.get_instance(inst.instance_id).args == ["-c", "8192"]
