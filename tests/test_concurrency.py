"""Cross-process lifecycle safety, with real processes sharing one state dir.

Each test here was checked to fail with its lock removed (see the README's
Development section).
"""

import json

from conftest import run_workers


def test_lease_writes_from_many_processes_are_not_lost(tmp_path, fake_luallm):
    state = tmp_path / "state"
    state.mkdir()
    results = run_workers(state, fake_luallm, [["leases", "25"]] * 6)
    assert all(r["ok"] for r in results)
    saved = json.loads((state / "state.json").read_text())
    assert len(saved["leases"]) == 150


def test_concurrent_acquires_start_the_model_once(tmp_path, fake_luallm):
    state = tmp_path / "state"
    state.mkdir()
    env = fake_luallm.env(FAKE_LUALLM_START_DELAY="0.2")
    results = run_workers(state, fake_luallm, [["acquire", "tiny"]] * 5, env=env)

    assert [r.get("error") for r in results] == [None] * 5
    assert len({r["instance_id"] for r in results}) == 1
    assert sum(1 for r in results if not r["reused"]) == 1
    assert len(fake_luallm.lines("starts.log")) == 1


def test_release_never_stops_a_model_another_process_leased(tmp_path, fake_luallm):
    """With no grace period, the last release stops the server. Another
    process acquiring at the same moment must either get a fresh start or
    keep the server running: never a lease on a server being stopped."""
    state = tmp_path / "state"
    state.mkdir()
    env = fake_luallm.env(LUALLM_MCP_IDLE_GRACE_SECONDS="0",
                          LUALLM_MCP_MEMORY_BUDGET_GB="1000",
                          FAKE_LUALLM_START_DELAY="0.02")
    results = run_workers(state, fake_luallm, [["cycle", "tiny", "15"]] * 3,
                          env=env, timeout=120)
    for r in results:
        assert r["violations"] == []
        assert r["errors"] == []
    saved = json.loads((state / "state.json").read_text())
    assert saved["leases"] == []
