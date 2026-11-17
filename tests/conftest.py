import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

FAKES = Path(__file__).resolve().parent / "fakes"


class FakeLuaLLM:
    """A fake luallm executable with its own state dir, and cleanup of every
    fake server it started."""

    def __init__(self, root: Path):
        self.dir = root / "luallm"
        self.dir.mkdir()
        self.path = root / "luallm-fake"
        self.path.write_text(
            f"#!/bin/sh\nexec {sys.executable} {FAKES / 'fake_luallm.py'} \"$@\"\n")
        self.path.chmod(self.path.stat().st_mode | stat.S_IXUSR)

    def env(self, **extra) -> dict:
        env = dict(os.environ)
        env["FAKE_LUALLM_DIR"] = str(self.dir)
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def lines(self, name: str) -> list[str]:
        p = self.dir / name
        return p.read_text().splitlines() if p.exists() else []

    def events(self, model: str) -> list[dict]:
        return [json.loads(l) for l in self.lines(f"{model}.events")]

    def servers(self) -> list[dict]:
        p = self.dir / "servers.json"
        return json.loads(p.read_text()) if p.exists() else []

    def cleanup(self) -> None:
        # pids.log, not servers.json: a duplicate start (the race some tests
        # look for) replaces its predecessor's entry in servers.json.
        for pid in {int(p) for p in self.lines("pids.log")}:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def fake_luallm(tmp_path, monkeypatch):
    fake = FakeLuaLLM(tmp_path)
    monkeypatch.setenv("FAKE_LUALLM_DIR", str(fake.dir))
    yield fake
    fake.cleanup()


def run_workers(state_dir: Path, fake: FakeLuaLLM, commands: list[list[str]],
                env: dict | None = None, timeout: float = 60) -> list[dict]:
    """Start one lifecycle_worker per command, release them together, and
    return each one's JSON result."""
    go = state_dir / "go"
    procs = [subprocess.Popen(
        [sys.executable, str(FAKES / "lifecycle_worker.py"), str(state_dir),
         str(fake.path), str(go), *cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env or fake.env()) for cmd in commands]
    go.touch()
    results = []
    for p in procs:
        out, err = p.communicate(timeout=timeout)
        assert p.returncode == 0, err
        results.append(json.loads(out.strip().splitlines()[-1]))
    return results
