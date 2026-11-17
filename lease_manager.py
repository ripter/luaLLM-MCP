"""Lease tracking and state management for luaLLM MCP server.

Several processes share one state file: every stdio MCP client gets its own
server process, and the OpenAI gateway is another. Two locks keep them
consistent:

- The state lock guards one read-modify-write of state.json. Every public
  method reloads the file under it, so a change made by another process is
  never overwritten with a stale copy.
- The lifecycle lock serializes decisions that start or stop model servers
  (acquire's start-or-reuse decision, release that stops, eviction, idle
  reaping, force stop). Without it two processes can both see "not running"
  and both start the model, or one can stop a server another has just
  leased. It is held across the luallm start/stop commands, which return
  quickly, but never across waiting for a model to load.

Lock order is always lifecycle, then state. Both are flock() locks on files
in the state dir, so they also exclude threads within one process (each
acquisition opens its own descriptor). Both are re-entrant within a thread.
"""

import fcntl
import json
import os
import threading
import uuid
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict, fields


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class SchemaVersion:
    V1 = "1.0"


def get_state_dir() -> str:
    """Get OS-appropriate application data directory following luaLLM pattern."""
    env_path = os.environ.get("LUALLM_MCP_STATE_DIR")
    if env_path:
        return env_path

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.environ.get("APPDATA", "~/.config"))
        return str(Path(base).expanduser() / "luaLLM-MCP")
    else:
        base = os.environ.get("XDG_CONFIG_HOME", "~/.config")
        return str(Path(base).expanduser() / "luaLLM-MCP")


def _known_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
    # A state file written by a newer version may carry fields this one
    # doesn't know; drop them rather than refuse to load the whole file.
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class Instance:
    instance_id: str
    model: str
    preset: str | None
    managed_by_mcp: bool
    started_at: str
    port: int | None = None
    # Profile name when the server was started through a curated profile
    # (profiles.py). None means "no profile requested"; older state files
    # that predate profiles load with the default and keep matching.
    profile: str | None = None
    # ISO timestamp set when the last lease was released and the server was
    # kept warm for the idle grace period; None while leased. Older state
    # files load with the default.
    idle_since: str | None = None
    # The extra llama.cpp flags the server was started with. A profile can
    # be edited while its server runs; the gateway compares these with the
    # profile's current flags so a request never runs under stale ones.
    # None: unknown (adopted external server, or an older record).
    args: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        return cls(**_known_fields(cls, data))


@dataclass
class Lease:
    lease_id: str
    instance_id: str
    client_id: str | None
    purpose: str | None
    acquired_at: str
    renewed_at: str
    # Set for leases tied to one live process (the gateway holds one per
    # request). If that process dies the lease is dropped by the next sweep
    # instead of pinning the model until the TTL. None for MCP leases, which
    # deliberately outlive the stdio process that took them.
    owner_pid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lease":
        return cls(**_known_fields(cls, data))


@dataclass
class State:
    schema_version: str = SchemaVersion.V1
    instances: list[Instance] = field(default_factory=list)
    leases: list[Lease] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "instances": [i.to_dict() for i in self.instances],
            "leases": [l.to_dict() for l in self.leases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        instances = [Instance.from_dict(i) for i in data.get("instances", [])]
        leases = [Lease.from_dict(l) for l in data.get("leases", [])]
        return cls(
            schema_version=data.get("schema_version", SchemaVersion.V1),
            instances=instances,
            leases=leases,
        )


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but not ours to signal
    return True


class _FileLock:
    """A re-entrant (per thread) exclusive flock on one file."""

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()

    @contextmanager
    def hold(self) -> Iterator[None]:
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth -= 1
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._local.depth = 1
            try:
                yield
            finally:
                self._local.depth = 0
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class LeaseManager:
    def __init__(self, state_dir: str | None = None):
        self.state_dir = Path(state_dir or get_state_dir())
        self.state_file = self.state_dir / "state.json"
        self._state: State | None = None
        self._state_lock = _FileLock(self.state_dir / "state.lock")
        self._lifecycle_lock = _FileLock(self.state_dir / "lifecycle.lock")
        self._txn_local = threading.local()

    @contextmanager
    def lifecycle(self) -> Iterator[None]:
        """Hold the cross-process lifecycle lock (see module docstring)."""
        with self._lifecycle_lock.hold():
            yield

    @contextmanager
    def _txn(self, write: bool = True) -> Iterator[State]:
        """Reload state under the state lock, and save it afterwards if
        write. Nested calls in one thread share the outer transaction. An
        exception discards the changes."""
        if getattr(self._txn_local, "depth", 0):
            self._txn_local.depth += 1
            try:
                yield self._state
            finally:
                self._txn_local.depth -= 1
            return
        with self._state_lock.hold():
            self._txn_local.depth = 1
            try:
                state = self._read_file()
                self._state = state
                yield state
                if write:
                    self._atomic_save(state)
            finally:
                self._txn_local.depth = 0

    def _get_lease_ttl(self) -> int:
        """Get lease TTL in seconds from environment."""
        return int(os.environ.get("LUALLM_MCP_LEASE_TTL_SECONDS", "7200"))

    def idle_grace_seconds(self) -> float:
        """How long an MCP-managed model stays running after its last lease
        is released, so a release → re-acquire cycle reuses the warm
        server instead of a stop + cold start. 0 stops immediately."""
        return max(0.0, float(os.environ.get("LUALLM_MCP_IDLE_GRACE_SECONDS", "300")))

    def _atomic_save(self, state: State) -> None:
        """Write state atomically using a unique temp file + rename, so a
        reader never sees a partial file."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(temp_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        shutil.move(str(temp_file), str(self.state_file))

    def _read_file(self) -> State:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return State.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError, TypeError):
                return State()
        return State()

    def load(self) -> State:
        """Load state from file, creating default if not exists."""
        with self._txn(write=False) as state:
            return state

    def save(self) -> None:
        """Save the state last loaded by this manager. Internal code uses
        transactions instead; this stays for callers that edit a loaded
        record directly."""
        with self._state_lock.hold():
            if self._state is None:
                self._state = self._read_file()
            self._atomic_save(self._state)

    @staticmethod
    def _find_instance(state: State, model: str, preset: str | None,
                       profile: str | None) -> Instance | None:
        for instance in state.instances:
            if (instance.model == model and instance.preset == preset
                    and instance.profile == profile):
                return instance
        return None

    def find_instance(self, model: str, preset: str | None = None,
                      profile: str | None = None) -> Instance | None:
        """Find running instance matching model, preset, and profile.

        The (preset, profile) pair identifies the launch configuration:
        same model under a different preset or profile is a different
        instance and must not be reused.
        """
        with self._txn(write=False) as state:
            return self._find_instance(state, model, preset, profile)

    def find_instance_for_model(self, model: str) -> Instance | None:
        """First instance record for a model under any configuration.

        Used to detect that a running server belongs to this MCP (started
        under some other preset/profile) so callers get told to stop it
        instead of adopting it under the wrong configuration.
        """
        with self._txn(write=False) as state:
            return next((i for i in state.instances if i.model == model), None)

    def find_lease(self, lease_id: str) -> Lease | None:
        """Find lease by ID."""
        with self._txn(write=False) as state:
            return next((l for l in state.leases if l.lease_id == lease_id), None)

    def acquire_lease(self, model: str, preset: str | None,
                      client_id: str | None = None,
                      purpose: str | None = None,
                      port: int | None = None,
                      managed_by_mcp: bool = True,
                      profile: str | None = None,
                      live_port: int | None = None,
                      args: list[str] | None = None,
                      owner_pid: int | None = None) -> tuple[Instance, Lease]:
        """Acquire a lease for a model instance.

        port/managed_by_mcp/args describe the instance when a new record is
        created: managed_by_mcp=False adopts an externally started server,
        which release_lease will never stop. profile records the curated
        profile the server was started with and is part of the instance
        identity alongside preset. live_port overrides the recorded port of
        an existing instance (the server may have moved). owner_pid ties the
        lease to a process (see Lease.owner_pid).
        """
        now = _now_iso()
        with self._txn() as state:
            instance = self._find_instance(state, model, preset, profile)
            if instance is None:
                instance = Instance(
                    instance_id=str(uuid.uuid4()),
                    model=model,
                    preset=preset,
                    managed_by_mcp=managed_by_mcp,
                    started_at=now,
                    port=port,
                    profile=profile,
                    args=list(args) if args is not None else None,
                )
                state.instances.append(instance)
            else:
                if live_port is not None:
                    instance.port = live_port
                elif instance.port is None and port is not None:
                    # Backfill ports on records created before ports were tracked.
                    instance.port = port
                # A new lease ends any idle grace period.
                instance.idle_since = None

            lease = Lease(
                lease_id=str(uuid.uuid4()),
                instance_id=instance.instance_id,
                client_id=client_id,
                purpose=purpose,
                acquired_at=now,
                renewed_at=now,
                owner_pid=owner_pid,
            )
            state.leases.append(lease)
            return instance, lease

    def remove_instance(self, instance_id: str) -> None:
        """Drop an instance record and any leases attached to it. Used when
        the server behind a record turns out to be gone (stale state)."""
        with self._txn() as state:
            state.instances = [i for i in state.instances if i.instance_id != instance_id]
            state.leases = [l for l in state.leases if l.instance_id != instance_id]

    def remove_model_records(self, model: str) -> dict[str, int]:
        """Drop every instance record and lease for a model (force stop)."""
        with self._txn() as state:
            instance_ids = {i.instance_id for i in state.instances if i.model == model}
            removed_leases = [l for l in state.leases if l.instance_id in instance_ids]
            state.instances = [i for i in state.instances if i.model != model]
            state.leases = [l for l in state.leases if l.instance_id not in instance_ids]
            return {
                "removed_instances": len(instance_ids),
                "removed_leases": len(removed_leases),
            }

    def release_lease(self, lease_id: str, keep_instance: bool = False) -> dict[str, Any]:
        """Release a lease by ID.

        When the last lease on an MCP-managed instance goes, the instance
        either stops now (idle grace 0 → record removed, instance_stopped
        True; the caller stops the server) or is kept warm: idle_since is
        stamped and the idle reaper stops it once the grace period lapses.
        keep_instance never stops — used when taking back a lease on a
        model that is still loading — but still marks the instance idle so
        it is reaped if nobody comes back for it.

        A caller that will stop the server must hold the lifecycle lock
        across this call and the stop, or another process could lease the
        server in between.
        """
        with self._txn() as state:
            lease = next((l for l in state.leases if l.lease_id == lease_id), None)
            if lease is None:
                return {"success": False, "error": "LEASE_NOT_FOUND", "lease_id": lease_id}

            state.leases = [l for l in state.leases if l.lease_id != lease_id]
            instance = next((i for i in state.instances
                             if i.instance_id == lease.instance_id), None)
            remaining = [l for l in state.leases if l.instance_id == lease.instance_id]

            should_stop = False
            idle = False
            if instance and instance.managed_by_mcp and len(remaining) == 0:
                if keep_instance or self.idle_grace_seconds() > 0:
                    instance.idle_since = _now_iso()
                    idle = True
                else:
                    should_stop = True
                    state.instances = [i for i in state.instances
                                       if i.instance_id != lease.instance_id]

        result = {
            "success": True,
            "instance_stopped": should_stop,
            "instance_id": lease.instance_id,
            "model": instance.model if instance else None,
            "remaining_leases": len(remaining),
            "external": bool(instance) and not instance.managed_by_mcp,
            "idle": idle,
        }
        if idle:
            result["idle_grace_s"] = self.idle_grace_seconds()
        return result

    def lease_count(self, instance_id: str) -> int:
        with self._txn(write=False) as state:
            return sum(1 for l in state.leases if l.instance_id == instance_id)

    def idle_instances(self) -> list[Instance]:
        """MCP-managed instances with no leases, longest-idle first.

        Any such instance is idle, even one never stamped (leases expired
        by TTL, or a state file from before idle tracking): it gets stamped
        now, so it receives a full grace period from first notice.
        """
        with self._txn() as state:
            leased = {l.instance_id for l in state.leases}
            idle = [i for i in state.instances
                    if i.managed_by_mcp and i.instance_id not in leased]
            for i in idle:
                if i.idle_since is None:
                    i.idle_since = _now_iso()
            return sorted(idle, key=lambda i: _parse_iso(i.idle_since))

    def expired_idle_instances(self, now: datetime | None = None) -> list[Instance]:
        """Idle instances whose grace period has lapsed, longest-idle first.
        Records are left in place; the caller stops each server and then
        removes the record, so a failed stop is retried on the next sweep."""
        now = now or datetime.now(timezone.utc)
        grace = self.idle_grace_seconds()
        return [i for i in self.idle_instances()
                if (now - _parse_iso(i.idle_since)).total_seconds() >= grace]

    def renew_lease(self, lease_id: str) -> bool:
        """Renew a lease's timestamp."""
        with self._txn() as state:
            lease = next((l for l in state.leases if l.lease_id == lease_id), None)
            if lease is None:
                return False
            lease.renewed_at = _now_iso()
            return True

    def expire_stale_leases(self) -> list[str]:
        """Expire leases older than TTL. Returns list of expired lease IDs."""
        ttl = self._get_lease_ttl()
        now = datetime.now(timezone.utc)
        with self._txn() as state:
            expired = []
            active = []
            for lease in state.leases:
                if (now - _parse_iso(lease.renewed_at)).total_seconds() > ttl:
                    expired.append(lease.lease_id)
                else:
                    active.append(lease)
            state.leases = active
            return expired

    def drop_orphaned_leases(self, alive: Callable[[int], bool] = pid_alive) -> list[str]:
        """Drop process-owned leases whose process is gone, or that have
        gone unrenewed past the TTL (the pid may have been reused). Leases
        without an owner (MCP leases) are left alone."""
        ttl = self._get_lease_ttl()
        now = datetime.now(timezone.utc)
        with self._txn() as state:
            dropped = []
            kept = []
            for lease in state.leases:
                if lease.owner_pid is not None and (
                        not alive(lease.owner_pid)
                        or (now - _parse_iso(lease.renewed_at)).total_seconds() > ttl):
                    dropped.append(lease.lease_id)
                else:
                    kept.append(lease)
            state.leases = kept
            return dropped

    def leases_owned_by(self, pid: int) -> list[str]:
        with self._txn(write=False) as state:
            return [l.lease_id for l in state.leases if l.owner_pid == pid]

    def list_leases(self) -> list[dict[str, Any]]:
        """List all leases with sanitized metadata."""
        with self._txn(write=False) as state:
            return [l.to_dict() for l in state.leases]

    def get_instance(self, instance_id: str) -> Instance | None:
        """Get instance by ID."""
        with self._txn(write=False) as state:
            return next((i for i in state.instances if i.instance_id == instance_id), None)
