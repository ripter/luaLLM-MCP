"""Lease tracking and state management for luaLLM MCP server."""

import json
import os
import uuid
import shutil
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict


class SchemaVersion:
    V1 = "1.0"


@dataclass
class Instance:
    instance_id: str
    model: str
    preset: str | None
    managed_by_mcp: bool
    started_at: str
    port: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        return cls(**data)


@dataclass
class Lease:
    lease_id: str
    instance_id: str
    client_id: str | None
    purpose: str | None
    acquired_at: str
    renewed_at: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lease":
        return cls(**data)


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


class LeaseManager:
    def __init__(self, state_dir: str | None = None):
        self.state_dir = Path(state_dir or self._get_state_dir())
        self.state_file = self.state_dir / "state.json"
        self._state: State | None = None
    
    def _get_state_dir(self) -> str:
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
    
    def _get_lease_ttl(self) -> int:
        """Get lease TTL in seconds from environment."""
        return int(os.environ.get("LUALLM_MCP_LEASE_TTL_SECONDS", "7200"))
    
    def _atomic_save(self, state: State) -> None:
        """Write state atomically using temp file + rename."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        shutil.move(str(temp_file), str(self.state_file))
    
    def load(self) -> State:
        """Load state from file, creating default if not exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self._state = State.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                self._state = State()
        else:
            self._state = State()
        return self._state
    
    def save(self) -> None:
        """Save current state to file."""
        if self._state is None:
            self.load()
        self._atomic_save(self._state)
    
    def find_instance(self, model: str, preset: str | None = None) -> Instance | None:
        """Find running instance matching model and preset."""
        if self._state is None:
            self.load()
        for instance in self._state.instances:
            if instance.model == model and instance.preset == preset:
                return instance
        return None
    
    def find_lease(self, lease_id: str) -> Lease | None:
        """Find lease by ID."""
        if self._state is None:
            self.load()
        for lease in self._state.leases:
            if lease.lease_id == lease_id:
                return lease
        return None
    
    def acquire_lease(self, model: str, preset: str | None,
                      client_id: str | None = None,
                      purpose: str | None = None) -> tuple[Instance, Lease]:
        """Acquire a lease for a model instance."""
        if self._state is None:
            self.load()
        
        now = datetime.now(timezone.utc).isoformat()
        
        instance = self.find_instance(model, preset)
        if instance is None:
            instance = Instance(
                instance_id=str(uuid.uuid4()),
                model=model,
                preset=preset,
                managed_by_mcp=True,
                started_at=now,
                port=None,
            )
            self._state.instances.append(instance)
        
        lease = Lease(
            lease_id=str(uuid.uuid4()),
            instance_id=instance.instance_id,
            client_id=client_id,
            purpose=purpose,
            acquired_at=now,
            renewed_at=now,
        )
        self._state.leases.append(lease)
        self.save()
        
        return instance, lease
    
    def release_lease(self, lease_id: str) -> dict[str, Any]:
        """Release a lease by ID."""
        if self._state is None:
            self.load()
        
        lease = self.find_lease(lease_id)
        if lease is None:
            return {"success": False, "error": "LEASE_NOT_FOUND", "lease_id": lease_id}
        
        self._state.leases = [l for l in self._state.leases if l.lease_id != lease_id]
        
        instance = next((i for i in self._state.instances 
                        if i.instance_id == lease.instance_id), None)
        
        should_stop = False
        if instance and instance.managed_by_mcp:
            remaining_leases = [l for l in self._state.leases 
                              if l.instance_id == lease.instance_id]
            if len(remaining_leases) == 0:
                should_stop = True
        
        if should_stop:
            self._state.instances = [i for i in self._state.instances 
                                   if i.instance_id != lease.instance_id]
        
        self.save()
        
        return {
            "success": True,
            "instance_stopped": should_stop,
            "instance_id": lease.instance_id,
            "model": instance.model if instance else None,
        }
    
    def renew_lease(self, lease_id: str) -> bool:
        """Renew a lease's timestamp."""
        if self._state is None:
            self.load()
        
        lease = self.find_lease(lease_id)
        if lease is None:
            return False
        
        lease.renewed_at = datetime.now(timezone.utc).isoformat()
        self.save()
        return True
    
    def expire_stale_leases(self) -> list[str]:
        """Expire leases older than TTL. Returns list of expired lease IDs."""
        if self._state is None:
            self.load()
        
        ttl = self._get_lease_ttl()
        now = datetime.now(timezone.utc)
        
        expired = []
        active_leases = []
        
        for lease in self._state.leases:
            renewed = datetime.fromisoformat(lease.renewed_at.replace("Z", "+00:00"))
            age = (now - renewed).total_seconds()
            if age > ttl:
                expired.append(lease.lease_id)
            else:
                active_leases.append(lease)
        
        self._state.leases = active_leases
        self.save()
        return expired
    
    def list_leases(self) -> list[dict[str, Any]]:
        """List all leases with sanitized metadata."""
        if self._state is None:
            self.load()
        return [l.to_dict() for l in self._state.leases]
    
    def get_instance(self, instance_id: str) -> Instance | None:
        """Get instance by ID."""
        if self._state is None:
            self.load()
        for instance in self._state.instances:
            if instance.instance_id == instance_id:
                return instance
        return None