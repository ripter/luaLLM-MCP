"""Operator-curated profiles for starting models with preconfigured flags.

A profile bundles a name, description, optional luallm preset, and extra
llama.cpp flags. MCP clients can only apply a profile by name — they never
supply raw flags, so every flag that reaches llama-server is one the
operator put in profiles.json.

File discovery: profiles.json lives in the MCP state dir with the lease
state; the file hot-reloads, so operator edits apply without a restart.
"""

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

import lease_manager

logger = logging.getLogger(__name__)

_ALLOWED_ENTRY_KEYS = {"description", "preset", "args"}


@dataclass
class Profile:
    name: str
    args: list[str]
    preset: str | None = None
    description: str | None = None

    def to_dict(self) -> dict:
        out = {"name": self.name, "args": self.args}
        if self.preset is not None:
            out["preset"] = self.preset
        if self.description is not None:
            out["description"] = self.description
        return out


class ProfileRegistry:
    def __init__(self, state_dir: str | None = None):
        self.path = Path(state_dir or lease_manager.get_state_dir()) / "profiles.json"

    def load(self) -> tuple[list[Profile], list[str]]:
        """Parse profiles.json fresh each call (operator edits hot-reload).

        Returns (profiles, errors). A missing file is fine: the registry
        just has no profiles until the operator creates one. A broken file
        yields no profiles plus an error entry, so list_profiles and
        acquire_model can surface why instead of failing silently.
        """
        if not self.path.exists():
            return [], []

        try:
            with open(self.path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return [], [f"could not parse {self.path}: {e}"]

        if not isinstance(data, dict):
            return [], [f"{self.path} must contain a JSON object"]

        entries = data.get("profiles", {})
        if not isinstance(entries, dict):
            return [], [f"'profiles' in {self.path} must be an object of name -> profile"]

        profiles: list[Profile] = []
        errors: list[str] = []
        for name, spec in entries.items():
            if not isinstance(spec, dict):
                errors.append(f"profile '{name}': must be an object, got {type(spec).__name__}")
                continue
            unknown = set(spec) - _ALLOWED_ENTRY_KEYS
            if unknown:
                errors.append(
                    f"profile '{name}': unknown key(s) {sorted(unknown)} "
                    f"(allowed: {sorted(_ALLOWED_ENTRY_KEYS)})")
                continue

            description = spec.get("description")
            if description is not None and not isinstance(description, str):
                errors.append(f"profile '{name}': 'description' must be a string")
                continue
            preset = spec.get("preset")
            if preset is not None and not isinstance(preset, str):
                errors.append(f"profile '{name}': 'preset' must be a string")
                continue

            raw_args = spec.get("args", [])
            args: list[str] | None = None
            if isinstance(raw_args, str):
                # Operator convenience: "-c 32768" splits like a shell line.
                args = shlex.split(raw_args)
                if any(t in ("--preset", "--json") for t in args):
                    errors.append(
                        f"profile '{name}': --preset/--json are handled by the MCP; drop them from args")
                    continue
            elif isinstance(raw_args, list) and all(
                    isinstance(t, str) and t for t in raw_args):
                args = list(raw_args)
                if "--preset" in args or "--json" in args:
                    errors.append(
                        f"profile '{name}': --preset/--json are handled by the MCP; drop them from args")
                    continue
            if args is None:
                errors.append(
                    f"profile '{name}': 'args' must be a flag string or a list of flag strings")
                continue

            profiles.append(Profile(
                name=name, args=args, preset=preset, description=description))

        profiles.sort(key=lambda p: p.name)
        return profiles, errors

    def get(self, name: str | None) -> Profile | None:
        if name is None:
            return None
        profiles, errors = self.load()
        if errors:
            logger.warning("profiles.json problems: %s", errors)
        return next((p for p in profiles if p.name == name), None)
