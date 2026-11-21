"""The gateway's configuration: which models it exposes, its token, and the
capabilities verified from each model's own llama-server.

gateway.json (in the state dir, next to profiles.json):

    {
      "listen": "127.0.0.1:8090",
      "default_model": "qwen",
      "models": {
        "qwen": {"model": "Qwen3.8-27B-Uncensored-Q6_K", "profile": "director",
                 "max_output_tokens": 4096, "description": "..."}
      }
    }

Only names listed under "models" are served, and clients never see the
model files, presets, or flags behind them. The file and profiles.json are
read again for each request, so edits apply to the next request; a request
already running keeps the route it started with. "listen" is read once, at
startup.

"default_model" names the one clients should use when they have no reason
to choose. The gateway itself never routes to it: /v1/models marks it, and
a chat request still names its model.
"""

import hashlib
import ipaddress
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from profiles import Profile, ProfileRegistry

DEFAULT_LISTEN = "127.0.0.1:8090"
DEFAULT_MAX_REQUEST_MB = 32
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_ROUTE_KEYS = {"model", "profile", "max_output_tokens", "description"}
_TOP_KEYS = {"listen", "allow_non_loopback", "max_request_mb", "models", "default_model"}
TOKEN_FILE = "gateway.token"
MIN_TOKEN_LENGTH = 32


class ConfigError(Exception):
    """The gateway can't start with this configuration."""


@dataclass(frozen=True)
class Route:
    """One exposed name, resolved to a model and a snapshot of its profile."""
    name: str
    model: str
    profile: Profile | None
    max_output_tokens: int | None = None
    description: str | None = None

    @property
    def start_args(self) -> list[str]:
        return list(self.profile.args) if self.profile else []

    @property
    def fingerprint(self) -> str:
        """Changes whenever what would be started changes."""
        key = json.dumps([self.model, self.profile.name if self.profile else None,
                          self.profile.preset if self.profile else None, self.start_args])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @property
    def reasoning_disabled_by_flags(self) -> bool:
        """The profile's own flags turn thinking off. Flags luallm adds by
        itself (model_overrides) aren't visible here."""
        args = self.start_args
        for i, a in enumerate(args):
            nxt = args[i + 1] if i + 1 < len(args) else None
            if a in ("--reasoning", "-rea") and nxt == "off":
                return True
            if a in ("--reasoning=off", "-rea=off"):
                return True
            if a == "--reasoning-budget" and nxt == "0":
                return True
        return False


@dataclass
class GatewayConfig:
    host: str
    port: int
    max_request_bytes: int
    routes: dict[str, Route]
    warnings: list[str]
    # A name in routes, or None: unset, or naming one that isn't served.
    default_model: str | None = None


def _parse_listen(value: Any, allow_non_loopback: bool) -> tuple[str, int]:
    if not isinstance(value, str) or ":" not in value:
        raise ConfigError(f"'listen' must be \"host:port\", got {value!r}")
    host, _, port_s = value.rpartition(":")
    host = host.strip("[]")
    try:
        port = int(port_s)
    except ValueError:
        raise ConfigError(f"'listen' port must be a number, got {port_s!r}")
    if not 0 < port < 65536:
        raise ConfigError(f"'listen' port out of range: {port}")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback and not allow_non_loopback:
        raise ConfigError(
            f"'listen' is {host}, which isn't loopback. The gateway starts models on "
            f"this Mac for anyone who has the token; to listen beyond loopback, set "
            f"\"allow_non_loopback\": true deliberately.")
    return host, port


def load_config(state_dir: str | Path, profiles: ProfileRegistry) -> GatewayConfig:
    """Parse gateway.json. Broken model entries are skipped with a warning,
    never guessed at; a broken file or listen address is a ConfigError."""
    path = Path(state_dir) / "gateway.json"
    if not path.exists():
        raise ConfigError(f"No gateway configuration: create {path} (see README).")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"could not parse {path}: {e}")
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")

    warnings = [f"unknown top-level key '{k}' ignored" for k in sorted(set(data) - _TOP_KEYS)]
    allow = data.get("allow_non_loopback", False) is True
    host, port = _parse_listen(data.get("listen", DEFAULT_LISTEN), allow)

    max_mb = data.get("max_request_mb", DEFAULT_MAX_REQUEST_MB)
    if not isinstance(max_mb, (int, float)) or isinstance(max_mb, bool) or not 0 < max_mb <= 512:
        raise ConfigError("'max_request_mb' must be a number between 0 and 512")

    entries = data.get("models", {})
    if not isinstance(entries, dict):
        raise ConfigError("'models' must be an object of name -> model")

    known_profiles, profile_errors = profiles.load()
    by_name = {p.name: p for p in known_profiles}
    warnings += [f"profiles.json: {e}" for e in profile_errors]

    routes: dict[str, Route] = {}
    for name, spec in entries.items():
        if not _NAME.match(name):
            warnings.append(f"model '{name}': names are letters, digits, and . _ : - "
                            f"(up to 64), starting with a letter or digit")
            continue
        if not isinstance(spec, dict):
            warnings.append(f"model '{name}': must be an object")
            continue
        unknown = set(spec) - _ROUTE_KEYS
        if unknown:
            warnings.append(f"model '{name}': unknown key(s) {sorted(unknown)} "
                            f"(allowed: {sorted(_ROUTE_KEYS)})")
            continue
        model = spec.get("model")
        if not isinstance(model, str) or not model:
            warnings.append(f"model '{name}': 'model' must be a luallm model name")
            continue
        profile = None
        if spec.get("profile") is not None:
            profile = by_name.get(spec["profile"])
            if profile is None:
                warnings.append(f"model '{name}': profile '{spec['profile']}' "
                                f"isn't defined in profiles.json")
                continue
        max_out = spec.get("max_output_tokens")
        if max_out is not None and (not isinstance(max_out, int) or isinstance(max_out, bool)
                                    or max_out < 1):
            warnings.append(f"model '{name}': 'max_output_tokens' must be a positive integer")
            continue
        description = spec.get("description")
        if description is not None and not isinstance(description, str):
            warnings.append(f"model '{name}': 'description' must be a string")
            continue
        routes[name] = Route(name=name, model=model, profile=profile,
                             max_output_tokens=max_out, description=description)

    default = data.get("default_model")
    if default is not None and not isinstance(default, str):
        raise ConfigError("'default_model' must be the name of one of 'models'")
    if default is not None and default not in routes:
        # Warned, not refused, like a broken entry: its own entry may be the broken one, and the
        # other models still work. Clients are told there's no default rather than given another.
        warnings.append(f"default_model '{default}' isn't one of the models served, so there's "
                        f"no default")
        default = None

    return GatewayConfig(host=host, port=port, max_request_bytes=int(max_mb * 1024 * 1024),
                         routes=routes, warnings=warnings, default_model=default)


def load_token(state_dir: str | Path) -> str:
    """The bearer token: $LUALLM_GATEWAY_TOKEN, or gateway.token in the state
    dir, which must be readable by its owner only."""
    env = os.environ.get("LUALLM_GATEWAY_TOKEN")
    if env is not None:
        token = env.strip()
        source = "LUALLM_GATEWAY_TOKEN"
    else:
        path = Path(state_dir) / TOKEN_FILE
        if not path.exists():
            raise ConfigError(f"No token: set LUALLM_GATEWAY_TOKEN, or run "
                              f"`make gateway-token` to create {path}.")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ConfigError(f"{path} is readable by others (mode {oct(mode)}); "
                              f"run `chmod 600 {path}`.")
        token = path.read_text().strip()
        source = str(path)
    if len(token) < MIN_TOKEN_LENGTH:
        raise ConfigError(f"The token from {source} is shorter than "
                          f"{MIN_TOKEN_LENGTH} characters.")
    return token


def create_token(state_dir: str | Path) -> tuple[Path, bool]:
    """Write a new random token file if there isn't one. Returns (path, created)."""
    path = Path(state_dir) / TOKEN_FILE
    if path.exists():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_urlsafe(32) + "\n")
    return path, True


def capabilities_from_props(props: dict, route: Route) -> dict:
    """What this model's server says it can do, from llama-server's /props,
    narrowed by flags in the route's profile. Anything /props doesn't state
    is reported as unknown (None), never assumed."""
    caps = props.get("chat_template_caps") or {}
    modalities = props.get("modalities") or {}
    template = props.get("chat_template") or ""
    off = route.reasoning_disabled_by_flags

    levels = caps.get("reasoning_effort_levels") if caps.get("supports_reasoning_effort") else None
    if levels is not None and not (isinstance(levels, list) and all(isinstance(l, str) for l in levels)):
        levels = None
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")

    return {
        "images": modalities.get("vision") if isinstance(modalities.get("vision"), bool) else None,
        "context_tokens": n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None,
        "max_output_tokens": route.max_output_tokens,
        "reasoning": {
            "disabled_by_profile": off,
            # Accepted reasoning_effort values; [] when the template has none
            # or the profile turns thinking off.
            "effort_levels": [] if off else list(levels or []),
            # chat_template_kwargs.enable_thinking has an effect only if the
            # template reads it.
            "enable_thinking": (not off) and "enable_thinking" in template,
            "preserve_reasoning": bool(caps.get("supports_preserve_reasoning")),
        },
    }


class CapabilityCache:
    """Last verified capabilities per exposed name, kept on disk so
    /v1/models can report them for models that aren't loaded. An entry
    counts only while its route's fingerprint is unchanged."""

    def __init__(self, state_dir: str | Path):
        self.path = Path(state_dir) / "gateway-capabilities.json"

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def get(self, route: Route) -> dict | None:
        entry = self._read().get(route.name)
        if isinstance(entry, dict) and entry.get("fingerprint") == route.fingerprint:
            return entry
        return None

    def put(self, route: Route, caps: dict) -> dict:
        data = self._read()
        entry = {"fingerprint": route.fingerprint, "capabilities": caps,
                 "verified_at": datetime.now(timezone.utc).isoformat()}
        data[route.name] = entry
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path)
        return entry
