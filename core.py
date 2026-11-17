"""Model lifecycle shared by the MCP server and the OpenAI gateway.

Everything that decides whether to start, reuse, swap, evict, or stop a
model server runs under the LeaseManager's lifecycle lock, so separate
processes (each stdio MCP client, the gateway) can't start the same model
twice or stop a server another process has just leased. Waiting for a
model to load happens outside the lock: once the instance record and its
lease exist, any other acquirer reuses that record and waits too.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from lease_manager import Instance, Lease, LeaseManager
from profiles import Profile, ProfileRegistry

logger = logging.getLogger("luaLLM-MCP")


def error_detail(resp: dict) -> str:
    """Best-available error text from a lua_client response.

    luallm reports errors inconsistently (stderr, non-JSON stdout, or JSON
    on stdout with a non-zero exit), so combine everything present instead
    of returning an empty string.
    """
    data = resp.get("data")
    if isinstance(data, dict) and data.get("message"):
        # luallm --json failures are structured: {"error": code, "message": text}
        code = data.get("error")
        message = str(data["message"])
        return f"{code}: {message}" if code and code != message else message
    parts = []
    error = resp.get("error")
    if error:
        parts.append(str(error))
    stderr = (resp.get("stderr") or "").strip()
    if stderr:
        parts.append(f"stderr: {stderr[:500]}")
    stdout = (resp.get("stdout") or "").strip()
    if stdout:
        parts.append(f"stdout: {stdout[:500]}")
    if not parts and data:
        parts.append(data if isinstance(data, str) else json.dumps(data))
    if not parts:
        returncode = resp.get("returncode")
        parts.append(
            f"luallm exited with code {returncode} and no output"
            if returncode is not None
            else "UNKNOWN_ERROR"
        )
    return " | ".join(parts)


def ready_timeout() -> float:
    """How long an acquire waits for the model to finish loading."""
    return float(os.environ.get("LUALLM_MCP_READY_TIMEOUT_SECONDS", "600"))


def process_alive(pid: int | None) -> bool:
    """True if a process with this pid is running (best-effort, no pid = yes)."""
    if pid is None:
        return True
    try:
        result = subprocess.run(["ps", "-p", str(pid)],
                                capture_output=True, text=True,
                                check=False)
        return result.returncode == 0
    except (OSError, ValueError):
        return True


def probe_health(port: int) -> int:
    """Health state of the model server on port: 2 ready, 1 loading, 0 refused.

    llama.cpp answers 200 on /health once loaded and 503 while loading
    (urllib surfaces that as HTTPError — still proof the server is up).
    A refusal means the port is closed: normal for the first moment after
    launch, fatal once the server has ever accepted a connection, and
    inconclusive otherwise (pid-liveness resolves it via wait_until_ready).
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            return 2 if resp.status == 200 else 1
    except urllib.error.HTTPError:
        return 1
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0


async def wait_until_ready(port: int | None, timeout_s: float,
                           interval_s: float = 1.0,
                           pid: int | None = None,
                           probe: Callable[[int], int] = probe_health,
                           alive: Callable[[int | None], bool] = process_alive,
                           in_thread: bool = False) -> tuple[str, float]:
    """Poll the model server until it is ready to serve requests.

    A lease means "ready to use": llama.cpp accepts connections while the
    model is still loading, so acquire must wait for /health to go green.
    Returns ("ready" | "timeout" | "crashed", seconds waited).

    "crashed" fires when the server is provably gone: a port that had
    accepted connections starts refusing them, or a known-daemon pid
    disappears while the port keeps refusing. Either means the process
    died during load and will never become ready — most often a broken
    model file or an invalid flag from a profile; waiting out the full
    timeout would just stall the caller. "timeout" keeps the instance
    record because a slow load is still in progress; "crashed" callers
    must drop it. No port means nothing to probe.

    in_thread runs the blocking probes off the event loop, for a server
    that is streaming other requests meanwhile.
    """
    if port is None:
        return "ready", 0.0

    async def call(fn, arg):
        return await asyncio.to_thread(fn, arg) if in_thread else fn(arg)

    loop = asyncio.get_event_loop()
    started = loop.time()
    deadline = started + timeout_s
    ever_accepted = False
    while True:
        state = await call(probe, port)
        if state == 2:
            return "ready", loop.time() - started
        if state == 1:
            ever_accepted = True
        else:
            if ever_accepted:
                logger.error(f"Model server on port {port} stopped accepting "
                             f"connections — the process died during startup.")
                return "crashed", loop.time() - started
            if pid is not None and not await call(alive, pid):
                logger.error(f"Model server process {pid} is gone while port "
                             f"{port} refuses connections — died before the "
                             f"listener opened.")
                return "crashed", loop.time() - started
        if loop.time() >= deadline:
            return "timeout", loop.time() - started
        await asyncio.sleep(interval_s)


_SIZE_UNITS = {"": 1, "B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(size) -> int | None:
    """Bytes from a luallm size like "26.6GB" (or a raw number); None if unknown."""
    if isinstance(size, (int, float)):
        return int(size)
    if not isinstance(size, str):
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)i?B?\s*", size, re.IGNORECASE)
    if not m:
        return None
    try:
        return int(float(m.group(1)) * _SIZE_UNITS[m.group(2).upper()])
    except ValueError:
        return None


def total_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def memory_budget_bytes() -> int | None:
    """Memory the MCP may fill with model servers. LUALLM_MCP_MEMORY_BUDGET_GB
    overrides; the default is 75% of physical RAM (roughly what macOS lets
    Metal wire on Apple silicon)."""
    env = os.environ.get("LUALLM_MCP_MEMORY_BUDGET_GB")
    if env:
        return int(float(env) * 1024**3)
    total = total_memory_bytes()
    return int(total * 0.75) if total else None


def footprint(size_bytes: int | None) -> int:
    """Estimated resident memory of a loaded model: weights plus KV cache and
    compute buffers, approximated as a multiple of the GGUF file size."""
    if not size_bytes:
        return 0
    return int(size_bytes * float(os.environ.get("LUALLM_MCP_MEMORY_OVERHEAD", "1.2")))


@dataclass
class Ctx:
    """What the lifecycle functions act on. The MCP server builds one per
    call from its module globals, so tests can swap any part."""
    client: Any
    leases: LeaseManager
    profiles: ProfileRegistry
    probe: Callable[[int], int] = probe_health
    alive: Callable[[int | None], bool] = process_alive


def find_running_server(ctx: Ctx, model_name: str) -> dict | None:
    """Look up a running luallm server for the model, or None."""
    status_resp = ctx.client.get_server_status()
    if not status_resp["success"]:
        return None
    for srv in status_resp.get("data", {}).get("servers", []):
        if srv.get("model") == model_name and srv.get("state") == "running":
            return srv
    return None


def stop_idle_instance(ctx: Ctx, instance: Instance) -> bool:
    """Stop an idle MCP-managed server and drop its record. A server that is
    already gone just loses its record. False if it is still running.
    Callers hold the lifecycle lock."""
    with ctx.leases.lifecycle():
        stop_resp = ctx.client.stop_model(instance.model)
        if not stop_resp["success"] and find_running_server(ctx, instance.model) is not None:
            logger.warning(f"Could not stop idle {instance.model}: {error_detail(stop_resp)}")
            return False
        ctx.leases.remove_instance(instance.instance_id)
        return True


def make_room_for(ctx: Ctx, model_name: str, model_sizes: dict[str, int | None]) -> list[str]:
    """Before starting model_name, stop idle models (longest-idle first) until
    its estimated footprint fits the memory budget. Leased and external
    servers are never touched. Returns the models stopped. When even
    evicting every idle model is not enough, the start goes ahead anyway —
    the estimate is a heuristic, not a hard admission limit."""
    need = footprint(model_sizes.get(model_name))
    budget = memory_budget_bytes()
    if not need or budget is None:
        return []
    with ctx.leases.lifecycle():
        status_resp = ctx.client.get_server_status()
        servers = status_resp.get("data", {}).get("servers", []) if status_resp["success"] else []
        running = {s.get("model") for s in servers
                   if s.get("state") == "running" and s.get("model") != model_name}
        used = sum(footprint(model_sizes.get(m)) for m in running)

        evicted = []
        for idle in ctx.leases.idle_instances():
            if used + need <= budget:
                break
            if idle.model == model_name:
                continue
            if idle.model not in running:
                ctx.leases.remove_instance(idle.instance_id)
                continue
            if stop_idle_instance(ctx, idle):
                used -= footprint(model_sizes.get(idle.model))
                evicted.append(idle.model)
                logger.info(f"Evicted idle {idle.model} to make room for {model_name}")
        if used + need > budget:
            logger.warning(
                f"Starting {model_name} (~{need / 1024**3:.1f} GB) with only "
                f"~{max(0, budget - used) / 1024**3:.1f} GB of the memory budget free; "
                f"the remaining servers are leased or external.")
        return evicted


def reap_idle(ctx: Ctx) -> list[str]:
    """Drop leases whose owning process died, then stop idle models whose
    grace period has lapsed. Returns models stopped."""
    with ctx.leases.lifecycle():
        orphaned = ctx.leases.drop_orphaned_leases()
        if orphaned:
            logger.info(f"Dropped {len(orphaned)} lease(s) whose owning process is gone")
        stopped = []
        for idle in ctx.leases.expired_idle_instances():
            if stop_idle_instance(ctx, idle):
                stopped.append(idle.model)
                logger.info(f"Stopped {idle.model}: idle past "
                            f"{int(ctx.leases.idle_grace_seconds())}s grace period")
        return stopped


def stop_all_idle(ctx: Ctx) -> None:
    """On shutdown nothing is left to reap idle models, so stop them now."""
    with ctx.leases.lifecycle():
        for idle in ctx.leases.idle_instances():
            if stop_idle_instance(ctx, idle):
                logger.info(f"Stopped idle {idle.model} on shutdown")


def resolve_profile(ctx: Ctx, profile_name: str | None,
                    preset: str | None) -> tuple[Profile | None, dict | None]:
    """(profile, None) or (None, error result)."""
    if not profile_name:
        return None, None
    if preset:
        return None, {
            "success": False, "code": "PRESET_AND_PROFILE_EXCLUSIVE",
            "error": ("PRESET_AND_PROFILE_EXCLUSIVE: give preset or profile, "
                      "not both — a profile already specifies its own luallm "
                      "preset via profiles.json."),
        }
    profile = ctx.profiles.get(profile_name)
    if profile is not None:
        return profile, None
    profiles, errors = ctx.profiles.load()
    msg = (f"PROFILE_NOT_FOUND: '{profile_name}' is not a defined "
           f"profile (from {ctx.profiles.path}).")
    if errors:
        msg += " Profiles file problems: " + "; ".join(errors)
    elif profiles:
        msg += " Available profiles: " + ", ".join(p.name for p in profiles)
    else:
        msg += " No profiles are defined yet — create them in " \
               f"{ctx.profiles.path} (see README)."
    return None, {"success": False, "code": "PROFILE_NOT_FOUND", "error": msg}


@dataclass
class Begun:
    """An acquire whose lease exists but whose model may still be loading."""
    model: str
    instance: Instance
    lease: Lease
    effective_preset: str | None
    profile: Profile | None
    reused: bool
    pid: int | None = None
    evicted: list[str] = field(default_factory=list)


def _fail(code: str, message: str) -> dict:
    return {"success": False, "code": code, "error": message}


def acquire_begin(ctx: Ctx, model_name: str | None, preset: str | None,
                  client_id: str | None, purpose: str | None,
                  profile: Profile | None = None,
                  owner_pid: int | None = None,
                  expected_args: list[str] | None = None) -> Begun | dict:
    """Decide, under the lifecycle lock, whether to reuse, adopt, swap, or
    start the model, and take the lease. Returns a Begun, or an error result
    carrying a machine-readable "code".

    profile is an already-resolved snapshot: the caller decides which
    profile flags this acquire means, so an edit to profiles.json mid-way
    can't change them. expected_args (gateway only) also requires a managed
    instance to be running with exactly those flags: an idle one with other
    flags is restarted; a leased one is refused, never interrupted.
    """
    if not model_name:
        return _fail("MISSING_MODEL_NAME", "MISSING_MODEL_NAME")

    with ctx.leases.lifecycle():
        models_resp = ctx.client.list_models()
        if not models_resp["success"]:
            return _fail("CANNOT_LIST_MODELS", "CANNOT_LIST_MODELS")
        model_sizes = {m.get("name"): parse_size(m.get("size"))
                       for m in models_resp.get("data", [])}
        if model_name not in model_sizes:
            return _fail("MODEL_NOT_FOUND", "MODEL_NOT_FOUND")

        # The luallm preset actually applied: the caller's, or the one the
        # profile bundles with its extra flags.
        profile_name = profile.name if profile else None
        effective_preset = profile.preset if profile else preset
        start_args = list(profile.args) if profile else []

        instance = ctx.leases.find_instance(model_name, effective_preset, profile_name)
        running = find_running_server(ctx, model_name)
        evicted: list[str] = []

        if instance and running is None:
            # Stale record: the server behind it is gone (stopped or crashed
            # outside MCP). Drop it and start fresh below.
            logger.info(f"Dropping stale instance record for {model_name}")
            ctx.leases.remove_instance(instance.instance_id)
            instance = None

        if (instance is not None and expected_args is not None
                and instance.managed_by_mcp and instance.args != expected_args):
            if ctx.leases.lease_count(instance.instance_id) == 0:
                logger.info(f"Restarting idle {model_name}: its profile's flags "
                            f"changed since it was started")
                if not stop_idle_instance(ctx, instance):
                    return _fail("STOP_FAILED",
                                 f"STOP_FAILED: {model_name} is idle with outdated flags "
                                 f"and could not be stopped for the restart.")
                evicted.append(model_name)
                instance = None
                running = None
            else:
                return _fail("MODEL_RUNNING_DIFFERENT_CONFIG",
                             f"MODEL_RUNNING_DIFFERENT_CONFIG: {model_name} is running "
                             f"with flags that no longer match its profile, and is in "
                             f"use. It restarts with the new flags once released.")

        if instance is None and running is not None:
            other = ctx.leases.find_instance_for_model(model_name)
            if (other and other.managed_by_mcp
                    and ctx.leases.lease_count(other.instance_id) == 0):
                # Running only because of the idle grace period, under another
                # configuration: nobody is using it, so swap it out.
                logger.info(f"Stopping idle {model_name} to restart under the "
                            f"requested configuration")
                if not stop_idle_instance(ctx, other):
                    return _fail("STOP_FAILED",
                                 f"STOP_FAILED: {model_name} is idle under another "
                                 f"configuration and could not be stopped for the swap.")
                evicted.append(model_name)
                running = None

        if instance:
            # Trust live status over the record (server may have moved ports).
            port = running.get("port")
            instance, lease = ctx.leases.acquire_lease(
                model_name, effective_preset, client_id, purpose, profile=profile_name,
                live_port=int(port) if port is not None else None, owner_pid=owner_pid)
            return Begun(model_name, instance, lease, effective_preset, profile,
                         reused=True, evicted=evicted)

        if running:
            other = ctx.leases.find_instance_for_model(model_name)
            if other:
                # One server per model (luallm refuses a second start), so this
                # server belongs to the other record's configuration. Adopting it
                # under the requested config would claim flags it doesn't carry.
                cfg = (f"profile '{other.profile}'" if other.profile
                       else f"preset '{other.preset}'" if other.preset
                       else "the default configuration")
                return _fail("MODEL_RUNNING_DIFFERENT_CONFIG",
                             f"MODEL_RUNNING_DIFFERENT_CONFIG: {model_name} is already "
                             f"running with {cfg}; luallm cannot start a second server "
                             f"for the same model. Release its leases or force_stop_model, "
                             f"then acquire again with the requested configuration.")
            # Started outside the MCP (e.g. `luallm start` in a terminal).
            # Adopt it so workflows can use it; release will not stop it.
            if preset or profile:
                wanted = f"profile '{profile_name}'" if profile else f"preset '{preset}'"
                return _fail("MODEL_RUNNING_EXTERNALLY",
                             f"MODEL_RUNNING_EXTERNALLY: {model_name} is already running "
                             f"outside MCP management, so {wanted} cannot be applied. "
                             f"Acquire without a preset/profile to use it, or stop it first.")
            port = running.get("port")
            instance, lease = ctx.leases.acquire_lease(
                model_name, effective_preset, client_id, purpose,
                port=int(port) if port is not None else None,
                managed_by_mcp=False, owner_pid=owner_pid)
            logger.info(f"Adopted externally started instance of {model_name}")
            return Begun(model_name, instance, lease, effective_preset, profile,
                         reused=True, evicted=evicted)

        evicted += make_room_for(ctx, model_name, model_sizes)
        start_resp = ctx.client.start_model(
            model_name, effective_preset, extra_args=start_args or None)
        if not start_resp["success"]:
            return _fail("START_FAILED", error_detail(start_resp))
        start_data = start_resp.get("data", {})
        port = start_data.get("port")
        instance, lease = ctx.leases.acquire_lease(
            model_name, effective_preset, client_id, purpose,
            port=int(port) if port is not None else None, profile=profile_name,
            args=start_args, owner_pid=owner_pid)
        return Begun(model_name, instance, lease, effective_preset, profile,
                     reused=False, pid=start_data.get("pid"), evicted=evicted)


def acquire_failed(ctx: Ctx, begun: Begun, ready_state: str, timeout: float) -> dict:
    """Take back the lease of an acquire whose model never became ready,
    and say why. A slow load keeps the instance record (the model is still
    loading); a crashed server's record is garbage and goes."""
    with ctx.leases.lifecycle():
        released = ctx.leases.release_lease(
            begun.lease.lease_id, keep_instance=(ready_state == "timeout"))
        if ready_state == "crashed" and released.get("remaining_leases") == 0:
            # Never park a dead server in the idle grace period.
            ctx.leases.remove_instance(begun.instance.instance_id)
    if ready_state == "crashed":
        return _fail("SERVER_EXITED",
                     f"SERVER_EXITED: {begun.model} died during startup (port "
                     f"{begun.instance.port} stopped accepting connections). Check "
                     f"`luallm logs {begun.model}` for the actual error — a bad "
                     f"model file or an invalid profile flag are common causes.")
    return _fail("MODEL_NOT_READY",
                 f"MODEL_NOT_READY: {begun.model} did not become ready within "
                 f"{int(timeout)}s (still loading). It continues loading in the "
                 f"background — acquire again shortly, or force_stop_model to abort.")


def _public(result: dict) -> dict:
    # The MCP tool contract predates "code"; keep its errors as they were.
    return {k: v for k, v in result.items() if k != "code"}


async def acquire(ctx: Ctx, model_name: str | None, preset: str | None,
                  client_id: str | None, purpose: str | None,
                  profile_name: str | None = None) -> dict:
    """The acquire_model tool: reuse an MCP-managed instance, adopt an
    externally started one (never stopped on release), or start the model —
    then wait until the model is actually ready to serve before returning
    the lease."""
    profile, error = resolve_profile(ctx, profile_name, preset)
    if error:
        return _public(error)
    begun = acquire_begin(ctx, model_name, preset, client_id, purpose, profile=profile)
    if isinstance(begun, dict):
        return _public(begun)

    # A lease means "ready to use": block until the server can actually
    # serve (llama.cpp answers 503 on /health while loading weights).
    timeout = ready_timeout()
    ready_state, waited = await wait_until_ready(
        begun.instance.port, timeout, pid=begun.pid, probe=ctx.probe, alive=ctx.alive)
    if ready_state != "ready":
        return _public(acquire_failed(ctx, begun, ready_state, timeout))

    instance = begun.instance
    data = {
        "instance_id": instance.instance_id,
        "lease_id": begun.lease.lease_id,
        "model": begun.model,
        "preset": begun.effective_preset,
        "port": instance.port,
        "reused": begun.reused,
        "external": not instance.managed_by_mcp,
        "ready_after_s": round(waited, 1),
    }
    if begun.profile is not None:
        data["profile"] = begun.profile.name
        data["args"] = begun.profile.args
    if begun.pid is not None:
        data["pid"] = begun.pid
    if begun.evicted:
        data["evicted_idle"] = begun.evicted
    return {"success": True, "data": data}


def release(ctx: Ctx, lease_id: str | None) -> dict:
    """The release_model tool: drop the lease. When this was the last lease
    of an MCP-managed instance it is kept warm for the idle grace period
    (the reaper stops it later), or with grace 0 stopped now. A failed stop
    is an error, not a warning — otherwise models silently pile up."""
    if not lease_id:
        return {"success": False, "error": "MISSING_LEASE_ID"}
    with ctx.leases.lifecycle():
        release_result = ctx.leases.release_lease(lease_id)
        if not release_result["success"]:
            return {"success": False, "error": release_result.get("error", "RELEASE_FAILED")}

        stopped = False
        if release_result.get("instance_stopped"):
            stop_resp = ctx.client.stop_model(release_result["model"])
            if stop_resp["success"]:
                stopped = True
            else:
                return {
                    "success": False,
                    "error": (f"LEASE_RELEASED_BUT_STOP_FAILED: the lease was released but "
                              f"{release_result['model']} could not be stopped: "
                              f"{error_detail(stop_resp)}"),
                    "data": {**release_result, "stopped": False},
                }
        return {"success": True, "data": {**release_result, "stopped": stopped}}


def force_stop(ctx: Ctx, model_name: str | None) -> dict:
    """The force_stop_model tool: stop a model regardless of leases and purge
    its instance/lease records. The escape hatch for orphaned models."""
    if not model_name:
        return {"success": False, "error": "MISSING_MODEL_NAME"}
    with ctx.leases.lifecycle():
        removed = ctx.leases.remove_model_records(model_name)
        stop_resp = ctx.client.stop_model(model_name)
    if not stop_resp["success"]:
        return {
            "success": False,
            "error": f"STOP_FAILED: {error_detail(stop_resp)}",
            "data": {"model": model_name, "stopped": False, **removed},
        }
    logger.info(f"Force-stopped {model_name} "
                f"(removed {removed['removed_leases']} lease(s))")
    return {"success": True, "data": {"model": model_name, "stopped": True, **removed}}


def list_profiles(ctx: Ctx) -> dict:
    """The list_profiles tool: discover curated launch profiles and the file
    to edit to define more. Subprocess-free, so it doubles as a config
    sanity check."""
    profiles, errors = ctx.profiles.load()
    data: dict = {"profiles": [p.to_dict() for p in profiles], "file": str(ctx.profiles.path)}
    if errors:
        data["warnings"] = errors
    return {"success": True, "data": data}
