"""A stand-in for the luallm CLI, for tests that need real processes.

State lives in $FAKE_LUALLM_DIR:
  servers.json   what `status --json` reports
  starts.log     one line per `start` invocation (to count duplicate starts)
  stops.log      one line per `stop` invocation

`start` launches tests/fakes/fake_llama_server.py as a real daemon on a
free port, passing the profile flags through, so the gateway talks to a
real HTTP server and real sockets. Like luallm, it refuses to start a model
that is already running. It deliberately takes no lock of its own: any
duplicate start is the caller's race, and shows up in starts.log.

Environment:
  FAKE_LUALLM_MODELS        comma-separated model names (default "tiny")
  FAKE_LUALLM_START_DELAY   seconds `start` sleeps before acting (widens races)
  FAKE_LLAMA_*              passed through to the fake server
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = Path(os.environ["FAKE_LUALLM_DIR"])


def _servers() -> list[dict]:
    try:
        return json.loads((STATE / "servers.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(servers: list[dict]) -> None:
    tmp = STATE / f"servers.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(servers))
    os.replace(tmp, STATE / "servers.json")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); ask ps.
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log(name: str, line: str) -> None:
    with open(STATE / name, "a") as f:
        f.write(line + "\n")


def main(argv: list[str]) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    args = [a for a in argv if a != "--json"]
    cmd = args[0] if args else "help"

    if cmd == "help":
        print(json.dumps({"commands": ["list", "status", "start", "stop"]}))
        return 0

    if cmd == "list":
        names = os.environ.get("FAKE_LUALLM_MODELS", "tiny").split(",")
        print(json.dumps([{"name": n, "size": "1GB"} for n in names]))
        return 0

    if cmd == "status":
        servers = _servers()
        for s in servers:
            if s["state"] == "running" and not _alive(s["pid"]):
                s["state"] = "stopped"
        print(json.dumps({"servers": servers}))
        return 0

    if cmd == "info":
        print(json.dumps({"model": args[1]}))
        return 0

    if cmd == "logs":
        print(json.dumps({"log_file": str(STATE / f"{args[1]}.log")}))
        return 0

    if cmd == "start":
        model = args[1]
        extra = args[2:]
        if "--preset" in extra:
            i = extra.index("--preset")
            extra = extra[:i] + extra[i + 2:]
        time.sleep(float(os.environ.get("FAKE_LUALLM_START_DELAY", "0")))
        _log("starts.log", f"{model} {os.getpid()}")
        for s in _servers():
            if s["model"] == model and s["state"] == "running" and _alive(s["pid"]):
                print(f"Error: {model} is already running (PID {s['pid']}) on port {s['port']}")
                return 1
        port = _free_port()
        log = open(STATE / f"{model}.log", "a")
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "fake_llama_server.py"), "--port", str(port),
             "--model", model, *extra],
            stdout=log, stderr=log, start_new_session=True)
        _log("pids.log", str(proc.pid))  # every server ever started, for cleanup
        servers = [s for s in _servers() if s["model"] != model]
        servers.append({"model": model, "pid": proc.pid, "port": port,
                        "state": "running", "mode": "daemon", "args": extra})
        _save(servers)
        print(f"Starting daemon: {model}")
        print(json.dumps({"pid": proc.pid, "model": model, "port": port, "status": "started"}))
        return 0

    if cmd == "stop":
        model = args[1]
        _log("stops.log", model)
        servers = _servers()
        found = False
        for s in servers:
            if s["model"] == model and s["state"] == "running":
                found = True
                try:
                    os.kill(s["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
                s["state"] = "stopped"
        _save(servers)
        if not found:
            print(json.dumps({"error": "not_running", "message": f"{model} is not running"}))
            return 1
        print(json.dumps({"stopped": True, "model": model}))
        return 0

    print(json.dumps({"error": "unknown_command", "message": cmd}))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
