"""One process in a cross-process lifecycle test.

  lifecycle_worker.py <state_dir> <luallm> <go_file> leases <n>
  lifecycle_worker.py <state_dir> <luallm> <go_file> acquire <model>
  lifecycle_worker.py <state_dir> <luallm> <go_file> cycle <model> <n>

Each waits for go_file to exist so all workers start together, then prints
one JSON line with what it saw.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import core  # noqa: E402
from lease_manager import LeaseManager  # noqa: E402
from lua_client import LuaLLMClient  # noqa: E402
from profiles import ProfileRegistry  # noqa: E402


def main() -> None:
    state_dir, luallm, go_file, cmd, *rest = sys.argv[1:]
    leases = LeaseManager(state_dir=state_dir)
    ctx = core.Ctx(client=LuaLLMClient(luallm_path=luallm), leases=leases,
                   profiles=ProfileRegistry(state_dir=state_dir))
    while not os.path.exists(go_file):
        time.sleep(0.005)

    if cmd == "leases":
        for _ in range(int(rest[0])):
            leases.acquire_lease("model", None, client_id=str(os.getpid()))
        print(json.dumps({"ok": True}))

    elif cmd == "acquire":
        begun = core.acquire_begin(ctx, rest[0], None, str(os.getpid()), "test")
        if isinstance(begun, dict):
            print(json.dumps({"error": begun["error"]}))
        else:
            print(json.dumps({"instance_id": begun.instance.instance_id,
                              "reused": begun.reused}))

    elif cmd == "cycle":
        model, n = rest[0], int(rest[1])
        violations = []
        errors = []
        for i in range(n):
            begun = core.acquire_begin(ctx, model, None, str(os.getpid()), "test")
            if isinstance(begun, dict):
                errors.append(begun["error"])
                continue
            # While we hold a lease, nobody may stop the server under us.
            for check in range(2):
                if core.find_running_server(ctx, model) is None:
                    violations.append(f"iteration {i} check {check}: stopped while leased")
                time.sleep(0.01)
            rel = core.release(ctx, begun.lease.lease_id)
            if not rel["success"]:
                errors.append(rel["error"])
        print(json.dumps({"violations": violations, "errors": errors}))


if __name__ == "__main__":
    main()
