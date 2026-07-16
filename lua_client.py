"""luaLLM CLI wrapper with subprocess safety."""

import subprocess
import json
from typing import Any


class LuaLLMClient:
    def __init__(self, luallm_path: str = "luallm"):
        self.luallm_path = luallm_path
    
    def _run_command(self, args: list[str]) -> dict[str, Any]:
        """Run luallm command and return parsed JSON response."""
        result = subprocess.run(
            [self.luallm_path] + args,
            capture_output=True,
            text=True,
            shell=False
        )
        if result.returncode != 0 and not result.stdout:
            return {"success": False, "stderr": result.stderr, "returncode": result.returncode}
        try:
            data = json.loads(result.stdout) if result.stdout else {}
            return {"success": result.returncode == 0, "data": data, "stderr": result.stderr}
        except json.JSONDecodeError as e:
            return {"success": False, "error": str(e), "stderr": result.stderr}
    
    def list_models(self) -> dict[str, Any]:
        return self._run_command(["list", "--json"])
    
    def get_model_info(self, model_name: str) -> dict[str, Any]:
        return self._run_command(["info", model_name, "--json"])
    
    def get_server_status(self) -> dict[str, Any]:
        return self._run_command(["status", "--json"])
    
    def get_all_commands(self) -> dict[str, Any]:
        return self._run_command(["help", "--json"])
    
    def get_pinned(self) -> dict[str, Any]:
        return self._run_command(["pinned", "--json"])
    
    def get_notes(self) -> dict[str, Any]:
        return self._run_command(["notes", "--json"])
    
    def get_history(self) -> dict[str, Any]:
        return self._run_command(["history", "--json"])
    
    def start_model(self, model_name: str, preset: str | None = None,
                    extra_args: list[str] | None = None) -> dict[str, Any]:
        args = ["start", model_name]
        if preset:
            args.extend(["--preset", preset])
        if extra_args:
            args.extend(extra_args)
        args.append("--json")
        return self._run_command(args)
    
    def stop_model(self, model_name: str) -> dict[str, Any]:
        return self._run_command(["stop", model_name, "--json"])
    
    def get_logs(self, model_name: str) -> dict[str, Any]:
        return self._run_command(["logs", model_name, "--json"])
    
    def health_check(self) -> dict[str, Any]:
        """Check if luaLLM is available and working by running help --json."""
        return self._run_command(["help", "--json"])