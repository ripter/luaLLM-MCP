.PHONY: install run test clean check

install:
	uv venv
	uv pip install -e .
	uv pip install --group dev
	@uv run python -c "import mcp; print('✓ MCP installed')"

run:
	uv run python mcp_server.py

test:
	uv run pytest tests/ -v

check:
	@uv run python -c "import mcp; print('✓ MCP import OK')"
	@uv run python -c "import lua_client; print('✓ lua_client import OK')"
	@uv run python -c "import lease_manager; print('✓ lease_manager import OK')"

clean:
	rm -rf .venv __pycache__ .mcp.json
	rm -f *.pyc