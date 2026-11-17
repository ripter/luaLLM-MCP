.PHONY: install run test clean check gateway gateway-check gateway-token

install:
	uv venv
	uv pip install -e .
	uv pip install --group dev
	@uv run python -c "import mcp; print('✓ MCP installed')"

run:
	uv run python mcp_server.py

# The OpenAI-compatible gateway (see README). Runs until stopped with Ctrl-C.
gateway:
	uv run python gateway.py

gateway-check:
	uv run python gateway.py --check

gateway-token:
	uv run python gateway.py --init-token

test:
	uv run pytest tests/ -v

check:
	@uv run python -c "import mcp; print('✓ MCP import OK')"
	@uv run python -c "import lua_client; print('✓ lua_client import OK')"
	@uv run python -c "import lease_manager; print('✓ lease_manager import OK')"
	@uv run python -c "import gateway; print('✓ gateway import OK')"

clean:
	rm -rf .venv __pycache__ .mcp.json
	rm -f *.pyc