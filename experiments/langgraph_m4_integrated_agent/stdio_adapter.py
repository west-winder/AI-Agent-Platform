import os
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain.mcp import MCPAdapter


SERVER_ROOT = Path(__file__).resolve().parents[2] / "mcp_servers" / "web_page_fetch"
SERVER_PYTHON = SERVER_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
FIXTURE_SCRIPT = SERVER_ROOT / "tests" / "support" / "fixture_server.py"


def web_page_fetch_adapter(entry: Path = FIXTURE_SCRIPT) -> MCPAdapter:
    """用独立 Server 解释器启动本地 fixture，并交给已验证的 MCPAdapter。"""
    if not SERVER_PYTHON.is_file():
        raise FileNotFoundError(f"Create the Web Page Fetch server environment at {SERVER_PYTHON}")
    transport = StdioTransport(
        command=str(SERVER_PYTHON), args=["-B", "-X", "utf8", str(entry)],
        cwd=str(SERVER_ROOT), keep_alive=False,
    )
    return MCPAdapter(Client(transport, init_timeout=10.0, timeout=40.0, cache=False))
