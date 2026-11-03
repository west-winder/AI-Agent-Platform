"""Manual bounded subprocess probe for the SDK's Windows Selector fallback."""

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from backend.mcp_host.bootstrap import bootstrap_host
from backend.mcp_host.config import ServerConfig
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.dispatcher import ToolDispatcher


async def main():
    config = ServerConfig(server_id="selector", command=sys.executable,
        args=["-B", str(Path(__file__).with_name("protocol_server.py")), "pages"],
        cwd=str(ROOT))
    async with bootstrap_host([config], llm=SimpleNamespace()) as host:
        result = await ToolDispatcher(host.registry).dispatch("mcp_selector__last", {}, ToolContext())
        assert result.status == "success", result
    print("selector_success_and_shutdown")


with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
    runner.run(main())
