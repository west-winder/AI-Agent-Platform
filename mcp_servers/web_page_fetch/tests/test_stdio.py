"""Official SDK Client, actual processes and real stdio framing."""

import asyncio
from datetime import timedelta
from pathlib import Path
import sys
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client import stdio
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fixture", [False, True], ids=["production", "fake_fetcher"])
def test_real_stdio_protocol_and_graceful_shutdown(monkeypatch, fixture):
    processes, forced, notifications = [], [], []
    create = stdio._create_platform_compatible_process
    terminate = stdio._terminate_process_tree
    async def observed_create(*args, **kwargs):
        process = await create(*args, **kwargs)
        processes.append(process)
        return process
    async def observed_terminate(*args, **kwargs):
        forced.append(True)
        await terminate(*args, **kwargs)
    # Observation only: the official launch/framing/cleanup implementations still run.
    monkeypatch.setattr(stdio, "_create_platform_compatible_process", observed_create)
    monkeypatch.setattr(stdio, "_terminate_process_tree", observed_terminate)
    async def on_log(params):
        notifications.append(str(params))
    args = ["-B", str(ROOT / "tests/support/fixture_server.py")] if fixture else ["-B", "-m", "web_page_fetch"]
    params = StdioServerParameters(command=sys.executable, args=args, cwd=ROOT,
                                   env={"PYTHONUTF8": "1", "FASTMCP_LOG_LEVEL": "DEBUG"})
    # Avoid system temp assumptions, and always close the handle before reading on Windows.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=ROOT / ".venv") as stderr:
        async def run():
            async with asyncio.timeout(20):
                async with stdio.stdio_client(params, errlog=stderr) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=5),
                                             logging_callback=on_log) as client:
                        init = await client.initialize()
                        assert init.serverInfo.name == "Web Page Fetch MCP Server"
                        tools = (await client.list_tools()).tools
                        assert [t.name for t in tools] == ["fetch_web_page"]
                        tool = tools[0]
                        assert tool.inputSchema["required"] == ["url"]
                        assert set(tool.inputSchema["properties"]) == {"url"}
                        assert tool.inputSchema["properties"]["url"]["type"] == "string"
                        assert set(tool.outputSchema["required"]) == {"title", "url", "content", "truncated", "content_type"}
                        assert tool.outputSchema["additionalProperties"] is False
                        if fixture:
                            result = await client.call_tool("fetch_web_page", {"url": "https://example.com/?secret=query_canary"})
                            assert not result.isError
                            assert result.structuredContent == dict(title="Fixture 标题", url="https://example.com/",
                                content="Fixture 正文", truncated=False, content_type="text/html")
                            assert result.content[0].type == "text"
                            for path, code in [("failure", "NETWORK_ERROR"), ("unknown", "INTERNAL_ERROR")]:
                                failed = await client.call_tool("fetch_web_page", {"url": f"https://example.com/{path}"})
                                assert failed.isError and code in failed.content[0].text
                                assert "raw_secret_canary" not in str(failed)
                                assert "10.0.0.1" not in str(failed)
                            again = await client.call_tool("fetch_web_page", {"url": "https://example.com/"})
                            assert not again.isError and again.structuredContent["content"] == "Fixture 正文"
                        else:
                            for url in ("http://localhost/?query_canary", "http://127.0.0.1/", "https://10.0.0.1/"):
                                failed = await client.call_tool("fetch_web_page", {"url": url})
                                assert failed.isError and "TARGET_NOT_ALLOWED" in failed.content[0].text
                                assert url not in str(failed)
                        for args in ({}, {"url": 42}):
                            bad = await client.call_tool("fetch_web_page", args)
                            assert bad.isError
                        assert len((await client.list_tools()).tools) == 1
        asyncio.run(run())
        stderr.seek(0)
        logs = stderr.read()
    assert len(processes) == 1 and processes[0].returncode == 0
    assert forced == []
    for canary in ("raw_secret_canary", "10.0.0.1", "127.0.0.1", "query_canary", "https://", "Traceback"):
        assert canary not in logs and canary not in "".join(notifications)
    assert "fetch_failed code=" in logs
