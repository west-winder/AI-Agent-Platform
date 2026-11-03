import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp.client import stdio
import pytest

from backend.agent_runtime.contracts import LLMMessage, LLMResponse, LLMToolCall
from backend.mcp_host.bootstrap import bootstrap_host
from backend.mcp_host.config import REPO_ROOT, ServerConfig
from backend.mcp_host.manager import MCPClientManager
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.dispatcher import ToolDispatcher

SERVER_ROOT = REPO_ROOT / "mcp_servers/web_page_fetch"
CHILD_PYTHON = SERVER_ROOT / ".venv/Scripts/python.exe"
SUPPORT = Path(__file__).parent / "support/protocol_server.py"


def web_config(production=False):
    return ServerConfig(server_id="web_page_fetch", command=str(CHILD_PYTHON),
        args=["-B", "-m", "web_page_fetch"] if production else ["-B", str(SERVER_ROOT / "tests/support/fixture_server.py")],
        cwd=str(SERVER_ROOT), env={"PYTHONUTF8": "1"})


def protocol_config(mode="pages", *args):
    return ServerConfig(server_id="test", command=sys.executable, args=["-B", str(SUPPORT), mode, *args],
                        cwd=str(REPO_ROOT), env={"PYTHONUTF8": "1"}, call_timeout_seconds=0.4)


@pytest.fixture
def processes(monkeypatch):
    records = SimpleNamespace(items=[], forced=[])
    create, terminate = stdio._create_platform_compatible_process, stdio._terminate_process_tree
    async def observed(*args, **kwargs):
        p = await create(*args, **kwargs); records.items.append(p); return p
    async def forced(*args, **kwargs):
        records.forced.append(True); return await terminate(*args, **kwargs)
    monkeypatch.setattr(stdio, "_create_platform_compatible_process", observed)
    monkeypatch.setattr(stdio, "_terminate_process_tree", forced)
    return records


def test_full_agent_loop_web_fetch_and_safe_failure(processes, caplog):
    name = "mcp_web_page_fetch__fetch_web_page"
    async def run():
        llm = SimpleNamespace(complete=AsyncMock(side_effect=[
            LLMResponse(None, "tool_calls", [LLMToolCall("1", name, '{"url":"https://example.com/"}', {"url": "https://example.com/"})]),
            LLMResponse("已读取 Fixture 正文", "stop"),
        ]))
        async with bootstrap_host([web_config()], llm=llm) as host:
            result = await host.runtime.run([LLMMessage("user", "读取网页")], ToolContext(user_id=7))
            assert result.final_answer == "已读取 Fixture 正文"
            tool_message = next(m for m in result.messages if m.role == "tool")
            payload = json.loads(tool_message.content)
            assert payload["status"] == "success" and payload["data"]["content"] == "Fixture 正文"
            assert set(payload["data"]) == {"title", "url", "content", "truncated", "content_type"}
            assert name in [t.name for t in llm.complete.call_args_list[0].args[0].tools]
            dispatcher = ToolDispatcher(host.registry)
            for path in ("failure", "unknown"):
                failed = await dispatcher.dispatch(name, {"url": f"https://example.com/{path}"}, ToolContext())
                assert failed.error.code == "MCP_TOOL_ERROR" and not failed.error.retryable
            good = await dispatcher.dispatch(name, {"url": "https://example.com/"}, ToolContext())
            assert good.status == "success"
        assert all(c.task.done() for c in host.manager.connections.values())
    asyncio.run(run())
    assert [p.returncode for p in processes.items] == [0]
    assert not processes.forced
    assert "raw_secret_canary" not in caplog.text


def test_real_pagination_cache(processes):
    async def run():
        async with bootstrap_host([protocol_config()], llm=SimpleNamespace()) as host:
            assert host.statuses["test"] == "registered"
            dispatcher = ToolDispatcher(host.registry)
            for _ in range(3):
                result = await dispatcher.dispatch("mcp_test__last", {}, ToolContext())
                assert result.data == {"ok": True, "lists": 2, "remote": "last"}
    asyncio.run(run())
    assert [p.returncode for p in processes.items] == [0]


@pytest.mark.parametrize("already_exited", [False, True])
def test_invalid_utf8_server_does_not_block_healthy_server(processes, monkeypatch, already_exited):
    if already_exited:
        create = stdio._create_platform_compatible_process
        async def wait_for_bad_process(*args, **kwargs):
            process = await create(*args, **kwargs)
            if kwargs.get("args", [])[:2] == ["-B", "-c"]:
                # Real bytes stay in the real pipe; make the natural-exit case
                # deterministic before SDK readers start. The live case is unmodified.
                await process.wait()
            return process
        monkeypatch.setattr(stdio, "_create_platform_compatible_process", wait_for_bad_process)
    async def run():
        bad = ServerConfig(server_id="invalid_utf8", command=sys.executable,
            args=["-B", "-c", "import os; os.write(1, bytes([255,10])); os._exit(0)"],
            cwd=str(REPO_ROOT), startup_timeout_seconds=5.0)
        async with bootstrap_host([bad, web_config()], llm=SimpleNamespace()) as host:
            assert host.statuses["invalid_utf8"] == "initialization_failed"
            connection = host.manager.connections["invalid_utf8"]
            assert connection.state == "failed"
            assert not connection.cleanup_failed
            assert connection.task.done()
            if already_exited:
                assert processes.items[0].returncode == 0
            else:
                assert processes.items[0].returncode is not None
            assert host.registry.get("get_weather")
            assert [t.name for t in host.registry.list_tools()] == [
                "get_weather", "mcp_web_page_fetch__fetch_web_page"]
            result = await ToolDispatcher(host.registry).dispatch(
                "mcp_web_page_fetch__fetch_web_page", {"url": "https://example.com/"}, ToolContext())
            assert result.status == "success"
            assert any(d.phase == "transport_exit" and d.kind == "communication"
                       and d.error_type == "UnicodeDecodeError" for d in connection.diagnostics)
        assert all(c.task.done() for c in host.manager.connections.values())
    asyncio.run(run())
    assert len(processes.items) == 2
    assert all(p.returncode is not None for p in processes.items)
    assert processes.items[1].returncode == 0
    print("invalid_utf8_isolation", {"already_exited": already_exited,
        "A_cleanup_failed": False, "B_call": "success",
        "returncodes": [p.returncode for p in processes.items], "all_owners_finished": True})


@pytest.mark.parametrize("mode,expected", [("crash", "MCP_SERVER_UNAVAILABLE"),
    ("hang", "MCP_TIMEOUT"), ("bad_output", "MCP_INVALID_RESULT")])
def test_real_failures_keep_registration(processes, mode, expected):
    async def run():
        async with bootstrap_host([protocol_config(mode)], llm=SimpleNamespace()) as host:
            dispatcher = ToolDispatcher(host.registry)
            result = await dispatcher.dispatch("mcp_test__last", {}, ToolContext())
            assert result.error.code == expected
            assert host.registry.get("mcp_test__last")
            if mode == "crash":
                again = await dispatcher.dispatch("mcp_test__last", {}, ToolContext())
                assert again.error.code == expected and len(processes.items) == 1
    asyncio.run(run())
    assert all(p.returncode is not None for p in processes.items)


def test_malformed_protocol_logging_is_private(processes, caplog):
    async def run():
        async with bootstrap_host([protocol_config("malformed")], llm=SimpleNamespace()) as host:
            result = await ToolDispatcher(host.registry).dispatch("mcp_test__last", {}, ToolContext())
            assert result.error.code == "MCP_PROTOCOL_ERROR"
    asyncio.run(run())
    assert "private-malformed-canary" not in caplog.text
    assert "Traceback" not in caplog.text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object evidence")
def test_forced_exit_and_grandchild_job_cleanup(processes, tmp_path):
    import win32api
    import win32process
    pidfile = tmp_path / "child.pid"
    async def run():
        async with bootstrap_host([protocol_config("stubborn", str(pidfile))], llm=SimpleNamespace()) as host:
            assert host.statuses["test"] == "registered"
            assert getattr(processes.items[0], "_job_object", None) is not None
    child_pid = None
    try:
        asyncio.run(run())
        child_pid = int(pidfile.read_text())
        assert processes.forced and all(p.returncode is not None for p in processes.items)
        try:
            handle = win32api.OpenProcess(0x1000, False, child_pid)
        except Exception:
            return  # PID no longer exists.
        try:
            assert win32process.GetExitCodeProcess(handle) != 259
        finally:
            handle.Close()
    finally:
        # Only a test-created child may be terminated if an assertion exposed an SDK failure.
        if child_pid is None and pidfile.exists(): child_pid = int(pidfile.read_text())
        if child_pid:
            try:
                handle = win32api.OpenProcess(1, False, child_pid)
                try: win32api.TerminateProcess(handle, 1)
                finally: handle.Close()
            except Exception: pass


def test_real_rejected_server_isolated_and_startup_cancel(processes):
    async def run():
        broken = protocol_config().model_copy(update={"server_id": "bad", "command": str(REPO_ROOT / "no-such-python.exe")})
        async with bootstrap_host([web_config(), broken], llm=SimpleNamespace()) as host:
            assert host.statuses["bad"] == "initialization_failed"
            good = await ToolDispatcher(host.registry).dispatch("mcp_web_page_fetch__fetch_web_page",
                {"url": "https://example.com/"}, ToolContext())
            assert good.status == "success"
        manager = MCPClientManager()
        created = asyncio.Event()
        original = stdio._create_platform_compatible_process
        async def signal(*args, **kwargs):
            p = await original(*args, **kwargs); created.set(); return p
        stdio._create_platform_compatible_process = signal
        try:
            startup = asyncio.create_task(manager.connect(protocol_config("no_initialize")))
            await created.wait(); startup.cancel()
            with pytest.raises(asyncio.CancelledError): await startup
            await manager.aclose()
            assert all(c.task.done() for c in manager.connections.values())
        finally:
            stdio._create_platform_compatible_process = original
    asyncio.run(run())
    assert all(p.returncode is not None for p in processes.items)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process handle evidence")
def test_repeated_stdio_handle_count():
    import ctypes
    from ctypes import wintypes
    import gc
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    counts = []
    async def cycle():
        async with bootstrap_host([protocol_config()], llm=SimpleNamespace()) as host:
            assert host.statuses["test"] == "registered"
    async def run():
        for _ in range(5):
            await cycle(); await asyncio.sleep(0); gc.collect()
            count = wintypes.DWORD()
            assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(count))
            counts.append(count.value)
    asyncio.run(run())
    assert counts[-1] <= counts[0] + 4, counts
    print("handle_counts", counts)
