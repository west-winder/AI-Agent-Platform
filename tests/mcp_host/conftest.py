import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from mcp import types
import pytest

from backend.mcp_host.config import ServerConfig
import backend.mcp_host.manager as manager_module


@pytest.fixture
def config():
    return ServerConfig(server_id="one", command="unused", startup_timeout_seconds=0.15,
                        call_timeout_seconds=0.15)


@pytest.fixture
def fake_sdk(monkeypatch):
    state = SimpleNamespace(sessions=[], events=[], tools_by_command={}, pages_by_command={},
                            exit_release=None, exit_error=None, transport_exit_error=None,
                            exit_started=None)

    @asynccontextmanager
    async def transport(params, errlog):
        if "entry_fail" in params.args:
            raise OSError("secret")
        owner = asyncio.current_task()
        state.events.append((params.command, "transport_enter", owner))
        try:
            yield params, None
        finally:
            assert asyncio.current_task() is owner
            state.events.append((params.command, "transport_exit", owner))
            assert not errlog.closed
            if state.transport_exit_error is not None:
                raise state.transport_exit_error
        state.stderr = errlog

    class Session:
        def __init__(self, params, write, **kwargs):
            self.params = params
            self.tools = state.tools_by_command.get(params.command, [types.Tool(name="echo", inputSchema={"type": "object"})])
            self.calls = []
            self.pages = state.pages_by_command.get(params.command)
            self.call_error = None
            self.result = types.CallToolResult(content=[], structuredContent={"ok": True})
            state.sessions.append(self)

        async def __aenter__(self):
            self.owner = asyncio.current_task()
            state.events.append((self.params.command, "session_enter", self.owner))
            return self

        async def __aexit__(self, *args):
            assert asyncio.current_task() is self.owner
            if state.exit_started is not None:
                state.exit_started.set()
            if state.exit_release is not None:
                await state.exit_release.wait()
            if state.exit_error is not None:
                raise state.exit_error
            state.events.append((self.params.command, "session_exit", self.owner))

        async def initialize(self):
            if "init_fail" in self.params.args:
                raise ValueError("secret")
            if "hang" in self.params.args:
                await asyncio.Event().wait()

        def get_server_capabilities(self):
            return types.ServerCapabilities(tools=types.ToolsCapability())

        async def list_tools(self, **kwargs):
            self.calls.append(("list", kwargs.get("params")))
            if self.pages is not None:
                return self.pages.pop(0)
            return types.ListToolsResult(tools=self.tools)

        async def call_tool(self, name, arguments, **kwargs):
            self.calls.append((name, arguments))
            if self.call_error:
                raise self.call_error
            return self.result

    monkeypatch.setattr(manager_module, "stdio_client", transport)
    monkeypatch.setattr(manager_module, "ClientSession", Session)
    return state
