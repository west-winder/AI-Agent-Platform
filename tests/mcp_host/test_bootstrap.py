import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp import types
import pytest

import backend.mcp_host.bootstrap as bootstrap
from backend.mcp_host.config import ServerConfig
from backend.mcp_host.manager import OwnerError
from backend.mcp_host.schema_adapter import build_args_model
from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext
from backend.tools.infrastructure.registry import ToolRegistry


def llm():
    return SimpleNamespace(complete=AsyncMock())


def remote(name="echo", schema=None):
    return types.Tool(name=name, inputSchema=schema or {"type": "object"})


def test_namespaces_atomic_partial_and_local(config, fake_sdk):
    fake_sdk.tools_by_command["bad"] = [remote("first"), remote("last", {"type": "array"})]
    async def run():
        configs = [config, config.model_copy(update={"server_id": "bad", "command": "bad"}),
                   config.model_copy(update={"server_id": "two"})]
        async with bootstrap.bootstrap_host(configs, llm=llm()) as host:
            assert [t.name for t in host.registry.list_tools()] == ["get_weather", "mcp_one__echo", "mcp_two__echo"]
            assert host.statuses["bad"] == "initialization_failed"
            assert host.manager.connections["bad"].task.done()
            for name in ("mcp_one__echo", "mcp_two__echo"):
                tool = host.registry.get(name)
                assert await tool.handler(tool.args_schema.model_validate({}), ToolContext()) == {"ok": True}
        assert all(c.task.done() for c in host.manager.connections.values())
    asyncio.run(run())


@pytest.mark.parametrize("tools", [[remote("x"), remote("x")], [remote("a" * 60)], [remote("bad.name")]])
def test_duplicate_or_long_invalid_name(config, fake_sdk, tools):
    fake_sdk.tools_by_command["unused"] = tools
    async def run():
        async with bootstrap.bootstrap_host([config], llm=llm()) as host:
            assert [t.name for t in host.registry.list_tools()] == ["get_weather"]
            assert host.statuses["one"] == "initialization_failed"
    asyncio.run(run())


def test_collision_with_local_registry(config, fake_sdk, monkeypatch):
    registry = ToolRegistry()
    async def handler(args, context): return {}
    registry.register(RegisteredTool("mcp_one__echo", "local", build_args_model({"type": "object"}), handler))
    monkeypatch.setattr(bootstrap, "build_tool_registry", lambda: registry)
    async def run():
        async with bootstrap.bootstrap_host([config], llm=llm()) as host:
            assert host.registry is registry
            assert host.statuses["one"] == "initialization_failed"
            assert host.registry.get("mcp_one__echo").description == "local"
    asyncio.run(run())


def test_pagination_empty_cursor_and_no_relist(config, fake_sdk):
    fake_sdk.pages_by_command["unused"] = [types.ListToolsResult(tools=[], nextCursor=""),
        types.ListToolsResult(tools=[remote("last")])]
    async def run():
        async with bootstrap.bootstrap_host([config], llm=llm()) as host:
            tool = host.registry.get("mcp_one__last")
            for _ in range(2):
                await tool.handler(tool.args_schema.model_validate({}), ToolContext())
            assert [c[0] for c in fake_sdk.sessions[0].calls] == ["list", "list", "last", "last"]
            assert fake_sdk.sessions[0].calls[1][1].cursor == ""
    asyncio.run(run())


@pytest.mark.parametrize("pages", [
    [types.ListToolsResult(tools=[], nextCursor="x"), types.ListToolsResult(tools=[], nextCursor="x")],
    [types.ListToolsResult(tools=[remote()], nextCursor="x"), types.ListToolsResult(tools=[remote()])],
    [types.ListToolsResult(tools=[remote()], nextCursor="x"), types.ListToolsResult(tools=[remote("bad", {"type": "array"})])],
])
def test_bad_pagination_discards_whole_server(config, fake_sdk, pages):
    fake_sdk.pages_by_command["unused"] = pages
    async def run():
        async with bootstrap.bootstrap_host([config], llm=llm()) as host:
            assert len(host.registry.list_tools()) == 1
            assert host.manager.connections["one"].task.done()
    asyncio.run(run())


def test_empty_disabled_and_all_failed(config, fake_sdk):
    fake_sdk.tools_by_command["unused"] = []
    async def run():
        async with bootstrap.bootstrap_host([config], llm=llm()) as host:
            assert host.statuses["one"] == "registered"
        async with bootstrap.bootstrap_host([config.model_copy(update={"enabled": False})], llm=llm()) as host:
            assert host.statuses["one"] == "disabled" and not host.manager.connections
        async with bootstrap.bootstrap_host([config.model_copy(update={"args": ["init_fail"]})], llm=llm()) as host:
            assert len(host.registry.list_tools()) == 1
    asyncio.run(run())


def test_runtime_build_failure_cleans(config, fake_sdk, monkeypatch):
    def fail(**kwargs): raise ValueError("infrastructure")
    monkeypatch.setattr(bootstrap, "build_agent_runtime", fail)
    async def run():
        with pytest.raises(ValueError, match="infrastructure"):
            async with bootstrap.bootstrap_host([config], llm=llm()): pass
        assert [x[1] for x in fake_sdk.events][-2:] == ["session_exit", "transport_exit"]
    asyncio.run(run())


def test_discovery_limits():
    async def run():
        session = SimpleNamespace(list_tools=AsyncMock(return_value=types.ListToolsResult(
            tools=[remote("one"), remote("two")], nextCursor="more")))
        with pytest.raises(ValueError):
            await bootstrap.discover_tools(session, max_tools=1)
        with pytest.raises(ValueError):
            await bootstrap.discover_tools(session, max_pages=1)
    asyncio.run(run())


def test_owner_infrastructure_failure_still_aborts_bootstrap(config, fake_sdk, monkeypatch):
    import backend.mcp_host.manager as manager_module
    def failed_open(*args, **kwargs): raise OSError("secret-open")
    monkeypatch.setattr(manager_module, "open", failed_open, raising=False)
    async def run():
        with pytest.raises(OwnerError):
            async with bootstrap.bootstrap_host([config], llm=llm()):
                pytest.fail("Owner infrastructure failure must not become partial availability")
        assert not fake_sdk.sessions
    asyncio.run(run())
