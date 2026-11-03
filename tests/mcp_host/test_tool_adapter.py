import asyncio
from types import SimpleNamespace

import anyio
from mcp import types
from mcp.shared.exceptions import McpError
import pytest

from backend.mcp_host.manager import MCPClientManager
from backend.mcp_host.schema_adapter import build_args_model
from backend.mcp_host.tool_adapter import adapt_tool
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.errors import ToolExecutionError


def make(manager, timeout=1):
    remote = types.Tool(name="echo", inputSchema={"type": "object"})
    model = build_args_model(remote.inputSchema)
    return adapt_tool("one", remote, model, manager, timeout), model


def test_success_binding_context_and_no_discovery(config, fake_sdk):
    async def run():
        manager = MCPClientManager(); await manager.connect(config)
        tool, model = make(manager)
        assert tool.name == "mcp_one__echo"
        result = await tool.handler(model.model_validate({"x": None}), ToolContext(user_id=123))
        assert result == {"ok": True}
        assert fake_sdk.sessions[0].calls == [("echo", {"x": None})]
        assert tool.handler.mcp_binding.remote_tool_name == "echo"
        await manager.aclose()
    asyncio.run(run())


def test_outgoing_payload_preserves_colliding_extras(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        try:
            await manager.connect(config)
            remote = types.Tool(name="echo", inputSchema={"type": "object", "properties": {
                "optional_text": {"type": "string", "default": "never-insert"},
                "nullable": {"type": ["string", "null"]},
            }})
            model = build_args_model(remote.inputSchema)
            tool = adapt_tool("one", remote, model, manager, 1)
            payload = {"nullable": None, "field_0": "extra-0", "field_1": "extra-1"}
            await tool.handler(model.model_validate(payload), ToolContext())
            assert fake_sdk.sessions[0].calls == [("echo", payload)]
        finally:
            await manager.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("exc,code", [
    (McpError(types.ErrorData(code=408, message="secret")), "MCP_TIMEOUT"),
    (McpError(types.ErrorData(code=types.CONNECTION_CLOSED, message="secret")), "MCP_SERVER_UNAVAILABLE"),
    (anyio.BrokenResourceError(), "MCP_SERVER_UNAVAILABLE"),
    (McpError(types.ErrorData(code=-32602, message="secret")), "MCP_PROTOCOL_ERROR"),
    (ValueError("secret"), "MCP_PROTOCOL_ERROR"),
    (RuntimeError("secret-output-schema"), "MCP_INVALID_RESULT"),
    (ExceptionGroup("secret", [ValueError("secret")]), "MCP_PROTOCOL_ERROR"),
])
def test_errors_safe_no_retry(config, fake_sdk, exc, code):
    async def run():
        manager = MCPClientManager(); await manager.connect(config)
        fake_sdk.sessions[0].call_error = exc
        tool, model = make(manager)
        with pytest.raises(ToolExecutionError) as caught:
            await tool.handler(model.model_validate({}), ToolContext())
        assert caught.value.code == code and not caught.value.retryable
        assert "secret" not in str(caught.value)
        assert len(fake_sdk.sessions[0].calls) == 1
        if code == "MCP_SERVER_UNAVAILABLE":
            with pytest.raises(ToolExecutionError) as again:
                await tool.handler(model.model_validate({}), ToolContext())
            assert again.value.code == code and len(fake_sdk.sessions[0].calls) == 1
        await manager.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("result,code", [
    (SimpleNamespace(isError=True, structuredContent={"secret": True}), "MCP_TOOL_ERROR"),
    (SimpleNamespace(isError=False, structuredContent=None), "MCP_INVALID_RESULT"),
    (SimpleNamespace(isError=False, structuredContent=[]), "MCP_INVALID_RESULT"),
    (SimpleNamespace(isError=False, structuredContent={"x": float("nan")}), "MCP_INVALID_RESULT"),
    (SimpleNamespace(isError=False, structuredContent={"x": object()}), "MCP_INVALID_RESULT"),
])
def test_invalid_results(config, fake_sdk, result, code):
    async def run():
        manager = MCPClientManager(); await manager.connect(config)
        fake_sdk.sessions[0].result = result
        tool, model = make(manager)
        with pytest.raises(ToolExecutionError) as caught:
            await tool.handler(model.model_validate({}), ToolContext())
        assert caught.value.code == code
        await manager.aclose()
    asyncio.run(run())


def test_lock_timeout_and_cancel(config, fake_sdk):
    async def run():
        manager = MCPClientManager(); await manager.connect(config)
        tool, model = make(manager, 0.02)
        async with manager.session("one"):
            with pytest.raises(ToolExecutionError) as caught:
                await tool.handler(model.model_validate({}), ToolContext())
            assert caught.value.code == "MCP_TIMEOUT"
            task = asyncio.create_task(tool.handler(model.model_validate({}), ToolContext()))
            await asyncio.sleep(0); task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert fake_sdk.sessions[0].calls == []
        await manager.aclose()
    asyncio.run(run())


def test_cancel_group_propagates(config, fake_sdk):
    async def run():
        manager = MCPClientManager(); await manager.connect(config)
        fake_sdk.sessions[0].call_error = BaseExceptionGroup("cancel", [asyncio.CancelledError(), ValueError()])
        tool, model = make(manager)
        with pytest.raises(BaseExceptionGroup):
            await tool.handler(model.model_validate({}), ToolContext())
        await manager.aclose()
    asyncio.run(run())
