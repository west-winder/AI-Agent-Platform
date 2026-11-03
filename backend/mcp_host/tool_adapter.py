"""Remote tool calls expressed as ordinary asynchronous platform handlers."""

import asyncio
from dataclasses import dataclass
from datetime import timedelta
import re

import anyio
from jsonschema import validators
from mcp import types
from mcp.shared.exceptions import McpError
from pydantic import BaseModel

from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext
from backend.tools.infrastructure.errors import ToolExecutionError
from .manager import MCPClientManager, ServerUnavailable
from .schema_adapter import check_json

MESSAGES = {
    "MCP_TOOL_ERROR": "远程工具执行失败",
    "MCP_TIMEOUT": "远程工具调用超时",
    "MCP_SERVER_UNAVAILABLE": "远程工具服务不可用",
    "MCP_PROTOCOL_ERROR": "远程工具通信失败",
    "MCP_INVALID_RESULT": "远程工具返回结果不符合约定",
}


def _error(code):
    return ToolExecutionError(code, MESSAGES[code], retryable=False)


def _classify(exc):
    if isinstance(exc, BaseExceptionGroup):
        codes = [_classify(child) for child in exc.exceptions]
        return "MCP_SERVER_UNAVAILABLE" if "MCP_SERVER_UNAVAILABLE" in codes else "MCP_PROTOCOL_ERROR"
    if isinstance(exc, (ServerUnavailable, anyio.BrokenResourceError, anyio.ClosedResourceError,
                        EOFError, BrokenPipeError, ConnectionError)):
        return "MCP_SERVER_UNAVAILABLE"
    if isinstance(exc, McpError):
        if exc.error.code == types.CONNECTION_CLOSED:
            return "MCP_SERVER_UNAVAILABLE"
        if exc.error.code == 408:
            return "MCP_TIMEOUT"
    return "MCP_PROTOCOL_ERROR"


@dataclass(frozen=True)
class ToolBinding:
    server_id: str
    remote_tool_name: str

    @property
    def local_name(self):
        name = f"mcp_{self.server_id}__{self.remote_tool_name}"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.server_id) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Invalid namespaced tool name")
        return name


def adapt_tool(server_id: str, remote_tool: types.Tool, args_model: type[BaseModel],
               manager: MCPClientManager, call_timeout: float) -> RegisteredTool:
    binding = ToolBinding(server_id, remote_tool.name)
    name = binding.local_name
    if remote_tool.outputSchema is not None:
        check_json(remote_tool.outputSchema)
        validators.validator_for(remote_tool.outputSchema).check_schema(remote_tool.outputSchema)

    async def handler(arguments: BaseModel, context: ToolContext) -> dict:
        budget = asyncio.timeout(call_timeout)
        try:
            async with budget:
                payload = arguments.model_dump(mode="python", by_alias=True, exclude_unset=True)
                check_json(payload)
                async with manager.session(binding.server_id) as session:
                    try:
                        result = await session.call_tool(
                            binding.remote_tool_name, payload,
                            read_timeout_seconds=timedelta(seconds=call_timeout),
                        )
                    except RuntimeError:
                        raise _error("MCP_INVALID_RESULT") from None
                if result.isError:
                    raise _error("MCP_TOOL_ERROR")
                if type(result.structuredContent) is not dict:
                    raise _error("MCP_INVALID_RESULT")
                try:
                    check_json(result.structuredContent)
                except ValueError:
                    raise _error("MCP_INVALID_RESULT") from None
                return result.structuredContent
        except asyncio.CancelledError:
            raise
        except ToolExecutionError:
            raise
        except TimeoutError:
            raise _error("MCP_TIMEOUT" if budget.expired() else "MCP_PROTOCOL_ERROR") from None
        except Exception as exc:
            code = _classify(exc)
            if code == "MCP_SERVER_UNAVAILABLE":
                manager.mark_unavailable(binding.server_id)
            raise _error(code) from None

    # Binding is explicit and reviewable without changing RegisteredTool's contract.
    handler.mcp_binding = binding
    return RegisteredTool(name=name, description=remote_tool.description or "",
                          args_schema=args_model, handler=handler)
