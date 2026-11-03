"""Server-isolated discovery and atomic assembly at the Host boundary."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncio
import logging
from pathlib import Path
import sys

from mcp import types

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.runtime import AgentRuntime
from backend.tools.bootstrap import build_tool_registry
from backend.tools.infrastructure.registry import ToolRegistry
from .config import DEFAULT_CONFIG, ConfigEntry, ServerConfig, load_config
from .manager import CleanupError, MCPClientManager, OwnerError
from .schema_adapter import build_args_model
from .tool_adapter import adapt_tool

logger = logging.getLogger(__name__)


async def discover_tools(session, *, max_pages=100, max_tools=1000):
    tools, names, cursors = [], set(), set()
    cursor = None
    for _ in range(max_pages):
        page = await session.list_tools() if cursor is None else await session.list_tools(
            params=types.PaginatedRequestParams(cursor=cursor))
        for tool in page.tools:
            if tool.name in names or len(tools) >= max_tools:
                raise ValueError("Duplicate tools or discovery limit")
            names.add(tool.name)
            tools.append(tool)
        cursor = page.nextCursor
        if cursor is None:
            return tuple(tools)
        if cursor in cursors:
            raise ValueError("Repeated discovery cursor")
        cursors.add(cursor)
    raise ValueError("Discovery page limit")


@dataclass
class Host:
    manager: MCPClientManager
    registry: ToolRegistry
    runtime: AgentRuntime
    statuses: dict[str, str]


@asynccontextmanager
async def bootstrap_host(configs: list[ServerConfig | ConfigEntry] | None = None, *,
                         config_path: Path = DEFAULT_CONFIG, llm=None):
    entries = load_config(config_path) if configs is None else [
        ConfigEntry(server_id=c.server_id, config=c, enabled=c.enabled) if isinstance(c, ServerConfig) else c
        for c in configs
    ]
    if len({c.server_id for c in entries}) != len(entries):
        raise ValueError("Duplicate MCP server identity")
    registry = build_tool_registry()
    manager = MCPClientManager()
    statuses = {}
    try:
        for entry in entries:
            if not entry.enabled:
                statuses[entry.server_id] = "disabled"
                continue
            if entry.config is None:
                statuses[entry.server_id] = "invalid_config"
                continue
            config = entry.config
            try:
                async with asyncio.timeout(config.startup_timeout_seconds):
                    await manager.connect(config)
                    async with manager.session(config.server_id) as session:
                        caps = session.get_server_capabilities()
                        if caps is None or caps.tools is None:
                            raise ValueError("Server has no tools capability")
                        tools = await discover_tools(session)
                    candidates = [adapt_tool(config.server_id, tool, build_args_model(tool.inputSchema),
                                             manager, config.call_timeout_seconds) for tool in tools]
                    proposed = ToolRegistry()
                    for tool in [*registry.list_tools(), *candidates]:
                        proposed.register(tool)
                    registry = proposed
                statuses[entry.server_id] = "registered"
            except (CleanupError, OwnerError):
                raise
            except Exception:
                statuses[entry.server_id] = "initialization_failed"
                logger.warning("mcp_server_initialization_failed", extra={"server_id": entry.server_id})
                await manager.close_server(entry.server_id)
        runtime = build_agent_runtime(registry=registry, llm=llm)
        yield Host(manager, registry, runtime, statuses)
    finally:
        pending_error = sys.exc_info()[1]
        try:
            await manager.aclose()
        except CleanupError as cleanup:
            if isinstance(pending_error, asyncio.CancelledError):
                raise pending_error from cleanup
            raise


@asynccontextmanager
async def lifespan(app):
    async with bootstrap_host() as host:
        app.state.mcp_host = host
        app.state.tool_registry = host.registry
        app.state.agent_runtime = host.runtime
        try:
            yield
        finally:
            # Retain host/statuses for shutdown diagnostics; stop publishing a Runtime.
            del app.state.agent_runtime
            del app.state.tool_registry
