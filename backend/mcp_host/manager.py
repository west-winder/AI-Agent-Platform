"""Own SDK contexts in one task per server; never exit them from a borrower."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
import os

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ServerConfig
from .safe_logging import SDKLogGuard


class ServerUnavailable(Exception):
    pass


@dataclass(frozen=True)
class LifecycleDiagnostic:
    phase: str
    kind: str
    error_type: str


class CleanupError(Exception):
    def __init__(self, message: str, *, diagnostics=()):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


class OwnerError(Exception):
    """An unexpected owner infrastructure failure, distinct from cleanup failure."""


class SessionProtocolError(Exception):
    pass


async def finish_cleanup(task: asyncio.Task):
    """Finish saved cleanup despite repeated caller cancellation, then propagate it."""
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            break
    try:
        result = task.result()
    except Exception as error:
        if cancelled is not None:
            raise cancelled from error
        raise
    if cancelled is not None:
        raise cancelled
    return result


@dataclass
class Connection:
    config: ServerConfig
    ready: asyncio.Future
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: str = "connecting"
    session: ClientSession | None = None
    task: asyncio.Task | None = None
    closing: asyncio.Task | None = None
    cleanup_failed: bool = False
    protocol_failed: bool = False
    diagnostics: list[LifecycleDiagnostic] = field(default_factory=list)


def _leaves(error):
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            yield from _leaves(child)
    elif error is not None:
        yield error


def _record(connection, observed, phase, kind, error):
    for leaf in _leaves(error):
        if not isinstance(leaf, Exception) or any(leaf is old for old in observed):
            continue
        observed.append(leaf)
        connection.diagnostics.append(LifecycleDiagnostic(phase, kind, type(leaf).__name__))
        if kind == "cleanup":
            connection.cleanup_failed = True


async def _enter_resource(stack, context, connection, observed, phase):
    """Keep SDK exit propagation intact while attributing only NEW exit failures."""
    try:
        resource = await context.__aenter__()
    except BaseException as error:
        _record(connection, observed, f"{phase}_enter", "initialization", error)
        raise

    async def exit_resource(exc_type, exc_value, traceback):
        try:
            return await context.__aexit__(exc_type, exc_value, traceback)
        except BaseException as error:
            incoming = tuple(_leaves(exc_value))
            for leaf in _leaves(error):
                if any(leaf is old for old in incoming):
                    continue  # Re-raising/wrapping the body exception is not cleanup failure.
                # SDK 1.26 stdio reports reader/writer task failures from TaskGroup
                # exit. Only these known communication leaves are reclassified;
                # direct exit errors and unknown/mixed leaves remain cleanup errors.
                communication = phase == "transport" and isinstance(error, BaseExceptionGroup) and isinstance(
                    leaf, (UnicodeDecodeError, anyio.BrokenResourceError, anyio.ClosedResourceError,
                           anyio.EndOfStream, ConnectionError),
                )
                _record(connection, observed, f"{phase}_exit",
                        "communication" if communication else "cleanup", leaf)
            raise

    stack.push_async_exit(exit_resource)
    return resource


class MCPClientManager:
    def __init__(self, *, shutdown_timeout_seconds: float = 10):
        self.connections: dict[str, Connection] = {}
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self._closing = False
        self._close_task = None
        self._log_guard = SDKLogGuard()
        self._log_guard.__enter__()

    async def _owner(self, connection: Connection):
        failed = False
        observed = []
        try:
            async with AsyncExitStack() as stack:
                # A sink prevents SDK/server stderr from reaching platform logs and disk.
                stderr = open(os.devnull, "w", encoding="utf-8")

                def close_stderr():
                    try:
                        stderr.close()
                    except Exception as error:
                        _record(connection, observed, "stderr_exit", "cleanup", error)
                        raise

                stack.callback(close_stderr)
                try:
                    config = connection.config
                    streams = await _enter_resource(stack, stdio_client(
                        StdioServerParameters(command=config.command, args=config.args,
                                              cwd=config.cwd, env=config.env), errlog=stderr,
                    ), connection, observed, "transport")
                    if connection.stop.is_set():
                        return
                    async def on_message(message):
                        if isinstance(message, Exception):
                            connection.protocol_failed = True
                            self.mark_unavailable(config.server_id)

                    session = await _enter_resource(stack, ClientSession(
                        *streams, read_timeout_seconds=timedelta(seconds=config.startup_timeout_seconds),
                        message_handler=on_message,
                    ), connection, observed, "session")
                    # Requests may be made in a child task; SDK contexts stay in owner.
                    initialize = asyncio.create_task(session.initialize())
                    stopped = asyncio.create_task(connection.stop.wait())
                    try:
                        done, _ = await asyncio.wait({initialize, stopped},
                            timeout=config.startup_timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
                        if stopped in done:
                            return
                        if initialize not in done:
                            raise TimeoutError
                        await initialize
                    finally:
                        for task in (initialize, stopped):
                            if not task.done(): task.cancel()
                        await asyncio.gather(initialize, stopped, return_exceptions=True)
                    if connection.stop.is_set():
                        return
                    connection.session = session
                    connection.state = "ready"
                    connection.ready.set_result(True)
                    await connection.stop.wait()
                except Exception as error:
                    _record(connection, observed, "initialize", "initialization", error)
                    failed = True
        except asyncio.CancelledError:
            failed = True
            raise
        except Exception as error:
            failed = True
            # Exit callbacks have already attributed their own errors. Anything
            # left here originated in owner infrastructure, not in cleanup.
            _record(connection, observed, "owner", "owner", error)
        finally:
            connection.session = None
            connection.state = "failed" if failed else "closed"
            if not connection.ready.done():
                connection.ready.set_result(False)
            if self._closing and all(c.task is None or c is connection or c.task.done()
                                     for c in self.connections.values()):
                self._log_guard.__exit__(None, None, None)

    async def connect(self, config: ServerConfig) -> None:
        if self._closing or config.server_id in self.connections:
            raise ServerUnavailable("MCP connection unavailable")
        connection = Connection(config=config, ready=asyncio.get_running_loop().create_future())
        self.connections[config.server_id] = connection
        connection.task = asyncio.create_task(self._owner(connection), name=f"mcp-owner-{config.server_id}")
        # Consume task failures even if a cancelled startup caller never gets ready.
        connection.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        try:
            async with asyncio.timeout(config.startup_timeout_seconds):
                ready = await asyncio.shield(connection.ready)
            if not ready:
                if any(d.kind == "owner" for d in connection.diagnostics):
                    raise OwnerError("MCP owner infrastructure failed")
                raise ServerUnavailable("MCP initialization failed")
        except asyncio.CancelledError as cancelled:
            try:
                await self.close_server(config.server_id)
            except CleanupError as cleanup:
                raise cancelled from cleanup
            raise
        except BaseException:
            await self.close_server(config.server_id)
            raise

    @asynccontextmanager
    async def session(self, server_id: str):
        connection = self.connections.get(server_id)
        if self._closing or connection is None or connection.state != "ready":
            raise ServerUnavailable("MCP session unavailable")
        async with connection.lock:
            if self._closing or connection.state != "ready" or connection.session is None:
                raise ServerUnavailable("MCP session unavailable")
            yield connection.session
            if connection.protocol_failed:
                raise SessionProtocolError("Invalid MCP protocol message")

    def mark_unavailable(self, server_id: str):
        connection = self.connections.get(server_id)
        if connection is not None and connection.state == "ready":
            connection.state = "unavailable"

    async def _close_connection(self, connection):
        if connection.task.done():
            if connection.cleanup_failed:
                raise CleanupError("MCP owner cleanup failed", diagnostics=connection.diagnostics)
            return
        connection.state = "closing"
        # Drain a current lease, but do not admit queued/new leases.
        try:
            async with asyncio.timeout(self.shutdown_timeout_seconds):
                async with connection.lock:
                    pass
        except TimeoutError:
            pass
        connection.stop.set()
        done, _ = await asyncio.wait({connection.task}, timeout=self.shutdown_timeout_seconds)
        if not done:
            connection.cleanup_failed = True
            connection.diagnostics.append(LifecycleDiagnostic("owner_wait", "cleanup", "TimeoutError"))
            raise CleanupError("MCP owner cleanup did not complete", diagnostics=connection.diagnostics)
        if connection.cleanup_failed:
            raise CleanupError("MCP owner cleanup failed", diagnostics=connection.diagnostics)

    async def close_server(self, server_id: str):
        connection = self.connections.get(server_id)
        if connection is None:
            return
        if connection.closing is None:
            if not connection.task.done():
                connection.state = "closing"
            connection.closing = asyncio.create_task(self._close_connection(connection))
        await finish_cleanup(connection.closing)

    async def _close_all(self):
        results = await asyncio.gather(*(self.close_server(key) for key in self.connections), return_exceptions=True)
        if all(c.task.done() for c in self.connections.values()):
            self._log_guard.__exit__(None, None, None)
        if any(isinstance(result, BaseException) for result in results):
            raise CleanupError("One or more MCP owners failed cleanup", diagnostics=[
                diagnostic for c in self.connections.values() for diagnostic in c.diagnostics
            ])

    async def aclose(self):
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_all())
        await finish_cleanup(self._close_task)
