import asyncio

import pytest

from backend.mcp_host.manager import MCPClientManager, ServerUnavailable, CleanupError, OwnerError


def test_owner_lease_lock_and_close(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        await manager.connect(config)
        entered, release = asyncio.Event(), asyncio.Event()
        async def borrower():
            async with manager.session("one"):
                entered.set(); await release.wait()
        task = asyncio.create_task(borrower())
        await entered.wait()
        close = asyncio.create_task(manager.aclose())
        await asyncio.sleep(0)
        with pytest.raises(ServerUnavailable):
            async with manager.session("one"):
                pass
        assert not close.done()
        release.set(); await task; await close; await manager.aclose()
        assert manager.connections["one"].task.done()
        assert fake_sdk.stderr.closed
        assert [e[1] for e in fake_sdk.events] == ["transport_enter", "session_enter", "session_exit", "transport_exit"]
        assert len({e[2] for e in fake_sdk.events}) == 1
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["init_fail", "entry_fail", "hang"])
def test_failed_start_cleanup(config, fake_sdk, mode):
    async def run():
        manager = MCPClientManager()
        with pytest.raises((ServerUnavailable, TimeoutError)):
            await manager.connect(config.model_copy(update={"args": [mode]}))
        await manager.aclose()
        connection = manager.connections["one"]
        assert connection.task.done() and not connection.cleanup_failed
        assert any(d.kind == "initialization" for d in connection.diagnostics)
    asyncio.run(run())


def test_cancel_start_and_repeated_cancel(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        task = asyncio.create_task(manager.connect(config.model_copy(update={"args": ["hang"]})))
        while not fake_sdk.sessions:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await manager.aclose()
        assert all(c.task.done() for c in manager.connections.values())
    asyncio.run(run())


def test_serial_leases_and_unavailable(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        await manager.connect(config)
        active = 0
        async def use():
            nonlocal active
            async with manager.session("one"):
                active += 1
                assert active == 1
                await asyncio.sleep(0)
                active -= 1
        await asyncio.gather(*(use() for _ in range(5)))
        manager.mark_unavailable("one")
        with pytest.raises(ServerUnavailable):
            async with manager.session("one"):
                pass
        await manager.aclose()
    asyncio.run(run())


def test_cleanup_timeout_observable_without_cancelling_owner(config, fake_sdk):
    async def run():
        manager = MCPClientManager(shutdown_timeout_seconds=0.02)
        await manager.connect(config)
        release = asyncio.Event()
        fake_sdk.exit_release = release
        with pytest.raises(CleanupError):
            await manager.aclose()
        connection = manager.connections["one"]
        assert connection.cleanup_failed and not connection.task.done()
        release.set(); await connection.task
        assert not connection.task.cancelled()
    asyncio.run(run())


def test_exit_error_reported_not_silenced(config, fake_sdk):
    async def run():
        manager = MCPClientManager(); await manager.connect(config)
        fake_sdk.exit_error = OSError("secret-cleanup")
        with pytest.raises(CleanupError):
            await manager.aclose()
        assert manager.connections["one"].cleanup_failed
        assert "transport_exit" in [e[1] for e in fake_sdk.events]
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["session", "transport"])
def test_exit_failure_has_phase_diagnostics(config, fake_sdk, phase):
    async def run():
        manager = MCPClientManager()
        await manager.connect(config)
        setattr(fake_sdk, "exit_error" if phase == "session" else "transport_exit_error",
                OSError("secret-exit"))
        with pytest.raises(CleanupError) as caught:
            await manager.aclose()
        connection = manager.connections["one"]
        assert connection.cleanup_failed and connection.task.done()
        assert any(d.phase == f"{phase}_exit" and d.kind == "cleanup" and d.error_type == "OSError"
                   for d in caught.value.diagnostics)
        assert "secret-exit" not in str(caught.value)
    asyncio.run(run())


def test_mixed_transport_group_does_not_hide_real_cleanup_error(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        await manager.connect(config)
        fake_sdk.transport_exit_error = ExceptionGroup("secret-group", [
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "secret-bytes"), OSError("secret-cleanup")])
        with pytest.raises(CleanupError) as caught:
            await manager.aclose()
        diagnostics = caught.value.diagnostics
        assert any(d.kind == "communication" and d.error_type == "UnicodeDecodeError" for d in diagnostics)
        assert any(d.kind == "cleanup" and d.error_type == "OSError" for d in diagnostics)
        assert manager.connections["one"].cleanup_failed
    asyncio.run(run())


def test_owner_cancellation_passes_through_exits_without_false_cleanup_error(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        await manager.connect(config)
        connection = manager.connections["one"]
        connection.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connection.task
        await manager.aclose()
        assert not connection.cleanup_failed
        assert [e[1] for e in fake_sdk.events][-2:] == ["session_exit", "transport_exit"]
    asyncio.run(run())


def test_owner_infrastructure_failure_is_distinct_from_cleanup(config, fake_sdk, monkeypatch):
    import backend.mcp_host.manager as module
    def failed_open(*args, **kwargs): raise OSError("secret-open")
    monkeypatch.setattr(module, "open", failed_open, raising=False)
    async def run():
        manager = MCPClientManager()
        with pytest.raises(OwnerError):
            await manager.connect(config)
        await manager.aclose()
        connection = manager.connections["one"]
        assert not connection.cleanup_failed and connection.state == "failed"
        assert [(d.phase, d.kind, d.error_type) for d in connection.diagnostics] == [("owner", "owner", "OSError")]
    asyncio.run(run())


def test_cancelled_start_keeps_cleanup_error_as_cause(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        fake_sdk.exit_error = OSError("secret-cleanup")
        startup = asyncio.create_task(manager.connect(config.model_copy(update={"args": ["hang"]})))
        while not fake_sdk.sessions:
            await asyncio.sleep(0)
        startup.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await startup
        assert isinstance(caught.value.__cause__, CleanupError)
        assert any(d.phase == "session_exit" for d in caught.value.__cause__.diagnostics)
        with pytest.raises(CleanupError):
            await manager.aclose()
        assert manager.connections["one"].task.done()
        assert [e[1] for e in fake_sdk.events][-1] == "transport_exit"
        assert len({e[2] for e in fake_sdk.events}) == 1
    asyncio.run(run())


def test_repeated_cancel_during_close_preserves_cleanup_error(config, fake_sdk):
    async def run():
        manager = MCPClientManager()
        await manager.connect(config)
        fake_sdk.exit_started = asyncio.Event()
        fake_sdk.exit_release = asyncio.Event()
        fake_sdk.exit_error = OSError("secret-cleanup")
        close = asyncio.create_task(manager.aclose())
        await fake_sdk.exit_started.wait()
        close.cancel(); await asyncio.sleep(0); close.cancel()
        fake_sdk.exit_release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await close
        assert isinstance(caught.value.__cause__, CleanupError)
        assert any(d.phase == "session_exit" and d.kind == "cleanup"
                   for d in caught.value.__cause__.diagnostics)
        assert manager.connections["one"].task.done()
        assert len({e[2] for e in fake_sdk.events}) == 1
    asyncio.run(run())
