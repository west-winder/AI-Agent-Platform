import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI
import pytest

import backend.mcp_host.bootstrap as bootstrap
from backend.mcp_host.manager import CleanupError


def test_asgi_startup_shutdown_partial_inflight(config, fake_sdk, monkeypatch):
    original = bootstrap.bootstrap_host
    monkeypatch.setattr(bootstrap, "bootstrap_host", lambda: original([
        config, config.model_copy(update={"server_id": "bad", "args": ["init_fail"]})], llm=SimpleNamespace()))
    async def run():
        app = FastAPI(lifespan=bootstrap.lifespan)
        queue = asyncio.Queue(); outputs = asyncio.Queue()
        async def send(message): await outputs.put(message)
        task = asyncio.create_task(app({"type": "lifespan", "state": {}}, queue.get, send))
        await queue.put({"type": "lifespan.startup"})
        assert (await outputs.get())["type"] == "lifespan.startup.complete"
        host = app.state.mcp_host
        assert app.state.agent_runtime is host.runtime
        assert host.statuses["bad"] == "initialization_failed"
        entered, release = asyncio.Event(), asyncio.Event()
        async def inflight():
            async with host.manager.session("one"):
                entered.set(); await release.wait()
        borrower = asyncio.create_task(inflight()); await entered.wait()
        await queue.put({"type": "lifespan.shutdown"}); await asyncio.sleep(0)
        release.set(); await borrower
        assert (await outputs.get())["type"] == "lifespan.shutdown.complete"
        await task
        assert not hasattr(app.state, "agent_runtime")
        assert all(c.task.done() for c in host.manager.connections.values())
    asyncio.run(run())


def test_asgi_startup_failure(monkeypatch):
    @asynccontextmanager
    async def failed():
        raise ValueError("infrastructure")
        yield
    monkeypatch.setattr(bootstrap, "bootstrap_host", failed)
    async def run():
        app = FastAPI(lifespan=bootstrap.lifespan)
        outputs = []
        async def receive(): return {"type": "lifespan.startup"}
        async def send(value): outputs.append(value)
        with pytest.raises(ValueError):
            await app({"type": "lifespan", "state": {}}, receive, send)
        assert outputs[0]["type"] == "lifespan.startup.failed"
        assert not hasattr(app.state, "mcp_host")
    asyncio.run(run())


def test_main_routes_and_no_new_api():
    # AST verifies the actual production connection without importing its live DB.
    import ast
    from pathlib import Path
    tree = ast.parse(Path("backend/main.py").read_text(encoding="utf-8"))
    app_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "FastAPI")
    assert any(k.arg == "lifespan" and isinstance(k.value, ast.Name) and k.value.id == "lifespan" for k in app_call.keywords)
    routes = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "include_router"]
    assert len(routes) == 7


def test_cancel_asgi_startup_cleans_owner(config, fake_sdk, monkeypatch):
    original = bootstrap.bootstrap_host
    monkeypatch.setattr(bootstrap, "bootstrap_host", lambda: original([
        config.model_copy(update={"args": ["hang"]})], llm=SimpleNamespace()))
    async def run():
        app = FastAPI(lifespan=bootstrap.lifespan)
        async def receive(): return {"type": "lifespan.startup"}
        async def send(message): pass
        task = asyncio.create_task(app({"type": "lifespan", "state": {}}, receive, send))
        while not fake_sdk.sessions: await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert not hasattr(app.state, "mcp_host")
        assert [e[1] for e in fake_sdk.events][-2:] == ["session_exit", "transport_exit"]
    asyncio.run(run())


def test_cancelled_lifespan_retains_cleanup_failure(config, fake_sdk, monkeypatch):
    original = bootstrap.bootstrap_host
    monkeypatch.setattr(bootstrap, "bootstrap_host", lambda: original([config], llm=SimpleNamespace()))
    async def run():
        app = FastAPI(lifespan=bootstrap.lifespan)
        ready = asyncio.Event()
        async def lifecycle():
            async with bootstrap.lifespan(app):
                ready.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(lifecycle())
        await ready.wait()
        fake_sdk.exit_error = OSError("secret-cleanup")
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert isinstance(caught.value.__cause__, CleanupError)
        assert any(d.phase == "session_exit" for d in caught.value.__cause__.diagnostics)
        host = app.state.mcp_host
        assert host.manager.connections["one"].cleanup_failed
        assert host.manager.connections["one"].task.done()
        assert not hasattr(app.state, "agent_runtime")
        assert len({e[2] for e in fake_sdk.events}) == 1
    asyncio.run(run())
