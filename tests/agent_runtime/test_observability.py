"""观测只记录事实，不改变重试、工具失败、异常与取消语义。"""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import httpx2
from openai import APIStatusError
import pytest
from pydantic import BaseModel

from backend.agent_runtime import observability
from backend.agent_runtime import runtime as runtime_module
from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import (
    AgentRunStatus, LLMMessage, LLMResponse, LLMToolCall,
)
from backend.agent_runtime.observability import (
    LoggingObserver, NoOpObserver, ObservationEvent, RunContext, safe_emit,
)
from backend.agent_runtime.runtime import AgentRuntime
from backend.exceptions.external_exceptions import ExternalServiceError, ExternalTimeoutError
from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext, ToolSuccess
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.errors import ToolExecutionError
from backend.tools.infrastructure.registry import ToolRegistry


class InMemoryObserver:
    def __init__(self):
        self.events: list[ObservationEvent] = []

    def emit(self, event):
        self.events.append(event)

    def of_type(self, event_type):
        return [event for event in self.events if event.event_type == event_type]


class BrokenObserver:
    def emit(self, event):
        raise OSError("private-observer-message")


class EchoArgs(BaseModel):
    text: str


def tool_call(call_id="original-id", **overrides):
    return LLMToolCall(**{
        "id": call_id, "name": "echo", "raw_arguments": '{"text":"private-arguments"}',
        "arguments": {"text": "private-arguments"}, **overrides,
    })


def tools(*calls):
    return LLMResponse("private-model-response", "tool_calls", list(calls))


def make_runtime(*responses, observer=None, handler=None, max_iterations=5, llm=None):
    async def echo(args, context):
        return {"text": "private-tool-result"}

    registry = ToolRegistry()
    registry.register(RegisteredTool("echo", "test", EchoArgs, handler or echo))
    if llm is None:
        llm = SimpleNamespace(complete=AsyncMock(side_effect=list(responses)))
    return AgentRuntime(llm, registry, ToolDispatcher(registry), max_iterations, observer=observer)


def run(runtime):
    return asyncio.run(runtime.run([LLMMessage("user", "private-prompt")], ToolContext()))


def test_events_and_run_context_are_immutable_and_utc():
    attributes = {"attempt": 1}
    local = datetime(2026, 11, 7, tzinfo=timezone(timedelta(hours=8)))
    event = ObservationEvent("run", "llm.started", local, attributes)
    attributes["attempt"] = 2
    assert event.attributes["attempt"] == 1
    assert event.occurred_at.tzinfo is UTC
    assert event.occurred_at == local
    with pytest.raises(TypeError):
        event.attributes["attempt"] = 3
    with pytest.raises(FrozenInstanceError):
        event.run_id = "other"
    context = RunContext.create()
    assert UUID(context.run_id).version == 4
    assert context.started_at_utc.tzinfo is UTC
    with pytest.raises(FrozenInstanceError):
        context.run_id = "other"


@pytest.mark.parametrize("attributes", [
    {"prompt": "private"}, {"arguments": {}}, {"result": "private"},
    {"status": {"nested": "private"}}, {"status": ["private"]},
    {"elapsed_seconds": float("inf")}, {"elapsed_seconds": float("nan")},
])
def test_event_rejects_non_metadata_and_mutable_values(attributes):
    with pytest.raises((TypeError, ValueError)):
        ObservationEvent("run", "run.started", datetime.now(UTC), attributes)


def test_event_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone-aware"):
        ObservationEvent("run", "run.started", datetime(2026, 11, 7), {})


def test_concurrent_runs_on_same_runtime_keep_all_associations_separate():
    async def scenario():
        observer = InMemoryObserver()
        entered = 0
        both_entered = asyncio.Event()

        class LLM:
            async def complete(self, request):
                nonlocal entered
                if not any(message.role == "tool" for message in request.messages):
                    entered += 1
                    if entered == 2:
                        both_entered.set()
                    await both_entered.wait()
                    # 原始 ID 可以在不同 Run 重复，关联必须包含 run_id。
                    return tools(tool_call("shared-raw-id"))
                return LLMResponse("done", "stop")

        runtime = make_runtime(observer=observer, llm=LLM())
        results = await asyncio.gather(*[
            runtime.run([LLMMessage("user", str(index))], ToolContext()) for index in (1, 2)
        ])
        assert all(result.status is AgentRunStatus.COMPLETED for result in results)
        run_ids = {event.run_id for event in observer.events}
        assert len(run_ids) == 2
        call_ids_by_run = []
        for run_id in run_ids:
            assert UUID(run_id).version == 4
            events = [event for event in observer.events if event.run_id == run_id]
            assert [event.event_type for event in events] == [
                "run.started", "llm.started", "llm.completed", "tool.started",
                "tool.completed", "llm.started", "llm.completed", "run.finished",
            ]
            llm_ids = {event.attributes["llm_call_id"] for event in events if "llm_call_id" in event.attributes}
            assert len(llm_ids) == 2
            call_ids_by_run.append(llm_ids)
            parent = events[1].attributes["llm_call_id"]
            for event in events[3:5]:
                assert event.attributes["tool_call_id"] == "shared-raw-id"
                assert event.attributes["parent_llm_call_id"] == parent
        assert call_ids_by_run[0].isdisjoint(call_ids_by_run[1])
        assert "run_id" not in vars(runtime) and "_run_id" not in vars(runtime)

    asyncio.run(scenario())


@pytest.mark.parametrize("recovers", [True, False])
def test_retries_share_logical_id_and_report_attempts(recovers):
    observer = InMemoryObserver()
    first = ExternalTimeoutError("private-first-error")
    final = ValueError("private-final-error")
    runtime = make_runtime(first, LLMResponse("done", "stop") if recovers else final, observer=observer)
    if recovers:
        assert run(runtime).iterations == 1
    else:
        with pytest.raises(ValueError) as raised:
            run(runtime)
        assert raised.value is final
    events = [event for event in observer.events if event.event_type.startswith("llm.")]
    assert len({event.attributes["llm_call_id"] for event in events}) == 1
    assert len(observer.of_type("llm.started")) == 1
    failures = observer.of_type("llm.attempt_failed")
    assert [event.attributes["attempt"] for event in failures] == ([1] if recovers else [1, 2])
    assert failures[0].attributes["category"] == "timeout"
    assert failures[0].attributes["will_retry"] is True
    if not recovers:
        assert failures[1].attributes["will_retry"] is False
        assert failures[1].attributes["error_type"] == "ValueError"
    assert events[-1].event_type == ("llm.completed" if recovers else "llm.exception")
    assert events[-1].attributes["attempts"] == 2
    assert len(observer.of_type("run.finished" if recovers else "run.exception")) == 1


def test_multiple_tools_keep_raw_ids_and_share_parent():
    observer = InMemoryObserver()
    result = run(make_runtime(tools(tool_call("a"), tool_call("b")), LLMResponse("done", "stop"), observer=observer))
    parent = observer.of_type("llm.started")[0].attributes["llm_call_id"]
    assert [message.tool_call_id for message in result.messages if message.role == "tool"] == ["a", "b"]
    for event_type in ("tool.started", "tool.completed"):
        events = observer.of_type(event_type)
        assert [event.attributes["tool_call_id"] for event in events] == ["a", "b"]
        assert all(event.attributes["parent_llm_call_id"] == parent for event in events)
    assert all(event.attributes["status"] == "success" for event in observer.of_type("tool.completed"))


@pytest.mark.parametrize("failure", ["parse", "schema", "missing", "handler"])
def test_tool_failure_completes_and_loop_continues(failure):
    observer = InMemoryObserver()

    async def handler(args, context):
        raise ToolExecutionError("TEMP_UNAVAILABLE", "private-handler-message", retryable=True)

    call = tool_call()
    expected = "TEMP_UNAVAILABLE"
    if failure == "parse":
        call = tool_call(raw_arguments="private-invalid-json", arguments=None, parse_error="private-parse-error")
        expected = "INVALID_ARGUMENTS"
    elif failure == "schema":
        call = tool_call(arguments={})
        expected = "INVALID_ARGUMENTS"
    elif failure == "missing":
        call = tool_call(name="missing")
        expected = "TOOL_NOT_FOUND"
    runtime = make_runtime(tools(call), LLMResponse("done", "stop"), observer=observer, handler=handler)
    assert run(runtime).status is AgentRunStatus.COMPLETED
    completed, = observer.of_type("tool.completed")
    assert completed.attributes["status"] == "error"
    assert completed.attributes["error_code"] == expected
    assert not observer.of_type("tool.exception")
    assert len(observer.of_type("llm.started")) == 2


def test_unexpected_tool_exception_retains_identity_and_stops_batch():
    observer = InMemoryObserver()
    error = RuntimeError("private-handler-error")
    handler = AsyncMock(side_effect=error)
    runtime = make_runtime(tools(tool_call("a"), tool_call("b")), observer=observer, handler=handler)
    with pytest.raises(RuntimeError) as raised:
        run(runtime)
    assert raised.value is error
    handler.assert_awaited_once()
    assert observer.of_type("tool.exception")[0].attributes["error_type"] == "RuntimeError"
    assert observer.of_type("run.exception")[0].attributes["error_type"] == "RuntimeError"
    assert not observer.of_type("tool.completed") and not observer.of_type("run.finished")


@pytest.mark.parametrize("phase", ["result_serialization", "history_write"])
def test_tool_boundary_includes_serialization_and_history_write(monkeypatch, phase):
    observer = InMemoryObserver()
    error = RuntimeError("private-post-dispatch-error")
    if phase == "result_serialization":
        def broken_dump(self, **kwargs):
            raise error

        monkeypatch.setattr(ToolSuccess, "model_dump", broken_dump)
    else:
        original = runtime_module.LLMMessage

        def broken_message(*args, **kwargs):
            if kwargs.get("role") == "tool":
                raise error
            return original(*args, **kwargs)

        monkeypatch.setattr(runtime_module, "LLMMessage", broken_message)
    with pytest.raises(RuntimeError) as raised:
        run(make_runtime(tools(tool_call()), observer=observer))
    assert raised.value is error
    assert len(observer.of_type("tool.exception")) == 1
    assert not observer.of_type("tool.completed")


@pytest.mark.parametrize("original_failure", [None, "llm", "tool", "context"])
def test_observer_failure_and_failed_diagnostic_do_not_change_result(monkeypatch, original_failure):
    error = RuntimeError("private-original-error")
    diagnostic = Mock(side_effect=OSError("private-diagnostic-message"))
    monkeypatch.setattr(observability.logger, "warning", diagnostic)
    responses = [LLMResponse("done", "stop")]
    handler = None
    if original_failure == "llm":
        responses = [error]
    elif original_failure == "tool":
        responses = [tools(tool_call())]
        handler = AsyncMock(side_effect=error)
    runtime = make_runtime(*responses, observer=BrokenObserver(), handler=handler)
    if original_failure == "context":
        runtime._context_builder = SimpleNamespace(build=Mock(side_effect=error))
    if original_failure:
        with pytest.raises(RuntimeError) as raised:
            run(runtime)
        assert raised.value is error
    else:
        assert run(runtime).status is AgentRunStatus.COMPLETED
    assert diagnostic.call_count >= 2


@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
def test_safe_emit_does_not_catch_base_exception(error):
    class InterruptingObserver:
        def emit(self, event):
            raise error

    with pytest.raises(type(error)) as raised:
        safe_emit(InterruptingObserver(), ObservationEvent("run", "run.started", datetime.now(UTC), {}))
    assert raised.value is error


@pytest.mark.parametrize("phase", ["llm_first", "llm_second", "llm_wait", "tool"])
def test_active_cancellation_has_cancel_events_and_propagates(monkeypatch, phase):
    async def scenario():
        observer = InMemoryObserver()
        entered = asyncio.Event()

        async def block(*args):
            entered.set()
            await asyncio.Event().wait()

        handler = block if phase == "tool" else None
        llm = SimpleNamespace(complete=AsyncMock(side_effect=block))
        if phase == "tool":
            llm.complete = AsyncMock(return_value=tools(tool_call("a"), tool_call("b")))
        elif phase == "llm_second":
            calls = 0

            async def second(request):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise ExternalTimeoutError("private")
                await block()

            llm.complete = AsyncMock(side_effect=second)
        elif phase == "llm_wait":
            failure = rate_limit_failure()
            llm.complete = AsyncMock(side_effect=failure)
            monkeypatch.setattr(runtime_module.asyncio, "sleep", block)
        runtime = make_runtime(llm=llm, observer=observer, handler=handler)
        task = asyncio.create_task(runtime.run([LLMMessage("user", "private-prompt")], ToolContext()))
        await entered.wait()
        task.cancel("private-cancellation-message")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("private-cancellation-message",)
        assert observer.events[-1].event_type == "run.cancelled"
        cancelled = observer.of_type("tool.cancelled" if phase == "tool" else "llm.cancelled")
        assert len(cancelled) == 1
        assert not observer.of_type("run.exception") and not observer.of_type("run.finished")
        assert not observer.of_type("tool.exception") and not observer.of_type("llm.exception")
        if phase == "tool":
            assert len(observer.of_type("tool.started")) == 1
            assert not observer.of_type("tool.completed")
        else:
            assert not observer.of_type("llm.completed")

    asyncio.run(scenario())


def rate_limit_failure():
    request = httpx2.Request("POST", "https://example.invalid")
    error = ExternalServiceError("private-error")
    error.__cause__ = APIStatusError("private-upstream", response=httpx2.Response(
        429, headers={"retry-after": "30"}, request=request,
    ), body=None)
    return error


@pytest.mark.parametrize("phase", ["first", "second", "wait"])
def test_budget_expiry_is_failure_not_cancellation_without_duplicate_attempts(monkeypatch, phase):
    async def scenario():
        observer = InMemoryObserver()
        loop = asyncio.get_running_loop()
        started = loop.time()
        clock = [started]
        first = rate_limit_failure()
        count = 0

        async def block_until_deadline(*args):
            clock[0] = started + 180
            await asyncio.Event().wait()

        async def complete(request):
            nonlocal count
            count += 1
            if phase != "first" and count == 1:
                raise first
            await block_until_deadline()

        runtime = make_runtime(llm=SimpleNamespace(complete=complete), observer=observer)
        with monkeypatch.context() as patch:
            patch.setattr(loop, "time", lambda: clock[0])
            if phase == "wait":
                patch.setattr(runtime_module.asyncio, "sleep", block_until_deadline)
            else:
                async def no_wait(delay):
                    pass

                patch.setattr(runtime_module.asyncio, "sleep", no_wait)
            with pytest.raises(ExternalServiceError) as raised:
                await runtime.run([LLMMessage("user", "private")], ToolContext())
        if phase == "wait":
            assert raised.value is first
        else:
            assert isinstance(raised.value, ExternalTimeoutError)
        failures = observer.of_type("llm.attempt_failed")
        assert [event.attributes["attempt"] for event in failures] == ([1, 2] if phase == "second" else [1])
        assert failures[0].attributes["status_code"] == (None if phase == "first" else 429)
        assert observer.of_type("llm.exception")[0].attributes["attempts"] == count
        assert not observer.of_type("llm.cancelled") and not observer.of_type("run.cancelled")

    asyncio.run(scenario())


@pytest.mark.parametrize("status", list(AgentRunStatus))
def test_normal_terminal_status_is_emitted_exactly_once(status):
    observer = InMemoryObserver()
    response = LLMResponse("done", "stop") if status is AgentRunStatus.COMPLETED else tools(tool_call())
    result = run(make_runtime(response, observer=observer, max_iterations=1))
    assert result.status is status
    terminals = [event for event in observer.events if event.event_type in {
        "run.finished", "run.exception", "run.cancelled",
    }]
    assert len(terminals) == 1
    assert terminals[0].event_type == "run.finished"
    assert terminals[0].attributes["status"] == status.value
    assert terminals[0].attributes["iterations"] == 1
    assert not observer.of_type("tool.started")


def test_input_validation_precedes_observation():
    observer = InMemoryObserver()
    runtime = make_runtime(observer=observer)
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(runtime.run([], ToolContext()))
    assert observer.events == []


def test_elapsed_time_uses_monotonic_even_if_wall_clock_moves_back(monkeypatch):
    observer = InMemoryObserver()
    clock = [10.0]
    wall = [datetime(2026, 11, 7, tzinfo=UTC)]

    class Clock:
        @staticmethod
        def now(tz):
            return wall[0]

    async def complete(request):
        clock[0] += 2
        wall[0] -= timedelta(days=1)
        return tools(tool_call()) if len(request.messages) == 1 else LLMResponse("done", "stop")

    async def handler(args, context):
        clock[0] += 3
        return {"private": "result"}

    original_dump = ToolSuccess.model_dump

    def dump(self, **kwargs):
        clock[0] += 5
        return original_dump(self, **kwargs)

    original_message = runtime_module.LLMMessage

    def message(*args, **kwargs):
        if kwargs.get("role") == "tool":
            clock[0] += 7
        return original_message(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(observability, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime_module, "datetime", Clock)
    monkeypatch.setattr(observability, "datetime", Clock)
    monkeypatch.setattr(ToolSuccess, "model_dump", dump)
    monkeypatch.setattr(runtime_module, "LLMMessage", message)
    run(make_runtime(llm=SimpleNamespace(complete=complete), handler=handler, observer=observer))
    assert [event.attributes["elapsed_seconds"] for event in observer.of_type("llm.completed")] == [2.0, 2.0]
    assert observer.of_type("tool.completed")[0].attributes["elapsed_seconds"] == 15.0
    assert observer.of_type("run.finished")[0].attributes["elapsed_seconds"] == 19.0
    assert observer.events[-1].occurred_at < observer.events[0].occurred_at


def test_json_logs_and_diagnostics_exclude_all_raw_content(caplog):
    async def handler(args, context):
        raise ToolExecutionError("TEMP_UNAVAILABLE", "private-tool-error")

    with caplog.at_level(logging.INFO):
        run(make_runtime(tools(tool_call()), LLMResponse("private-final-answer", "stop")))
        run(make_runtime(tools(tool_call()), LLMResponse("private-final-answer", "stop"), handler=handler))
        with pytest.raises(RuntimeError):
            run(make_runtime(RuntimeError("private-llm-error")))
        with pytest.raises(RuntimeError):
            run(make_runtime(tools(tool_call()), handler=AsyncMock(side_effect=RuntimeError("private-tool-error"))))
        run(make_runtime(LLMResponse("private-answer", "stop"), observer=BrokenObserver()))
    assert "private-" not in str([record.__dict__ for record in caplog.records])
    events = [json.loads(record.message) for record in caplog.records if record.message.startswith("{")]
    assert {event["event_type"] for event in events} >= {"run.finished", "tool.completed", "llm.exception"}
    assert all(datetime.fromisoformat(event["occurred_at"]).utcoffset() == timedelta(0) for event in events)
    diagnostics = [record for record in caplog.records if record.message == "agent_runtime.observer_failure"]
    assert diagnostics and all(record.error_type == "OSError" and record.run_id and record.event_type for record in diagnostics)
    assert all(not record.exc_info for record in caplog.records)


def test_logging_observer_handler_failure_is_isolated(monkeypatch):
    monkeypatch.setattr(observability.logger, "info", Mock(side_effect=OSError("private-log-error")))
    assert run(make_runtime(LLMResponse("done", "stop"))).status is AgentRunStatus.COMPLETED


@pytest.mark.parametrize("original_failure", [None, "llm", "tool"])
def test_shared_log_handler_failure_does_not_override_business_result(caplog, original_failure):
    class BrokenHandler(logging.Handler):
        def emit(self, record):
            raise OSError("private-shared-log-handler")

    error = RuntimeError("private-original-error")
    handler = BrokenHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            if original_failure is None:
                result = run(make_runtime(tools(tool_call()), LLMResponse("done", "stop")))
                assert result.status is AgentRunStatus.COMPLETED
            else:
                runtime = make_runtime(error) if original_failure == "llm" else make_runtime(
                    tools(tool_call()), handler=AsyncMock(side_effect=error),
                )
                with pytest.raises(RuntimeError) as raised:
                    run(runtime)
                assert raised.value is error
    finally:
        root.removeHandler(handler)


def test_bootstrap_observer_injection_default_and_noop():
    observer = InMemoryObserver()
    llm = SimpleNamespace(complete=AsyncMock(return_value=LLMResponse("done", "stop")))
    runtime = build_agent_runtime(llm=llm, registry=ToolRegistry(), observer=observer)
    assert runtime._observer is observer
    assert run(runtime).status is AgentRunStatus.COMPLETED
    assert observer.of_type("run.finished")
    default = build_agent_runtime(llm=llm, registry=ToolRegistry())
    assert isinstance(default._observer, LoggingObserver)
    noop = NoOpObserver()
    disabled = build_agent_runtime(llm=llm, registry=ToolRegistry(), observer=noop)
    assert disabled._observer is noop
    assert run(disabled).status is AgentRunStatus.COMPLETED
