"""Metrics V1 的统计口径、固定桶、快照隔离和真实 Runtime 集成。"""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
from openai import APIStatusError
import pytest
from pydantic import BaseModel

from backend.agent_runtime import observability
from backend.agent_runtime import runtime as runtime_module
from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import (
    AgentRunStatus,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
)
from backend.agent_runtime.metrics import (
    RUN_DURATION_BUCKETS,
    LLM_DURATION_BUCKETS,
    TOOL_DURATION_BUCKETS,
)
from backend.agent_runtime.observability import (
    LoggingObserver,
    NoOpObserver,
    ObservationEvent,
    get_default_observer,
)
from backend.agent_runtime.runtime import AgentRuntime
from backend.exceptions.external_exceptions import (
    ExternalServiceError,
    ExternalTimeoutError,
)
from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.errors import ToolExecutionError
from backend.tools.infrastructure.registry import ToolRegistry


def emit(observer, kind, run_id="run", **attributes):
    observer.emit(ObservationEvent(run_id, kind, datetime.now(UTC), attributes))


def call(call_id="raw-tool-id", name="echo"):
    return LLMToolCall(
        call_id, name, '{"text":"private-arguments"}', {"text": "private-arguments"}
    )


def tool_response(*calls):
    return LLMResponse("private-response", "tool_calls", list(calls))


def make_runtime(observer, *responses, handler=None, llm=None, max_iterations=5):
    class Args(BaseModel):
        text: str

    async def echo(args, context):
        return {"text": "private-result"}

    registry = ToolRegistry()
    registry.register(RegisteredTool("echo", "echo", Args, handler or echo))
    if llm is None:
        llm = SimpleNamespace(complete=AsyncMock(side_effect=list(responses)))
    return AgentRuntime(
        llm, registry, ToolDispatcher(registry), max_iterations, observer=observer
    )


def run(runtime):
    return asyncio.run(
        runtime.run([LLMMessage("user", "private-prompt")], ToolContext())
    )


def rate_limit_failure():
    error = ExternalServiceError("private-upstream-message")
    error.__cause__ = APIStatusError(
        "private-cause",
        response=httpx2.Response(
            429,
            headers={"retry-after": "30"},
            request=httpx2.Request("POST", "https://example.invalid"),
        ),
        body=None,
    )
    return error


@pytest.mark.parametrize(
    "outcome", ["completed", "max_iterations_reached", "exception", "cancelled"]
)
def test_run_gauge_and_terminal_outcomes_in_real_runtime(outcome):
    async def scenario():
        observer = LoggingObserver()
        entered, release = asyncio.Event(), asyncio.Event()
        error = RuntimeError("private-runtime-exception")

        async def complete(request):
            entered.set()
            await release.wait()
            if outcome == "exception":
                raise error
            if outcome == "max_iterations_reached":
                return tool_response(call())
            return LLMResponse("done", "stop")

        runtime = make_runtime(
            observer, llm=SimpleNamespace(complete=complete), max_iterations=1
        )
        task = asyncio.create_task(
            runtime.run([LLMMessage("user", "private")], ToolContext())
        )
        await entered.wait()
        started = observer.snapshot()
        assert started.agent_runs_started_total == started.agent_runs_active == 1
        assert sum(started.agent_runs_terminal_total.values()) == 0
        if outcome == "cancelled":
            task.cancel("private-cancel")
            with pytest.raises(asyncio.CancelledError) as raised:
                await task
            assert raised.value.args == ("private-cancel",)
        else:
            release.set()
            if outcome == "exception":
                with pytest.raises(RuntimeError) as raised:
                    await task
                assert raised.value is error
            else:
                result = await task
                assert result.status.value.lower() == outcome
        snapshot = observer.snapshot()
        assert snapshot.agent_runs_active == 0
        assert snapshot.agent_runs_terminal_total[outcome] == 1
        assert sum(snapshot.agent_runs_terminal_total.values()) == 1
        assert snapshot.agent_run_duration_seconds[outcome].count == 1

    asyncio.run(scenario())


def test_repeated_start_and_terminal_do_not_duplicate_run_accounting():
    observer = LoggingObserver()
    emit(observer, "run.started")
    emit(observer, "run.started")
    assert observer.snapshot().agent_runs_active == 1
    emit(observer, "run.finished", status="COMPLETED", elapsed_seconds=2.5)
    emit(observer, "run.finished", status="COMPLETED", elapsed_seconds=2.5)
    emit(observer, "run.exception", error_type="RuntimeError", elapsed_seconds=3.0)
    emit(observer, "run.cancelled", elapsed_seconds=3.0)
    emit(
        observer,
        "run.finished",
        run_id="orphan",
        status="COMPLETED",
        elapsed_seconds=1.0,
    )
    snapshot = observer.snapshot()
    assert snapshot.agent_runs_started_total == 1
    assert snapshot.agent_runs_active == 0
    assert snapshot.agent_runs_terminal_total == {
        "completed": 1,
        "max_iterations_reached": 0,
        "exception": 0,
        "cancelled": 0,
    }
    assert snapshot.agent_run_duration_seconds["completed"].sum == 2.5
    assert not observer._metrics._active_run_ids


@pytest.mark.parametrize(
    "retries,final_failure", [(False, False), (True, False), (True, True)]
)
def test_logical_calls_actual_attempts_and_retries_are_distinct(retries, final_failure):
    observer = LoggingObserver()
    error = ValueError("private-final-error")
    responses = [ExternalTimeoutError("private-timeout")] if retries else []
    responses.append(error if final_failure else LLMResponse("done", "stop"))
    runtime = make_runtime(observer, *responses)
    if final_failure:
        with pytest.raises(ValueError) as raised:
            run(runtime)
        assert raised.value is error
    else:
        run(runtime)
    snapshot = observer.snapshot()
    assert snapshot.agent_llm_calls_total == 1
    assert snapshot.agent_llm_attempts_total == (2 if retries else 1)
    assert snapshot.agent_llm_retries_total == int(retries)
    assert snapshot.agent_llm_call_duration_seconds.count == 1


def test_attempt_count_is_already_visible_inside_each_actual_request():
    observer = LoggingObserver()
    seen = []

    async def complete(request):
        snapshot = observer.snapshot()
        seen.append(
            (
                snapshot.agent_llm_calls_total,
                snapshot.agent_llm_attempts_total,
                snapshot.agent_llm_retries_total,
            )
        )
        if len(seen) == 1:
            raise ExternalTimeoutError("private-first-error")
        return LLMResponse("done", "stop")

    run(make_runtime(observer, llm=SimpleNamespace(complete=complete)))
    assert seen == [(1, 1, 0), (1, 2, 1)]


def test_cancellation_during_retry_wait_does_not_invent_attempt_or_retry(monkeypatch):
    async def scenario():
        observer = LoggingObserver()
        waiting = asyncio.Event()

        async def sleep(delay):
            assert delay == 30
            waiting.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(runtime_module.asyncio, "sleep", sleep)
        runtime = make_runtime(observer, rate_limit_failure())
        task = asyncio.create_task(
            runtime.run([LLMMessage("user", "private")], ToolContext())
        )
        await waiting.wait()
        before = observer.snapshot()
        assert (
            before.agent_llm_calls_total,
            before.agent_llm_attempts_total,
            before.agent_llm_retries_total,
        ) == (1, 1, 0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        after = observer.snapshot()
        assert (after.agent_llm_attempts_total, after.agent_llm_retries_total) == (1, 0)
        assert after.agent_llm_call_duration_seconds.count == 1
        assert after.agent_runs_active == 0
        assert after.agent_runs_terminal_total["cancelled"] == 1

    asyncio.run(scenario())


def test_llm_logical_duration_includes_retry_wait_using_monotonic_clock(monkeypatch):
    observer = LoggingObserver()
    clock = [100.0]
    requests = 0

    async def complete(request):
        nonlocal requests
        requests += 1
        clock[0] += 2 if requests == 1 else 3
        if requests == 1:
            raise rate_limit_failure()
        return LLMResponse("done", "stop")

    async def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(runtime_module, "monotonic", lambda: clock[0])
    monkeypatch.setattr(observability, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime_module.asyncio, "sleep", sleep)
    run(make_runtime(observer, llm=SimpleNamespace(complete=complete)))
    snapshot = observer.snapshot()
    assert snapshot.agent_llm_call_duration_seconds.sum == 35.0
    assert snapshot.agent_run_duration_seconds["completed"].sum == 35.0
    assert (
        snapshot.agent_llm_attempts_total == 2 and snapshot.agent_llm_retries_total == 1
    )


def test_request_copy_failure_does_not_count_an_actual_request(monkeypatch):
    observer = LoggingObserver()
    error = TypeError("private-copy-error")
    llm = SimpleNamespace(complete=AsyncMock())
    runtime = make_runtime(observer, llm=llm)

    def broken_copy(value):
        raise error

    monkeypatch.setattr(runtime_module, "deepcopy", broken_copy)

    async def scenario():
        with pytest.raises(TypeError) as raised:
            await runtime._complete(
                SimpleNamespace(), 1, run_id="run", llm_call_id="llm"
            )
        assert raised.value is error

    asyncio.run(scenario())
    snapshot = observer.snapshot()
    assert snapshot.agent_llm_calls_total == 1
    assert snapshot.agent_llm_attempts_total == snapshot.agent_llm_retries_total == 0
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize("outcome", ["success", "failure", "exception", "cancelled"])
def test_tool_outcomes_and_full_duration_from_runtime(monkeypatch, outcome):
    async def scenario():
        observer = LoggingObserver()
        clock = [100.0]
        entered = asyncio.Event()
        error = RuntimeError("private-handler-error")

        async def handler(args, context):
            clock[0] += 1.25
            if outcome == "failure":
                raise ToolExecutionError("TEMP_UNAVAILABLE", "private-tool-failure")
            if outcome == "exception":
                raise error
            if outcome == "cancelled":
                entered.set()
                await asyncio.Event().wait()
            return {"private": "result"}

        monkeypatch.setattr(runtime_module, "monotonic", lambda: clock[0])
        monkeypatch.setattr(observability, "monotonic", lambda: clock[0])
        runtime = make_runtime(
            observer,
            tool_response(call()),
            LLMResponse("done", "stop"),
            handler=handler,
        )
        task = asyncio.create_task(
            runtime.run([LLMMessage("user", "private")], ToolContext())
        )
        if outcome == "cancelled":
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome == "exception":
            with pytest.raises(RuntimeError) as raised:
                await task
            assert raised.value is error
        else:
            assert (await task).status is AgentRunStatus.COMPLETED
        snapshot = observer.snapshot()
        assert snapshot.agent_tool_calls_total[outcome] == 1
        assert sum(snapshot.agent_tool_calls_total.values()) == 1
        histogram = snapshot.agent_tool_duration_seconds[outcome]
        assert histogram.count == 1 and histogram.sum == 1.25
        if outcome in ("success", "failure"):
            assert snapshot.agent_runs_terminal_total["completed"] == 1
            assert snapshot.agent_runs_terminal_total["exception"] == 0
            assert snapshot.agent_llm_calls_total == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "metric,bounds,kind,attributes",
    [
        (
            "agent_run_duration_seconds",
            RUN_DURATION_BUCKETS,
            "run.finished",
            {"status": "COMPLETED"},
        ),
        ("agent_llm_call_duration_seconds", LLM_DURATION_BUCKETS, "llm.completed", {}),
        (
            "agent_tool_duration_seconds",
            TOOL_DURATION_BUCKETS,
            "tool.completed",
            {"status": "success"},
        ),
    ],
)
def test_histogram_count_sum_inclusive_cumulative_buckets_and_infinity(
    metric, bounds, kind, attributes
):
    observer = LoggingObserver()
    values = [0.0, bounds[0], (bounds[0] + bounds[1]) / 2, bounds[-1], bounds[-1] + 10]
    for index, value in enumerate(values):
        if kind == "run.finished":
            emit(observer, "run.started", run_id=str(index))
        emit(observer, kind, run_id=str(index), elapsed_seconds=value, **attributes)
    histogram = getattr(observer.snapshot(), metric)
    if kind != "llm.completed":
        histogram = histogram["completed" if kind == "run.finished" else "success"]
    assert histogram.count == len(values)
    assert histogram.sum == sum(values)
    assert list(histogram.buckets) == [*(str(bound) for bound in bounds), "+Inf"]
    assert list(histogram.buckets.values()) == [
        *(sum(value <= bound for value in values) for bound in bounds),
        len(values),
    ]
    assert histogram.buckets[str(bounds[0])] == 2
    assert histogram.buckets[str(bounds[-1])] == 4
    assert histogram.buckets["+Inf"] == 5
    assert histogram.sum / histogram.count == sum(values) / len(values)


def test_concurrent_runs_share_observer_without_cross_contamination():
    async def scenario():
        observer = LoggingObserver()
        entered = 0
        ready, release = asyncio.Event(), asyncio.Event()
        error = RuntimeError("private-error")

        async def complete(request):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await release.wait()
            if request.messages[0].content == "fails":
                raise error
            return LLMResponse("done", "stop")

        runtime = make_runtime(observer, llm=SimpleNamespace(complete=complete))
        tasks = [
            asyncio.create_task(
                runtime.run([LLMMessage("user", content)], ToolContext())
            )
            for content in ("ok", "fails")
        ]
        await ready.wait()
        active = observer.snapshot()
        assert active.agent_runs_started_total == active.agent_runs_active == 2
        assert active.agent_llm_attempts_total == 2
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert results[0].status is AgentRunStatus.COMPLETED and results[1] is error
        snapshot = observer.snapshot()
        assert snapshot.agent_runs_active == 0
        assert (
            snapshot.agent_runs_terminal_total["completed"]
            == snapshot.agent_runs_terminal_total["exception"]
            == 1
        )
        assert snapshot.agent_llm_calls_total == snapshot.agent_llm_attempts_total == 2
        assert snapshot.agent_llm_retries_total == 0

    asyncio.run(scenario())


def test_default_observer_and_metrics_are_shared_across_runtime_instances():
    observer = get_default_observer()
    before = observer.snapshot()
    llm = SimpleNamespace(complete=AsyncMock(return_value=LLMResponse("done", "stop")))
    runtimes = [build_agent_runtime(llm=llm, registry=ToolRegistry()) for _ in range(2)]
    runtimes.append(AgentRuntime(llm, ToolRegistry(), ToolDispatcher(ToolRegistry())))
    assert all(runtime._observer is observer for runtime in runtimes)
    for runtime in runtimes:
        run(runtime)
    after = observer.snapshot()
    assert after.agent_runs_started_total - before.agent_runs_started_total == 3
    assert (
        after.agent_runs_terminal_total["completed"]
        - before.agent_runs_terminal_total["completed"]
        == 3
    )
    assert after.agent_runs_active == before.agent_runs_active
    # 显式 NoOp 不把事件送入共享指标；显式 Observer 可隔离测试状态。
    run(make_runtime(NoOpObserver(), LLMResponse("done", "stop")))
    isolated = LoggingObserver()
    run(make_runtime(isolated, LLMResponse("done", "stop")))
    assert isolated.snapshot().agent_runs_started_total == 1
    assert observer.snapshot() == after


def test_metric_labels_are_finite_and_exclude_ids_names_and_error_codes():
    observer = LoggingObserver()
    for index in range(1000):
        run_id = f"private-run-{index}"
        emit(observer, "run.started", run_id=run_id)
        emit(observer, "llm.started", run_id=run_id, llm_call_id=f"private-llm-{index}")
        emit(
            observer,
            "tool.completed",
            run_id=run_id,
            tool_call_id=f"private-call-{index}",
            parent_llm_call_id=f"private-llm-{index}",
            tool_name=f"private-unknown-tool-{index}",
            status="error",
            error_code=f"private-error-{index}",
            elapsed_seconds=0.5,
        )
        emit(
            observer,
            "run.finished",
            run_id=run_id,
            status="COMPLETED",
            elapsed_seconds=1.0,
        )
    snapshot = observer.snapshot()
    assert set(snapshot.agent_tool_calls_total) == {
        "success",
        "failure",
        "exception",
        "cancelled",
    }
    assert snapshot.agent_tool_calls_total["failure"] == 1000
    assert (
        len(snapshot.agent_run_duration_seconds)
        == len(snapshot.agent_tool_duration_seconds)
        == 4
    )
    assert not observer._metrics._active_run_ids
    projection = json.dumps(snapshot.to_dict(), allow_nan=False)
    assert "private-" not in projection
    assert not any(
        name in projection
        for name in (
            "run_id",
            "llm_call_id",
            "tool_call_id",
            "user_id",
            "conversation_id",
            "tool_name",
            "error_code",
        )
    )


def test_unknown_tool_is_failure_while_run_can_complete():
    observer = LoggingObserver()
    run(
        make_runtime(
            observer, tool_response(call(name="unknown")), LLMResponse("done", "stop")
        )
    )
    snapshot = observer.snapshot()
    assert snapshot.agent_tool_calls_total["failure"] == 1
    assert snapshot.agent_runs_terminal_total["completed"] == 1
    assert snapshot.agent_runs_terminal_total["exception"] == 0


def test_snapshot_is_detached_deeply_readonly_and_json_export_is_independent():
    observer = LoggingObserver()
    emit(observer, "run.started")
    emit(observer, "run.finished", status="COMPLETED", elapsed_seconds=2.0)
    snapshot = observer.snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.agent_runs_started_total = 9
    with pytest.raises(TypeError):
        snapshot.agent_runs_terminal_total["completed"] = 9
    with pytest.raises(TypeError):
        snapshot.agent_run_duration_seconds["completed"] = None
    with pytest.raises(TypeError):
        snapshot.agent_tool_calls_total["success"] = 9
    with pytest.raises(TypeError):
        snapshot.agent_tool_duration_seconds["success"].buckets["+Inf"] = 9
    with pytest.raises(TypeError):
        snapshot.agent_llm_call_duration_seconds.buckets["+Inf"] = 9
    with pytest.raises(FrozenInstanceError):
        snapshot.agent_run_duration_seconds["completed"].count = 9
    exported = snapshot.to_dict()
    assert set(exported) == {"counters", "gauges", "histograms"}
    assert sum(len(group) for group in exported.values()) == 10
    exported["counters"]["agent_runs_terminal_total"]["completed"] = 99
    exported["histograms"]["agent_run_duration_seconds"]["completed"]["buckets"][
        "+Inf"
    ] = 99
    emit(observer, "run.started", run_id="next")
    emit(
        observer, "run.finished", run_id="next", status="COMPLETED", elapsed_seconds=3.0
    )
    assert (
        snapshot.agent_runs_started_total
        == snapshot.agent_runs_terminal_total["completed"]
        == 1
    )
    assert snapshot.agent_run_duration_seconds["completed"].sum == 2.0
    current = observer.snapshot()
    assert current.agent_runs_terminal_total["completed"] == 2
    assert current.agent_run_duration_seconds["completed"].buckets["+Inf"] == 2


def test_logging_and_backup_diagnostic_failure_do_not_block_metrics(monkeypatch):
    observer = LoggingObserver()
    monkeypatch.setattr(
        observability.logger, "info", Mock(side_effect=OSError("private-log-error"))
    )
    monkeypatch.setattr(
        observability.logger,
        "warning",
        Mock(side_effect=OSError("private-diagnostic-error")),
    )
    assert (
        run(make_runtime(observer, LLMResponse("done", "stop"))).status
        is AgentRunStatus.COMPLETED
    )
    snapshot = observer.snapshot()
    assert (
        snapshot.agent_runs_started_total
        == snapshot.agent_runs_terminal_total["completed"]
        == 1
    )
    assert snapshot.agent_llm_calls_total == snapshot.agent_llm_attempts_total == 1
    assert snapshot.agent_runs_active == 0


@pytest.mark.parametrize("original_failure", [False, True])
def test_metrics_failure_does_not_block_logging_or_override_runtime(
    caplog, monkeypatch, original_failure
):
    observer = LoggingObserver()
    monkeypatch.setattr(
        observer._metrics, "record", Mock(side_effect=OSError("private-metrics-error"))
    )
    error = RuntimeError("private-runtime-error")
    runtime = make_runtime(
        observer, error if original_failure else LLMResponse("done", "stop")
    )
    with caplog.at_level(logging.INFO, logger="backend.agent_runtime.observability"):
        if original_failure:
            with pytest.raises(RuntimeError) as raised:
                run(runtime)
            assert raised.value is error
        else:
            assert run(runtime).status is AgentRunStatus.COMPLETED
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.message.startswith("{")
    ]
    kinds = [event["event_type"] for event in events]
    assert "run.started" in kinds and "llm.attempt_started" in kinds
    assert ("run.exception" if original_failure else "run.finished") in kinds
    assert any(
        record.message == "agent_runtime.observer_failure" for record in caplog.records
    )
    assert "private-" not in str([record.__dict__ for record in caplog.records])


@pytest.mark.parametrize("branch", ["logging", "metrics"])
def test_base_exception_in_either_branch_still_propagates(monkeypatch, branch):
    observer = LoggingObserver()
    error = asyncio.CancelledError("private-cancel")
    if branch == "logging":
        monkeypatch.setattr(observer, "_write_log", Mock(side_effect=error))
    else:
        monkeypatch.setattr(observer._metrics, "record", Mock(side_effect=error))
    with pytest.raises(asyncio.CancelledError) as raised:
        emit(observer, "run.started")
    assert raised.value is error


def test_unknown_events_and_outcomes_do_not_create_metric_series():
    observer = LoggingObserver()
    before = observer.snapshot()
    emit(observer, "custom.event")
    emit(observer, "run.finished", status="unknown", elapsed_seconds=1.0)
    emit(observer, "tool.completed", status="unknown", elapsed_seconds=1.0)
    emit(observer, "tool.started", tool_name="unknown")
    assert observer.snapshot() == before


def test_invalid_metric_duration_does_not_partially_terminate_run(caplog):
    observer = LoggingObserver()
    emit(observer, "run.started")
    with caplog.at_level(logging.WARNING):
        emit(observer, "run.finished", status="COMPLETED", elapsed_seconds=-1.0)
    assert observer.snapshot().agent_runs_active == 1
    assert observer.snapshot().agent_runs_terminal_total["completed"] == 0
    emit(observer, "run.finished", status="COMPLETED", elapsed_seconds=1.0)
    assert observer.snapshot().agent_runs_active == 0
    assert observer.snapshot().agent_runs_terminal_total["completed"] == 1
