"""F-01：合法字符串子类兼容，以及事件构造故障的旁路隔离。"""

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel
import pytest

from backend.agent_runtime import observability, runtime as runtime_module
from backend.agent_runtime.contracts import (
    AgentRunStatus,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    validate_tool_calls,
)
from backend.agent_runtime.observability import (
    LoggingObserver,
    NoOpObserver,
    ObservationEvent,
    safe_emit_factory,
)
from backend.agent_runtime.runtime import AgentRuntime
from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.registry import ToolRegistry


class ToolName(StrEnum):
    ECHO = "echo"


class CallId(StrEnum):
    ORIGINAL = "original-call-id"


class WireText(str):
    def __str__(self):
        raise AssertionError("Normalization must not invoke custom __str__")


class Number(IntEnum):
    ONE = 1


class Integer(int):
    pass


class Float(float):
    pass


class Stringifiable:
    def __str__(self):
        raise AssertionError("Arbitrary objects must not be stringified")


class Args(BaseModel):
    text: str


def make_runtime(observer, responses, handler=None):
    registry = ToolRegistry()
    actual_handler = (
        handler if handler is not None else AsyncMock(return_value={"text": "done"})
    )
    registry.register(RegisteredTool("echo", "echo", Args, actual_handler))
    dispatcher = ToolDispatcher(registry)
    dispatcher.dispatch = AsyncMock(wraps=dispatcher.dispatch)
    llm = SimpleNamespace(complete=AsyncMock(side_effect=responses))
    return (
        AgentRuntime(llm, registry, dispatcher, observer=observer),
        dispatcher,
        actual_handler,
    )


def run(runtime):
    return asyncio.run(
        runtime.run([LLMMessage("user", "private-prompt")], ToolContext())
    )


@pytest.mark.parametrize(
    "name,call_id",
    [
        ("echo", "original-call-id"),
        (ToolName.ECHO, CallId.ORIGINAL),
        (WireText("echo"), WireText("original-call-id")),
    ],
)
@pytest.mark.parametrize("observer_type", [NoOpObserver, LoggingObserver])
def test_string_subclasses_complete_real_runtime_and_dispatcher(
    name, call_id, observer_type
):
    observer = observer_type()
    observer.emit = Mock(wraps=observer.emit)
    call = LLMToolCall(call_id, name, '{"text":"value"}', {"text": "value"})
    validate_tool_calls([call])
    runtime, dispatcher, handler = make_runtime(
        observer,
        [
            LLMResponse(None, "tool_calls", [call]),
            LLMResponse("done", "stop"),
        ],
    )
    result = run(runtime)
    assert result.status is AgentRunStatus.COMPLETED and result.final_answer == "done"
    dispatcher.dispatch.assert_awaited_once()
    handler.assert_awaited_once()
    assert dispatcher.dispatch.await_args.kwargs["tool_name"] == name
    assert handler.await_args.args[0] == Args(text="value")
    (tool_message,) = [message for message in result.messages if message.role == "tool"]
    assert tool_message.tool_call_id == "original-call-id"
    # 仅正规化事件元数据，不替换模型提供的调用 ID 或业务 History 中的类型。
    assert type(tool_message.tool_call_id) is type(call_id)
    tool_events = [
        args.args[0]
        for args in observer.emit.call_args_list
        if args.args[0].event_type.startswith("tool.")
    ]
    assert [event.event_type for event in tool_events] == [
        "tool.started",
        "tool.completed",
    ]
    for event in tool_events:
        assert type(event.attributes["tool_name"]) is str
        assert type(event.attributes["tool_call_id"]) is str
        assert event.attributes["tool_name"] == "echo"
        assert event.attributes["tool_call_id"] == "original-call-id"
    if isinstance(observer, LoggingObserver):
        snapshot = observer.snapshot()
        assert snapshot.agent_runs_terminal_total["completed"] == 1
        assert snapshot.agent_runs_active == 0
        assert snapshot.agent_tool_calls_total["success"] == 1


def test_normalization_keeps_metadata_copy_readonly_and_other_scalar_types():
    attributes = {
        "tool_name": ToolName.ECHO,
        "tool_call_id": WireText("original-call-id"),
        "attempt": 1,
        "will_retry": True,
        "elapsed_seconds": 1.25,
        "status_code": None,
    }
    event = ObservationEvent("run", "tool.started", datetime.now(UTC), attributes)
    assert type(event.attributes["tool_name"]) is str
    assert type(event.attributes["tool_call_id"]) is str
    assert type(attributes["tool_name"]) is ToolName
    assert type(attributes["tool_call_id"]) is WireText
    for key in ("attempt", "will_retry", "elapsed_seconds", "status_code"):
        assert type(event.attributes[key]) is type(attributes[key])
    attributes["tool_name"] = "changed"
    assert event.attributes["tool_name"] == "echo"
    with pytest.raises(TypeError):
        event.attributes["tool_name"] = "changed"
    with pytest.raises(FrozenInstanceError):
        event.event_type = "changed"


@pytest.mark.parametrize(
    "attributes",
    [
        {"tool_name": Stringifiable()},
        {"tool_call_id": object()},
        {"status": {"nested": "private"}},
        {"status": ["private"]},
        {"prompt": "private"},
        {"elapsed_seconds": float("nan")},
        {"elapsed_seconds": float("inf")},
        {"elapsed_seconds": float("-inf")},
        {"attempt": Number.ONE},
        {"attempt": Integer(1)},
        {"elapsed_seconds": Float(1.0)},
    ],
)
def test_non_string_objects_and_existing_invalid_metadata_remain_rejected(attributes):
    with pytest.raises((TypeError, ValueError)):
        ObservationEvent("run", "tool.started", datetime.now(UTC), attributes)


def test_invalid_event_is_diagnosed_without_reaching_observer(caplog):
    observer = SimpleNamespace(emit=Mock())
    with caplog.at_level(logging.WARNING, logger=observability.__name__):
        safe_emit_factory(
            observer,
            lambda: ObservationEvent(
                "run", "tool.started", datetime.now(UTC), {"prompt": "private-prompt"}
            ),
            run_id="run",
            event_type="tool.started",
        )
    observer.emit.assert_not_called()
    (diagnostic,) = caplog.records
    assert (diagnostic.run_id, diagnostic.event_type, diagnostic.error_type) == (
        "run",
        "tool.started",
        "ValueError",
    )
    assert "private-" not in str(diagnostic.__dict__)


@pytest.mark.parametrize("phase", ["success", "llm_error", "tool_error"])
def test_constructor_failure_never_changes_business_result_or_exception(
    monkeypatch, caplog, phase
):
    observer = SimpleNamespace(emit=Mock())
    error = RuntimeError("private-original-error")
    call = LLMToolCall("id", "echo", '{"text":"value"}', {"text": "value"})
    responses = (
        [error]
        if phase == "llm_error"
        else [LLMResponse(None, "tool_calls", [call]), LLMResponse("done", "stop")]
    )
    handler = AsyncMock(side_effect=error) if phase == "tool_error" else None
    runtime, dispatcher, _ = make_runtime(observer, responses, handler)
    monkeypatch.setattr(
        runtime_module,
        "ObservationEvent",
        Mock(side_effect=OSError("private-constructor-error")),
    )
    with caplog.at_level(logging.WARNING):
        if phase == "success":
            assert run(runtime).status is AgentRunStatus.COMPLETED
        else:
            with pytest.raises(RuntimeError) as raised:
                run(runtime)
            assert raised.value is error
    observer.emit.assert_not_called()
    assert dispatcher.dispatch.await_count == (0 if phase == "llm_error" else 1)
    assert any(
        record.message == "agent_runtime.observer_failure" for record in caplog.records
    )
    assert "private-" not in str([record.__dict__ for record in caplog.records])


def test_constructor_and_backup_diagnostic_failures_are_both_isolated(monkeypatch):
    runtime, _, _ = make_runtime(NoOpObserver(), [LLMResponse("done", "stop")])
    monkeypatch.setattr(
        runtime_module,
        "ObservationEvent",
        Mock(side_effect=OSError("private-constructor")),
    )
    monkeypatch.setattr(
        observability.logger, "warning", Mock(side_effect=OSError("private-diagnostic"))
    )
    assert run(runtime).status is AgentRunStatus.COMPLETED


@pytest.mark.parametrize("phase", ["llm", "tool"])
def test_constructor_failure_does_not_replace_active_task_cancellation(
    monkeypatch, phase
):
    async def scenario():
        entered = asyncio.Event()

        async def block(*args):
            entered.set()
            await asyncio.Event().wait()

        observer = SimpleNamespace(emit=Mock())
        call = LLMToolCall("id", "echo", '{"text":"value"}', {"text": "value"})
        runtime, _, _ = make_runtime(
            observer,
            [LLMResponse(None, "tool_calls", [call])],
            block if phase == "tool" else None,
        )
        if phase == "llm":
            runtime._llm.complete = block
        monkeypatch.setattr(
            runtime_module,
            "ObservationEvent",
            Mock(side_effect=OSError("private-constructor")),
        )
        monkeypatch.setattr(
            observability.logger,
            "warning",
            Mock(side_effect=OSError("private-diagnostic")),
        )
        task = asyncio.create_task(
            runtime.run([LLMMessage("user", "private")], ToolContext())
        )
        await entered.wait()
        task.cancel("private-cancel")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("private-cancel",)
        observer.emit.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_constructor_base_exceptions_still_propagate(monkeypatch, error_type):
    runtime, _, _ = make_runtime(NoOpObserver(), [LLMResponse("done", "stop")])
    error = error_type("private-interrupt")
    monkeypatch.setattr(runtime_module, "ObservationEvent", Mock(side_effect=error))
    with pytest.raises(error_type) as raised:
        run(runtime)
    assert raised.value is error
