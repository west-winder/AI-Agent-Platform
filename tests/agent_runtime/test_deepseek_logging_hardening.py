"""F-02：真实 Adapter 与 Runtime 在故障日志 Handler 下保持原业务语义。"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
from openai import APIConnectionError, APIStatusError
import pytest

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import LLMMessage, LLMRequest
from backend.agent_runtime.deepseek_adapter import DeepSeekAdapter
from backend.agent_runtime.observability import LoggingObserver
from backend.exceptions.external_exceptions import (
    ExternalServiceError,
    ExternalTimeoutError,
)
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.registry import ToolRegistry


class BrokenHandler(logging.Handler):
    def __init__(self, broken):
        super().__init__()
        self.broken = broken
        self.records = []

    def emit(self, record):
        self.records.append(record)
        if self.broken:
            raise OSError("private-handler-failure")


@pytest.fixture(params=[False, True], ids=["healthy-log", "broken-log"])
def log_sink(request, caplog):
    caplog.set_level(logging.INFO)
    root = logging.getLogger()
    sink = BrokenHandler(request.param)
    root.addHandler(sink)
    try:
        yield sink
    finally:
        root.removeHandler(sink)


def sdk_response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant", content="done", tool_calls=[]
                ),
            )
        ]
    )


def connection_error():
    error = APIConnectionError(
        request=httpx2.Request("POST", "https://example.invalid")
    )
    error.__cause__ = ConnectionResetError("private-network-cause")
    return error


def make_adapter(*responses, translated=None, timeout=1.0):
    create = AsyncMock(side_effect=list(responses))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    translator = Mock(
        return_value=translated
        if translated is not None
        else ExternalServiceError("private-translated")
    )
    adapter = DeepSeekAdapter(
        client,
        model="mock",
        overall_timeout_seconds=timeout,
        translate_error=translator,
    )
    return adapter, create, translator


def request():
    return LLMRequest([LLMMessage("user", "private-prompt")])


def test_successful_sdk_response_survives_shared_log_sink_failure(log_sink, caplog):
    adapter, create, translator = make_adapter(sdk_response())
    response = asyncio.run(adapter.complete(request()))
    assert response.content == "done" and response.finish_reason == "stop"
    create.assert_awaited_once()
    translator.assert_not_called()
    assert any(
        record.getMessage() == "agent_llm.response" for record in log_sink.records
    )
    assert "private-" not in str(
        [record.__dict__ for record in [*caplog.records, *log_sink.records]]
    )


def test_api_error_preserves_translated_exception_identity_and_cause_chain(
    log_sink, caplog
):
    upstream = connection_error()
    original_cause = upstream.__cause__
    translated = ExternalServiceError("private-translated")
    adapter, create, translator = make_adapter(upstream, translated=translated)
    with pytest.raises(ExternalServiceError) as raised:
        asyncio.run(adapter.complete(request()))
    assert raised.value is translated
    assert raised.value.__cause__ is upstream
    assert upstream.__cause__ is original_cause
    translator.assert_called_once_with(upstream)
    create.assert_awaited_once()
    (diagnostic,) = [
        record
        for record in log_sink.records
        if record.getMessage() == "agent_llm.external_error"
    ]
    assert diagnostic.error_type == "APIConnectionError"
    assert not diagnostic.exc_info
    assert "private-" not in str(
        [record.__dict__ for record in [*caplog.records, *log_sink.records]]
    )


def test_timeout_translation_is_unchanged_under_shared_log_failure(log_sink):
    upstream = TimeoutError("private-timeout")
    translated = ExternalTimeoutError("private-translated-timeout")
    adapter, create, translator = make_adapter(upstream, translated=translated)
    with pytest.raises(ExternalTimeoutError) as raised:
        asyncio.run(adapter.complete(request()))
    assert raised.value is translated and raised.value.__cause__ is upstream
    translator.assert_called_once_with(upstream)
    create.assert_awaited_once()


def test_actual_adapter_budget_expiry_keeps_timeout_and_cleans_up_sdk_call(log_sink):
    async def scenario():
        cleaned = []

        async def blocked(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.append(True)

        translated = ExternalTimeoutError("private-timeout")
        adapter, create, translator = make_adapter(translated=translated, timeout=0.01)
        create.side_effect = blocked
        with pytest.raises(ExternalTimeoutError) as raised:
            await adapter.complete(request())
        assert raised.value is translated and isinstance(
            raised.value.__cause__, TimeoutError
        )
        translator.assert_called_once_with(raised.value.__cause__)
        assert cleaned == [True]
        create.assert_awaited_once()

    asyncio.run(scenario())


def test_active_sdk_cancellation_is_not_translated_or_swallowed(log_sink):
    async def scenario():
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        adapter, create, translator = make_adapter()
        create.side_effect = blocked
        task = asyncio.create_task(adapter.complete(request()))
        await entered.wait()
        task.cancel("private-cancel")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("private-cancel",)
        translator.assert_not_called()
        create.assert_awaited_once()

    asyncio.run(scenario())


def test_runtime_retry_policy_and_metrics_survive_faulty_shared_handler(log_sink):
    upstream = APIStatusError(
        "private-503",
        response=httpx2.Response(
            503,
            request=httpx2.Request("POST", "https://example.invalid"),
        ),
        body=None,
    )
    translated = ExternalServiceError("private-translated")
    adapter, create, translator = make_adapter(
        upstream, sdk_response(), translated=translated
    )
    observer = LoggingObserver()
    runtime = build_agent_runtime(
        llm=adapter, registry=ToolRegistry(), observer=observer
    )
    result = asyncio.run(runtime.run(request().messages, ToolContext()))
    assert result.final_answer == "done" and result.iterations == 1
    translator.assert_called_once_with(upstream)
    assert translated.__cause__ is upstream
    assert create.await_count == 2
    snapshot = observer.snapshot()
    assert (
        snapshot.agent_llm_calls_total,
        snapshot.agent_llm_attempts_total,
        snapshot.agent_llm_retries_total,
    ) == (1, 2, 1)
    assert snapshot.agent_runs_terminal_total["completed"] == 1
    assert snapshot.agent_runs_active == 0


def test_unexpected_sdk_business_exception_still_propagates_unchanged(log_sink):
    error = ValueError("private-sdk-program-error")
    adapter, create, translator = make_adapter(error)
    with pytest.raises(ValueError) as raised:
        asyncio.run(adapter.complete(request()))
    assert raised.value is error
    translator.assert_not_called()
    create.assert_awaited_once()


@pytest.mark.parametrize(
    "error_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("phase", ["response", "external_error"])
def test_diagnostic_handler_base_exception_is_not_swallowed(caplog, error_type, phase):
    caplog.set_level(logging.INFO)
    error = error_type("private-interrupt")

    class InterruptingHandler(logging.Handler):
        def emit(self, record):
            raise error

    sink = InterruptingHandler()
    root = logging.getLogger()
    root.addHandler(sink)
    try:
        adapter, create, translator = make_adapter(
            sdk_response() if phase == "response" else connection_error()
        )
        with pytest.raises(error_type) as raised:
            asyncio.run(adapter.complete(request()))
        assert raised.value is error
        create.assert_awaited_once()
        translator.assert_not_called()
    finally:
        root.removeHandler(sink)
