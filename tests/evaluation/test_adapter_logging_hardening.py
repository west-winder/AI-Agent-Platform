"""F-02：Mock SDK → 真 Adapter → 真 Runtime → 真 Runner 的日志故障回归。"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
from openai import APIConnectionError
import pytest

from backend.agent_runtime.deepseek_adapter import DeepSeekAdapter
from backend.agent_runtime.observability import get_default_observer
from backend.evaluation.contracts import EvaluationDataset, EvaluationRunConfig
from backend.evaluation.runner import EvaluationRunner, ExecutionFailurePhase
from backend.exceptions.external_exceptions import ExternalServiceError


class BrokenHandler(logging.Handler):
    def emit(self, record):
        raise OSError("private-log-sink")


@pytest.fixture(params=[False, True], ids=["healthy-log", "broken-log"])
def log_sink(request, caplog):
    caplog.set_level(logging.INFO)
    root = logging.getLogger()
    sink = BrokenHandler() if request.param else logging.NullHandler()
    root.addHandler(sink)
    try:
        yield
    finally:
        root.removeHandler(sink)


def dataset(*case_ids):
    return EvaluationDataset.model_validate(
        {
            "purpose": "AGENT_QUALITY",
            "cases": [
                {
                    "case_id": case_id,
                    "input_messages": [{"role": "user", "content": "private-prompt"}],
                    "criteria": [
                        {
                            "criterion_id": "quality",
                            "evaluator_type": "JUDGE",
                            "criterion_type": "grounding",
                            "expected": None,
                            "rubric": "Check the supplied answer.",
                        }
                    ],
                }
                for case_id in case_ids
            ],
        }
    )


def make_adapter():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant", content="done", tool_calls=[]
                ),
            )
        ]
    )
    create = AsyncMock(return_value=response)
    translator = Mock(return_value=ExternalServiceError("private-translated"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    adapter = DeepSeekAdapter(
        client, model="mock", overall_timeout_seconds=1, translate_error=translator
    )
    return adapter, create, translator


@pytest.mark.parametrize("upstream_failure", [False, True])
def test_runner_records_original_execution_outcome_under_shared_logging_failure(
    log_sink, caplog, upstream_failure
):
    adapter, create, translator = make_adapter()
    upstream = APIConnectionError(
        request=httpx2.Request("POST", "https://example.invalid")
    )
    if upstream_failure:
        create.side_effect = upstream
    observer = get_default_observer()
    before = observer.snapshot()
    config = EvaluationRunConfig(mode="MODEL", model_ref="mock")
    runner = EvaluationRunner(model_factories={"mock": lambda: adapter})
    (record,) = asyncio.run(runner.run(dataset("case"), config))
    execution = record.execution_result
    after = observer.snapshot()
    assert create.await_count == 1
    assert after.agent_runs_started_total - before.agent_runs_started_total == 1
    assert after.agent_runs_active == before.agent_runs_active
    assert after.agent_llm_attempts_total - before.agent_llm_attempts_total == 1
    assert after.agent_llm_retries_total == before.agent_llm_retries_total
    if upstream_failure:
        assert execution.execution_status == "RAISED"
        assert record.failure_phase is ExecutionFailurePhase.RUNTIME
        assert execution.exception_info.exception_type == "ExternalServiceError"
        translator.assert_called_once_with(upstream)
        assert translator.return_value.__cause__ is upstream
        assert (
            after.agent_runs_terminal_total["exception"]
            - before.agent_runs_terminal_total["exception"]
            == 1
        )
    else:
        assert execution.execution_status == "RETURNED"
        assert execution.agent_result.final_answer == "done"
        assert execution.exception_info is None
        assert record.failure_phase is None
        translator.assert_not_called()
        assert (
            after.agent_runs_terminal_total["completed"]
            - before.agent_runs_terminal_total["completed"]
            == 1
        )
    assert "private-" not in str([log.__dict__ for log in caplog.records])


def test_runner_cancellation_cleans_up_active_metric_and_does_not_start_next_case(
    log_sink,
):
    async def scenario():
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        adapter, create, translator = make_adapter()
        create.side_effect = blocked
        factory = Mock(return_value=adapter)
        runner = EvaluationRunner(model_factories={"mock": factory})
        observer = get_default_observer()
        before = observer.snapshot()
        task = asyncio.create_task(
            runner.run(
                dataset("first", "second"),
                EvaluationRunConfig(mode="MODEL", model_ref="mock"),
            )
        )
        await entered.wait()
        assert observer.snapshot().agent_runs_active == before.agent_runs_active + 1
        task.cancel("private-cancel")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("private-cancel",)
        after = observer.snapshot()
        assert after.agent_runs_active == before.agent_runs_active
        assert (
            after.agent_runs_terminal_total["cancelled"]
            - before.agent_runs_terminal_total["cancelled"]
            == 1
        )
        factory.assert_called_once()
        create.assert_awaited_once()
        translator.assert_not_called()

    asyncio.run(scenario())
