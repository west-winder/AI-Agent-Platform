import asyncio
import json

import pytest
from pydantic import ValidationError

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import (
    AgentRunResult,
    AgentRunStatus,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
)
from backend.evaluation.contracts import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationExecutionResult,
    EvaluationRunConfig,
    EvidenceAvailability,
    ExceptionInfo,
    ExecutionEvidence,
    ExecutionStatus,
    ToolCallEvidence,
    ToolFixture,
)
from backend.tools.infrastructure.contracts import ToolContext, ToolFailure, ToolSuccess
from backend.tools.infrastructure.registry import ToolRegistry


@pytest.fixture
def case_data():
    return {
        "case_id": "weather-1",
        "input_messages": [
            {"role": "system", "content": "Use the supplied weather data."},
            {"role": "user", "content": "武汉天气如何？"},
        ],
        "tool_fixtures": [
            {
                "tool_name": "get_weather",
                "arguments": {"city": "武汉"},
                "result": {
                    "status": "error",
                    "error": {"code": "TEMPORARY", "message": "Please retry", "retryable": True},
                },
            },
            {
                "tool_name": "get_weather",
                "arguments": {"city": "武汉"},
                "result": {"status": "success", "data": {"temperature_c": 21}},
            },
        ],
        "criteria": [
            {"criterion_id": "status", "evaluator_type": "RULE", "criterion_type": "run_status",
             "expected": "COMPLETED"},
            {"criterion_id": "grounding", "evaluator_type": "JUDGE", "criterion_type": "grounding",
             "expected": None, "rubric": "Check claims against the fixture.", "required": False},
        ],
        "reference_facts": ["武汉 21 °C"],
        "metadata": {"category": "weather", "version": 1},
    }


def test_case_and_dataset_round_trip_preserve_fixture_order(case_data, tmp_path):
    case = EvaluationCase.model_validate(case_data)
    assert case.input_messages[-1] == LLMMessage(role="user", content="武汉天气如何？")
    assert [type(fixture.result) for fixture in case.tool_fixtures] == [ToolFailure, ToolSuccess]
    assert EvaluationCase.model_validate_json(case.model_dump_json()) == case
    dataset = EvaluationDataset(cases=[case])
    path = tmp_path / "dataset.json"
    path.write_text(dataset.model_dump_json(), encoding="utf-8")
    assert EvaluationDataset.model_validate_json(path.read_bytes()) == dataset


def test_case_ids_are_unique_in_dataset(case_data):
    with pytest.raises(ValidationError, match="Case IDs must be unique"):
        EvaluationDataset(cases=[case_data, case_data])


@pytest.mark.parametrize("version", [2, True, 1.0])
def test_dataset_rejects_unsupported_or_non_integer_schema_version(case_data, version):
    with pytest.raises(ValidationError, match="schema_version"):
        EvaluationDataset.model_validate_json(json.dumps({"schema_version": version, "cases": [case_data]}))


@pytest.mark.parametrize("invalid_id", ["", "  ", "\t"])
def test_case_id_must_be_non_blank(case_data, invalid_id):
    with pytest.raises(ValidationError, match="case_id"):
        EvaluationCase.model_validate(case_data | {"case_id": invalid_id})


def test_criterion_ids_are_unique_after_trimming(case_data):
    case_data["criteria"][1]["criterion_id"] = " status "
    with pytest.raises(ValidationError, match="Criterion IDs must be unique"):
        EvaluationCase.model_validate(case_data)


@pytest.mark.parametrize("empty", [True, False])
def test_case_requires_at_least_one_required_criterion(case_data, empty):
    case_data["criteria"] = [] if empty else case_data["criteria"][1:]
    with pytest.raises(ValidationError, match="at least one required"):
        EvaluationCase.model_validate(case_data)


@pytest.mark.parametrize("updates", [
    {"criterion_type": "package.callable"},
    {"criterion_type": "lambda: True"},
    {"expected": lambda: True},
    {"expected": {"non_finite": float("inf")}},
    {"expression": "x + 1"},
    {"required": "yes"},
    {"rubric": "Unused by rule"},
    {"evaluator_type": "JUDGE"},
])
def test_criterion_contract_rejects_invalid_or_executable_fields(case_data, updates):
    case_data["criteria"][0].update(updates)
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(case_data)


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": "done"}]])
def test_case_requires_a_runtime_compatible_new_turn(case_data, messages):
    with pytest.raises(ValidationError, match="input_messages"):
        EvaluationCase.model_validate(case_data | {"input_messages": messages})


@pytest.mark.parametrize("result", [
    {"status": "unsupported", "data": {}},
    {"status": "success", "data": []},
    {"status": "success", "data": {"bad": object()}},
    {"status": "success", "data": {"bad": float("nan")}},
    {"status": "success", "data": {}, "handler": "some.module"},
    {"status": "success", "data": {}, "error": {"code": "X", "message": "Bad"}},
    {"status": "error", "data": {}, "error": {"code": "X", "message": "Bad"}},
    {"status": "error"},
    {"status": "error", "error": {"code": "", "message": "Bad"}},
    {"status": "error", "error": {"code": "X", "message": "Bad", "retryable": "true"}},
    {"status": "error", "error": {"code": "X", "message": "Bad", "extra": True}},
])
def test_invalid_fixture_results_rejected(result):
    with pytest.raises(ValidationError):
        ToolFixture(tool_name="get_weather", arguments={}, result=result)


@pytest.mark.parametrize("arguments", [[], {"city": object()}, {"city": float("nan")}])
def test_invalid_fixture_arguments_rejected(arguments):
    with pytest.raises(ValidationError, match="arguments"):
        ToolFixture(tool_name="get_weather", arguments=arguments, result=ToolSuccess(data={}))


@pytest.mark.parametrize("fields", [
    {"mode": "DETERMINISTIC", "fake_llm_script_ref": "weather-v1"},
    {"mode": "MODEL", "model_ref": "test-model", "judge_ref": "test-judge"},
    {"mode": "MODEL", "model_ref": "test-model", "max_iterations": 1},
])
def test_run_config_round_trip(fields):
    config = EvaluationRunConfig.model_validate(fields)
    assert EvaluationRunConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize("fields", [
    {"mode": "UNSUPPORTED"},
    {"mode": "DETERMINISTIC"},
    {"mode": "DETERMINISTIC", "fake_llm_script_ref": "script", "model_ref": "test-model"},
    {"mode": "DETERMINISTIC", "fake_llm_script_ref": "script", "judge_ref": "test-judge"},
    {"mode": "MODEL"},
    {"mode": "MODEL", "model_ref": " ", "judge_ref": "test-judge"},
    {"mode": "MODEL", "model_ref": "test-model", "fake_llm_script_ref": "script"},
    {"mode": "MODEL", "model_ref": "https://example.invalid/profile"},
    {"mode": "MODEL", "model_ref": "test-model", "api_key": "secret-marker"},
    {"mode": "MODEL", "model_ref": {"name": "model", "api_key": "secret-marker"}},
    *[
        {"mode": "MODEL", "model_ref": "test-model", "max_iterations": value}
        for value in [True, 0, -1, 1.5, "2"]
    ],
])
def test_invalid_run_config_rejected(fields):
    with pytest.raises(ValidationError) as exc:
        EvaluationRunConfig.model_validate(fields)
    assert "secret-marker" not in str(exc.value)


@pytest.mark.parametrize("status", list(AgentRunStatus))
def test_execution_result_wraps_existing_runtime_result(status):
    agent_result = AgentRunResult(
        status=status, final_answer="done" if status is AgentRunStatus.COMPLETED else None,
        iterations=1, messages=[LLMMessage(role="user", content="hello")],
    )
    execution = EvaluationExecutionResult(
        case_id="case", execution_status="RETURNED", agent_result=agent_result, duration_ms=1,
    )
    assert execution.agent_result is agent_result
    assert execution.execution_status is ExecutionStatus.RETURNED
    assert EvaluationExecutionResult.model_validate_json(execution.model_dump_json()) == execution


@pytest.mark.parametrize(("status", "has_result", "has_exception"), [
    ("RETURNED", False, False),
    ("RETURNED", True, True),
    ("RAISED", False, False),
    ("RAISED", True, True),
    ("RAISED", True, False),
])
def test_execution_status_consistency(status, has_result, has_exception):
    with pytest.raises(ValidationError):
        EvaluationExecutionResult(
            case_id="case", execution_status=status, duration_ms=0,
            agent_result=AgentRunResult(AgentRunStatus.COMPLETED, "done", 1, []) if has_result else None,
            exception_info=ExceptionInfo(exception_type="TimeoutError") if has_exception else None,
        )


@pytest.mark.parametrize("duration", [True, -1, float("nan"), float("inf")])
def test_invalid_execution_duration(duration):
    with pytest.raises(ValidationError, match="duration_ms"):
        EvaluationExecutionResult(
            case_id="case", execution_status="RAISED", duration_ms=duration,
            exception_info=ExceptionInfo(exception_type="TimeoutError"),
        )


@pytest.mark.parametrize("exception_info", [
    RuntimeError("secret-marker"),
    {"exception_type": "RuntimeError", "traceback": "secret-marker"},
    {"exception_type": "RuntimeError", "message": "secret-marker"},
    {"exception_type": "RuntimeError: secret-marker"},
    {"exception_type": "module.RuntimeError"},
])
def test_exception_summary_rejects_raw_exceptions_and_sensitive_details(exception_info):
    with pytest.raises(ValidationError) as exc:
        EvaluationExecutionResult(
            case_id="case", execution_status="RAISED", duration_ms=0, exception_info=exception_info,
        )
    assert "secret-marker" not in str(exc.value)


def test_tool_evidence_distinguishes_absent_partial_unknown_and_skipped():
    call = LLMToolCall(id="call-1", name="echo", raw_arguments="{}", arguments={})
    incomplete_call = ToolCallEvidence(call=call)
    no_calls = ExecutionEvidence(tool_call_availability="COMPLETE")
    partial = ExecutionEvidence(tool_call_availability="PARTIAL", tool_calls=[incomplete_call])
    partial_without_records = ExecutionEvidence(tool_call_availability="PARTIAL")
    unknown = ExecutionEvidence()
    skipped = ExecutionEvidence(
        run_status="MAX_ITERATIONS_REACHED", iterations=1,
        tool_call_availability="COMPLETE", tool_calls=[ToolCallEvidence(call=call, skipped=True)],
    )
    observed = ExecutionEvidence(
        tool_call_availability="COMPLETE",
        tool_calls=[ToolCallEvidence(call=call, tool_result=ToolSuccess(data={"text": "hello"}))],
    )
    assert no_calls.tool_calls == unknown.tool_calls == partial_without_records.tool_calls == []
    assert unknown.tool_call_availability is EvidenceAvailability.UNAVAILABLE
    assert partial.tool_calls[0].tool_result is None
    assert skipped.tool_calls[0].skipped
    for evidence in (no_calls, partial, partial_without_records, unknown, skipped, observed):
        assert ExecutionEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    with pytest.raises(ValidationError, match="UNAVAILABLE"):
        ExecutionEvidence(tool_calls=[incomplete_call])
    with pytest.raises(ValidationError, match="COMPLETE"):
        ExecutionEvidence(tool_call_availability="COMPLETE", tool_calls=[incomplete_call])
    with pytest.raises(ValidationError, match="skipped call"):
        ToolCallEvidence(call=call, skipped=True, tool_result=ToolSuccess(data={}))


def test_dataset_input_works_with_unmodified_runtime(case_data):
    class FakeLLM:
        async def complete(self, request):
            assert request.messages[-1].content == "武汉天气如何？"
            return LLMResponse(content="武汉 21 °C", finish_reason="stop")

    case = EvaluationCase.model_validate_json(json.dumps(case_data))
    runtime = build_agent_runtime(llm=FakeLLM(), registry=ToolRegistry())
    result = asyncio.run(runtime.run(case.input_messages, ToolContext()))
    assert isinstance(result, AgentRunResult)
    assert result.status is AgentRunStatus.COMPLETED
    assert result.messages[:-1] == case.input_messages
