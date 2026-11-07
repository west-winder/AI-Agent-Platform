import asyncio
import json

import pytest
from pydantic import BaseModel, JsonValue, ValidationError

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
    ExecutionStatus,
    ToolFixture,
)
from backend.evaluation.evidence import collect_evidence
from backend.evaluation.fixtures import FixtureViolation, FixtureViolationCode
from backend.evaluation.runner import EvaluationRunner, ExecutionFailurePhase
from backend.tools.infrastructure.contracts import (
    RegisteredTool,
    ToolErrorInfo,
    ToolFailure,
    ToolSuccess,
)
from backend.tools.infrastructure.registry import ToolRegistry


class EchoArgs(BaseModel):
    payload: JsonValue
    optional: str = "default"


@pytest.fixture
def registry():
    async def real_handler(arguments, context):
        pytest.fail("The production handler was called")

    source = ToolRegistry()
    source.register(RegisteredTool("echo", "Echo a payload", EchoArgs, real_handler))
    return source


def case(case_id="example", fixtures=(), messages=None, *, quality=False):
    criteria = [
        {"criterion_id": "status", "evaluator_type": "RULE",
         "criterion_type": "run_status", "expected": "COMPLETED"},
    ]
    if quality:
        criteria.append({
            "criterion_id": "meaning", "evaluator_type": "JUDGE", "criterion_type": "grounding",
            "expected": None, "rubric": "Check the answer against the task.",
        })
    return EvaluationCase(
        case_id=case_id,
        input_messages=messages or [LLMMessage("user", case_id)],
        tool_fixtures=list(fixtures),
        criteria=criteria,
    )


def fixture(payload, result=None, tool_name="echo"):
    return ToolFixture(
        tool_name=tool_name,
        arguments={"payload": payload},
        result=ToolSuccess(data={"echo": payload}) if result is None else result,
    )


def call(payload, call_id="call", tool_name="echo"):
    arguments = {"payload": payload}
    return LLMToolCall(call_id, tool_name, json.dumps(arguments), arguments)


def calls(*tool_calls):
    return LLMResponse(None, "tool_calls", list(tool_calls))


def answer(text="done"):
    return LLMResponse(text, "stop")


def run_script(registry, cases, scripts, max_iterations=5):
    # Exercise the same boundary as a dataset loaded from disk.
    dataset = EvaluationDataset(cases=cases)
    loaded = EvaluationDataset.model_validate_json(dataset.model_dump_json())
    config = EvaluationRunConfig(
        mode="DETERMINISTIC", fake_llm_script_ref="script-v1", max_iterations=max_iterations,
    )
    runner = EvaluationRunner(tool_registry=registry, scripts={"script-v1": scripts})
    return asyncio.run(runner.run(loaded, config))


def test_deterministic_runs_sequential_fixture_outcomes_and_preserves_batch_scope(registry):
    failure = ToolFailure(error=ToolErrorInfo(code="TEMPORARY", message="Retry", retryable=True))
    prior_call = call("prior")
    prior = [
        LLMMessage("user", "older turn"),
        LLMMessage("assistant", tool_calls=[prior_call]),
        LLMMessage("tool", ToolSuccess(data={"old": True}).model_dump_json(), tool_call_id="call"),
        LLMMessage("assistant", "old answer"),
        LLMMessage("user", "current turn"),
    ]
    first = {"first": [1, False], "second": "text"}
    reversed_keys = {"second": "text", "first": [1, False]}
    sample = case(fixtures=[fixture(first, failure), fixture("again"), fixture("batch")], messages=prior)
    [result] = run_script(
        registry,
        [sample],
        {"example": [calls(call(reversed_keys)), calls(call("again"), call("batch", "other")), answer()]},
    )
    assert result.execution_result.execution_status is ExecutionStatus.RETURNED
    assert result.execution_result.duration_ms >= 0
    assert result.failure_phase is None
    assert result.fixture_violations == result.unconsumed_fixture_indices == ()
    evidence = result.evidence
    assert evidence.final_answer == "done"
    assert evidence.run_status is AgentRunStatus.COMPLETED
    assert evidence.iterations == 3
    assert evidence.tool_call_availability is EvidenceAvailability.COMPLETE
    assert [record.call.id for record in evidence.tool_calls] == ["call", "call", "other"]
    assert evidence.tool_calls[0].call.arguments == {"payload": reversed_keys}
    assert [record.tool_result for record in evidence.tool_calls] == [
        failure, ToolSuccess(data={"echo": "again"}), ToolSuccess(data={"echo": "batch"}),
    ]


def test_fixture_violation_is_recoverable_and_does_not_consume_or_disappear(registry):
    sample = case(fixtures=[fixture(1), fixture([1, 2])])
    script = [calls(call(True)), calls(call(1)), calls(call([2, 1])), answer("recovered")]
    [result] = run_script(registry, [sample], {"example": script})
    assert result.evidence.final_answer == "recovered"
    assert result.fixture_violations == (
        FixtureViolation("echo", FixtureViolationCode.ARGUMENT_MISMATCH, 0),
        FixtureViolation("echo", FixtureViolationCode.ARGUMENT_MISMATCH, 1),
    )
    assert result.unconsumed_fixture_indices == (1,)
    mismatch, match, wrong_order = [record.tool_result for record in result.evidence.tool_calls]
    assert mismatch.error.code == wrong_order.error.code == "FIXTURE_ARGUMENT_MISMATCH"
    assert match == ToolSuccess(data={"echo": 1})


def test_exhaustion_is_reported_and_fixture_state_is_fresh_per_case(registry):
    cases = [case(case_id, [fixture("x")]) for case_id in ("first", "second")]
    sequence = [calls(call("x")), calls(call("x")), answer()]
    results = run_script(registry, cases, {"first": sequence, "second": sequence})
    assert [result.execution_result.case_id for result in results] == ["first", "second"]
    for result in results:
        matched, exhausted = [record.tool_result for record in result.evidence.tool_calls]
        assert matched == ToolSuccess(data={"echo": "x"})
        assert exhausted.error.code == "FIXTURE_EXHAUSTED"
        assert result.fixture_violations == (FixtureViolation("echo", FixtureViolationCode.EXHAUSTED),)
        assert result.unconsumed_fixture_indices == ()


def test_model_injection_preserves_definitions_and_runs_serially(registry):
    events = []

    class StubLLM:
        async def complete(self, request):
            [definition] = request.tools
            source = registry.get("echo")
            assert (definition.name, definition.description, definition.parameters) == (
                source.name, source.description, source.args_schema.model_json_schema(),
            )
            messages = request.messages
            case_id = messages[0].content
            if messages[-1].role == "user":
                events.append(f"start:{case_id}")
                await asyncio.sleep(0)
                return calls(call(case_id))
            events.append(f"finish:{case_id}")
            assert json.loads(messages[-1].content)["data"] == {"echo": case_id}
            return answer(case_id)

    dataset = EvaluationDataset(
        purpose="AGENT_QUALITY", cases=[case(case_id, [fixture(case_id)], quality=True) for case_id in ("a", "b")],
    )
    config = EvaluationRunConfig(mode="MODEL", model_ref="stub", judge_ref="unused-judge")
    runner = EvaluationRunner(tool_registry=registry, model_factories={"stub": StubLLM})
    results = asyncio.run(runner.run(dataset, config))
    assert events == ["start:a", "finish:a", "start:b", "finish:b"]
    assert [result.evidence.final_answer for result in results] == ["a", "b"]
    assert all(not result.fixture_violations for result in results)


def test_case_assembly_failure_isolated_and_does_not_lose_unconsumed_fixture(registry):
    samples = [case("unknown-tool", [fixture("x", tool_name="missing")]), case("ok")]
    failed, successful = run_script(registry, samples, {"ok": [answer()]})
    assert failed.failure_phase is ExecutionFailurePhase.ASSEMBLY
    assert failed.execution_result.execution_status is ExecutionStatus.RAISED
    assert failed.execution_result.exception_info.exception_type == "ToolNotFoundError"
    assert failed.unconsumed_fixture_indices == (0,)
    assert failed.evidence.tool_call_availability is EvidenceAvailability.UNAVAILABLE
    assert successful.execution_result.execution_status is ExecutionStatus.RETURNED


def test_model_factory_error_is_summarized_and_does_not_fall_back(registry):
    factories = iter([None, StubAnswer()])
    runner = EvaluationRunner(tool_registry=registry, model_factories={"test": lambda: next(factories)})
    dataset = EvaluationDataset(purpose="AGENT_QUALITY", cases=[case("bad-factory", quality=True), case("ok", quality=True)])
    failed, successful = asyncio.run(runner.run(dataset, EvaluationRunConfig(mode="MODEL", model_ref="test")))
    assert failed.failure_phase is ExecutionFailurePhase.ASSEMBLY
    assert failed.execution_result.exception_info.exception_type == "TypeError"
    assert successful.evidence.final_answer == "ok"


class StubAnswer:
    async def complete(self, request):
        return answer("ok")


def test_runtime_exception_without_history_keeps_only_safe_summary_and_continues(registry):
    class RaisingLLM:
        async def complete(self, request):
            raise RuntimeError("sensitive runtime exception details")

    factories = iter([RaisingLLM(), StubAnswer()])
    runner = EvaluationRunner(tool_registry=registry, model_factories={"test": lambda: next(factories)})
    config = EvaluationRunConfig(mode="MODEL", model_ref="test")
    dataset = EvaluationDataset(purpose="AGENT_QUALITY", cases=[case("raised", quality=True), case("ok", quality=True)])
    failed, successful = asyncio.run(runner.run(dataset, config))
    assert failed.failure_phase is ExecutionFailurePhase.RUNTIME
    assert failed.execution_result.execution_status is ExecutionStatus.RAISED
    assert failed.execution_result.agent_result is None
    assert failed.execution_result.exception_info.model_dump() == {"exception_type": "RuntimeError"}
    assert "sensitive" not in failed.execution_result.model_dump_json()
    assert failed.evidence.tool_call_availability is EvidenceAvailability.UNAVAILABLE
    assert failed.evidence.tool_calls == []
    assert successful.evidence.tool_call_availability is EvidenceAvailability.COMPLETE
    assert successful.evidence.tool_calls == []


def test_runtime_raise_after_tool_activity_keeps_violation_without_fabricating_calls(registry):
    samples = [case("raised", [fixture("expected")]), case("ok")]
    scripts = {"raised": [calls(call("wrong"))], "ok": [answer()]}
    failed, successful = run_script(registry, samples, scripts)
    assert failed.failure_phase is ExecutionFailurePhase.RUNTIME
    assert failed.execution_result.exception_info.exception_type == "FakeLLMScriptExhausted"
    assert failed.evidence.tool_call_availability is EvidenceAvailability.PARTIAL
    assert failed.evidence.run_status is failed.evidence.iterations is None
    assert failed.evidence.tool_calls == []
    assert failed.fixture_violations == (FixtureViolation("echo", FixtureViolationCode.ARGUMENT_MISMATCH, 0),)
    assert failed.unconsumed_fixture_indices == (0,)
    assert successful.evidence.final_answer == "done"


@pytest.mark.parametrize("limit", [1, 2])
def test_iteration_limit_omits_unexecuted_final_batch_and_marks_evidence_partial(registry, limit):
    sample = case(fixtures=[fixture("first"), fixture("second")])
    script = {"example": [calls(call("first", "id-1")), calls(call("second", "id-2"))]}
    [result] = run_script(registry, [sample], script, max_iterations=limit)
    assert result.evidence.run_status is AgentRunStatus.MAX_ITERATIONS_REACHED
    assert result.evidence.iterations == limit
    assert result.evidence.tool_call_availability is EvidenceAvailability.PARTIAL
    assert [record.call.id for record in result.evidence.tool_calls] == ["id-1"][:limit - 1]
    assert result.unconsumed_fixture_indices == (0, 1)[limit - 1:]
    assert result.fixture_violations == ()


def test_dispatcher_failures_are_collected_without_inventing_fixture_violations(registry):
    sample = case(fixtures=[fixture("valid")])
    invalid = LLMToolCall("parse", "echo", "{", parse_error="invalid JSON")
    missing = LLMToolCall("schema", "echo", "{}", {})
    unknown = call("unknown", "unknown", "other-tool")
    [result] = run_script(
        registry, [sample], {"example": [calls(invalid, missing, unknown), answer()]},
    )
    assert result.evidence.tool_call_availability is EvidenceAvailability.COMPLETE
    assert [record.tool_result.error.code for record in result.evidence.tool_calls] == [
        "INVALID_ARGUMENTS", "INVALID_ARGUMENTS", "TOOL_NOT_FOUND",
    ]
    assert result.evidence.tool_calls[0].call.arguments is None
    assert result.fixture_violations == ()
    assert result.unconsumed_fixture_indices == (0,)


@pytest.mark.parametrize("tool_content", ["{", "null", '{"status":"error"}'])
def test_collector_keeps_call_and_marks_unreadable_result_partial(tool_content):
    execution = EvaluationExecutionResult(
        case_id="example", execution_status="RETURNED", duration_ms=0,
        agent_result=AgentRunResult(
            AgentRunStatus.COMPLETED, "done", 2,
            [
                LLMMessage("user", "example"),
                LLMMessage("assistant", tool_calls=[call("x")]),
                LLMMessage("tool", tool_content, tool_call_id="call"),
                LLMMessage("assistant", "done"),
            ],
        ),
    )
    evidence = collect_evidence(execution, input_message_count=1)
    assert evidence.tool_call_availability is EvidenceAvailability.PARTIAL
    [record] = evidence.tool_calls
    assert record.call.id == "call"
    assert record.tool_result is None


def test_external_task_cancellation_propagates_without_starting_next_case(registry):
    async def scenario():
        started = asyncio.Event()
        events = []

        class WaitingLLM:
            async def complete(self, request):
                events.append(request.messages[-1].content)
                started.set()
                await asyncio.Event().wait()

        runner = EvaluationRunner(tool_registry=registry, model_factories={"wait": WaitingLLM})
        config = EvaluationRunConfig(mode="MODEL", model_ref="wait")
        dataset = EvaluationDataset(purpose="AGENT_QUALITY", cases=[case("first", quality=True), case("second", quality=True)])
        task = asyncio.create_task(runner.run(dataset, config))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["first"]

    asyncio.run(scenario())


def test_invalid_dataset_and_config_are_rejected_before_any_case(registry):
    def no_factory():
        pytest.fail("Run validation must happen before assembly")

    runner = EvaluationRunner(tool_registry=registry, model_factories={"stub": no_factory})
    dataset = EvaluationDataset(purpose="AGENT_QUALITY", cases=[case(quality=True)])
    config = EvaluationRunConfig(mode="MODEL", model_ref="stub")
    invalid = config.model_copy(update={"max_iterations": 0})
    with pytest.raises(ValidationError, match="max_iterations"):
        asyncio.run(runner.run(dataset, invalid))
    with pytest.raises(ValueError, match="Unknown model profile"):
        asyncio.run(runner.run(dataset, EvaluationRunConfig(mode="MODEL", model_ref="missing")))
    dataset.cases.clear()
    with pytest.raises(ValidationError, match="cases"):
        asyncio.run(runner.run(dataset, config))
