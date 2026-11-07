import asyncio
from dataclasses import replace
from unittest.mock import patch

import pytest

from backend.agent_runtime.contracts import AgentRunResult, AgentRunStatus, LLMMessage, LLMResponse, LLMToolCall
from backend.evaluation.contracts import (
    CaseVerdict,
    CriterionDecision,
    EvaluationCase,
    EvaluationCriterion,
    EvaluationDataset,
    EvaluationExecutionResult,
    EvaluationResult,
    EvaluationRunConfig,
    ExceptionInfo,
    ExecutionEvidence,
    ToolCallEvidence,
)
from backend.evaluation.engine import EvaluationEngine
from backend.evaluation.fixtures import FixtureViolation, FixtureViolationCode
from backend.evaluation.judge_port import JudgeVerdict
from backend.evaluation.rule_evaluator import RuleEvaluator
from backend.evaluation.runner import CaseExecution, EvaluationRunner, ExecutionFailurePhase
from backend.tools.infrastructure.contracts import ToolFailure, ToolSuccess


DETERMINISTIC = EvaluationRunConfig(mode="DETERMINISTIC", fake_llm_script_ref="script")
MODEL = EvaluationRunConfig(mode="MODEL", model_ref="agent", judge_ref="judge")


def rule(kind, expected, *, required=True, criterion_id=None):
    return EvaluationCriterion(
        criterion_id=criterion_id or kind, evaluator_type="RULE",
        criterion_type=kind, expected=expected, required=required,
    )


def semantic(criterion_id="grounding", *, required=True):
    return EvaluationCriterion(
        criterion_id=criterion_id, evaluator_type="JUDGE", criterion_type="grounding",
        expected=None, rubric="Check the answer against the facts.", required=required,
    )


def sample(*criteria, case_id="case"):
    return EvaluationCase(
        case_id=case_id, input_messages=[LLMMessage("system", "private history"), LLMMessage("user", "Task")],
        criteria=list(criteria), reference_facts=["Reference"],
    )


def evaluation_dataset(config, *cases):
    purpose = "AGENT_QUALITY" if config.mode == "MODEL" else "HARNESS_REGRESSION"
    return EvaluationDataset(cases=list(cases), purpose=purpose)


async def evaluate_records(engine, dataset, config, records):
    """Inject synthetic evidence at the case boundary, keeping real batch issuance."""
    by_id = {record.execution_result.case_id: record for record in records}
    runner = EvaluationRunner(scripts={"script": {}}, model_factories={"agent": lambda: None})

    async def case_record(case, *args):
        return by_id[case.case_id]

    with patch.object(runner, "_run_case", case_record):
        batch = await runner.run(dataset, config)
    return await engine.evaluate(dataset, config, batch)


def returned(case_id="case", *, availability="COMPLETE", records=(), status=AgentRunStatus.COMPLETED):
    answer = "Answer" if status is AgentRunStatus.COMPLETED else None
    result = AgentRunResult(status, answer, 1, [LLMMessage("assistant", "private history")])
    return CaseExecution(
        EvaluationExecutionResult(
            case_id=case_id, execution_status="RETURNED", agent_result=result, duration_ms=0,
        ),
        ExecutionEvidence(
            final_answer=answer, run_status=status, iterations=1,
            tool_call_availability=availability, tool_calls=list(records),
        ),
        None, (), (),
    )


def raised(phase=ExecutionFailurePhase.RUNTIME):
    return CaseExecution(
        EvaluationExecutionResult(
            case_id="case", execution_status="RAISED",
            exception_info=ExceptionInfo(exception_type="TimeoutError"), duration_ms=0,
        ),
        ExecutionEvidence(), phase, (), (),
    )


def observed(result=None, *, name="weather", skipped=False):
    return ToolCallEvidence(
        call=LLMToolCall("call", name, "private raw text", {"city": "武汉", "options": [1, True]}),
        tool_result=result, skipped=skipped,
    )


@pytest.mark.parametrize(("kind", "expected", "decision"), [
    ("execution_status", "RETURNED", "PASS"),
    ("execution_status", "RAISED", "FAIL"),
    ("run_status", "COMPLETED", "PASS"),
    ("run_status", "MAX_ITERATIONS_REACHED", "FAIL"),
    ("tool_called", "weather", "PASS"),
    ("tool_called", "other", "FAIL"),
    ("tool_not_called", "weather", "FAIL"),
    ("tool_not_called", "other", "PASS"),
    ("tool_arguments_match", {"tool_name": "weather", "arguments": {"options": [1, True], "city": "武汉"}}, "PASS"),
    ("tool_arguments_match", {"tool_name": "weather", "arguments": {"options": [True, True], "city": "武汉"}}, "FAIL"),
    ("tool_arguments_match", {"tool_name": "weather", "arguments": {"options": [True, 1], "city": "武汉"}}, "FAIL"),
    ("tool_arguments_match", {"tool_name": "weather", "arguments": {"options": [1.0, True], "city": "武汉"}}, "FAIL"),
    ("tool_result_status", {"tool_name": "weather", "status": "success"}, "PASS"),
    ("tool_result_status", {"tool_name": "weather", "status": "error"}, "FAIL"),
    ("fixture_no_violations", None, "PASS"),
    ("fixture_all_consumed", None, "PASS"),
])
def test_rules_with_complete_evidence(kind, expected, decision):
    execution = returned(records=[observed(ToolSuccess(data={}))])
    assert RuleEvaluator().evaluate(rule(kind, expected), execution).decision == decision


def test_partial_evidence_proves_observed_facts_only():
    execution = returned(
        availability="PARTIAL", status=AgentRunStatus.MAX_ITERATIONS_REACHED,
        records=[observed(ToolFailure(error={"code": "TEMPORARY", "message": "retry"}))],
    )
    evaluator = RuleEvaluator()
    assert evaluator.evaluate(rule("run_status", "MAX_ITERATIONS_REACHED"), execution).decision == "PASS"
    assert evaluator.evaluate(rule("tool_called", "weather"), execution).decision == "PASS"
    assert evaluator.evaluate(rule("tool_not_called", "weather"), execution).decision == "FAIL"
    assert evaluator.evaluate(rule("tool_not_called", "other"), execution).decision == "UNDETERMINED"
    assert evaluator.evaluate(rule("tool_called", "other"), execution).decision == "UNDETERMINED"
    status = {"tool_name": "weather", "status": "error"}
    assert evaluator.evaluate(rule("tool_result_status", status), execution).decision == "PASS"
    status["status"] = "success"
    assert evaluator.evaluate(rule("tool_result_status", status), execution).decision == "UNDETERMINED"


def test_unavailable_evidence_and_known_skips_are_distinct():
    evaluator = RuleEvaluator()
    assert evaluator.evaluate(rule("tool_not_called", "weather"), raised()).decision == "UNDETERMINED"
    assert evaluator.evaluate(rule("run_status", "COMPLETED"), raised()).decision == "UNDETERMINED"
    known_skip = returned(records=[observed(skipped=True)])
    assert evaluator.evaluate(rule("tool_not_called", "weather"), known_skip).decision == "PASS"


@pytest.mark.parametrize("code", list(FixtureViolationCode))
def test_fixture_violations_and_unconsumed_are_facts_even_without_tool_evidence(code):
    execution = replace(
        raised(), fixture_violations=(FixtureViolation("weather", code),), unconsumed_fixture_indices=(0,),
    )
    evaluator = RuleEvaluator()
    assert evaluator.evaluate(rule("fixture_no_violations", None), execution).decision == "FAIL"
    assert evaluator.evaluate(rule("fixture_all_consumed", None), execution).decision == "FAIL"


def test_expected_exception_phase_and_required_pass_on_raise():
    criterion = rule("expected_exception", {"exception_type": "TimeoutError"})
    dataset = EvaluationDataset(cases=[sample(criterion)])
    engine = EvaluationEngine()
    [result] = asyncio.run(evaluate_records(engine, dataset, DETERMINISTIC, [raised()]))
    assert result.case_verdict is CaseVerdict.PASS
    assembly = raised(ExecutionFailurePhase.ASSEMBLY)
    assert RuleEvaluator().evaluate(criterion, assembly).decision == "FAIL"
    phase_explicit = rule("expected_exception", {"exception_type": "TimeoutError", "phase": "ASSEMBLY"})
    assert RuleEvaluator().evaluate(phase_explicit, assembly).decision == "PASS"
    assert RuleEvaluator().evaluate(rule("fixture_no_violations", None), assembly).decision == "UNDETERMINED"


@pytest.mark.parametrize(("kind", "expected", "error"), [
    ("unsupported", None, "UNKNOWN_RULE"),
    ("tool_called", {"tool_name": "weather"}, "INVALID_RULE_EXPECTED"),
    ("tool_arguments_match", {"tool_name": "weather", "arguments": []}, "INVALID_RULE_EXPECTED"),
    ("tool_result_status", {"tool_name": "weather", "status": "ERROR"}, "INVALID_RULE_EXPECTED"),
    ("expected_exception", {"exception_type": "TimeoutError", "unexpected": True}, "INVALID_RULE_EXPECTED"),
    ("fixture_all_consumed", True, "INVALID_RULE_EXPECTED"),
])
def test_rule_errors_are_scoped_and_other_criteria_and_cases_continue(kind, expected, error):
    dataset = EvaluationDataset(cases=[
        sample(rule(kind, expected, criterion_id="bad"), rule("execution_status", "RETURNED")),
        sample(rule("execution_status", "RETURNED"), case_id="next"),
    ])
    results = asyncio.run(evaluate_records(
        EvaluationEngine(), dataset, DETERMINISTIC, [returned("next"), returned()],
    ))
    assert [result.case_id for result in results] == ["case", "next"]
    assert [result.case_verdict for result in results] == ["UNDETERMINED", "PASS"]
    assert [item.decision for item in results[0].criteria_results] == ["UNDETERMINED", "PASS"]
    [failure] = results[0].evaluation_errors
    assert (failure.criterion_id, failure.evaluator_type, failure.error_code) == ("bad", "RULE", error)


class FakeJudge:
    def __init__(self, decisions=None, *, error=None):
        self.requests = []
        self.decisions = decisions or {}
        self.error = error

    async def judge(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return JudgeVerdict(criteria_results=[
            {"criterion_id": item, "decision": decision, "reason": "Assessment"}
            for item, decision in self.decisions.items()
        ])


def test_engine_combines_rules_and_multiple_judge_criteria_with_minimized_evidence():
    judge = FakeJudge({"uncertain": "UNDETERMINED", "optional": "FAIL", "grounding": "PASS"})
    engine = EvaluationEngine(judge=judge, judge_ref="judge", tool_evidence_fields={"weather": ["temperature", "large"]})
    dataset = evaluation_dataset(MODEL, sample(
        semantic(), rule("execution_status", "RETURNED"),
        semantic("optional", required=False), semantic("uncertain", required=False),
    ))
    execution = returned(records=[
        observed(ToolSuccess(data={"temperature": 21, "secret": "sensitive token", "large": "x" * 257})),
        observed(ToolSuccess(data={"value": "unrelated"}), name="other"),
    ])
    [result] = asyncio.run(evaluate_records(engine, dataset, MODEL, [execution]))
    assert result.execution_result is execution.execution_result
    assert result.case_verdict is CaseVerdict.PASS
    assert [criterion.decision for criterion in result.criteria_results] == ["PASS", "PASS", "FAIL", "UNDETERMINED"]
    assert EvaluationResult.model_validate_json(result.model_dump_json()) == result
    [request] = judge.requests
    assert (request.user_query, request.agent_answer, request.reference_facts) == ("Task", "Answer", ("Reference",))
    assert request.relevant_evidence["projected_tool_calls"] == [
        {"tool_name": "weather", "result_status": "success", "skipped": False, "data": {"temperature": 21}},
    ]
    assert request.relevant_evidence["tool_projection_limited"] is True
    assert "private" not in repr(request)
    assert "sensitive token" not in repr(request)


@pytest.mark.parametrize(("config", "expected", "errors"), [
    (DETERMINISTIC, "NOT_APPLICABLE", []),
    (EvaluationRunConfig(mode="MODEL", model_ref="agent"), "UNDETERMINED", ["JUDGE_NOT_CONFIGURED"]),
    (EvaluationRunConfig(mode="MODEL", model_ref="agent", judge_ref="other"), "UNDETERMINED", ["JUDGE_NOT_CONFIGURED"]),
])
def test_modes_with_no_applicable_judge_do_not_pass_or_call_judge(config, expected, errors):
    judge = FakeJudge({"grounding": "PASS"})
    dataset = evaluation_dataset(config, sample(semantic()))
    [result] = asyncio.run(evaluate_records(
        EvaluationEngine(judge=judge, judge_ref="judge"), dataset, config, [returned()],
    ))
    assert not judge.requests
    assert result.case_verdict is CaseVerdict.UNDETERMINED
    assert result.criteria_results[0].decision == expected
    assert [error.error_code for error in result.evaluation_errors] == errors


def test_judge_missing_answer_and_failed_case_do_not_call_judge():
    judge = FakeJudge({"grounding": "PASS"})
    dataset = evaluation_dataset(MODEL, sample(semantic()))
    engine = EvaluationEngine(judge=judge, judge_ref="judge")
    for execution in (raised(), returned(status=AgentRunStatus.MAX_ITERATIONS_REACHED)):
        [result] = asyncio.run(evaluate_records(engine, dataset, MODEL, [execution]))
        assert result.case_verdict is CaseVerdict.UNDETERMINED
        assert result.evaluation_errors == []
    assert not judge.requests


def test_judge_error_preserves_rule_fail_and_hides_exception():
    judge = FakeJudge(error=RuntimeError("sk-sensitive-error"))
    dataset = evaluation_dataset(MODEL, sample(semantic(), rule("execution_status", "RAISED")))
    [result] = asyncio.run(evaluate_records(
        EvaluationEngine(judge=judge, judge_ref="judge"), dataset, MODEL, [returned()],
    ))
    assert result.case_verdict is CaseVerdict.FAIL
    assert [item.decision for item in result.criteria_results] == ["UNDETERMINED", "FAIL"]
    assert result.evaluation_errors[0].error_code == "JUDGE_EXECUTION_FAILED"
    assert "sk-sensitive-error" not in result.model_dump_json()


def test_judge_port_id_mismatch_is_an_error_and_later_cases_still_evaluate():
    judge = FakeJudge({"valid": "PASS"})
    dataset = evaluation_dataset(MODEL, sample(semantic()), sample(semantic("valid"), case_id="next"))
    results = asyncio.run(evaluate_records(
        EvaluationEngine(judge=judge, judge_ref="judge"), dataset, MODEL, [returned(), returned("next")],
    ))
    assert results[0].evaluation_errors[0].error_code == "JUDGE_CRITERION_IDS"
    assert [result.case_verdict for result in results] == ["UNDETERMINED", "PASS"]


@pytest.mark.parametrize("ids", [(), ("case",), ("case", "case"), ("different",), ("case", "different")])
def test_unissued_records_are_rejected_before_any_judge_call(ids):
    judge = FakeJudge({"grounding": "PASS"})
    dataset = evaluation_dataset(MODEL, sample(semantic()))
    with pytest.raises(ValueError, match="runner-issued"):
        asyncio.run(EvaluationEngine(judge=judge, judge_ref="judge").evaluate(
            dataset, MODEL, [returned(case_id) for case_id in ids],
        ))
    assert not judge.requests


def test_evaluation_reuses_deterministic_runner_output_without_rerunning(monkeypatch):
    dataset = EvaluationDataset(cases=[sample(rule("run_status", "COMPLETED"))])
    runner = EvaluationRunner(scripts={"script": {"case": [LLMResponse("answer", "stop")]}})
    executions = asyncio.run(runner.run(dataset, DETERMINISTIC))

    async def fail_if_reexecuted(*args, **kwargs):
        pytest.fail("Engine must not rerun the Agent")

    monkeypatch.setattr("backend.agent_runtime.runtime.AgentRuntime.run", fail_if_reexecuted)
    [result] = asyncio.run(EvaluationEngine().evaluate(dataset, DETERMINISTIC, executions))
    assert result.case_verdict is CaseVerdict.PASS
    assert result.execution_result is executions[0].execution_result


def test_judge_cancellation_propagates_through_engine():
    judge = FakeJudge(error=asyncio.CancelledError())
    dataset = evaluation_dataset(MODEL, sample(semantic()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(evaluate_records(
            EvaluationEngine(judge=judge, judge_ref="judge"), dataset, MODEL, [returned()],
        ))
