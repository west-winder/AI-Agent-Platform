import pytest
from pydantic import ValidationError

from backend.agent_runtime.contracts import AgentRunResult, AgentRunStatus
from backend.evaluation.contracts import (
    CaseVerdict,
    CriterionResult,
    EvaluationError,
    EvaluationExecutionResult,
    EvaluationResult,
    ExceptionInfo,
    aggregate_case_verdict,
)


@pytest.mark.parametrize(("decisions", "expected"), [
    (["PASS", "PASS"], CaseVerdict.PASS),
    (["PASS", "FAIL"], CaseVerdict.FAIL),
    (["FAIL", "UNDETERMINED"], CaseVerdict.FAIL),
    (["FAIL", "NOT_APPLICABLE"], CaseVerdict.FAIL),
    (["PASS", "UNDETERMINED"], CaseVerdict.UNDETERMINED),
    (["PASS", "NOT_APPLICABLE"], CaseVerdict.UNDETERMINED),
    (["PASS"], CaseVerdict.UNDETERMINED),
    (["FAIL"], CaseVerdict.FAIL),
    ([], CaseVerdict.UNDETERMINED),
])
def test_required_criteria_aggregation(decisions, expected):
    results = [
        CriterionResult(criterion_id=criterion_id, decision=decision)
        for criterion_id, decision in zip(["first", "second"], decisions)
    ]
    assert aggregate_case_verdict(["first", "second"], results) is expected


def test_optional_failure_does_not_veto_required_pass():
    results = [
        CriterionResult(criterion_id="required", decision="PASS"),
        CriterionResult(criterion_id="optional", decision="FAIL"),
    ]
    assert aggregate_case_verdict(["required"], results) is CaseVerdict.PASS


@pytest.mark.parametrize("required_ids", [[], ["required", "required"]])
def test_aggregation_rejects_invalid_required_id_sets(required_ids):
    with pytest.raises(ValueError, match="Required criterion IDs"):
        aggregate_case_verdict(required_ids, [])


def test_aggregation_rejects_duplicate_results():
    results = [
        CriterionResult(criterion_id="required", decision="PASS"),
        CriterionResult(criterion_id="required", decision="FAIL"),
    ]
    with pytest.raises(ValueError, match="Criterion result IDs"):
        aggregate_case_verdict(["required"], results)


@pytest.mark.parametrize(("decision", "verdict"), [
    ("PASS", CaseVerdict.UNDETERMINED),
    ("UNDETERMINED", CaseVerdict.UNDETERMINED),
    ("FAIL", CaseVerdict.FAIL),
])
@pytest.mark.parametrize("error_id", [None, "rule", "unknown"])
def test_blocking_evaluation_error_preserves_confirmed_failure(decision, verdict, error_id):
    execution = EvaluationExecutionResult(
        case_id="case", execution_status="RETURNED", duration_ms=0,
        agent_result=AgentRunResult(AgentRunStatus.COMPLETED, "answer", 1, []),
    )
    results = [CriterionResult(criterion_id="rule", decision=decision)]
    errors = [
        EvaluationError(evaluator_type="RULE", error_code="TIMEOUT", criterion_id=error_id),
    ]
    assert aggregate_case_verdict(["rule"], results, errors) is verdict
    result = EvaluationResult(
        case_id="case", evaluation_mode="MODEL", execution_result=execution,
        required_criterion_ids=["rule"], criteria_results=results,
        case_verdict=verdict, evaluation_errors=errors,
    )
    assert result.execution_result.exception_info is None
    assert result.evaluation_errors == errors
    assert EvaluationResult.model_validate_json(result.model_dump_json()) == result


def test_known_optional_evaluation_error_does_not_block_required_pass():
    results = [
        CriterionResult(criterion_id="required", decision="PASS"),
        CriterionResult(criterion_id="optional", decision="UNDETERMINED"),
    ]
    error = EvaluationError(evaluator_type="JUDGE", error_code="TIMEOUT", criterion_id="optional")
    assert aggregate_case_verdict(["required"], results, [error]) is CaseVerdict.PASS


def test_expected_runtime_exception_can_coexist_with_passing_evaluation():
    execution = EvaluationExecutionResult(
        case_id="case", execution_status="RAISED", duration_ms=0,
        exception_info=ExceptionInfo(exception_type="TimeoutError"),
    )
    result = EvaluationResult(
        case_id="case", evaluation_mode="DETERMINISTIC", execution_result=execution,
        required_criterion_ids=["expected_exception"], case_verdict="PASS",
        criteria_results=[CriterionResult(criterion_id="expected_exception", decision="PASS")],
    )
    assert result.execution_result.agent_result is None
    assert result.execution_result.exception_info.exception_type == "TimeoutError"
    assert result.evaluation_errors == []
    assert EvaluationResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(("updates", "match"), [
    ({"case_id": "other"}, "case IDs must match"),
    ({"case_verdict": "PASS"}, "Case verdict must match"),
])
def test_evaluation_result_rejects_inconsistent_link_or_verdict(updates, match):
    data = {
        "case_id": "case",
        "evaluation_mode": "MODEL",
        "execution_result": EvaluationExecutionResult(
            case_id="case", execution_status="RAISED", duration_ms=0,
            exception_info=ExceptionInfo(exception_type="TimeoutError"),
        ),
        "required_criterion_ids": ["missing"],
        "criteria_results": [],
        "case_verdict": "UNDETERMINED",
    }
    with pytest.raises(ValidationError, match=match):
        EvaluationResult.model_validate(data | updates)
