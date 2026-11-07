"""Deterministic rules over the facts already captured by the Runner."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from backend.agent_runtime.contracts import AgentRunStatus
from backend.evaluation.contracts import (
    CriterionDecision,
    CriterionResult,
    EvaluationCriterion,
    EvidenceAvailability,
    ExecutionStatus,
    JsonObject,
    NonBlank,
    Symbol,
)
from backend.evaluation.fixtures import same_json
from backend.evaluation.runner import CaseExecution, ExecutionFailurePhase


class RuleEvaluationError(Exception):
    """A safe, fixed classification for an invalid rule specification."""


class _ExceptionExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exception_type: Symbol
    phase: ExecutionFailurePhase = ExecutionFailurePhase.RUNTIME


class _ToolExpected(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    tool_name: NonBlank


class _ArgumentsExpected(_ToolExpected):
    arguments: JsonObject


class _StatusExpected(_ToolExpected):
    status: Literal["success", "error"]


_EXECUTION_STATUS = TypeAdapter(ExecutionStatus)
_RUN_STATUS = TypeAdapter(AgentRunStatus)
_TOOL_NAME = TypeAdapter(NonBlank)
_NULL = TypeAdapter(type(None))


def _decision(matches: bool) -> CriterionDecision:
    return CriterionDecision.PASS if matches else CriterionDecision.FAIL


class RuleEvaluator:
    def evaluate(self, criterion: EvaluationCriterion, execution: CaseExecution) -> CriterionResult:
        try:
            decision = self._decide(criterion, execution)
        except ValidationError:
            raise RuleEvaluationError("INVALID_RULE_EXPECTED") from None
        return CriterionResult(criterion_id=criterion.criterion_id, decision=decision)

    def _decide(self, criterion: EvaluationCriterion, execution: CaseExecution) -> CriterionDecision:
        expected = criterion.expected
        evidence = execution.evidence
        kind = criterion.criterion_type
        if kind == "execution_status":
            status = _EXECUTION_STATUS.validate_python(expected)
            return _decision(execution.execution_result.execution_status is status)
        if kind == "run_status":
            run_status = _RUN_STATUS.validate_python(expected)
            if evidence.run_status is None:
                return CriterionDecision.UNDETERMINED
            return _decision(evidence.run_status is run_status)
        if kind == "expected_exception":
            exception = _ExceptionExpected.model_validate(expected)
            actual = execution.execution_result.exception_info
            if actual is None or actual.exception_type != exception.exception_type:
                return CriterionDecision.FAIL
            if execution.failure_phase is None:
                return CriterionDecision.UNDETERMINED
            return _decision(execution.failure_phase is exception.phase)
        if kind in ("fixture_no_violations", "fixture_all_consumed"):
            _NULL.validate_python(expected)
            diagnostics = (
                execution.fixture_violations
                if kind == "fixture_no_violations"
                else execution.unconsumed_fixture_indices
            )
            if diagnostics:
                return CriterionDecision.FAIL
            if execution.failure_phase is ExecutionFailurePhase.ASSEMBLY:
                return CriterionDecision.UNDETERMINED
            return CriterionDecision.PASS
        return self._tool_decision(kind, expected, execution)

    def _tool_decision(self, kind: str, expected: object, execution: CaseExecution) -> CriterionDecision:
        evidence = execution.evidence
        calls = [record for record in evidence.tool_calls if not record.skipped]
        if kind in ("tool_called", "tool_not_called"):
            name = _TOOL_NAME.validate_python(expected)
            observed = any(record.call.name == name for record in calls)
            if observed or evidence.tool_call_availability is EvidenceAvailability.COMPLETE:
                return _decision(observed == (kind == "tool_called"))
            return CriterionDecision.UNDETERMINED
        if kind == "tool_arguments_match":
            arguments = _ArgumentsExpected.model_validate(expected)
            matched = any(
                record.call.name == arguments.tool_name
                and same_json(record.call.arguments, arguments.arguments)
                for record in calls
            )
        elif kind == "tool_result_status":
            result = _StatusExpected.model_validate(expected)
            matched = any(
                record.call.name == result.tool_name
                and record.tool_result is not None
                and record.tool_result.status == result.status
                for record in calls
            )
        else:
            raise RuleEvaluationError("UNKNOWN_RULE")
        if matched or evidence.tool_call_availability is EvidenceAvailability.COMPLETE:
            return _decision(matched)
        return CriterionDecision.UNDETERMINED
