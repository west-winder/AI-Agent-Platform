"""Evaluate semantic criteria with a narrow, caller-authorized evidence projection."""

from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from backend.evaluation.contracts import (
    CriterionDecision,
    CriterionResult,
    EvaluationCase,
    EvaluationError,
    EvaluationMode,
    EvaluatorType,
    ExecutionStatus,
    JsonObject,
)
from backend.evaluation.judge_port import JudgeError, JudgePort, JudgeRequest, validate_judge_verdict
from backend.evaluation.runner import CaseExecution
from backend.tools.infrastructure.contracts import ToolSuccess


_MAX_TOOL_RECORDS = 12
_MAX_FIELDS_PER_RECORD = 8
_MAX_FIELD_CHARS = 256


class JudgeEvaluator:
    def __init__(self, tool_evidence_fields: Mapping[str, Sequence[str]] | None = None) -> None:
        self._fields = {
            name: tuple(dict.fromkeys(fields))
            for name, fields in (tool_evidence_fields or {}).items()
        }

    async def evaluate(
        self,
        case: EvaluationCase,
        execution: CaseExecution,
        mode: EvaluationMode,
        judge: JudgePort | None,
    ) -> tuple[list[CriterionResult], list[EvaluationError]]:
        criteria = tuple(item for item in case.criteria if item.evaluator_type is EvaluatorType.JUDGE)
        if not criteria:
            return [], []

        def unresolved(
            decision: CriterionDecision, rationale: str, error_code: str | None = None,
        ) -> tuple[list[CriterionResult], list[EvaluationError]]:
            results = [
                CriterionResult(criterion_id=item.criterion_id, decision=decision, rationale=rationale)
                for item in criteria
            ]
            if error_code is None:
                return results, []
            errors = [
                EvaluationError(
                    evaluator_type=EvaluatorType.JUDGE, error_code=error_code, criterion_id=item.criterion_id
                )
                for item in criteria
            ]
            return results, errors

        if mode is EvaluationMode.DETERMINISTIC:
            return unresolved(CriterionDecision.NOT_APPLICABLE, "Judge is disabled in deterministic mode")
        if judge is None:
            return unresolved(
                CriterionDecision.UNDETERMINED, "Judge profile is not configured", "JUDGE_NOT_CONFIGURED"
            )
        answer = execution.evidence.final_answer
        if (
            execution.execution_result.execution_status is not ExecutionStatus.RETURNED
            or answer is None
            or not answer.strip()
        ):
            return unresolved(CriterionDecision.UNDETERMINED, "No final answer is available")

        needs_context = case.judge_context_required or any(
            message.role != "system" for message in case.input_messages[:-1]
        )
        if needs_context and case.judge_context is None:
            return unresolved(CriterionDecision.UNDETERMINED, "Curated task context is missing")

        try:
            request = JudgeRequest(
                user_query=case.input_messages[-1].content or "",
                agent_answer=answer,
                reference_facts=tuple(case.reference_facts),
                relevant_evidence=self._project_evidence(execution),
                semantic_criteria=criteria,
                judge_context=case.judge_context,
            )
            verdict = validate_judge_verdict(request, await judge.judge(request))
        except JudgeError as exc:
            return unresolved(CriterionDecision.UNDETERMINED, "Judge evaluation failed", exc.code.value)
        except Exception:
            return unresolved(
                CriterionDecision.UNDETERMINED, "Judge evaluation failed", "JUDGE_EXECUTION_FAILED"
            )
        return [
            CriterionResult(criterion_id=item.criterion_id, decision=item.decision, rationale=item.reason)
            for item in verdict.criteria_results
        ], []

    def _project_evidence(self, execution: CaseExecution) -> JsonObject:
        evidence = execution.evidence
        records: list[JsonValue] = []
        limited = False
        for record in evidence.tool_calls:
            fields = self._fields.get(record.call.name)
            if fields is None:
                continue
            if len(records) == _MAX_TOOL_RECORDS:
                limited = True
                break
            result = record.tool_result
            data: JsonObject = {}
            if isinstance(result, ToolSuccess):
                limited = limited or len(fields) > _MAX_FIELDS_PER_RECORD
                for field in fields[:_MAX_FIELDS_PER_RECORD]:
                    if field not in result.data:
                        continue
                    value = result.data[field]
                    if isinstance(value, (dict, list)) or (
                        isinstance(value, str) and len(value) > _MAX_FIELD_CHARS
                    ):
                        limited = True
                    else:
                        data[field] = value
            records.append({
                "tool_name": record.call.name,
                "skipped": record.skipped,
                "result_status": result.status if result is not None else None,
                "data": data,
            })
        return {
            "run_status": evidence.run_status.value if evidence.run_status is not None else None,
            "iterations": evidence.iterations,
            "tool_call_availability": evidence.tool_call_availability.value,
            "tool_projection_limited": limited,
            "projected_tool_calls": records,
        }
