"""Evaluate existing CaseExecution records and assemble A1 results."""

from collections.abc import Mapping, Sequence

from backend.evaluation.contracts import (
    CriterionDecision,
    CriterionResult,
    EvaluationDataset,
    EvaluationError,
    EvaluationMode,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluatorType,
    aggregate_case_verdict,
    validate_dataset_mode,
)
from backend.evaluation.judge_evaluator import JudgeEvaluator
from backend.evaluation.judge_port import JudgePort
from backend.evaluation.rule_evaluator import RuleEvaluationError, RuleEvaluator
from backend.evaluation.runner import ExecutionBatch, verify_execution_batch


class EvaluationEngine:
    def __init__(
        self,
        *,
        judge: JudgePort | None = None,
        judge_ref: str | None = None,
        tool_evidence_fields: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if (judge is None) != (judge_ref is None):
            raise ValueError("Supply both a JudgePort and its profile reference")
        self._judge = judge
        self._judge_ref = judge_ref
        self._rule_evaluator = RuleEvaluator()
        self._judge_evaluator = JudgeEvaluator(tool_evidence_fields)

    async def evaluate(
        self,
        dataset: EvaluationDataset,
        config: EvaluationRunConfig,
        executions: ExecutionBatch,
    ) -> list[EvaluationResult]:
        dataset = EvaluationDataset.model_validate(dataset.model_dump())
        config = EvaluationRunConfig.model_validate(config.model_dump())
        verify_execution_batch(executions, dataset, config)
        validate_dataset_mode(dataset, config.mode)
        by_id = {execution.execution_result.case_id: execution for execution in executions}
        if len(by_id) != len(executions) or by_id.keys() != {case.case_id for case in dataset.cases}:
            raise ValueError("CaseExecution IDs must match the dataset exactly once")
        judge = (
            self._judge
            if config.mode is EvaluationMode.MODEL and config.judge_ref == self._judge_ref
            else None
        )

        results: list[EvaluationResult] = []
        for case in dataset.cases:
            execution = by_id[case.case_id]
            decisions: dict[str, CriterionResult] = {}
            errors: list[EvaluationError] = []
            for criterion in case.criteria:
                if criterion.evaluator_type is not EvaluatorType.RULE:
                    continue
                try:
                    decisions[criterion.criterion_id] = self._rule_evaluator.evaluate(criterion, execution)
                    continue
                except RuleEvaluationError as exc:
                    code = str(exc)
                except Exception:
                    code = "RULE_EVALUATION_FAILED"
                decisions[criterion.criterion_id] = CriterionResult(
                    criterion_id=criterion.criterion_id, decision=CriterionDecision.UNDETERMINED
                )
                errors.append(EvaluationError(
                    criterion_id=criterion.criterion_id, evaluator_type=EvaluatorType.RULE, error_code=code
                ))

            semantic_results, semantic_errors = await self._judge_evaluator.evaluate(
                case, execution, config.mode, judge
            )
            decisions.update((result.criterion_id, result) for result in semantic_results)
            errors.extend(semantic_errors)
            criteria_results = [decisions[item.criterion_id] for item in case.criteria]
            required = [item.criterion_id for item in case.criteria if item.required]
            results.append(EvaluationResult(
                case_id=case.case_id,
                evaluation_mode=config.mode,
                execution_result=execution.execution_result,
                required_criterion_ids=required,
                criteria_results=criteria_results,
                case_verdict=aggregate_case_verdict(required, criteria_results, errors),
                evaluation_errors=errors,
            ))
        return results
