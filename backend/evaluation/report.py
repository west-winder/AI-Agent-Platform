"""Dataset-bound metrics and an allowlist projection for shareable reports."""

from collections import Counter
from collections.abc import Sequence
from typing import TypeVar

from backend.agent_runtime.contracts import AgentRunStatus
from backend.evaluation.contracts import (
    CaseVerdict,
    CriterionDecision,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluatorType,
    ExecutionStatus,
    JsonObject,
    aggregate_case_verdict,
    validate_dataset_mode,
)
from backend.evaluation.judge_port import JudgeErrorCode
from backend.evaluation.runner import (
    CaseExecution, ExecutionBatch, content_fingerprint, verify_execution_batch,
)


_KNOWN_ERRORS = {code.value for code in JudgeErrorCode} | {
    "JUDGE_NOT_CONFIGURED", "UNKNOWN_RULE", "INVALID_RULE_EXPECTED", "RULE_EVALUATION_FAILED",
}
_CaseRecord = TypeVar("_CaseRecord", EvaluationResult, CaseExecution)


def _by_case_id(records: Sequence[_CaseRecord], dataset: EvaluationDataset) -> dict[str, _CaseRecord]:
    indexed = {record.execution_result.case_id: record for record in records}
    expected = {case.case_id for case in dataset.cases}
    if (
        not expected
        or len(expected) != len(dataset.cases)
        or len(indexed) != len(records)
        or indexed.keys() != expected
    ):
        raise ValueError("Case records must match the dataset exactly once")
    return indexed


def aggregate_metrics(
    dataset: EvaluationDataset, config: EvaluationRunConfig, results: Sequence[EvaluationResult],
) -> JsonObject:
    """Summarize one complete batch for one dataset and run configuration."""
    validate_dataset_mode(dataset, config.mode)
    indexed = _by_case_id(results, dataset)
    verdicts: Counter[CaseVerdict] = Counter()
    executions: Counter[ExecutionStatus] = Counter()
    statuses: Counter[AgentRunStatus] = Counter()
    decisions: dict[EvaluatorType, Counter[CriterionDecision]] = {
        evaluator: Counter() for evaluator in EvaluatorType
    }
    evaluation_error_count = evaluation_error_cases = 0
    for case in dataset.cases:
        result = indexed[case.case_id]
        criteria = {item.criterion_id: item for item in case.criteria}
        required = {item.criterion_id for item in case.criteria if item.required}
        observed = {item.criterion_id for item in result.criteria_results}
        if (
            result.case_id != case.case_id
            or result.evaluation_mode is not config.mode
            or set(result.required_criterion_ids) != required
            or len(result.required_criterion_ids) != len(required)
            or len(result.criteria_results) != len(observed)
            or observed != criteria.keys()
        ):
            raise ValueError("Case result does not match its dataset criteria or run mode")
        for error in result.evaluation_errors:
            if error.criterion_id is not None:
                criterion = criteria.get(error.criterion_id)
                if criterion is None or criterion.evaluator_type is not error.evaluator_type:
                    raise ValueError("Evaluation error does not match a dataset criterion")
        verdict = aggregate_case_verdict(tuple(required), result.criteria_results, result.evaluation_errors)
        if verdict is not result.case_verdict:
            raise ValueError("Case verdict does not match the dataset criteria")

        verdicts[verdict] += 1
        execution = result.execution_result
        executions[execution.execution_status] += 1
        if execution.agent_result is not None:
            statuses[execution.agent_result.status] += 1
        for item in result.criteria_results:
            decisions[criteria[item.criterion_id].evaluator_type][item.decision] += 1
        evaluation_error_cases += bool(result.evaluation_errors)
        evaluation_error_count += len(result.evaluation_errors)

    total = len(dataset.cases)
    passed = verdicts[CaseVerdict.PASS]
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": verdicts[CaseVerdict.FAIL],
        "undetermined_cases": verdicts[CaseVerdict.UNDETERMINED],
        "confirmed_pass_rate": passed / total,
        "returned_cases": executions[ExecutionStatus.RETURNED],
        "raised_cases": executions[ExecutionStatus.RAISED],
        "completed_cases": statuses[AgentRunStatus.COMPLETED],
        "max_iterations_cases": statuses[AgentRunStatus.MAX_ITERATIONS_REACHED],
        "evaluation_error_cases": evaluation_error_cases,
        "evaluation_error_count": evaluation_error_count,
        "criteria": {
            evaluator.value: {
                "total_criteria": counts.total(),
                **{decision.value: counts[decision] for decision in CriterionDecision},
            }
            for evaluator, counts in decisions.items()
        },
    }


def build_report(
    dataset: EvaluationDataset,
    config: EvaluationRunConfig,
    executions: ExecutionBatch,
    results: Sequence[EvaluationResult],
) -> JsonObject:
    provenance = verify_execution_batch(executions, dataset, config)
    metrics = aggregate_metrics(dataset, config, results)
    by_case = _by_case_id(results, dataset)
    executions_by_case = _by_case_id(executions, dataset)
    cases = []
    for case_index, case in enumerate(dataset.cases):
        result = by_case[case.case_id]
        record = executions_by_case[case.case_id]
        execution = result.execution_result
        if execution != record.execution_result:
            raise ValueError("Case result does not match its execution record")
        criterion_indices = {item.criterion_id: index for index, item in enumerate(case.criteria)}
        decisions = {item.criterion_id: item.decision for item in result.criteria_results}
        errors = [
            {
                "evaluator_type": error.evaluator_type.value,
                "code": error.error_code if error.error_code in _KNOWN_ERRORS else "OTHER_EVALUATION_ERROR",
                "criterion_index": criterion_indices.get(error.criterion_id) if error.criterion_id else None,
            }
            for error in result.evaluation_errors
        ]
        cases.append({
            "case_index": case_index,
            "verdict": result.case_verdict.value,
            "reason": {
                CaseVerdict.PASS: None,
                CaseVerdict.FAIL: "REQUIRED_CRITERION_FAILED",
                CaseVerdict.UNDETERMINED: "REQUIRED_EVALUATION_UNRESOLVED",
            }[result.case_verdict],
            "execution_status": execution.execution_status.value,
            "run_status": execution.agent_result.status.value if execution.agent_result else None,
            "iterations": execution.agent_result.iterations if execution.agent_result else None,
            "failure_phase": record.failure_phase.value if record.failure_phase else None,
            "tool_evidence": record.evidence.tool_call_availability.value,
            "fixture_diagnostics": {
                "violation_counts": dict(Counter(item.code.value for item in record.fixture_violations)),
                "unconsumed_count": len(record.unconsumed_fixture_indices),
            },
            "criteria": [
                {
                    "criterion_index": index,
                    "evaluator_type": criterion.evaluator_type.value,
                    "required": criterion.required,
                    "decision": decisions[criterion.criterion_id].value,
                }
                for index, criterion in enumerate(case.criteria)
            ],
            "evaluation_errors": errors,
        })
    return {
        "report_schema_version": 1,
        "dataset": {
            "schema_version": dataset.schema_version,
            "purpose": dataset.purpose.value,
            "sha256": provenance.dataset_sha256,
        },
        "evaluation_mode": provenance.mode.value,
        "run_config": {
            "sha256": provenance.run_config_sha256,
            "max_iterations": config.max_iterations,
            "model_ref_sha256": content_fingerprint(config.model_ref) if config.model_ref is not None else None,
            "judge_ref_sha256": content_fingerprint(config.judge_ref) if config.judge_ref is not None else None,
            "script_sha256": provenance.script_sha256,
            "script_ref_sha256": (
                content_fingerprint(config.fake_llm_script_ref) if config.fake_llm_script_ref is not None else None
            ),
        },
        "case_count": len(cases),
        "metrics": metrics,
        "model_usage": {"tokens": None, "cost": None, "latency_ms": None},
        "cases": cases,
    }
