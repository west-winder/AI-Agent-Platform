"""Compose existing execution, evaluation, and report stages."""

from backend.evaluation.contracts import (
    EvaluationDataset, EvaluationMode, EvaluationRunConfig, JsonObject, validate_dataset_mode,
)
from backend.evaluation.engine import EvaluationEngine
from backend.evaluation.report import build_report
from backend.evaluation.runner import EvaluationRunner


async def run_evaluation(
    dataset: EvaluationDataset,
    config: EvaluationRunConfig,
    *,
    runner: EvaluationRunner,
    engine: EvaluationEngine,
    model_case_limit: int | None = None,
) -> JsonObject:
    """Run one batch. An explicit case budget opts in to injected MODEL dependencies."""
    dataset = EvaluationDataset.model_validate(dataset.model_dump())
    config = EvaluationRunConfig.model_validate(config.model_dump())
    if config.mode is EvaluationMode.MODEL and (
        type(model_case_limit) is not int or not len(dataset.cases) <= model_case_limit <= 20
    ):
        raise ValueError("MODEL requires an explicit case limit from 1 to 20 covering the batch")
    validate_dataset_mode(dataset, config.mode)
    executions = await runner.run(dataset, config)
    results = await engine.evaluate(dataset, config, executions)
    return build_report(dataset, config, executions, results)
