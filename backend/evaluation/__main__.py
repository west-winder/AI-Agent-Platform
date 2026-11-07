"""Run the bundled offline regression and write its projected JSON report."""

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from backend.evaluation.contracts import EvaluationDataset, EvaluationMode, EvaluationRunConfig
from backend.evaluation.engine import EvaluationEngine
from backend.evaluation.harness import run_evaluation
from backend.evaluation.runner import EvaluationRunner
from backend.evaluation.v1_scripts import PROFILE, SCRIPTS


def main() -> None:
    data = Path(__file__).with_name("data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=data / "dataset_v1.json")
    parser.add_argument("--config", type=Path, default=data / "run_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        dataset = EvaluationDataset.model_validate_json(args.dataset.read_bytes())
        config = EvaluationRunConfig.model_validate_json(args.config.read_bytes())
    except (OSError, ValidationError):
        parser.error("Unable to read or validate the dataset or run config")
    if config.mode is not EvaluationMode.DETERMINISTIC:
        parser.error("The CLI is offline; MODEL requires injected dependencies and a case limit")
    if config.fake_llm_script_ref != PROFILE or any(case.case_id not in SCRIPTS for case in dataset.cases):
        parser.error("The dataset and run config must match the bundled script profile")
    if args.output.resolve() in {args.dataset.resolve(), args.config.resolve()}:
        parser.error("Output must be separate from input files")

    report = asyncio.run(run_evaluation(
        dataset, config, runner=EvaluationRunner(scripts={PROFILE: SCRIPTS}), engine=EvaluationEngine(),
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
