import asyncio
import json
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from backend.agent_runtime.contracts import LLMResponse, LLMToolCall
from backend.evaluation import __main__ as cli
from backend.evaluation.contracts import (
    EvaluationDataset,
    EvaluationError,
    EvaluationRunConfig,
)
from backend.evaluation.engine import EvaluationEngine
from backend.evaluation.harness import run_evaluation
from backend.evaluation.judge_port import JudgeVerdict
from backend.evaluation.report import aggregate_metrics, build_report
from backend.evaluation.runner import EvaluationRunner
from backend.evaluation.v1_scripts import PROFILE, SCRIPTS


DATA = Path(cli.__file__).with_name("data")


@pytest.fixture
def dataset():
    return EvaluationDataset.model_validate_json((DATA / "dataset_v1.json").read_bytes())


@pytest.fixture
def offline_config():
    return EvaluationRunConfig.model_validate_json((DATA / "run_v1.json").read_bytes())


def test_dataset_version_and_exact_script_coverage(dataset):
    assert dataset.schema_version == 1
    assert set(SCRIPTS) == {case.case_id for case in dataset.cases}
    with pytest.raises(ValidationError, match="schema_version"):
        EvaluationDataset.model_validate(dataset.model_dump() | {"schema_version": 2})


def test_bundled_cli_runs_offline_and_counts_all_cases(dataset, tmp_path, monkeypatch):
    async def reject_network(*args, **kwargs):
        pytest.fail("Offline runs must use fixture handlers")

    monkeypatch.setattr("backend.tools.bootstrap.get_weather_handler", reject_network)
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluation", "--output", str(output)])
    cli.main()
    report = json.loads(output.read_bytes())
    metrics = report["metrics"]
    assert report["report_schema_version"] == 1
    assert report["evaluation_mode"] == "DETERMINISTIC"
    assert report["case_count"] == metrics["total_cases"] == 15
    assert (metrics["passed_cases"], metrics["failed_cases"], metrics["undetermined_cases"]) == (6, 6, 3)
    assert metrics["confirmed_pass_rate"] == 6 / 15
    assert (metrics["returned_cases"], metrics["raised_cases"]) == (14, 1)
    assert (metrics["completed_cases"], metrics["max_iterations_cases"]) == (13, 1)
    assert metrics["evaluation_error_cases"] == metrics["evaluation_error_count"] == 0
    assert metrics["criteria"]["JUDGE"] == {
        "total_criteria": 5, "PASS": 0, "FAIL": 0, "UNDETERMINED": 0, "NOT_APPLICABLE": 5,
    }
    assert metrics["criteria"]["RULE"]["FAIL"] > 0
    assert metrics["criteria"]["RULE"]["UNDETERMINED"] == 1
    cases = {item.case_id: report["cases"][index] for index, item in enumerate(dataset.cases)}
    assert cases["expected_runtime_exception"]["execution_status"] == "RAISED"
    assert cases["expected_runtime_exception"]["verdict"] == "PASS"
    assert cases["fixture_mismatch"]["fixture_diagnostics"] == {
        "violation_counts": {"FIXTURE_ARGUMENT_MISMATCH": 1}, "unconsumed_count": 1,
    }
    assert cases["fixture_exhausted"]["fixture_diagnostics"]["violation_counts"] == {"FIXTURE_EXHAUSTED": 1}
    assert report["model_usage"] == {"tokens": None, "cost": None, "latency_ms": None}
    first = output.read_bytes()
    cli.main()
    assert output.read_bytes() == first


def test_cli_rejects_model_before_assembly(tmp_path, monkeypatch, capsys):
    config = tmp_path / "model.json"
    config.write_text('{"mode":"MODEL","model_ref":"model"}', encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluation", "--config", str(config), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "CLI is offline" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("limit", [None, 0, True, 14, 21])
def test_model_harness_requires_explicit_bounded_opt_in(dataset, limit):
    class NoRunner:
        async def run(self, *args):
            pytest.fail("Invalid MODEL budget must be rejected before execution")

    config = EvaluationRunConfig(mode="MODEL", model_ref="test")
    with pytest.raises(ValueError, match="explicit case limit"):
        asyncio.run(run_evaluation(
            dataset, config, runner=NoRunner(), engine=EvaluationEngine(), model_case_limit=limit,
        ))


@pytest.mark.parametrize("judge_fails", [False, True])
def test_model_report_omits_raw_history_payloads_rationale_and_refs(judge_fails):
    dataset = EvaluationDataset.model_validate({"purpose": "AGENT_QUALITY", "cases": [{
        "case_id": "SECRET_CASE",
        "input_messages": [{"role": "user", "content": "SECRET_HISTORY"}],
        "reference_facts": ["SECRET_FACT"],
        "tool_fixtures": [{
            "tool_name": "get_weather",
            "arguments": {"city": "SECRET_ARGUMENT"},
            "result": {"status": "success", "data": {"payload": "SECRET_TOOL_RESULT"}},
        }],
        "criteria": [
            {"criterion_id": "SECRET_RULE_ID", "evaluator_type": "RULE",
             "criterion_type": "run_status", "expected": "COMPLETED"},
            {"criterion_id": "SECRET_JUDGE_ID", "evaluator_type": "JUDGE",
             "criterion_type": "grounding", "expected": None, "rubric": "SECRET_RUBRIC", "required": True},
        ],
    }]})
    config = EvaluationRunConfig(mode="MODEL", model_ref="SECRET_MODEL", judge_ref="SECRET_JUDGE")

    class StubModel:
        async def complete(self, request):
            if request.messages[-1].role == "tool":
                return LLMResponse("SECRET_ANSWER", "stop")
            call = LLMToolCall("SECRET_CALL_ID", "get_weather", '{"city":"SECRET_ARGUMENT"}',
                               {"city": "SECRET_ARGUMENT"})
            return LLMResponse(None, "tool_calls", [call])

    class StubJudge:
        async def judge(self, request):
            if judge_fails:
                raise RuntimeError("SECRET_EXCEPTION")
            return JudgeVerdict(criteria_results=[{
                "criterion_id": "SECRET_JUDGE_ID", "decision": "PASS", "reason": "SECRET_REASON",
            }])

    runner = EvaluationRunner(model_factories={config.model_ref: StubModel})
    engine = EvaluationEngine(judge=StubJudge(), judge_ref=config.judge_ref)
    report = asyncio.run(run_evaluation(dataset, config, runner=runner, engine=engine, model_case_limit=1))
    assert "SECRET_" not in json.dumps(report)
    assert report["metrics"]["criteria"]["RULE"]["PASS"] == 1
    assert report["metrics"]["criteria"]["JUDGE"]["UNDETERMINED" if judge_fails else "PASS"] == 1
    assert report["metrics"]["evaluation_error_count"] == int(judge_fails)
    assert report["cases"][0]["verdict"] == ("UNDETERMINED" if judge_fails else "PASS")
    assert report["run_config"]["model_ref_sha256"] != report["run_config"]["judge_ref_sha256"]


@pytest.fixture
def batch(dataset, offline_config):
    runner = EvaluationRunner(scripts={PROFILE: SCRIPTS})

    async def execute():
        executions = await runner.run(dataset, offline_config)
        results = await EvaluationEngine().evaluate(dataset, offline_config, executions)
        return executions, results

    return asyncio.run(execute())


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "unknown", "mode", "required", "criterion", "error"])
def test_report_rejects_inconsistent_case_results(dataset, offline_config, batch, tamper):
    _, results = batch
    first, *rest = results
    if tamper == "missing":
        results = rest
    elif tamper == "duplicate":
        results = [first, first, *rest]
    else:
        update = {
            "unknown": {"execution_result": first.execution_result.model_copy(update={"case_id": "unknown"})},
            "mode": {"evaluation_mode": "MODEL"},
            "required": {"required_criterion_ids": first.required_criterion_ids[:1]},
            "criterion": {"criteria_results": first.criteria_results[:1]},
            "error": {"evaluation_errors": [
                EvaluationError(evaluator_type="RULE", criterion_id="unknown", error_code="OTHER"),
            ]},
        }[tamper]
        results = [first.model_copy(update=update), *rest]
    with pytest.raises(ValueError, match="dataset"):
        aggregate_metrics(dataset, offline_config, results)


def test_optional_error_is_counted_and_unknown_code_is_safe(dataset, offline_config, batch):
    executions, results = batch
    [index] = [i for i, result in enumerate(results) if result.case_id == "correct_tool"]
    results[index] = results[index].model_copy(update={"evaluation_errors": [
        EvaluationError(evaluator_type="JUDGE", criterion_id="grounding", error_code="SECRET_CODE"),
    ]})
    report = build_report(dataset, offline_config, executions, results)
    assert report["metrics"]["evaluation_error_count"] == report["metrics"]["evaluation_error_cases"] == 1
    assert report["cases"][index]["verdict"] == "PASS"
    assert report["cases"][index]["evaluation_errors"][0]["code"] == "OTHER_EVALUATION_ERROR"
    assert "SECRET_CODE" not in json.dumps(report)
