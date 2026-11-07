import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.agent_runtime.contracts import LLMResponse, LLMToolCall
from backend.evaluation.contracts import EvaluationDataset, EvaluationRunConfig
from backend.evaluation.deepseek_judge_adapter import DeepSeekJudgeAdapter
from backend.evaluation.engine import EvaluationEngine
from backend.evaluation.harness import run_evaluation
from backend.evaluation.judge_port import JudgeError, JudgeRequest, JudgeVerdict
from backend.evaluation.report import build_report
from backend.evaluation.runner import EvaluationRunner, ExecutionBatch


OFFLINE = EvaluationRunConfig(mode="DETERMINISTIC", fake_llm_script_ref="script")
MODEL = EvaluationRunConfig(mode="MODEL", model_ref="stub", judge_ref="judge")
SEMANTIC = {
    "criterion_id": "meaning", "evaluator_type": "JUDGE", "criterion_type": "grounding",
    "expected": None, "rubric": "PASS only if the answer gives the supplied reference code.",
}
RULE = {
    "criterion_id": "returned", "evaluator_type": "RULE", "criterion_type": "execution_status",
    "expected": "RETURNED",
}


def dataset(*, query="Question", criteria=(RULE,), **fields):
    return EvaluationDataset.model_validate({"cases": [{
        "case_id": "same", "input_messages": [{"role": "user", "content": query}],
        "criteria": list(criteria), **fields,
    }]})


def scripted(*responses):
    return EvaluationRunner(scripts={"script": {"same": responses}})


class CodeJudge:
    def __init__(self):
        self.requests = []

    async def judge(self, request):
        self.requests.append(request)
        expected = request.judge_context.rsplit(" ", 1)[-1]
        return JudgeVerdict(criteria_results=[{
            "criterion_id": "meaning",
            "decision": "PASS" if request.agent_answer == expected else "FAIL",
            "reason": "Compare against the supplied code",
        }])


@pytest.mark.parametrize("change", ["query", "mode", "iterations"])
def test_runner_provenance_rejects_reuse_before_judge(change):
    original = dataset(criteria=(SEMANTIC,))
    target = dataset(query="Different question", criteria=(SEMANTIC,)) if change == "query" else original
    config = {
        "query": OFFLINE, "mode": MODEL,
        "iterations": OFFLINE.model_copy(update={"max_iterations": 2}),
    }[change]
    judge = CodeJudge()

    async def exercise():
        batch = await scripted(LLMResponse("17", "stop")).run(original, OFFLINE)
        with pytest.raises(ValueError, match="provenance"):
            await EvaluationEngine(judge=judge, judge_ref="judge").evaluate(target, config, batch)
        assert judge.requests == []

    asyncio.run(exercise())


def test_judge_receives_distinct_curated_context_and_skips_missing_context():
    class Agent:
        async def complete(self, request):
            return LLMResponse("17", "stop")

    judge = CodeJudge()
    engine = EvaluationEngine(judge=judge, judge_ref="judge")
    runner = EvaluationRunner(model_factories={"stub": Agent})
    cases = []
    for index, code in enumerate(("17", "42", None)):
        cases.append({
            "case_id": str(index),
            "input_messages": [
                {"role": "user", "content": f"Private history with code {code}"},
                {"role": "assistant", "content": "Understood"},
                {"role": "user", "content": "Repeat that code."},
            ],
            "criteria": [SEMANTIC],
            "judge_context": f"Reference code: {code}" if code is not None else None,
        })

    async def exercise():
        data = EvaluationDataset.model_validate({"purpose": "AGENT_QUALITY", "cases": cases})
        batch = await runner.run(data, MODEL)
        return await engine.evaluate(data, MODEL, batch)

    results = asyncio.run(exercise())
    assert [result.case_verdict for result in results] == ["PASS", "FAIL", "UNDETERMINED"]
    assert [request.judge_context for request in judge.requests] == ["Reference code: 17", "Reference code: 42"]
    assert "Private history" not in repr(judge.requests)


def test_behavior_dataset_is_rejected_before_model_execution():
    root = Path(__file__).parents[2]
    bundled = EvaluationDataset.model_validate_json(
        (root / "backend/evaluation/data/dataset_v1.json").read_bytes()
    )

    def unexpected_model():
        pytest.fail("Regression dataset must be rejected before model assembly")

    with pytest.raises(ValueError, match="AGENT_QUALITY"):
        asyncio.run(run_evaluation(
            bundled, MODEL, runner=EvaluationRunner(model_factories={"stub": unexpected_model}),
            engine=EvaluationEngine(), model_case_limit=15,
        ))


@pytest.mark.parametrize("criteria", [(RULE,), (RULE, SEMANTIC | {"required": False})])
def test_quality_case_rejects_missing_mandatory_answer_evaluation(criteria):
    with pytest.raises(ValidationError, match="mandatory Judge criterion"):
        EvaluationDataset(purpose="AGENT_QUALITY", cases=dataset(criteria=criteria).cases)


def test_wrong_factual_answer_fails_quality_even_when_runtime_returns():
    class WrongAgent:
        async def complete(self, request):
            return LLMResponse("Berlin", "stop")

    rubric = SEMANTIC | {"rubric": "PASS only if the answer names Paris as the capital of France."}
    data = dataset(query="What is the capital of France?", criteria=(RULE, rubric), judge_context="Capital: Paris")
    quality = EvaluationDataset(purpose="AGENT_QUALITY", cases=data.cases)
    judge = CodeJudge()
    report = asyncio.run(run_evaluation(
        quality, MODEL, runner=EvaluationRunner(model_factories={"stub": WrongAgent}),
        engine=EvaluationEngine(judge=judge, judge_ref="judge"), model_case_limit=1,
    ))
    [case] = report["cases"]
    assert [criterion["decision"] for criterion in case["criteria"]] == ["PASS", "FAIL"]
    assert case["verdict"] == "FAIL"


@pytest.mark.parametrize("context", ["", " " * 3, "x" * 2049])
def test_curated_context_requires_bounded_nonblank_text(context):
    with pytest.raises(ValidationError, match="judge_context"):
        dataset(judge_context=context)


@pytest.mark.parametrize("arguments", [{"city": " Wuhan "}, {"town": "Wuhan"}])
def test_fixture_assembly_uses_real_schema(arguments):
    call = LLMToolCall("id", "get_weather", '{"city":" Wuhan "}', {"city": " Wuhan "})
    runner = scripted(LLMResponse(None, "tool_calls", [call]), LLMResponse("21 C", "stop"))
    data = dataset(tool_fixtures=[{
        "tool_name": "get_weather", "arguments": arguments,
        "result": {"status": "success", "data": {"temperature_c": 21}},
    }])
    if "town" in arguments:
        with pytest.raises(ValueError, match="Fixture"):
            asyncio.run(runner.run(data, OFFLINE))
    else:
        [record] = asyncio.run(runner.run(data, OFFLINE))
        assert record.fixture_violations == ()
        assert record.unconsumed_fixture_indices == ()


def test_judge_rejects_conflicting_decision_keys():
    content = (
        '{"criteria_results":[{"criterion_id":"meaning","decision":"FAIL",'
        '"decision":"PASS","reason":"conflicting keys"}]}'
    )

    async def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(role="assistant", content=content),
        )])

    client = SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    criterion = dataset(criteria=(SEMANTIC,)).cases[0].criteria[0]
    request = JudgeRequest("Question", "Answer", (), {}, (criterion,))
    with pytest.raises(JudgeError, match="JUDGE_PROTOCOL_ERROR"):
        asyncio.run(DeepSeekJudgeAdapter(client, model="stub").judge(request))


def test_report_fingerprints_actual_script_content_without_exposing_it():
    async def report(text):
        return await run_evaluation(
            dataset(), OFFLINE, runner=scripted(LLMResponse(text, "stop")), engine=EvaluationEngine(),
        )

    one = asyncio.run(report("secret answer one"))
    same = asyncio.run(report("secret answer one"))
    two = asyncio.run(report("secret answer two"))
    assert one == same
    assert one["run_config"].get("script_sha256") != two["run_config"].get("script_sha256")
    assert "secret answer" not in repr(one)


def test_batch_membership_cannot_be_recreated_from_copied_provenance():
    async def exercise():
        data = dataset()
        runner = scripted(LLMResponse("one", "stop"))
        other_runner = scripted(LLMResponse("two", "stop"))
        batch = await runner.run(data, OFFLINE)
        other = await other_runner.run(data, OFFLINE)
        assert batch.provenance.dataset_sha256 == other.provenance.dataset_sha256
        assert batch.provenance.script_sha256 != other.provenance.script_sha256
        with pytest.raises(TypeError, match="issued by EvaluationRunner"):
            ExecutionBatch(batch.provenance, tuple(other))
        results = await EvaluationEngine().evaluate(data, OFFLINE, batch)
        with pytest.raises(ValueError, match="execution record"):
            build_report(data, OFFLINE, other, results)
        changed_config = OFFLINE.model_copy(update={"max_iterations": 1})
        with pytest.raises(ValueError, match="provenance"):
            build_report(data, changed_config, batch, results)

    asyncio.run(exercise())


def test_script_fingerprint_covers_selected_responses_in_order():
    async def exercise():
        data = dataset()
        first = LLMResponse("first", "stop")
        second = LLMResponse("second", "stop")
        normal = await scripted(first, second).run(data, OFFLINE)
        reordered = await scripted(second, first).run(data, OFFLINE)
        extra = await EvaluationRunner(scripts={
            "script": {"same": (first, second), "not_selected": (LLMResponse("extra", "stop"),)},
        }).run(data, OFFLINE)
        assert normal.provenance.script_sha256 != reordered.provenance.script_sha256
        assert normal.provenance.script_sha256 == extra.provenance.script_sha256

    asyncio.run(exercise())
