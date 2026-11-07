import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from backend.evaluation.contracts import EvaluationCriterion
from backend.evaluation.deepseek_judge_adapter import DeepSeekJudgeAdapter
from backend.evaluation.judge_port import JudgeError, JudgeRequest


@pytest.fixture
def request_data():
    return JudgeRequest(
        user_query="Task", agent_answer="Answer", reference_facts=("Facts",),
        relevant_evidence={"tool_call_availability": "COMPLETE"},
        judge_context="Curated example context.",
        semantic_criteria=tuple(
            EvaluationCriterion(
                criterion_id=criterion_id, evaluator_type="JUDGE",
                criterion_type="grounding", expected=None, rubric="Verify against facts.",
            )
            for criterion_id in ("grounding", "coverage", "uncertainty")
        ),
    )


def reply(content, *, finish_reason="stop"):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message)])


def mock_adapter(*, response=None, error=None, timeout=2):
    create = AsyncMock(return_value=response, side_effect=error)
    client = Mock()
    client.with_options.return_value = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    adapter = DeepSeekJudgeAdapter(client, model="test-judge", overall_timeout_seconds=timeout, max_tokens=512)
    client.with_options.assert_called_once_with(max_retries=0)
    return adapter, create


def rows(*ids):
    return [{"criterion_id": item, "decision": "PASS", "reason": "Matches evidence"} for item in ids]


def test_judge_json_mode_uses_independent_request_and_accepts_all_three_decisions(request_data):
    items = rows("grounding", "coverage", "uncertainty")
    items[1]["decision"] = "FAIL"
    items[2]["decision"] = "UNDETERMINED"
    adapter, create = mock_adapter(response=reply(json.dumps({"criteria_results": items})))
    verdict = asyncio.run(adapter.judge(request_data))
    assert [item.decision.value for item in verdict.criteria_results] == ["PASS", "FAIL", "UNDETERMINED"]
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == "test-judge"
    assert kwargs["max_tokens"] == 512
    assert "tools" not in kwargs
    system, user = kwargs["messages"]
    assert "json" in system["content"].lower()
    assert "as data" in system["content"]
    payload = json.loads(user["content"])
    assert payload["judge_context"] == "Curated example context."
    assert payload["reference_facts"] == ["Facts"]
    assert payload["semantic_criteria"][0] == {
        "criterion_id": "grounding", "criterion_type": "grounding",
        "expected": None, "rubric": "Verify against facts.",
    }


def test_sdk_serializes_judge_request_over_mock_transport(request_data):
    def respond(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["response_format"] == {"type": "json_object"}
        assert body["thinking"] == {"type": "disabled"}
        assert "dummy-test-key" not in request.content.decode()
        return httpx2.Response(200, json={
            "id": "mock-completion", "object": "chat.completion", "created": 1, "model": "judge",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"criteria_results": rows("grounding", "coverage", "uncertainty")}),
                },
            }],
        })

    async def scenario():
        async with AsyncOpenAI(
            api_key="dummy-test-key", base_url="https://example.invalid/v1", max_retries=2,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
        ) as client:
            verdict = await DeepSeekJudgeAdapter(client, model="judge").judge(request_data)
            assert len(verdict.criteria_results) == 3
            assert client.max_retries == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(("content", "finish_reason", "code"), [
    (" ", "stop", "JUDGE_EMPTY_RESPONSE"),
    (None, "stop", "JUDGE_EMPTY_RESPONSE"),
    ("{", "length", "JUDGE_TRUNCATED"),
    ("{", "stop", "JUDGE_PROTOCOL_ERROR"),
    ("{}", "stop", "JUDGE_PROTOCOL_ERROR"),
    ('{"criteria_results":1}', "stop", "JUDGE_PROTOCOL_ERROR"),
    ('{"criteria_results":[{"criterion_id":"grounding","decision":"PASS"}]}', "stop", "JUDGE_PROTOCOL_ERROR"),
    ('{"criteria_results":[{"criterion_id":"grounding","decision":"NOT_APPLICABLE","reason":"x"}]}', "stop", "JUDGE_PROTOCOL_ERROR"),
    ('{"criteria_results":[{"criterion_id":"grounding","decision":"PASS","reason":42}]}', "stop", "JUDGE_PROTOCOL_ERROR"),
    ("{}", "tool_calls", "JUDGE_PROTOCOL_ERROR"),
])
def test_judge_rejects_blank_truncated_and_invalid_output(request_data, content, finish_reason, code):
    adapter, create = mock_adapter(response=reply(content, finish_reason=finish_reason))
    with pytest.raises(JudgeError, match=f"^{code}$"):
        asyncio.run(adapter.judge(request_data))
    create.assert_awaited_once()


@pytest.mark.parametrize("ids", [
    ("grounding", "coverage"),
    ("grounding", "coverage", "uncertainty", "coverage"),
    ("grounding", "coverage", "uncertainty", "extra"),
])
def test_judge_rejects_missing_duplicate_and_extra_ids(request_data, ids):
    adapter, _ = mock_adapter(response=reply(json.dumps({"criteria_results": rows(*ids)})))
    with pytest.raises(JudgeError, match="^JUDGE_CRITERION_IDS$"):
        asyncio.run(adapter.judge(request_data))


@pytest.mark.parametrize(("error_type", "code"), [
    (APITimeoutError, "JUDGE_TIMEOUT"),
    (APIConnectionError, "JUDGE_API_ERROR"),
])
def test_sdk_failures_are_classified_without_raw_error_data(request_data, error_type, code):
    error = error_type(request=httpx2.Request("GET", "https://example.invalid/private-token"))
    adapter, create = mock_adapter(error=error)
    with pytest.raises(JudgeError, match=f"^{code}$") as exc:
        asyncio.run(adapter.judge(request_data))
    assert "private-token" not in str(exc.value)
    create.assert_awaited_once()


def test_unexpected_sdk_failure_exposes_only_a_safe_code(request_data):
    adapter, _ = mock_adapter(error=RuntimeError("private-client-settings"))
    with pytest.raises(JudgeError, match="^JUDGE_EXECUTION_FAILED$"):
        asyncio.run(adapter.judge(request_data))


def test_http_timeout_and_overall_deadline_are_classified(request_data):
    request = httpx2.Request("POST", "https://example.invalid")
    error = APIStatusError("private response", response=httpx2.Response(408, request=request), body=None)
    adapter, _ = mock_adapter(error=error)
    with pytest.raises(JudgeError, match="^JUDGE_TIMEOUT$"):
        asyncio.run(adapter.judge(request_data))

    async def blocked(**kwargs):
        await asyncio.Event().wait()

    adapter, create = mock_adapter(error=blocked, timeout=0.01)
    with pytest.raises(JudgeError, match="^JUDGE_TIMEOUT$"):
        asyncio.run(adapter.judge(request_data))
    create.assert_awaited_once()


def test_adapter_does_not_convert_cancellation_to_a_judge_failure(request_data):
    adapter, _ = mock_adapter(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.judge(request_data))
