import json

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from e6_structured_output import EXAMPLE, INVALID, QUESTION, PaperDecision, review_paper
from mock_deepseek import fixed_completion


def test_pydantic_schema_and_local_validation():
    schema = PaperDecision.model_json_schema()
    assert set(schema["required"]) == {"paper_title", "needs_review", "reason"}
    assert schema["properties"]["needs_review"]["type"] == "boolean"
    assert schema["properties"]["paper_title"]["type"] == "string"
    assert schema["properties"]["reason"]["type"] == "string"
    decision = PaperDecision.model_validate(EXAMPLE)
    assert decision.paper_title == EXAMPLE["paper_title"]
    assert decision.needs_review is True
    assert decision.reason == EXAMPLE["reason"]
    with pytest.raises(ValidationError) as invalid:
        PaperDecision.model_validate(INVALID)
    assert invalid.value.errors()[0]["loc"] == ("needs_review",)
    assert invalid.value.errors()[0]["type"] == "bool_type"


def test_json_mode_returns_raw_and_parsed_on_success():
    content = json.dumps(EXAMPLE)
    with fixed_completion({"content": content}) as (model, requests):
        result = review_paper(model, QUESTION)

    (request,) = requests
    assert request["response_format"] == {"type": "json_object"}
    assert "JSON" in request["messages"][0]["content"]
    assert "tools" not in request
    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].content == content
    assert isinstance(result["parsed"], PaperDecision)
    assert result["parsed"].model_dump() == EXAMPLE
    assert result["parsing_error"] is None


@pytest.mark.parametrize(
    "content",
    [json.dumps(INVALID), "invalid JSON"],
    ids=["valid-json-invalid-schema", "malformed-json"],
)
def test_json_mode_returns_raw_and_error_when_parsing_fails(content):
    with fixed_completion({"content": content}) as (model, _):
        result = review_paper(model, QUESTION)

    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].content == content
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)


@pytest.mark.parametrize("method", ["function_calling", "json_schema"])
def test_adapter_sends_tool_schema_for_function_mode_and_json_schema_alias(method):
    reply = {
        "content": None,
        "tool_calls": [
            {
                "id": "paper-1",
                "type": "function",
                "function": {"name": "PaperDecision", "arguments": json.dumps(EXAMPLE)},
            }
        ],
    }
    with fixed_completion(reply, finish_reason="tool_calls") as (model, requests):
        decision = model.with_structured_output(PaperDecision, method=method).invoke(QUESTION)

    assert isinstance(decision, PaperDecision)
    assert decision.model_dump() == EXAMPLE
    (request,) = requests
    (tool_def,) = request["tools"]
    assert tool_def["function"]["name"] == "PaperDecision"
    assert set(tool_def["function"]["parameters"]["required"]) == set(EXAMPLE)
    assert "response_format" not in request
