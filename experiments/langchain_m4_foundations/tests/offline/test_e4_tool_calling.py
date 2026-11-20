import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from e4_tool_calling import EXAMPLE_CALL, QUESTION, get_weather, request_weather
from mock_deepseek import fixed_completion


def test_weather_metadata_schema_and_standalone_execution():
    schema = get_weather.get_input_schema().model_json_schema()
    assert get_weather.name == "get_weather"
    assert "simulated weather" in get_weather.description
    assert schema["required"] == ["city"]
    assert schema["properties"]["city"]["type"] == "string"
    assert get_weather.invoke({"city": "Beijing"}) == "sunny"
    with pytest.raises(ValidationError, match="city"):
        get_weather.invoke({})


def test_bind_tools_sends_definition_and_parses_call_without_execution(monkeypatch):
    def unexpected_call(city):
        pytest.fail(f"bind_tools must not execute the weather handler for {city}")

    monkeypatch.setattr(get_weather, "func", unexpected_call)
    with fixed_completion(
        {"content": None, "tool_calls": [EXAMPLE_CALL]}, finish_reason="tool_calls"
    ) as (model, requests):
        response = request_weather(model, QUESTION)

    (request,) = requests
    assert request["messages"] == [{"role": "user", "content": QUESTION}]
    (tool_def,) = request["tools"]
    assert tool_def["type"] == "function"
    assert tool_def["function"]["name"] == get_weather.name
    assert tool_def["function"]["description"] == get_weather.description
    schema = tool_def["function"]["parameters"]
    assert schema["required"] == ["city"]
    assert schema["properties"]["city"]["type"] == "string"
    assert isinstance(response, AIMessage)
    assert response.tool_calls == [
        {
            "name": "get_weather",
            "args": {"city": "Beijing"},
            "id": "weather-1",
            "type": "tool_call",
        },
    ]
