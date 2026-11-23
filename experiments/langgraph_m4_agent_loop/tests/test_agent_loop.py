from unittest.mock import call

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent_loop import RECURSION_LIMIT, SKIPPED_TOOL, build_agent_graph
from scripted_model import ScriptedChatModel, weather_request


def test_tool_result_returns_to_model_with_same_call_id(weather_handler):
    question = HumanMessage(content="What is the weather in Beijing?", id="human-1")
    answer = "The simulated weather in Beijing is sunny."
    model = ScriptedChatModel(
        messages=iter([weather_request(("weather-1", "Beijing")), answer]), cache=False
    )
    graph = build_agent_graph(model)
    assert not model.requests

    result = graph.invoke({"messages": [question]})

    human, request, tool_result, reply = result["messages"]
    assert human == question
    assert isinstance(request, AIMessage)
    assert request.tool_calls[0]["id"] == "weather-1"
    assert isinstance(tool_result, ToolMessage)
    assert (tool_result.tool_call_id, tool_result.status, tool_result.content) == (
        "weather-1", "success", "sunny"
    )
    assert isinstance(reply, AIMessage) and reply.content == answer
    assert result["final_answer"] == f"{answer}\n成功结果 [weather-1] sunny"
    assert result["stop_reason"] == "answered"
    assert result["llm_calls"] == len(model.requests) == 2
    (initial, schemas), (followup, followup_schemas) = model.requests
    assert isinstance(initial[0], SystemMessage) and initial[1:] == [question]
    assert followup == [initial[0], human, request, tool_result]
    assert schemas == followup_schemas
    (schema,) = schemas
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["parameters"]["required"] == ["city"]
    weather_handler.assert_called_once_with(city="Beijing")


def test_tool_value_error_is_visible_to_model_and_final_answer(weather_handler):
    answer = "The lookup failed. The simulated weather is unavailable."
    model = ScriptedChatModel(
        messages=iter([weather_request(("unknown-1", "Unknown")), answer]), cache=False
    )

    result = build_agent_graph(model).invoke({"messages": [HumanMessage(content="Unknown weather")]})

    failure = model.requests[1][0][-1]
    assert isinstance(failure, ToolMessage)
    assert failure == result["messages"][-2]
    assert (failure.tool_call_id, failure.status) == ("unknown-1", "error")
    assert "No simulated weather for Unknown" in failure.content
    assert answer in result["final_answer"]
    assert "工具失败 [unknown-1]" in result["final_answer"]
    assert "成功结果" not in result["final_answer"]
    assert result["stop_reason"] == "answered"
    assert result["llm_calls"] == len(model.requests) == 2
    weather_handler.assert_called_once_with(city="Unknown")


def test_at_llm_limit_records_every_pending_call_without_executing_it(weather_handler):
    model = ScriptedChatModel(
        messages=iter([
            weather_request(("first", "Beijing")),
            weather_request(("failed", "Unknown")),
            weather_request(("pending-1", "Shanghai"), ("pending-2", "Beijing")),
        ]),
        cache=False,
    )

    result = build_agent_graph(model).invoke({"messages": [HumanMessage(content="Weather for three cities")]})

    assert result["stop_reason"] == "llm_limit"
    assert result["llm_calls"] == len(model.requests) == 3
    assert weather_handler.call_args_list == [call(city="Beijing"), call(city="Unknown")]
    last_request, *skipped = result["messages"][-3:]
    assert isinstance(last_request, AIMessage)
    assert [call["id"] for call in last_request.tool_calls] == ["pending-1", "pending-2"]
    assert all(isinstance(message, ToolMessage) for message in skipped)
    assert [(message.tool_call_id, message.status, message.content) for message in skipped] == [
        ("pending-1", "error", SKIPPED_TOOL),
        ("pending-2", "error", SKIPPED_TOOL),
    ]
    feedback = [request[-1] for request, _ in model.requests[1:]]
    assert [(message.tool_call_id, message.status) for message in feedback] == [
        ("first", "success"), ("failed", "error")
    ]
    assert "任务未完成" in result["final_answer"]
    assert "成功结果 [first] sunny" in result["final_answer"]
    assert "工具失败 [failed]" in result["final_answer"]
    assert f"未完成 [pending-1] {SKIPPED_TOOL}" in result["final_answer"]
    assert f"未完成 [pending-2] {SKIPPED_TOOL}" in result["final_answer"]


def test_third_llm_call_can_finish_with_an_answer(weather_handler):
    model = ScriptedChatModel(
        messages=iter([
            weather_request(("beijing", "Beijing")),
            weather_request(("shanghai", "Shanghai")),
            "Beijing is sunny. Shanghai is cloudy.",
        ]),
        cache=False,
    )

    result = build_agent_graph(model).invoke({"messages": [HumanMessage(content="Weather for both cities")]})

    assert result["stop_reason"] == "answered"
    assert result["llm_calls"] == len(model.requests) == 3
    assert weather_handler.call_args_list == [call(city="Beijing"), call(city="Shanghai")]
    assert isinstance(result["messages"][-1], AIMessage)
    assert "成功结果 [shanghai] cloudy" in result["final_answer"]


def test_independent_invokes_isolate_state_while_sharing_model(weather_handler):
    model = ScriptedChatModel(
        messages=iter([weather_request(("first", "Beijing")), "Sunny.", "Hello!"]), cache=False
    )
    graph = build_agent_graph(model)
    first = graph.invoke({"messages": [HumanMessage(content="Beijing weather")]})
    greeting = HumanMessage(content="Hello", id="second-human")
    second = graph.invoke({"messages": [greeting]})

    assert first["llm_calls"] == 2 and len(first["messages"]) == 4
    assert second["llm_calls"] == 1 and second["final_answer"] == "Hello!"
    assert second["stop_reason"] == "answered"
    assert len(second["messages"]) == 2 and second["messages"][0] == greeting
    assert len(model.requests) == 3 and model.requests[2][0][1:] == [greeting]
    weather_handler.assert_called_once_with(city="Beijing")


def test_finish_does_not_summarize_input_history_tool_results():
    history = [
        HumanMessage(content="Earlier weather question"),
        weather_request(("old-call", "Shanghai")),
        ToolMessage(content="cloudy", tool_call_id="old-call", status="success"),
        AIMessage(content="Shanghai is cloudy."),
    ]
    current = HumanMessage(content="Say hello")
    model = ScriptedChatModel(messages=iter(["Hello!"]), cache=False)

    result = build_agent_graph(model).invoke({"messages": [*history, current]})

    assert result["messages"][:-1] == [*history, current]
    assert model.requests[0][0][1:] == [*history, current]
    assert result["llm_calls"] == 1 and result["final_answer"] == "Hello!"


def test_graph_configures_recursion_limit_as_a_separate_guard():
    model = ScriptedChatModel(messages=iter([weather_request(("first", "Beijing"))]), cache=False)
    graph = build_agent_graph(model)
    assert graph.config["recursion_limit"] == RECURSION_LIMIT

    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage(content="Beijing weather")]}, config={"recursion_limit": 1})

    assert len(model.requests) == 1
