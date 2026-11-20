import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from e4_tool_calling import get_weather
from e5_tool_node import execute_calls


def tool_call(call_id, args):
    return {"name": get_weather.name, "args": args, "id": call_id}


def test_single_call_has_result_and_matching_id():
    request = AIMessage(content="", tool_calls=[tool_call("one", {"city": "Beijing"})])
    (result,) = execute_calls(request, ToolNode([get_weather]))
    assert isinstance(result, ToolMessage)
    assert result.name == get_weather.name
    assert (result.tool_call_id, result.status, result.content) == ("one", "success", "sunny")


def test_multiple_calls_match_results_by_id():
    request = AIMessage(
        content="",
        tool_calls=[
            tool_call("clouds", {"city": "Shanghai"}),
            tool_call("sun", {"city": "Beijing"}),
        ],
    )
    outputs = execute_calls(request, ToolNode([get_weather]))
    assert len(outputs) == 2
    assert all(isinstance(message, ToolMessage) for message in outputs)
    by_call_id = {
        message.tool_call_id: (message.status, message.content) for message in outputs
    }
    assert by_call_id == {
        "clouds": ("success", "cloudy"),
        "sun": ("success", "sunny"),
    }


def test_missing_parameter_returns_error_tool_message():
    request = AIMessage(content="", tool_calls=[tool_call("missing-city", {})])
    (result,) = execute_calls(request, ToolNode([get_weather]))
    assert isinstance(result, ToolMessage)
    assert (result.tool_call_id, result.status) == ("missing-city", "error")
    assert "city" in result.content and "Field required" in result.content


def test_tool_execution_exception_propagates_by_default():
    request = AIMessage(content="", tool_calls=[tool_call("unknown", {"city": "Unknown"})])
    with pytest.raises(ValueError, match="No simulated weather for Unknown"):
        execute_calls(request, ToolNode([get_weather]))


def test_configured_error_handler_turns_exception_into_message():
    request = AIMessage(content="", tool_calls=[tool_call("unknown", {"city": "Unknown"})])
    node = ToolNode([get_weather], handle_tool_errors=ValueError)
    (result,) = execute_calls(request, node)
    assert isinstance(result, ToolMessage)
    assert (result.tool_call_id, result.status) == ("unknown", "error")
    assert "No simulated weather for Unknown" in result.content
