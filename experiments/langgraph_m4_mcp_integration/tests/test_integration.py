import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp import MCPError
from mcp_types import CONNECTION_CLOSED

from agent_graph import build_agent_graph
from fixture_model import FixtureChatModel
from stdio_adapter import FIXTURE_SCRIPT, web_page_fetch_adapter


DESCRIPTION = "Fetch readable text from a public HTTP/HTTPS HTML page without JavaScript."
PAGE = {
    "title": "Fixture 标题",
    "url": "https://example.com/",
    "content": "Fixture 正文",
    "truncated": False,
    "content_type": "text/html",
}


def test_production_stdio_negotiates_and_exposes_tool_schema():
    async def run():
        async with asyncio.timeout(25), web_page_fetch_adapter() as adapter:
            assert adapter.client.server_info.name == "Web Page Fetch MCP Server"
            assert adapter.client.protocol_version == "2025-11-25"
            (tool,) = await adapter.list_tools()
            assert tool.name == "fetch_web_page"
            assert tool.description == DESCRIPTION
            assert tool.args_schema["required"] == ["url"]
            assert set(tool.args_schema["properties"]) == {"url"}
            assert tool.args_schema["properties"]["url"]["type"] == "string"

    asyncio.run(run())


def test_fixture_tool_roundtrip_preserves_structured_content_and_call_id():
    async def run():
        async with asyncio.timeout(25), web_page_fetch_adapter(str(FIXTURE_SCRIPT)) as adapter:
            tools = await adapter.list_tools()
            model = FixtureChatModel(url=PAGE["url"], cache=False)
            graph = build_agent_graph(model, tools)
            question = HumanMessage(content="读取网页正文。", id="question")
            state = await graph.ainvoke({"messages": [question]})

        human, request, result, answer = state["messages"]
        assert human == question
        assert isinstance(request, AIMessage)
        (call,) = request.tool_calls
        assert (call["name"], call["args"], call["id"]) == (
            "fetch_web_page", {"url": PAGE["url"]}, "fetch-1"
        )
        assert isinstance(result, ToolMessage)
        assert (result.name, result.tool_call_id, result.status) == ("fetch_web_page", "fetch-1", "success")
        assert result.artifact == {"structured_content": PAGE}
        assert json.loads(result.text) == PAGE
        assert isinstance(answer, AIMessage) and answer.content == PAGE["content"]
        (first_input, schemas), (followup, _) = model.requests
        assert first_input == [human]
        assert followup == [human, request, result]
        (definition,) = schemas
        assert definition["function"]["name"] == "fetch_web_page"
        assert definition["function"]["description"] == DESCRIPTION
        assert definition["function"]["parameters"]["required"] == ["url"]
        assert definition["function"]["parameters"]["properties"]["url"]["type"] == "string"

    asyncio.run(run())


def test_server_is_error_reaches_model_as_failed_tool_message(capfd):
    async def run():
        url = "https://example.com/failure"
        async with asyncio.timeout(25), web_page_fetch_adapter(str(FIXTURE_SCRIPT)) as adapter:
            tools = await adapter.list_tools()
            raw = await adapter.client.call_tool("fetch_web_page", {"url": url}, raise_on_error=False)
            assert raw.is_error
            model = FixtureChatModel(url=url, cache=False)
            graph = build_agent_graph(model, tools)
            state = await graph.ainvoke({"messages": [HumanMessage(content="读取网页正文。")]})

        request, result, answer = state["messages"][-3:]
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == request.tool_calls[0]["id"] == "fetch-1"
        assert result.artifact is None
        assert "NETWORK_ERROR" in result.text
        assert answer.content.startswith("抓取失败：") and "NETWORK_ERROR" in answer.content
        assert len(model.requests) == 2 and model.requests[1][0][-1] == result
        assert "raw_secret_canary" not in result.text

    asyncio.run(run())
    stderr = capfd.readouterr().err
    assert "fetch_failed code=NETWORK_ERROR" in stderr
    assert "raw_secret_canary" not in stderr and "Traceback" not in stderr


def test_stdio_disconnect_during_tool_call_propagates_to_graph_caller():
    async def run():
        entry = Path(__file__).with_name("disconnecting_server.py")
        async with asyncio.timeout(25), web_page_fetch_adapter(str(entry)) as adapter:
            tools = await adapter.list_tools()
            model = FixtureChatModel(url=PAGE["url"], cache=False)
            graph = build_agent_graph(model, tools)

            with pytest.raises(MCPError, match="Connection closed") as raised:
                await graph.ainvoke({"messages": [HumanMessage(content="读取网页正文。")]})

            assert raised.value.code == CONNECTION_CLOSED
            assert len(model.requests) == 1

    asyncio.run(run())
