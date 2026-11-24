import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from mcp import MCPError
from mcp_types import CONNECTION_CLOSED

from backend.rag.contracts import Chunk, RagRetrievedChunk
from experiments.langgraph_m4_integrated_agent.agent_graph import (
    RECURSION_LIMIT,
    SKIPPED_TOOL,
    build_integrated_agent,
)
from experiments.langgraph_m4_integrated_agent.fakes import (
    FakeRetriever,
    ScriptedChatModel,
    memory_reply,
    tool_request,
)
from experiments.langgraph_m4_integrated_agent.memory import FakeMemoryService
from experiments.langgraph_m4_integrated_agent.stdio_adapter import SERVER_ROOT, web_page_fetch_adapter
from experiments.langgraph_m4_integrated_agent.tools import build_rag_tool, weather_tool


PREFERENCE = "I always prefer short Python examples."
PAGE_URL = "https://example.com/"


def scripted(responses):
    return ScriptedChatModel(messages=iter(responses), cache=False)


def invoke(graph, query, user_id=1):
    return asyncio.run(graph.ainvoke({"user_id": user_id, "messages": [HumanMessage(content=query)]}))


@pytest.fixture
def rag_data():
    chunk = Chunk(chunk_id="backup-1", document_id="policy", content="Backups are retained for 30 days.")
    hit = RagRetrievedChunk(**chunk.model_dump(), source="policy.md", retrieval_rank=1)
    return {chunk.chunk_id: chunk}, hit


def test_three_tools_return_to_same_llm_turn_and_memory_is_saved(rag_data, monkeypatch):
    chunk_store, hit = rag_data
    retriever = FakeRetriever({"retention": [hit]})
    service = FakeMemoryService({1: ["I study Python with notebooks."], 2: ["I use Python for games."]})
    writer = AsyncMock(wraps=service.save_memory)
    monkeypatch.setattr(service, "save_memory", writer)
    model = scripted([tool_request(
        ("rag-1", "retrieve_knowledge", {"query": "retention"}),
        ("web-1", "fetch_web_page", {"url": PAGE_URL}),
        ("weather-1", "get_weather", {"city": "Shanghai"}),
    ), "Backups last 30 days, the fixture has readable text, and Shanghai is cloudy."])
    extractor = scripted([memory_reply(True, PREFERENCE)])
    query = f"{PREFERENCE} Check retention, the fixture page, and Shanghai weather."

    async def run():
        async with asyncio.timeout(25), web_page_fetch_adapter() as adapter:
            assert adapter.client.protocol_version == "2025-11-25"
            (web_tool,) = await adapter.list_tools()
            graph = build_integrated_agent(model, [build_rag_tool(retriever, chunk_store), web_tool, weather_tool],
                                           service, extractor)
            state = await graph.ainvoke({"user_id": 1, "messages": [HumanMessage(content=query, id="user-1")]})

        human, request, rag, web, weather, answer = state["messages"]
        assert isinstance(human, HumanMessage) and human.content == query
        assert isinstance(request, AIMessage) and isinstance(answer, AIMessage)
        assert all(isinstance(message, ToolMessage) for message in (rag, web, weather))
        ids = [call["id"] for call in request.tool_calls]
        assert ids == [rag.tool_call_id, web.tool_call_id, weather.tool_call_id] == ["rag-1", "web-1", "weather-1"]
        assert [(result.name, result.status) for result in (rag, web, weather)] == [
            ("retrieve_knowledge", "success"), ("fetch_web_page", "success"), ("get_weather", "success")
        ]
        assert json.loads(rag.text) == {"documents": [{"chunk_id": hit.chunk_id, "source": hit.source, "content": hit.content}]}
        page = {"title": "Fixture 标题", "url": PAGE_URL, "content": "Fixture 正文",
                "truncated": False, "content_type": "text/html"}
        assert web.artifact == {"structured_content": page}
        assert json.loads(web.text) == page
        assert weather.content == "cloudy"
        assert retriever.calls == [("retention", chunk_store, 5)] and retriever.calls[0][1] is chunk_store
        assert state["final_answer"] == answer.content and state["stop_reason"] == "answered"
        assert state["llm_calls"] == len(model.requests) == 2
        assert state["memory_context"] == "I study Python with notebooks."
        assert state["memory_saved"] is True
        assert set(state) == {
            "messages", "user_id", "memory_context", "llm_calls", "final_answer", "stop_reason", "memory_saved"
        }
        (first, schemas), (second, _) = model.requests
        assert isinstance(first[0], SystemMessage) and first[1:] == [human]
        assert first[0] == second[0] and state["memory_context"] in first[0].content
        assert second[1:] == [human, request, rag, web, weather]
        assert [definition["function"]["name"] for definition in schemas] == [
            "retrieve_knowledge", "fetch_web_page", "get_weather"
        ]
        (extractor_prompt, extractor_schema), = extractor.requests
        assert extractor_prompt[1:] == [HumanMessage(content=query)]
        assert extractor_schema[0]["function"]["name"] == "MemoryDecision"
        writer.assert_awaited_once_with(user_id=1, content=PREFERENCE)
        assert await service.load_context(1, "examples") == PREFERENCE
        assert await service.load_context(2, "examples") == ""

    asyncio.run(run())


def test_mcp_tool_error_is_a_failed_tool_message():
    model = scripted([tool_request(("web-error", "fetch_web_page", {"url": PAGE_URL + "failure"})),
                      "The web fetch failed."])
    extractor = scripted([memory_reply(False, None)])

    async def run():
        async with asyncio.timeout(25), web_page_fetch_adapter() as adapter:
            graph = build_integrated_agent(model, await adapter.list_tools(), FakeMemoryService(), extractor)
            state = await graph.ainvoke({"user_id": 1, "messages": [HumanMessage(content="Fetch the page.")]})

        _, request, failed, answer = state["messages"]
        assert isinstance(failed, ToolMessage) and failed.status == "error"
        assert request.tool_calls[0]["id"] == failed.tool_call_id == "web-error"
        assert "NETWORK_ERROR" in failed.text and failed.artifact is None
        assert model.requests[1][0][-1] == failed
        assert answer.content == state["final_answer"] == "The web fetch failed."
        assert state["memory_saved"] is False

    asyncio.run(run())


def test_mcp_connection_loss_propagates_without_fabricated_result():
    entry = SERVER_ROOT.parents[1] / "experiments/langgraph_m4_mcp_integration/tests/disconnecting_server.py"
    model = scripted([tool_request(("web-closed", "fetch_web_page", {"url": PAGE_URL}))])
    extractor = scripted([])

    async def run():
        async with asyncio.timeout(25), web_page_fetch_adapter(entry) as adapter:
            graph = build_integrated_agent(model, await adapter.list_tools(), FakeMemoryService(), extractor)
            with pytest.raises(MCPError, match="Connection closed") as raised:
                await graph.ainvoke({"user_id": 1, "messages": [HumanMessage(content="Fetch the page.")]})
        assert raised.value.code == CONNECTION_CLOSED
        assert len(model.requests) == 1 and extractor.requests == []

    asyncio.run(run())


def test_empty_rag_returns_successful_empty_collection():
    model = scripted([tool_request(("rag-empty", "retrieve_knowledge", {"query": "missing"})),
                      "No matching knowledge-base passages."])
    retriever = FakeRetriever({"missing": []})
    graph = build_integrated_agent(model, [build_rag_tool(retriever, {})], FakeMemoryService(),
                                   scripted([memory_reply(False, None)]))

    state = invoke(graph, "Search for missing evidence.")

    _, request, result, _ = state["messages"]
    assert isinstance(result, ToolMessage) and result.status == "success"
    assert request.tool_calls[0]["id"] == result.tool_call_id == "rag-empty"
    assert json.loads(result.text) == {"documents": []}
    assert model.requests[1][0][-1] == result
    assert state["llm_calls"] == 2


def test_rag_retrieval_exception_reaches_caller_unchanged():
    error = RuntimeError("retriever offline")
    retriever = FakeRetriever({"failure": error})
    model = scripted([tool_request(("rag-error", "retrieve_knowledge", {"query": "failure"}))])
    extractor = scripted([])
    graph = build_integrated_agent(model, [build_rag_tool(retriever, {})], FakeMemoryService(), extractor)

    with pytest.raises(RuntimeError, match="retriever offline") as raised:
        invoke(graph, "Search the local knowledge base.")

    assert raised.value is error
    assert retriever.calls == [("failure", {}, 5)]
    assert len(model.requests) == 1 and extractor.requests == []


def test_budget_preserves_results_and_pairs_every_unexecuted_call(rag_data):
    chunk_store, hit = rag_data
    retriever = FakeRetriever({"retention": [hit]})
    model = scripted([
        tool_request(("already-rag", "retrieve_knowledge", {"query": "retention"})),
        tool_request(("already-weather", "get_weather", {"city": "Beijing"})),
        tool_request(
            ("pending-1", "retrieve_knowledge", {"query": "must not run 1"}),
            ("pending-2", "retrieve_knowledge", {"query": "must not run 2"}),
        ),
    ])
    extractor = scripted([memory_reply(False, None)])
    graph = build_integrated_agent(model, [build_rag_tool(retriever, chunk_store), weather_tool],
                                   FakeMemoryService(), extractor)

    state = invoke(graph, "Search retention and weather.")

    assert graph.config["recursion_limit"] == RECURSION_LIMIT
    assert state["llm_calls"] == len(model.requests) == 3 and state["stop_reason"] == "llm_limit"
    assert retriever.calls == [("retention", chunk_store, 5)]
    _, _, rag, _, weather, last_request, *pending = state["messages"]
    assert rag.tool_call_id == "already-rag" and hit.content in rag.text
    assert weather.tool_call_id == "already-weather" and weather.content == "sunny"
    assert [call["id"] for call in last_request.tool_calls] == ["pending-1", "pending-2"]
    assert all(isinstance(result, ToolMessage) for result in pending)
    assert [(result.tool_call_id, result.status, result.content) for result in pending] == [
        ("pending-1", "error", SKIPPED_TOOL), ("pending-2", "error", SKIPPED_TOOL)
    ]
    assert "已保留工具结果 already-rag、already-weather" in state["final_answer"]
    assert f"pending-2 {SKIPPED_TOOL}" in state["final_answer"]
    assert model.requests[2][0][-1] == weather
    assert extractor.requests[0][0][1:] == [HumanMessage(content="Search retention and weather.")]


def test_reused_graph_reads_only_current_users_memory_and_starts_fresh():
    memory = FakeMemoryService({2: ["I use Python for web apps."]})
    model = scripted(["Saved.", "Web app example.", "Short example."])
    extractor = scripted([memory_reply(True, PREFERENCE), memory_reply(False, None), memory_reply(False, None)])
    graph = build_integrated_agent(model, [weather_tool], memory, extractor)
    first = invoke(graph, PREFERENCE, user_id=1)
    other = invoke(graph, "Show Python examples.", user_id=2)
    again = invoke(graph, "Show Python examples.", user_id=1)

    assert first["memory_context"] == "" and first["memory_saved"] is True
    assert other["memory_context"] == "I use Python for web apps." and other["final_answer"] == "Web app example."
    assert again["memory_context"] == PREFERENCE and again["final_answer"] == "Short example."
    for state in (other, again):
        assert state["llm_calls"] == 1 and state["memory_saved"] is False
        assert len(state["messages"]) == 2 and state["messages"][0].content == "Show Python examples."


def test_memory_validation_rejects_fact_from_history_and_answer(monkeypatch):
    memory = FakeMemoryService()
    writer = AsyncMock(wraps=memory.save_memory)
    monkeypatch.setattr(memory, "save_memory", writer)
    model = scripted([PREFERENCE])
    extractor = scripted([memory_reply(True, PREFERENCE)])
    graph = build_integrated_agent(model, [weather_tool], memory, extractor)
    history = [HumanMessage(content=PREFERENCE), AIMessage(content="Okay."), HumanMessage(content="Hello!")]

    with pytest.raises(ValueError, match="current user message"):
        asyncio.run(graph.ainvoke({"user_id": 1, "messages": history}))

    assert extractor.requests[0][0][1:] == [HumanMessage(content="Hello!")]
    writer.assert_not_called()


def test_memory_write_exception_reaches_graph_caller(monkeypatch):
    memory = FakeMemoryService()
    error = RuntimeError("fake write failed")
    writer = AsyncMock(side_effect=error)
    monkeypatch.setattr(memory, "save_memory", writer)
    graph = build_integrated_agent(scripted(["Okay."]), [weather_tool], memory,
                                   scripted([memory_reply(True, PREFERENCE)]))

    with pytest.raises(RuntimeError, match="fake write failed") as raised:
        invoke(graph, PREFERENCE)

    assert raised.value is error
    writer.assert_awaited_once_with(user_id=1, content=PREFERENCE)
