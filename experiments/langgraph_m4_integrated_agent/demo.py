import asyncio

from langchain_core.messages import HumanMessage, ToolMessage
from langsmith import tracing_context

from backend.rag.contracts import Chunk, RagRetrievedChunk

from .agent_graph import build_integrated_agent
from .fakes import FakeRetriever, ScriptedChatModel, memory_reply, tool_request
from .memory import FakeMemoryService
from .stdio_adapter import web_page_fetch_adapter
from .tools import build_rag_tool, weather_tool


async def main() -> None:
    chunk = Chunk(chunk_id="backup-1", document_id="backup-policy", content="Backups are retained for 30 days.")
    retriever = FakeRetriever({"backup retention": [RagRetrievedChunk(
        **chunk.model_dump(), source="backup-policy.md", retrieval_rank=1
    )]})
    rag_tool = build_rag_tool(retriever, {chunk.chunk_id: chunk})
    memory = FakeMemoryService({1: ["I study Python with notebooks."]})
    preference = "I always prefer short Python examples."
    question = f"{preference} Check backup retention, the fixture page, and Beijing weather."
    model = ScriptedChatModel(messages=iter([
        tool_request(
            ("rag-1", "retrieve_knowledge", {"query": "backup retention"}),
            ("web-1", "fetch_web_page", {"url": "https://example.com/"}),
            ("weather-1", "get_weather", {"city": "Beijing"}),
        ),
        "Backups are retained for 30 days. The fixture contains Fixture 正文. Beijing is sunny.",
    ]), cache=False)
    extractor = ScriptedChatModel(messages=iter([memory_reply(True, preference)]), cache=False)

    with tracing_context(enabled=False):
        async with asyncio.timeout(25), web_page_fetch_adapter() as adapter:
            (web_tool,) = await adapter.list_tools()
            graph = build_integrated_agent(model, [rag_tool, web_tool, weather_tool], memory, extractor)
            state = await graph.ainvoke({"user_id": 1, "messages": [HumanMessage(content=question)]})
            print(f"MCP protocol={adapter.client.protocol_version}, discovered={web_tool.name}")

    print("Memory context:", state["memory_context"])
    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            print(f"ToolMessage {message.tool_call_id} ({message.status}): {message.text}")
    print("Answer:", state["final_answer"])
    print(f"llm_calls={state['llm_calls']} stop_reason={state['stop_reason']} memory_saved={state['memory_saved']}")
    print("Saved memory, user 1:", repr(await memory.load_context(1, "Python")))
    print("Saved memory, user 2:", repr(await memory.load_context(2, "Python")))


if __name__ == "__main__":
    asyncio.run(main())
