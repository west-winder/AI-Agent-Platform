import asyncio
from importlib.metadata import version

from langchain_core.messages import HumanMessage, ToolMessage
from langsmith import tracing_context

from agent_graph import build_agent_graph
from fixture_model import FixtureChatModel
from stdio_adapter import FIXTURE_SCRIPT, web_page_fetch_adapter


async def main() -> None:
    print(f"langchain={version('langchain')} fastmcp={version('fastmcp')} client_mcp={version('mcp')}")
    with tracing_context(enabled=False):
        async with asyncio.timeout(25), web_page_fetch_adapter(str(FIXTURE_SCRIPT)) as adapter:
            (tool,) = await adapter.list_tools()
            print(f"Server: {adapter.client.server_info.name} | protocol={adapter.client.protocol_version}")
            print(f"Tool: {tool.name} — {tool.description}")
            print("Input schema:", tool.args_schema)

            for url in ("https://example.com/", "https://example.com/failure"):
                model = FixtureChatModel(url=url, cache=False)
                graph = build_agent_graph(model, [tool])
                state = await graph.ainvoke({"messages": [HumanMessage(content="读取网页正文。")]})
                result = state["messages"][-2]
                assert isinstance(result, ToolMessage)
                print(f"\nToolMessage id={result.tool_call_id} status={result.status}")
                print("Structured content:" if result.artifact else "Tool error:", result.artifact or result.text)
                print("LLM answer:", state["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
