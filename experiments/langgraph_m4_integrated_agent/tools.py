import json
from typing import Protocol

from langchain_core.tools import BaseTool, StructuredTool, ToolException, tool

from backend.rag.contracts import Chunk, RagRetrievedChunk


class Retriever(Protocol):
    def retrieve(
        self, query: str, chunk_store: dict[str, Chunk], top_k: int = 5
    ) -> list[RagRetrievedChunk]: ...


def build_rag_tool(retriever: Retriever, chunk_store: dict[str, Chunk]) -> BaseTool:
    @tool
    def retrieve_knowledge(query: str) -> str:
        """Look up passages in the local knowledge base for the given query."""
        chunks = retriever.retrieve(query=query, chunk_store=chunk_store, top_k=5)
        documents = [
            {"chunk_id": chunk.chunk_id, "source": chunk.source, "content": chunk.content}
            for chunk in chunks
        ]
        return json.dumps({"documents": documents}, ensure_ascii=False)

    return retrieve_knowledge


WEATHER = {"Beijing": "sunny", "Shanghai": "cloudy"}


def get_weather(city: str) -> str:
    """Look up simulated weather for a city by its exact name."""
    if city not in WEATHER:
        raise ToolException(f"No simulated weather for {city}")
    return WEATHER[city]


weather_tool = StructuredTool.from_function(get_weather, handle_tool_error=True)
