from backend.rag.contracts import Chunk, RagRetrievedChunk


class FakeRetriever:
    """按问题返回预设的 Stage 2 结果，并保留实际调用参数。"""

    def __init__(self, results: dict[str, list[RagRetrievedChunk] | Exception]):
        self.results = results
        self.calls: list[tuple[str, dict[str, Chunk], int]] = []

    def retrieve(
        self, query: str, chunk_store: dict[str, Chunk], top_k: int = 5
    ) -> list[RagRetrievedChunk]:
        self.calls.append((query, chunk_store, top_k))
        result = self.results[query]
        if isinstance(result, Exception):
            raise result
        return result
