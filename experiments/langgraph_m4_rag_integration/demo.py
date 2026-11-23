from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langsmith import tracing_context

from backend.rag.contracts import Chunk, RagRetrievedChunk

from .fake_retriever import FakeRetriever
from .rag_graph import build_rag_graph


def main() -> None:
    chunk = Chunk(
        chunk_id="retention-1",
        document_id="backup-policy",
        content="Daily backups are retained for 30 days.",
    )
    hit = RagRetrievedChunk(
        **chunk.model_dump(), source="backup-policy.md", retrieval_rank=1
    )
    normal = "How long are daily backups retained?"
    empty = "What is the weekend support number?"
    failed = "What is the backup schedule?"
    retriever = FakeRetriever({normal: [hit], empty: [], failed: RuntimeError("模拟检索故障")})
    model = FakeListChatModel(responses=["Daily backups are retained for 30 days."], cache=False)
    graph = build_rag_graph(retriever, {chunk.chunk_id: chunk}, model)

    with tracing_context(enabled=False):
        for label, query in [("A 正常检索", normal), ("B 空检索", empty)]:
            result = graph.invoke({"query": query})
            print(label)
            print("  query:", result["query"])
            print("  retrieved_docs:", result["retrieved_docs"])
            print("  answer:", result["answer"])

        print("C 检索失败")
        try:
            graph.invoke({"query": failed})
        except RuntimeError as error:
            print(f"  向调用方传播 {type(error).__name__}: {error}")
        else:
            raise AssertionError("Expected the retrieval failure to reach the caller")


if __name__ == "__main__":
    main()
