import numpy as np

from backend.rag.context_assembler import ContextAssembler
from backend.rag.contracts import Document
from backend.rag.ingestion.chunk_embedding import build_chunk_vectors
from backend.rag.ingestion.chunker import chunk_document
from backend.rag.retrieval.exact_search import ExactDenseSearch
from backend.rag.retrieval.hydrator import RAGHydrator
from backend.rag.retrieval.retriever import RAGRetriever


class FakeRAGEmbedder:
    """
    Integration Test 使用的确定性 Fake Embedder。

    目的不是测试真实 Embedding Model，
    而是让 Retrieval Ranking 可预测。
    """

    TEXT_VECTORS = {
        "AAAAA": np.array([0.8, 0.6]),
        "BBBBB": np.array([0.0, 1.0]),
        "CCCCC": np.array([1.0, 0.0]),
    }

    QUERY_VECTORS = {
        "find BBBBB": np.array([0.0, 1.0]),
        "ranking query": np.array([1.0, 0.0]),
        "empty query": np.array([1.0, 0.0]),
    }

    def embed_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return np.array([
            self.TEXT_VECTORS[text]
            for text in texts
        ])

    def embed(
        self,
        text: str,
    ) -> np.ndarray:
        return self.QUERY_VECTORS[text]


def build_pipeline():
    embedder = FakeRAGEmbedder()

    retriever = RAGRetriever(
        embedder=embedder,
        exact_search=ExactDenseSearch(),
        hydrator=RAGHydrator(),
    )

    assembler = ContextAssembler()

    return embedder, retriever, assembler


def build_document() -> Document:
    return Document(
        document_id="doc_1",
        file_name="knowledge.txt",
        file_type="txt",
    )


def test_rag_integration_happy_path():
    """
    Happy Path：

    Document + raw_text
        ↓
    Chunker
        ↓
    Chunk Embedding
        ↓
    Retrieval
        ↓
    Hydration
        ↓
    Context Assembly
    """

    document = build_document()

    chunks = chunk_document(
        document=document,
        raw_text="AAAAABBBBBCCCCC",
        chunk_size=5,
    )

    embedder, retriever, assembler = (
        build_pipeline()
    )

    chunk_vectors = build_chunk_vectors(
        chunks=chunks,
        embedder=embedder,
    )

    chunk_store = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    document_store = {
        document.document_id: document
    }

    retrieved_chunks = retriever.retrieve(
        query="find BBBBB",
        chunk_vectors=chunk_vectors,
        chunk_store=chunk_store,
        document_store=document_store,
        top_k=1,
    )

    assembled_context = assembler.assemble(
        retrieved_chunks
    )

    assert len(retrieved_chunks) == 1

    assert (
        retrieved_chunks[0].chunk_id
        == "doc_1_chunk_1"
    )

    assert retrieved_chunks[0].content == "BBBBB"
    assert retrieved_chunks[0].source == "knowledge.txt"

    assert (
        assembled_context.context_text
        ==
        '<document_context>\n'
        '<chunk id="doc_1_chunk_1" source="knowledge.txt">\n'
        'BBBBB\n'
        '</chunk>\n'
        '</document_context>'
    )

    assert assembled_context.used_chunks == (
        retrieved_chunks
    )


def test_rag_integration_empty_path():
    """
    Empty Knowledge：

    Empty Text
        ↓
    []
        ↓
    {}
        ↓
    Empty Retrieval
        ↓
    Empty Context

    整条链合法，不应报错。
    """

    document = build_document()

    chunks = chunk_document(
        document=document,
        raw_text="",
        chunk_size=5,
    )

    embedder, retriever, assembler = (
        build_pipeline()
    )

    chunk_vectors = build_chunk_vectors(
        chunks=chunks,
        embedder=embedder,
    )

    chunk_store = {}

    document_store = {
        document.document_id: document
    }

    retrieved_chunks = retriever.retrieve(
        query="empty query",
        chunk_vectors=chunk_vectors,
        chunk_store=chunk_store,
        document_store=document_store,
        top_k=5,
    )

    assembled_context = assembler.assemble(
        retrieved_chunks
    )

    assert chunks == []
    assert chunk_vectors == {}
    assert retrieved_chunks == []

    assert assembled_context.context_text == ""
    assert assembled_context.used_chunks == []


def test_rag_integration_preserves_ranking():
    """
    Retrieval Ranking 必须穿过：

    Exact Search
        ↓
    Hydration
        ↓
    Retriever
        ↓
    ContextAssembler

    而不被重新排序。
    """

    document = build_document()

    chunks = chunk_document(
        document=document,
        raw_text="AAAAABBBBBCCCCC",
        chunk_size=5,
    )

    embedder, retriever, assembler = (
        build_pipeline()
    )

    chunk_vectors = build_chunk_vectors(
        chunks=chunks,
        embedder=embedder,
    )

    chunk_store = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    document_store = {
        document.document_id: document
    }

    retrieved_chunks = retriever.retrieve(
        query="ranking query",
        chunk_vectors=chunk_vectors,
        chunk_store=chunk_store,
        document_store=document_store,
        top_k=2,
    )

    assembled_context = assembler.assemble(
        retrieved_chunks
    )

    assert [
        chunk.chunk_id
        for chunk in retrieved_chunks
    ] == [
        "doc_1_chunk_2",
        "doc_1_chunk_0",
    ]

    assert [
        chunk.chunk_id
        for chunk in assembled_context.used_chunks
    ] == [
        "doc_1_chunk_2",
        "doc_1_chunk_0",
    ]

    assert (
        assembled_context.context_text.index("CCCCC")
        <
        assembled_context.context_text.index("AAAAA")
    )
