import asyncio

import numpy as np

import backend.rag.generation as generation_module

from backend.rag.context_assembler import ContextAssembler
from backend.rag.contracts import (
    Document,
    RAGAnswer,
    RAGGenerationOutput,
)
from backend.rag.ingestion.chunk_embedding import build_chunk_vectors
from backend.rag.ingestion.chunker import chunk_document
from backend.rag.retrieval.exact_search import ExactDenseSearch
from backend.rag.retrieval.hydrator import RAGHydrator
from backend.rag.retrieval.retriever import RAGRetriever


class FakeE2EEmbedder:
    """Deterministic embeddings for the Phase 2A E2E pipeline."""

    TEXT_VECTORS = {
        "AAAAA": np.array([0.8, 0.6]),
        "BBBBB": np.array([0.0, 1.0]),
        "CCCCC": np.array([1.0, 0.0]),
    }

    QUERY_VECTORS = {
        "find BBBBB": np.array([0.0, 1.0]),
        "find CCCCC": np.array([1.0, 0.0]),
        "empty query": np.array([1.0, 0.0]),
    }

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([
            self.TEXT_VECTORS[text]
            for text in texts
        ])

    def embed(self, text: str) -> np.ndarray:
        return self.QUERY_VECTORS[text]


def build_retriever(embedder: FakeE2EEmbedder) -> RAGRetriever:
    return RAGRetriever(
        embedder=embedder,
        exact_search=ExactDenseSearch(),
        hydrator=RAGHydrator(),
    )


def test_phase_2a_e2e_happy_path(monkeypatch):
    document = Document(
        document_id="doc_1",
        file_name="knowledge.txt",
        file_type="txt",
    )
    chunks = chunk_document(
        document=document,
        raw_text="AAAAABBBBBCCCCC",
        chunk_size=5,
    )

    embedder = FakeE2EEmbedder()
    chunk_vectors = build_chunk_vectors(
        chunks=chunks,
        embedder=embedder,
    )
    retrieved_chunks = build_retriever(embedder).retrieve(
        query="find BBBBB",
        chunk_vectors=chunk_vectors,
        chunk_store={chunk.chunk_id: chunk for chunk in chunks},
        document_store={document.document_id: document},
        top_k=2,
    )
    assembled_context = ContextAssembler().assemble(retrieved_chunks)

    assert [
        chunk.chunk_id
        for chunk in retrieved_chunks
    ] == [
        "doc_1_chunk_1",
        "doc_1_chunk_0",
    ]
    assert (
        assembled_context.context_text.index("doc_1_chunk_1")
        < assembled_context.context_text.index("doc_1_chunk_0")
    )

    async def fake_call_llm_structured(messages, output_model):
        assert output_model is RAGGenerationOutput

        llm_context = messages[1]["content"]
        assert (
            llm_context.index("doc_1_chunk_1")
            < llm_context.index("doc_1_chunk_0")
        )
        assert (
            '<chunk id="doc_1_chunk_1" '
            'source="knowledge.txt">'
        ) in llm_context

        return RAGGenerationOutput(
            answer="BBBBB 是检索到的最高相关内容。",
            supporting_chunk_ids=[retrieved_chunks[0].chunk_id],
        )

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    result = asyncio.run(
        generation_module.generate_rag_answer(
            query="find BBBBB",
            assembled_context=assembled_context,
        )
    )

    assert result == RAGAnswer(
        answer="BBBBB 是检索到的最高相关内容。",
        sources=["knowledge.txt"],
    )


def test_phase_2a_e2e_empty_knowledge_path(monkeypatch):
    document = Document(
        document_id="doc_empty",
        file_name="empty.txt",
        file_type="txt",
    )
    chunks = chunk_document(
        document=document,
        raw_text="",
        chunk_size=5,
    )

    embedder = FakeE2EEmbedder()
    chunk_vectors = build_chunk_vectors(
        chunks=chunks,
        embedder=embedder,
    )
    retrieved_chunks = build_retriever(embedder).retrieve(
        query="empty query",
        chunk_vectors=chunk_vectors,
        chunk_store={},
        document_store={document.document_id: document},
        top_k=5,
    )
    assembled_context = ContextAssembler().assemble(retrieved_chunks)

    assert chunks == []
    assert chunk_vectors == {}
    assert retrieved_chunks == []
    assert assembled_context.context_text == ""
    assert assembled_context.used_chunks == []

    async def fake_call_llm_structured(messages, output_model):
        assert output_model is RAGGenerationOutput
        assert "<document_context>" not in messages[1]["content"]

        return RAGGenerationOutput(
            answer=(
                "当前知识库没有检索到相关内容，"
                "以下回答基于模型已有知识。"
            ),
            supporting_chunk_ids=[],
        )

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    result = asyncio.run(
        generation_module.generate_rag_answer(
            query="empty query",
            assembled_context=assembled_context,
        )
    )

    assert result.answer
    assert result.sources == []


def test_phase_2a_e2e_citation_handoff(monkeypatch):
    document = Document(
        document_id="citation_doc",
        file_name="citation-guide.md",
        file_type="markdown",
    )
    chunks = chunk_document(
        document=document,
        raw_text="AAAAABBBBBCCCCC",
        chunk_size=5,
    )

    embedder = FakeE2EEmbedder()
    chunk_vectors = build_chunk_vectors(
        chunks=chunks,
        embedder=embedder,
    )
    retrieved_chunks = build_retriever(embedder).retrieve(
        query="find CCCCC",
        chunk_vectors=chunk_vectors,
        chunk_store={chunk.chunk_id: chunk for chunk in chunks},
        document_store={document.document_id: document},
        top_k=1,
    )
    assembled_context = ContextAssembler().assemble(retrieved_chunks)

    citation_handle = assembled_context.used_chunks[0].chunk_id
    assert citation_handle == chunks[2].chunk_id
    assert (
        f'<chunk id="{citation_handle}" '
        'source="citation-guide.md">'
    ) in assembled_context.context_text

    async def fake_call_llm_structured(messages, output_model):
        assert output_model is RAGGenerationOutput
        assert citation_handle in messages[1]["content"]

        return RAGGenerationOutput(
            answer="CCCCC 由检索到的文档内容支持。",
            supporting_chunk_ids=[citation_handle],
        )

    monkeypatch.setattr(
        generation_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    result = asyncio.run(
        generation_module.generate_rag_answer(
            query="find CCCCC",
            assembled_context=assembled_context,
        )
    )

    assert result == RAGAnswer(
        answer="CCCCC 由检索到的文档内容支持。",
        sources=["citation-guide.md"],
    )
