import numpy as np
import pytest

from backend.rag.contracts import (
    DenseRetrieverResult,
    RagRetrievedChunk,
)
from backend.rag.retrieval.postgres_retriever import (
    PostgreSQLRAGRetriever,
)


# ==================================================
# Fake Components
# ==================================================

class FakeEmbedder:
    model_name = "test-model"

    def __init__(self):
        self.received_query = None

    def embed(self, query):
        self.received_query = query
        return np.array([1.0, 0.0, 0.0])


class FakeVectorSearch:

    def __init__(self):
        self.received_query_vector = None
        self.received_embedding_model = None
        self.received_top_k = None

    def search(
        self,
        query_vector,
        embedding_model,
        top_k,
    ):
        self.received_query_vector = query_vector
        self.received_embedding_model = embedding_model
        self.received_top_k = top_k

        return [
            DenseRetrieverResult(
                chunk_id="chunk_2",
                dense_score=0.9,
                dense_rank=1,
            ),
            DenseRetrieverResult(
                chunk_id="chunk_1",
                dense_score=0.8,
                dense_rank=2,
            ),
        ]


class FakeHydrator:

    def __init__(self):
        self.received_candidates = None

    def hydrate(
        self,
        candidates,
    ):
        self.received_candidates = candidates

        return [
            RagRetrievedChunk(
                chunk_id="chunk_2",
                document_id="doc_1",
                content="Content 2",
                source="knowledge.txt",
                retrieval_rank=1,
            ),
            RagRetrievedChunk(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Content 1",
                source="knowledge.txt",
                retrieval_rank=2,
            ),
        ]


# ==================================================
# Case 1：完整编排流程
# ==================================================

def test_retriever_orchestration():

    embedder = FakeEmbedder()
    vector_search = FakeVectorSearch()
    hydrator = FakeHydrator()

    retriever = PostgreSQLRAGRetriever(
        embedder=embedder,
        vector_search=vector_search,
        hydrator=hydrator,
    )

    results = retriever.retrieve(
        query="What is RAG?",
        top_k=2,
    )

    # Query -> Embedder
    assert embedder.received_query == "What is RAG?"

    # Query Vector -> Vector Search
    np.testing.assert_allclose(
        vector_search.received_query_vector,
        [1.0, 0.0, 0.0],
    )

    # Embedder identity -> Vector Search
    assert (
        vector_search.received_embedding_model
        == "test-model"
    )

    # top_k 正确传递
    assert vector_search.received_top_k == 2

    # DenseRetrieverResult[] -> RetrievedCandidate[] -> Hydrator
    assert [
        item.chunk_id
        for item in hydrator.received_candidates
    ] == [
        "chunk_2",
        "chunk_1",
    ]

    # dense_rank 必须被适配成候选的业务排名
    assert [
        item.rank
        for item in hydrator.received_candidates
    ] == [
        1,
        2,
    ]

    # 最终 RagRetrievedChunk[]
    assert [
        item.chunk_id
        for item in results
    ] == [
        "chunk_2",
        "chunk_1",
    ]


# ==================================================
# Case 2：Empty Retrieval
# ==================================================

class EmptyVectorSearch:

    def search(
        self,
        query_vector,
        embedding_model,
        top_k,
    ):
        return []


class EmptyHydrator:

    def __init__(self):
        self.received_candidates = None

    def hydrate(
        self,
        candidates,
    ):
        self.received_candidates = candidates
        return []


def test_empty_retrieval():

    hydrator = EmptyHydrator()

    retriever = PostgreSQLRAGRetriever(
        embedder=FakeEmbedder(),
        vector_search=EmptyVectorSearch(),
        hydrator=hydrator,
    )

    results = retriever.retrieve(
        query="nothing",
        top_k=5,
    )

    assert hydrator.received_candidates == []
    assert results == []


# ==================================================
# Case 3：Search Failure 必须传播
# ==================================================

class FailingVectorSearch:

    def search(
        self,
        query_vector,
        embedding_model,
        top_k,
    ):
        raise RuntimeError(
            "simulated vector search failure"
        )


class ForbiddenHydrator:

    def hydrate(
        self,
        candidates,
    ):
        raise AssertionError(
            "Hydrator must not run after search failure"
        )


def test_search_failure_propagates():

    retriever = PostgreSQLRAGRetriever(
        embedder=FakeEmbedder(),
        vector_search=FailingVectorSearch(),
        hydrator=ForbiddenHydrator(),
    )

    with pytest.raises(
        RuntimeError,
        match="simulated vector search failure",
    ):
        retriever.retrieve(
            query="test",
            top_k=5,
        )