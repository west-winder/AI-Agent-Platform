"""
RAGRetriever + PostgreSQL Dense backend + Hydrator 的 Pipeline 测试。

验证唯一 Orchestrator（RAGRetriever）在持久化形态下的完整接线：

    Persistent Ingestion
        ↓
    RAGRetriever
        ├─ dense_retriever = PostgreSQLVectorSearch（DenseRetriever 持久化实现）
        └─ hydrator        = RAGHydrator（PostgreSQL Repository）

默认 skip：需要 RUN_RAG_PERSISTENT_E2E=1 与真实 PostgreSQL。
"""

import os
import uuid

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM

from backend.rag.contracts import Document
from backend.rag.ingestion.rag_ingestion import (
    RAGIngestionService,
)
from backend.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)
from backend.rag.retrieval.fusion import RRFFusion
from backend.rag.retrieval.hydrator import RAGHydrator
from backend.rag.retrieval.postgres_vector_search import (
    PostgreSQLVectorSearch,
)
from backend.rag.retrieval.repositories import (
    ChunkRepository,
    DocumentRepository,
)
from backend.rag.retrieval.rerank_document_builder import (
    RerankDocumentBuilder,
)
from backend.rag.retrieval.reranker_adapter import (
    RerankerAdapter,
)
from backend.rag.retrieval.retriever import RAGRetriever


# ==================================================
# Deterministic Fake Embedder
# ==================================================

class FakeEmbedder:
    """
    Corpus-side 和 Query-side 共用同一个模型身份与向量空间。
    """

    model_name = "test-model"
    dimension = 3

    def embed_batch(self, texts):

        vector_map = {
            "AA": [1.0, 0.0, 0.0],
            "BB": [0.8, 0.6, 0.0],
            "CC": [0.0, 1.0, 0.0],
        }

        return np.array([
            vector_map[text]
            for text in texts
        ])

    def embed(self, query):

        if query != "find AA":
            raise ValueError(
                f"unexpected query: {query}"
            )

        return np.array([
            1.0,
            0.0,
            0.0,
        ])


# ==================================================
# Fake Reranker
# ==================================================

class FakeRerankItem:
    """
    模拟 backend.reranking.reranker.RerankResult 的字段契约。
    """

    def __init__(self, index, score):
        self.index = index
        self.score = score


class PreservingReranker:
    """
    测试专用底层 Reranker：保持 Fusion 顺序。

    真实 Cross Encoder Reranker 需要本地模型文件，
    因此持久化 E2E 注入 Fake，只验证 Pipeline 编排
    与 PostgreSQL storage 的接入。
    """

    def rerank(self, query, texts):
        return [
            FakeRerankItem(
                index=index,
                score=1.0 / (index + 1),
            )
            for index, text in enumerate(texts)
        ]


# ==================================================
# PostgreSQL 隔离环境
# ==================================================

@pytest.fixture
def session_factory():

    if os.getenv("RUN_RAG_PERSISTENT_E2E") != "1":
        pytest.skip(
            "Persistent RAG E2E requires opt-in"
        )

    if engine.dialect.name != "postgresql":
        pytest.fail(
            "PostgreSQL is required"
        )

    schema_name = (
        f"rag_persistent_e2e_{uuid.uuid4().hex}"
    )

    tables = [
        DocumentORM.__table__,
        ChunkORM.__table__,
        ChunkEmbeddingORM.__table__,
    ]

    with engine.connect() as connection:

        transaction = connection.begin()

        try:
            database_name = (
                connection.exec_driver_sql(
                    "SELECT current_database()"
                ).scalar_one()
            )

            if database_name != "ai_agent_platform":
                pytest.fail(
                    "Unexpected database"
                )

            connection.exec_driver_sql(
                f'CREATE SCHEMA "{schema_name}"'
            )

            test_connection = (
                connection.execution_options(
                    schema_translate_map={
                        None: schema_name
                    }
                )
            )

            Base.metadata.create_all(
                bind=test_connection,
                tables=tables,
                checkfirst=False,
            )

            factory = sessionmaker(
                bind=test_connection,
                join_transaction_mode=(
                    "create_savepoint"
                ),
                expire_on_commit=False,
            )

            yield factory

        finally:
            if transaction.is_active:
                transaction.rollback()


# ==================================================
# E2E：Persistent Ingestion -> Retrieval
# ==================================================

def test_persistent_ingestion_to_retrieval(
    session_factory,
):

    embedder = FakeEmbedder()

    # --------------------------------------------------
    # 1. Persistent Ingestion
    # --------------------------------------------------

    ingestion = RAGIngestionService(
        embedder=embedder,
        session_factory=session_factory,
    )

    document = Document(
        document_id="doc_1",
        file_name="knowledge.txt",
        file_type="txt",
    )

    ingestion_result = ingestion.ingest(
        document=document,
        raw_text="AABBCC",
        chunk_size=2,
    )

    assert ingestion_result.status == "success"
    assert ingestion_result.chunk_count == 3

    # --------------------------------------------------
    # 2. Build Retrieval Pipeline
    #
    # 唯一的 Orchestrator 是 RAGRetriever：
    # Dense 路径注入 PostgreSQL 实现，
    # 语料文本与 Hydration 都走 Repository。
    # --------------------------------------------------

    chunk_repository = ChunkRepository(
        session_factory=session_factory,
    )

    document_repository = DocumentRepository(
        session_factory=session_factory,
    )

    retriever = RAGRetriever(
        embedder=embedder,
        dense_retriever=PostgreSQLVectorSearch(
            embedding_model=embedder.model_name,
            session_factory=session_factory,
        ),
        bm25_retriever=BM25Retriever(),
        fusion=RRFFusion(),
        rerank_document_builder=RerankDocumentBuilder(),
        reranker=RerankerAdapter(PreservingReranker()),
        hydrator=RAGHydrator(
            chunk_repository=chunk_repository,
            document_repository=document_repository,
        ),
    )

    # --------------------------------------------------
    # 3. BM25 / Rerank 需要的语料文本同样来自 Repository
    # --------------------------------------------------

    chunk_store = {
        chunk.chunk_id: chunk
        for chunk in chunk_repository.get_by_ids([
            "doc_1_chunk_0",
            "doc_1_chunk_1",
            "doc_1_chunk_2",
        ])
    }

    # --------------------------------------------------
    # 4. Retrieval
    # --------------------------------------------------

    results = retriever.retrieve(
        query="find AA",
        chunk_store=chunk_store,
        top_k=2,
    )

    # --------------------------------------------------
    # 5. Assert Ranking
    # --------------------------------------------------

    assert [
        item.chunk_id
        for item in results
    ] == [
        "doc_1_chunk_0",
        "doc_1_chunk_1",
    ]

    assert [
        item.content
        for item in results
    ] == [
        "AA",
        "BB",
    ]

    # --------------------------------------------------
    # 6. Assert Hydration
    # --------------------------------------------------

    assert all(
        item.document_id == "doc_1"
        for item in results
    )

    assert all(
        item.source == "knowledge.txt"
        for item in results
    )

    # --------------------------------------------------
    # 7. Assert Ranking
    # --------------------------------------------------

    assert [
        item.retrieval_rank
        for item in results
    ] == [
        1,
        2,
    ]
