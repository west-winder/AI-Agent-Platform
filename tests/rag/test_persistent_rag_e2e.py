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
from backend.rag.retrieval.postgres_vector_search import (
    PostgreSQLVectorSearch,
)
from backend.rag.retrieval.repositories import (
    ChunkRepository,
    DocumentRepository,
)
from backend.rag.retrieval.postgres_hydrator import (
    PostgreSQLHydrator,
)
from backend.rag.retrieval.postgres_retriever import (
    PostgreSQLRAGRetriever,
)


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
    # 2. Build Persistent Retrieval Pipeline
    # --------------------------------------------------

    vector_search = PostgreSQLVectorSearch(
        session_factory=session_factory,
    )

    chunk_repository = ChunkRepository(
        session_factory=session_factory,
    )

    document_repository = DocumentRepository(
        session_factory=session_factory,
    )

    hydrator = PostgreSQLHydrator(
        chunk_repository=chunk_repository,
        document_repository=document_repository,
    )

    retriever = PostgreSQLRAGRetriever(
        embedder=embedder,
        vector_search=vector_search,
        hydrator=hydrator,
    )

    # --------------------------------------------------
    # 3. Persistent Retrieval
    # --------------------------------------------------

    results = retriever.retrieve(
        query="find AA",
        top_k=2,
    )

    # --------------------------------------------------
    # 4. Assert Ranking
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
    # 5. Assert Hydration
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
    # 6. Assert Similarity
    # --------------------------------------------------

    assert [
        item.score
        for item in results
    ] == pytest.approx(
        [
            1.0,
            0.8,
        ],
        abs=1e-5,
    )