import os
import uuid

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.embedding.embedder import Embedder

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
# MANUAL KEEP
#
# 真实 Qwen Embedding
# +
# 真实 PostgreSQL
# +
# 真实 pgvector
#
# 默认不进入普通 regression
# ==================================================


@pytest.fixture(scope="module")
def session_factory():

    if os.getenv("RUN_REAL_QWEN_RAG_SMOKE") != "1":
        pytest.skip(
            "Real Qwen RAG smoke test requires explicit opt-in"
        )

    if engine.dialect.name != "postgresql":
        pytest.fail("PostgreSQL is required")

    schema_name = (
        f"rag_real_qwen_smoke_{uuid.uuid4().hex}"
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


@pytest.fixture(scope="module")
def embedder():

    if os.getenv("RUN_REAL_QWEN_RAG_SMOKE") != "1":
        pytest.skip(
            "Real Qwen RAG smoke test requires explicit opt-in"
        )

    # 真正加载 .env 中配置的 Qwen3-Embedding-0.6B
    return Embedder()


def test_real_qwen_persistent_rag_smoke(
    session_factory,
    embedder,
):

    # ==================================================
    # 1. Persistent Ingestion
    # ==================================================

    ingestion = RAGIngestionService(
        embedder=embedder,
        session_factory=session_factory,
    )

    documents = [
        (
            Document(
                document_id="doc_vector",
                file_name="vector_search.txt",
                file_type="txt",
            ),
            "PostgreSQL 可以通过 pgvector 扩展执行向量相似度检索。",
        ),
        (
            Document(
                document_id="doc_fastapi",
                file_name="fastapi.txt",
                file_type="txt",
            ),
            "FastAPI 是一个用于构建 Python Web API 的框架。",
        ),
        (
            Document(
                document_id="doc_cat",
                file_name="cat.txt",
                file_type="txt",
            ),
            "猫喜欢在温暖的阳光下休息和睡觉。",
        ),
    ]

    for document, raw_text in documents:

        result = ingestion.ingest(
            document=document,
            raw_text=raw_text,

            # 大于单条文本长度：
            # 每个 Document 只生成一个 Chunk
            chunk_size=200,
        )

        assert result.status == "success"
        assert result.chunk_count == 1

    # ==================================================
    # 2. Build Persistent Retrieval Pipeline
    # ==================================================

    vector_search = PostgreSQLVectorSearch(
        session_factory=session_factory,
    )

    hydrator = PostgreSQLHydrator(
        chunk_repository=ChunkRepository(
            session_factory=session_factory,
        ),
        document_repository=DocumentRepository(
            session_factory=session_factory,
        ),
    )

    retriever = PostgreSQLRAGRetriever(
        embedder=embedder,
        vector_search=vector_search,
        hydrator=hydrator,
    )

    # ==================================================
    # 3. Real Query Embedding + pgvector Search
    # ==================================================

    results = retriever.retrieve(
        query="PostgreSQL 怎么进行向量检索？",
        top_k=3,
    )

    # ==================================================
    # 4. Structural Assertions
    # ==================================================

    assert len(results) == 3

    assert {
        result.source
        for result in results
    } == {
        "vector_search.txt",
        "fastapi.txt",
        "cat.txt",
    }

    # ==================================================
    # 5. Semantic Smoke Assertion
    # ==================================================

    # 这不是 Retrieval Evaluation，
    # 只要求明显相关的知识排到第一位。
    assert (
        results[0].source
        == "vector_search.txt"
    )

    # ==================================================
    # 6. Manual Inspection
    # ==================================================

    print("\n")
    print(
        f"Embedding Model: {embedder.model_name}"
    )
    print(
        f"Embedding Dimension: {embedder.dimension}"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Rank {rank}: "
            f"source={result.source}, "
            f"score={result.score:.6f}"
        )
        print(
            f"content={result.content}"
        )