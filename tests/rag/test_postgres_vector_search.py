
import os
import uuid

import numpy as np
import pytest
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM
from backend.rag.retrieval.postgres_vector_search import (
    EmbeddingCompatibilityError,
    PostgreSQLVectorSearch,
)


# ==================================================
# 1. 数据库测试环境：隔离 Schema
# ==================================================

@pytest.fixture
def session_factory():
    if os.getenv("RUN_RAG_VECTOR_SEARCH_TEST") != "1":
        pytest.skip("Database integration test requires opt-in")

    if engine.dialect.name != "postgresql":
        pytest.fail("PostgreSQL is required")

    schema_name = f"rag_search_test_{uuid.uuid4().hex}"

    tables = [
        DocumentORM.__table__,
        ChunkORM.__table__,
        ChunkEmbeddingORM.__table__,
    ]

    with engine.connect() as connection:
        transaction = connection.begin()

        try:
            db_name = connection.exec_driver_sql(
                "SELECT current_database()"
            ).scalar_one()

            if db_name != "ai_agent_platform":
                pytest.fail("Unexpected database")

            connection.exec_driver_sql(
                f'CREATE SCHEMA "{schema_name}"'
            )

            test_connection = connection.execution_options(
                schema_translate_map={None: schema_name}
            )

            Base.metadata.create_all(
                bind=test_connection,
                tables=tables,
                checkfirst=False,
            )

            factory = sessionmaker(
                bind=test_connection,
                join_transaction_mode="create_savepoint",
                expire_on_commit=False,
            )

            yield factory

        finally:
            if transaction.is_active:
                transaction.rollback()


# ==================================================
# 2. 测试数据
# ==================================================

def seed_vectors(factory):
    """
    三个属于 model-A 的 Chunk Embedding：

    chunk_1 -> [1, 0, 0]
    chunk_2 -> [0.8, 0.6, 0]
    chunk_3 -> [0, 1, 0]
    """

    vectors = {
        "chunk_1": [1.0, 0.0, 0.0],
        "chunk_2": [0.8, 0.6, 0.0],
        "chunk_3": [0.0, 1.0, 0.0],
    }

    with factory.begin() as session:
        session.add(
            DocumentORM(
                document_id="doc_1",
                file_name="knowledge.txt",
                file_type="txt",
            )
        )
        session.flush()

        session.add_all([
            ChunkORM(
                chunk_id=chunk_id,
                document_id="doc_1",
                content=f"Content of {chunk_id}",
            )
            for chunk_id in vectors
        ])
        session.flush()

        session.add_all([
            ChunkEmbeddingORM(
                chunk_id=chunk_id,
                embedding_model="model-A",
                embedding=vector,
            )
            for chunk_id, vector in vectors.items()
        ])


# ==================================================
# Case 1：Exact Search + Ranking + Score
# ==================================================

def test_exact_search_ranking(session_factory):
    seed_vectors(session_factory)

    search = PostgreSQLVectorSearch(session_factory)

    results = search.search(
        query_vector=np.array([1.0, 0.0, 0.0]),
        embedding_model="model-A",
        top_k=3,
    )

    assert [r.chunk_id for r in results] == [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]

    assert [r.score for r in results] == pytest.approx(
        [1.0, 0.8, 0.0],
        abs=1e-5,
    )


# ==================================================
# Case 2：Top-K
# ==================================================

def test_top_k(session_factory):
    seed_vectors(session_factory)

    search = PostgreSQLVectorSearch(session_factory)

    results = search.search(
        query_vector=np.array([1.0, 0.0, 0.0]),
        embedding_model="model-A",
        top_k=2,
    )

    assert len(results) == 2
    assert [r.chunk_id for r in results] == [
        "chunk_1",
        "chunk_2",
    ]


# ==================================================
# Case 3：Empty Corpus
# ==================================================

def test_empty_corpus(session_factory):
    search = PostgreSQLVectorSearch(session_factory)

    results = search.search(
        query_vector=np.array([1.0, 0.0, 0.0]),
        embedding_model="model-A",
        top_k=5,
    )

    assert results == []


# ==================================================
# Case 4：Model Mismatch
# ==================================================

def test_model_mismatch(session_factory):
    seed_vectors(session_factory)

    search = PostgreSQLVectorSearch(session_factory)

    with pytest.raises(EmbeddingCompatibilityError):
        search.search(
            query_vector=np.array([1.0, 0.0, 0.0]),
            embedding_model="model-B",
            top_k=5,
        )


# ==================================================
# Case 5：Embedding Model Filtering
# ==================================================

def test_model_filtering(session_factory):
    seed_vectors(session_factory)

    # 第二份文档使用另一种 Embedding 模型
    with session_factory.begin() as session:
        session.add(
            DocumentORM(
                document_id="doc_2",
                file_name="other.txt",
                file_type="txt",
            )
        )
        session.flush()

        session.add(
            ChunkORM(
                chunk_id="chunk_other",
                document_id="doc_2",
                content="Other model content",
            )
        )
        session.flush()

        session.add(
            ChunkEmbeddingORM(
                chunk_id="chunk_other",
                embedding_model="model-B",
                embedding=[1.0, 0.0, 0.0],
            )
        )

    search = PostgreSQLVectorSearch(session_factory)

    results = search.search(
        query_vector=np.array([1.0, 0.0, 0.0]),
        embedding_model="model-A",
        top_k=10,
    )

    assert len(results) == 3
    assert "chunk_other" not in [
        r.chunk_id for r in results
    ]


# ==================================================
# Case 6：Dimension Mismatch
# ==================================================

def test_dimension_mismatch_propagates(session_factory):
    seed_vectors(session_factory)

    search = PostgreSQLVectorSearch(session_factory)

    # 数据库向量为 3 维，Query 为 2 维
    # PostgreSQL 应报告维度不匹配
    with pytest.raises(DBAPIError):
        search.search(
            query_vector=np.array([1.0, 0.0]),
            embedding_model="model-A",
            top_k=3,
        )


# ==================================================
# Case 7：Invalid Arguments（纯 Python）
# ==================================================

@pytest.mark.parametrize(
    "overrides, error_type",
    [
        ({"top_k": 0}, ValueError),
        ({"top_k": True}, TypeError),
        ({"embedding_model": ""}, ValueError),
        ({"query_vector": np.array([])}, ValueError),
        ({"query_vector": np.array([0.0, 0.0])}, ValueError),
        ({"query_vector": np.array([np.nan, 1.0])}, ValueError),
        ({"query_vector": np.array([np.inf, 1.0])}, ValueError),
        ({"query_vector": np.array([[1.0, 0.0]])}, ValueError),
    ],
)
def test_invalid_arguments(overrides, error_type):

    def forbidden_session_factory():
        raise AssertionError(
            "Invalid input must not access database"
        )

    search = PostgreSQLVectorSearch(
        forbidden_session_factory
    )

    arguments = {
        "query_vector": np.array([1.0, 0.0, 0.0]),
        "embedding_model": "model-A",
        "top_k": 5,
    }
    arguments.update(overrides)

    with pytest.raises(error_type):
        search.search(**arguments)
