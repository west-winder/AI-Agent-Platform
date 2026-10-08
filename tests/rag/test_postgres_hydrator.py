import os
import uuid

import pytest
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM

from backend.rag.contracts import SearchResult
from backend.rag.retrieval.repositories import (
    ChunkRepository,
    DocumentRepository,
)
from backend.rag.retrieval.postgres_hydrator import (
    PostgreSQLHydrator,
)


# ==================================================
# PostgreSQL 隔离测试环境
# ==================================================

@pytest.fixture
def session_factory():

    if os.getenv("RUN_RAG_HYDRATOR_TEST") != "1":
        pytest.skip(
            "Database integration test requires opt-in"
        )

    schema_name = (
        f"rag_hydrator_test_{uuid.uuid4().hex}"
    )

    tables = [
        DocumentORM.__table__,
        ChunkORM.__table__,
        ChunkEmbeddingORM.__table__,
    ]

    with engine.connect() as connection:

        transaction = connection.begin()

        try:
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
# Seed
# ==================================================

def seed_data(factory):

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
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Content 1",
            ),
            ChunkORM(
                chunk_id="chunk_2",
                document_id="doc_1",
                content="Content 2",
            ),
            ChunkORM(
                chunk_id="chunk_3",
                document_id="doc_1",
                content="Content 3",
            ),
        ])


def build_hydrator(factory):

    return PostgreSQLHydrator(
        chunk_repository=ChunkRepository(
            factory
        ),
        document_repository=DocumentRepository(
            factory
        ),
    )


# ==================================================
# Case 1：Repository Batch Query
# ==================================================

def test_repository_batch_query(session_factory):

    seed_data(session_factory)

    repository = ChunkRepository(
        session_factory
    )

    chunks = repository.get_by_ids([
        "chunk_1",
        "chunk_3",
    ])

    assert {
        chunk.chunk_id
        for chunk in chunks
    } == {
        "chunk_1",
        "chunk_3",
    }


# ==================================================
# Case 2：Normal Hydration + Order Preservation
# ==================================================

def test_hydration_preserves_search_order(
    session_factory,
):

    seed_data(session_factory)

    hydrator = build_hydrator(
        session_factory
    )

    search_results = [
        SearchResult(
            chunk_id="chunk_3",
            score=0.95,
        ),
        SearchResult(
            chunk_id="chunk_1",
            score=0.90,
        ),
        SearchResult(
            chunk_id="chunk_2",
            score=0.80,
        ),
    ]

    results = hydrator.hydrate(
        search_results
    )

    assert [
        item.chunk_id
        for item in results
    ] == [
        "chunk_3",
        "chunk_1",
        "chunk_2",
    ]

    assert [
        item.score
        for item in results
    ] == pytest.approx([
        0.95,
        0.90,
        0.80,
    ])

    assert all(
        item.source == "knowledge.txt"
        for item in results
    )


# ==================================================
# Case 3：Empty Retrieval
# ==================================================

def test_empty_retrieval(session_factory):

    hydrator = build_hydrator(
        session_factory
    )

    result = hydrator.hydrate([])

    assert result == []


# ==================================================
# Case 4：Missing Chunk
# ==================================================

def test_missing_chunk_raises(
    session_factory,
):

    seed_data(session_factory)

    hydrator = build_hydrator(
        session_factory
    )

    search_results = [
        SearchResult(
            chunk_id="chunk_1",
            score=0.9,
        ),
        SearchResult(
            chunk_id="missing_chunk",
            score=0.8,
        ),
    ]

    with pytest.raises(
        KeyError,
        match="missing chunks",
    ):
        hydrator.hydrate(
            search_results
        )


# ==================================================
# Case 5：Missing Document
# ==================================================

class EmptyDocumentRepository:
    """
    模拟数据库读取异常状态：
    Chunk 存在，但对应 Document 缺失。

    正常 PostgreSQL FK 会阻止这种状态，
    这里用于验证 Hydrator 自身的防御边界。
    """

    def get_by_ids(
        self,
        document_ids,
    ):
        return []


def test_missing_document_raises(
    session_factory,
):

    seed_data(session_factory)

    hydrator = PostgreSQLHydrator(
        chunk_repository=ChunkRepository(
            session_factory
        ),
        document_repository=(
            EmptyDocumentRepository()
        ),
    )

    search_results = [
        SearchResult(
            chunk_id="chunk_1",
            score=0.9,
        ),
    ]

    with pytest.raises(
        KeyError,
        match="missing documents",
    ):
        hydrator.hydrate(
            search_results
        )