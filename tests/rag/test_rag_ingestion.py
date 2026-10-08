
import os
import uuid

import numpy as np
import pytest

from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM
from backend.rag.contracts import Document
from backend.rag.ingestion.rag_ingestion import RAGIngestionService


# ==================================================
# Fake Embedder：不加载真实模型
# ==================================================

class FakeEmbedder:
    model_name = "test-model"
    dimension = 2

    def __init__(self):
        self.calls = 0

    def embed_batch(self, texts):
        self.calls += 1
        return np.array([
            [float(i + 1), 1.0]
            for i in range(len(texts))
        ])


class BadEmbedder(FakeEmbedder):
    def embed_batch(self, texts):
        self.calls += 1
        return np.full((len(texts), 2), np.nan)


class BeginSpy:
    """检查 Validation 前是否错误开启了事务。"""

    def __init__(self, factory):
        self.factory = factory
        self.calls = 0

    def begin(self):
        self.calls += 1
        return self.factory.begin()


# ==================================================
# Test Fixture：独立 Schema + 外层事务
# ==================================================

@pytest.fixture
def session_factory():
    if os.getenv("RUN_RAG_INGESTION_TEST") != "1":
        pytest.skip("Explicit database test opt-in required")

    if engine.dialect.name != "postgresql":
        pytest.fail("PostgreSQL required")

    schema = f"rag_ingest_test_{uuid.uuid4().hex}"

    tables = [
        DocumentORM.__table__,
        ChunkORM.__table__,
        ChunkEmbeddingORM.__table__,
    ]

    with engine.connect() as connection:
        transaction = connection.begin()

        try:
            database_name = connection.exec_driver_sql(
                "SELECT current_database()"
            ).scalar_one()

            if database_name != "ai_agent_platform":
                pytest.fail("Unexpected database")

            connection.exec_driver_sql(
                f'CREATE SCHEMA "{schema}"'
            )

            test_connection = connection.execution_options(
                schema_translate_map={None: schema}
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
# Helpers
# ==================================================

def make_document(
    document_id="doc_1",
    file_name="knowledge.txt",
):
    return Document(
        document_id=document_id,
        file_name=file_name,
        file_type="txt",
    )


def get_counts(factory):
    with factory() as session:
        return tuple(
            session.scalar(
                select(func.count()).select_from(model)
            )
            for model in (
                DocumentORM,
                ChunkORM,
                ChunkEmbeddingORM,
            )
        )


# ==================================================
# Case 1：Success
# ==================================================

def test_ingestion_success(session_factory):
    embedder = FakeEmbedder()
    service = RAGIngestionService(embedder, session_factory)

    result = service.ingest(
        document=make_document(),
        raw_text="ABCDEF",
        chunk_size=2,
    )

    assert result.status == "success"
    assert result.document_id == "doc_1"
    assert result.chunk_count == 3
    assert embedder.calls == 1

    # 从新 Session 读取，验证事务提交后的结果
    assert get_counts(session_factory) == (1, 3, 3)

    with session_factory() as session:
        chunk = session.get(ChunkORM, "doc_1_chunk_0")
        embedding = session.get(
            ChunkEmbeddingORM,
            ("doc_1_chunk_0", "test-model"),
        )

        assert chunk.content == "AB"
        assert embedding is not None
        assert len(embedding.embedding) == 2
        assert np.allclose(
            embedding.embedding,
            [1.0, 1.0],
        )


# ==================================================
# Case 2：Empty
# ==================================================

def test_ingestion_empty(session_factory):
    embedder = FakeEmbedder()
    spy = BeginSpy(session_factory)

    service = RAGIngestionService(embedder, spy)

    result = service.ingest(
        document=make_document(),
        raw_text="  \n  ",
        chunk_size=2,
    )

    assert result.status == "empty"
    assert result.chunk_count == 0
    assert embedder.calls == 0
    assert spy.calls == 0
    assert get_counts(session_factory) == (0, 0, 0)


# ==================================================
# Case 3：Duplicate Document
# ==================================================

def test_ingestion_duplicate(session_factory):
    service = RAGIngestionService(
        FakeEmbedder(),
        session_factory,
    )

    service.ingest(
        document=make_document(),
        raw_text="ABCDEF",
        chunk_size=2,
    )

    with pytest.raises(IntegrityError):
        service.ingest(
            document=make_document(
                file_name="replacement.txt"
            ),
            raw_text="ZZZZZZ",
            chunk_size=2,
        )

    assert get_counts(session_factory) == (1, 3, 3)

    with session_factory() as session:
        document = session.get(DocumentORM, "doc_1")
        chunk = session.get(ChunkORM, "doc_1_chunk_0")

        assert document.file_name == "knowledge.txt"
        assert chunk.content == "AB"


# ==================================================
# Case 4：Validation Failure
# ==================================================

def test_validation_failure(session_factory):
    embedder = BadEmbedder()
    spy = BeginSpy(session_factory)

    service = RAGIngestionService(embedder, spy)

    with pytest.raises(ValueError, match="NaN or Inf"):
        service.ingest(
            document=make_document(),
            raw_text="ABCDEF",
            chunk_size=2,
        )

    assert embedder.calls == 1
    assert spy.calls == 0
    assert get_counts(session_factory) == (0, 0, 0)


# ==================================================
# Case 5：Persistence Failure + Rollback
# ==================================================

def test_persistence_failure_rollback(session_factory):
    service = RAGIngestionService(
        FakeEmbedder(),
        session_factory,
    )

    def fail_embedding_insert(mapper, connection, target):
        raise RuntimeError("simulated embedding persistence failure")

    # 在 Embedding INSERT 前注入异常。
    # 此时 Document 和 Chunk 已经 flush。
    event.listen(
        ChunkEmbeddingORM,
        "before_insert",
        fail_embedding_insert,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="simulated embedding persistence failure",
        ):
            service.ingest(
                document=make_document(),
                raw_text="ABCDEF",
                chunk_size=2,
            )
    finally:
        event.remove(
            ChunkEmbeddingORM,
            "before_insert",
            fail_embedding_insert,
        )

    # 即使前两类记录已经 flush，也必须全部回滚
    assert get_counts(session_factory) == (0, 0, 0)
