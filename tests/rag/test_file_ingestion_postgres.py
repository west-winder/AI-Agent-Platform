"""Opt-in file ingestion tests using real PostgreSQL and pgvector."""

import os
import uuid

import numpy as np
import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM
from backend.models.rag_document import DocumentORM
from backend.rag.ingestion.file_ingestion import FileIngestionService
from backend.rag.ingestion.rag_ingestion import RAGIngestionService


MODELS = (DocumentORM, ChunkORM, ChunkEmbeddingORM)


class FakeEmbedder:
    """Content-based vectors in the same space as the persistent E2E test."""

    model_name = "test-model"
    dimension = 3
    vectors = {
        "AA": [1.0, 0.0, 0.0],
        "BB": [0.8, 0.6, 0.0],
        "CC": [0.0, 1.0, 0.0],
    }

    def __init__(self):
        self.batches = []

    def embed_batch(self, texts):
        self.batches.append(list(texts))
        return np.array([self.vectors[text] for text in texts], dtype=float)


@pytest.fixture
def session_factory():
    if os.getenv("RUN_RAG_FILE_INGESTION_TEST") != "1":
        pytest.skip("File ingestion PostgreSQL tests require explicit opt-in")

    if engine.dialect.name != "postgresql":
        pytest.fail("PostgreSQL is required")

    tables = [model.__table__ for model in MODELS]
    if any(table.schema is not None for table in tables):
        pytest.fail("Expected schema-less mappings for test schema translation")

    schema = f"rag_file_ingest_test_{uuid.uuid4().hex}"
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            # Verify the database before any DDL or ingestion writes.
            database = connection.exec_driver_sql(
                "SELECT current_database()"
            ).scalar_one()
            if database != "ai_agent_platform":
                pytest.fail("Unexpected database; no test schema created")

            can_create = connection.exec_driver_sql(
                "SELECT has_database_privilege("
                "current_user, current_database(), 'CREATE')"
            ).scalar_one()
            if not can_create:
                pytest.fail("Missing permission to create an isolated schema")

            vector_available = connection.exec_driver_sql(
                "SELECT EXISTS (SELECT 1 FROM pg_extension "
                "WHERE extname = 'vector') AND to_regtype('vector') IS NOT NULL"
            ).scalar_one()
            if not vector_available:
                pytest.fail("pgvector must already be installed and accessible")

            # PostgreSQL transactional DDL: this schema is rolled back too.
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
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

            assert get_counts(factory) == (0, 0, 0)
            yield factory
        finally:
            outer_transaction_active = transaction.is_active
            if outer_transaction_active:
                transaction.rollback()
            assert outer_transaction_active, "A session ended the outer transaction"
            assert not connection.in_transaction()

            # Verify that the outer rollback removed both data and test DDL.
            with connection.begin():
                schema_exists = connection.exec_driver_sql(
                    "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)",
                    (schema,),
                ).scalar_one()
                assert not schema_exists, "Test schema survived the outer rollback"


@pytest.fixture
def service(session_factory):
    embedder = FakeEmbedder()
    ingestion = RAGIngestionService(
        embedder=embedder,
        session_factory=session_factory,
    )
    entry = FileIngestionService(
        ingestion_service=ingestion,
        session_factory=session_factory,
    )
    return entry, embedder


def get_counts(factory):
    with factory() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in MODELS
        )


def assert_saved_document(factory, result, file_name):
    assert result.status == "success"
    assert isinstance(result.document_id, str) and result.document_id
    assert result.file_name == file_name
    assert result.chunk_count == 3

    # A fresh session must see the service's committed savepoint.
    with factory() as session:
        document = session.get(DocumentORM, result.document_id)
        assert document is not None
        assert document.document_id == result.document_id
        assert document.file_name == file_name
        assert document.file_type == "txt"

        chunks = session.scalars(
            select(ChunkORM)
            .where(ChunkORM.document_id == result.document_id)
            .order_by(ChunkORM.chunk_id)
        ).all()
        # IDs encode the three chunk positions; no fixed document UUID.
        expected_ids = [f"{result.document_id}_chunk_{i}" for i in range(3)]
        assert [chunk.chunk_id for chunk in chunks] == expected_ids
        assert [chunk.content for chunk in chunks] == ["AA", "BB", "CC"]
        assert all(chunk.document_id == result.document_id for chunk in chunks)

        rows = session.execute(
            select(
                ChunkEmbeddingORM,
                func.vector_dims(ChunkEmbeddingORM.embedding),
            )
            .join(ChunkORM, ChunkORM.chunk_id == ChunkEmbeddingORM.chunk_id)
            .where(ChunkORM.document_id == result.document_id)
            .order_by(
                ChunkEmbeddingORM.chunk_id,
                ChunkEmbeddingORM.embedding_model,
            )
        ).all()
        assert [embedding.chunk_id for embedding, _ in rows] == expected_ids
        for chunk, (embedding, stored_dimension) in zip(chunks, rows):
            assert embedding.embedding_model == FakeEmbedder.model_name
            assert stored_dimension == FakeEmbedder.dimension
            vector = np.asarray(embedding.embedding, dtype=float)
            assert vector.shape == (FakeEmbedder.dimension,)
            assert np.isfinite(vector).all()
            assert np.linalg.norm(vector) > 0
            np.testing.assert_allclose(
                vector,
                FakeEmbedder.vectors[chunk.content],
                rtol=1e-6,
                atol=1e-6,
            )


def test_txt_upload_persists_document_chunks_and_vectors(service, session_factory):
    entry, embedder = service
    result = entry.ingest_txt("rag.txt", b"AABBCC", chunk_size=2)

    assert get_counts(session_factory) == (1, 3, 3)
    assert_saved_document(session_factory, result, "rag.txt")
    assert embedder.batches == [["AA", "BB", "CC"]]


def test_sequential_duplicate_names_persist_distinct_documents(
    service, session_factory
):
    entry, embedder = service
    first = entry.ingest_txt("rag.txt", b"AABBCC", chunk_size=2)
    assert get_counts(session_factory) == (1, 3, 3)
    second = entry.ingest_txt("rag.txt", b"AABBCC", chunk_size=2)

    assert first.document_id != second.document_id
    assert [first.file_name, second.file_name] == ["rag.txt", "rag(1).txt"]
    assert get_counts(session_factory) == (2, 6, 6)
    assert_saved_document(session_factory, first, "rag.txt")
    assert_saved_document(session_factory, second, "rag(1).txt")
    assert embedder.batches == [["AA", "BB", "CC"], ["AA", "BB", "CC"]]


def test_empty_file_returns_empty_without_writes(service, session_factory):
    entry, embedder = service
    result = entry.ingest_txt("empty.txt", b"", chunk_size=2)

    assert result.status == "empty"
    assert isinstance(result.document_id, str) and result.document_id
    assert result.file_name == "empty.txt"
    assert result.chunk_count == 0
    assert embedder.batches == []
    assert get_counts(session_factory) == (0, 0, 0)


def test_invalid_utf8_propagates_without_writes(service, session_factory):
    entry, embedder = service
    with pytest.raises(UnicodeDecodeError):
        entry.ingest_txt("invalid.txt", b"\xff\xfe\xff", chunk_size=2)

    assert embedder.batches == []
    assert get_counts(session_factory) == (0, 0, 0)


def test_embedding_insert_failure_rolls_back_all_rows(service, session_factory):
    entry, embedder = service
    failure = RuntimeError("controlled file ingestion embedding insert failure")
    counts_before_failure = []

    def fail_embedding_insert(mapper, connection, target):
        # Same translated connection: Document and Chunk have already flushed.
        counts_before_failure.append(tuple(
            connection.scalar(select(func.count()).select_from(model))
            for model in MODELS
        ))
        raise failure

    event.listen(ChunkEmbeddingORM, "before_insert", fail_embedding_insert)
    try:
        with pytest.raises(
            RuntimeError,
            match="controlled file ingestion embedding insert failure",
        ) as raised:
            entry.ingest_txt("failed.txt", b"AABBCC", chunk_size=2)
        assert raised.value is failure
    finally:
        event.remove(ChunkEmbeddingORM, "before_insert", fail_embedding_insert)

    assert not event.contains(
        ChunkEmbeddingORM, "before_insert", fail_embedding_insert
    )
    assert counts_before_failure == [(1, 3, 0)]
    assert embedder.batches == [["AA", "BB", "CC"]]
    assert get_counts(session_factory) == (0, 0, 0)
