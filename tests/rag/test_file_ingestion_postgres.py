"""Opt-in file ingestion tests using real PostgreSQL and pgvector."""

import os
import uuid

import numpy as np
import pymupdf
import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import sessionmaker

from backend.database.database import Base, engine
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM
from backend.models.rag_document import DocumentORM
from backend.rag.ingestion.file_ingestion import FileIngestionService
from backend.rag.ingestion.pdf_loader import NoExtractableTextError
from backend.rag.ingestion.rag_ingestion import RAGIngestionService


MODELS = (
    DocumentORM,
    ChunkORM,
    ChunkEmbeddingORM,
)


class FakeEmbedder:
    """Content-based vectors in the same space as the persistent E2E test."""

    model_name = "test-model"
    dimension = 3

    vectors = {
        "AA": [1.0, 0.0, 0.0],
        "BB": [0.8, 0.6, 0.0],
        "CC": [0.0, 1.0, 0.0],

        "# RAG\n": [1.0, 0.0, 1.0],
        "- one\n": [0.0, 1.0, 1.0],
        "* two": [1.0, 1.0, 0.0],

        "# NEW\n": [0.5, 0.5, 1.0],
        "- two\n": [0.75, 1.0, 0.0],
        "* end": [1.0, 0.5, 0.5],
    }

    def __init__(self):
        self.batches = []

    def embed_batch(self, texts):
        self.batches.append(list(texts))

        return np.array(
            [
                self.vectors[text]
                for text in texts
            ],
            dtype=float,
        )


@pytest.fixture
def session_factory():
    if os.getenv("RUN_RAG_FILE_INGESTION_TEST") != "1":
        pytest.skip(
            "File ingestion PostgreSQL tests require explicit opt-in"
        )

    if engine.dialect.name != "postgresql":
        pytest.fail("PostgreSQL is required")

    tables = [
        model.__table__
        for model in MODELS
    ]

    if any(
        table.schema is not None
        for table in tables
    ):
        pytest.fail(
            "Expected schema-less mappings "
            "for test schema translation"
        )

    schema = (
        f"rag_file_ingest_test_"
        f"{uuid.uuid4().hex}"
    )

    with engine.connect() as connection:
        transaction = connection.begin()

        try:
            # Verify the database before any DDL or ingestion writes.
            database = connection.exec_driver_sql(
                "SELECT current_database()"
            ).scalar_one()

            if database != "ai_agent_platform":
                pytest.fail(
                    "Unexpected database; "
                    "no test schema created"
                )

            can_create = connection.exec_driver_sql(
                "SELECT has_database_privilege("
                "current_user, current_database(), 'CREATE')"
            ).scalar_one()

            if not can_create:
                pytest.fail(
                    "Missing permission to create "
                    "an isolated schema"
                )

            vector_available = connection.exec_driver_sql(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_extension "
                "WHERE extname = 'vector'"
                ") "
                "AND to_regtype('vector') IS NOT NULL"
            ).scalar_one()

            if not vector_available:
                pytest.fail(
                    "pgvector must already be installed "
                    "and accessible"
                )

            # PostgreSQL transactional DDL:
            # this schema is rolled back too.
            connection.exec_driver_sql(
                f'CREATE SCHEMA "{schema}"'
            )

            test_connection = (
                connection.execution_options(
                    schema_translate_map={
                        None: schema
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
                join_transaction_mode="create_savepoint",
                expire_on_commit=False,
            )

            assert get_counts(factory) == (
                0,
                0,
                0,
            )

            yield factory

        finally:
            outer_transaction_active = (
                transaction.is_active
            )

            if outer_transaction_active:
                transaction.rollback()

            assert outer_transaction_active, (
                "A session ended the outer transaction"
            )

            assert not connection.in_transaction()

            # Verify that the outer rollback removed
            # both data and test DDL.
            with connection.begin():
                schema_exists = (
                    connection.exec_driver_sql(
                        "SELECT EXISTS ("
                        "SELECT 1 "
                        "FROM pg_namespace "
                        "WHERE nspname = %s"
                        ")",
                        (schema,),
                    ).scalar_one()
                )

                assert not schema_exists, (
                    "Test schema survived "
                    "the outer rollback"
                )


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
            session.scalar(
                select(func.count())
                .select_from(model)
            )
            for model in MODELS
        )


def build_text_pdf(text: str) -> bytes:
    """
    Build a small deterministic text PDF fully in memory.
    """

    pdf = pymupdf.open()

    try:
        page = pdf.new_page()

        page.insert_text(
            (72, 72),
            text,
        )

        return pdf.tobytes()

    finally:
        pdf.close()


def build_blank_pdf() -> bytes:
    """
    Build a valid PDF containing one genuinely blank page.
    """

    pdf = pymupdf.open()

    try:
        pdf.new_page()

        return pdf.tobytes()

    finally:
        pdf.close()


def assert_saved_document(
    factory,
    result,
    file_name,
    *,
    file_type="txt",
    expected_contents=("AA", "BB", "CC"),
):
    assert result.status == "success"

    assert (
        isinstance(result.document_id, str)
        and result.document_id
    )

    assert result.file_name == file_name

    assert (
        result.chunk_count
        == len(expected_contents)
    )

    # A fresh session must see the service's
    # committed savepoint.
    with factory() as session:
        document = session.get(
            DocumentORM,
            result.document_id,
        )

        assert document is not None

        assert (
            document.document_id
            == result.document_id
        )

        assert (
            document.file_name
            == file_name
        )

        assert (
            document.file_type
            == file_type
        )

        chunks = session.scalars(
            select(ChunkORM)
            .where(
                ChunkORM.document_id
                == result.document_id
            )
            .order_by(
                ChunkORM.chunk_id
            )
        ).all()

        # IDs encode chunk positions;
        # no fixed document UUID.
        expected_ids = [
            (
                f"{result.document_id}"
                f"_chunk_{i}"
            )
            for i in range(
                len(expected_contents)
            )
        ]

        assert [
            chunk.chunk_id
            for chunk in chunks
        ] == expected_ids

        assert [
            chunk.content
            for chunk in chunks
        ] == list(expected_contents)

        assert all(
            chunk.document_id
            == result.document_id
            for chunk in chunks
        )

        rows = session.execute(
            select(
                ChunkEmbeddingORM,
                func.vector_dims(
                    ChunkEmbeddingORM.embedding
                ),
            )
            .join(
                ChunkORM,
                ChunkORM.chunk_id
                == ChunkEmbeddingORM.chunk_id,
            )
            .where(
                ChunkORM.document_id
                == result.document_id
            )
            .order_by(
                ChunkEmbeddingORM.chunk_id,
                ChunkEmbeddingORM.embedding_model,
            )
        ).all()

        assert [
            embedding.chunk_id
            for embedding, _ in rows
        ] == expected_ids

        for chunk, (
            embedding,
            stored_dimension,
        ) in zip(
            chunks,
            rows,
        ):
            assert (
                embedding.embedding_model
                == FakeEmbedder.model_name
            )

            assert (
                stored_dimension
                == FakeEmbedder.dimension
            )

            vector = np.asarray(
                embedding.embedding,
                dtype=float,
            )

            assert vector.shape == (
                FakeEmbedder.dimension,
            )

            assert np.isfinite(
                vector
            ).all()

            assert (
                np.linalg.norm(vector)
                > 0
            )

            np.testing.assert_allclose(
                vector,
                FakeEmbedder.vectors[
                    chunk.content
                ],
                rtol=1e-6,
                atol=1e-6,
            )


# =========================================================
# Phase 3.1 — TXT MANUAL KEEP
# =========================================================


def test_txt_upload_persists_document_chunks_and_vectors(
    service,
    session_factory,
):
    entry, embedder = service

    result = entry.ingest_txt(
        "rag.txt",
        b"AABBCC",
        chunk_size=2,
    )

    assert get_counts(
        session_factory
    ) == (
        1,
        3,
        3,
    )

    assert_saved_document(
        session_factory,
        result,
        "rag.txt",
    )

    assert embedder.batches == [
        [
            "AA",
            "BB",
            "CC",
        ]
    ]


def test_sequential_duplicate_names_persist_distinct_documents(
    service,
    session_factory,
):
    entry, embedder = service

    first = entry.ingest_txt(
        "rag.txt",
        b"AABBCC",
        chunk_size=2,
    )

    assert get_counts(
        session_factory
    ) == (
        1,
        3,
        3,
    )

    second = entry.ingest_txt(
        "rag.txt",
        b"AABBCC",
        chunk_size=2,
    )

    assert (
        first.document_id
        != second.document_id
    )

    assert [
        first.file_name,
        second.file_name,
    ] == [
        "rag.txt",
        "rag(1).txt",
    ]

    assert get_counts(
        session_factory
    ) == (
        2,
        6,
        6,
    )

    assert_saved_document(
        session_factory,
        first,
        "rag.txt",
    )

    assert_saved_document(
        session_factory,
        second,
        "rag(1).txt",
    )

    assert embedder.batches == [
        [
            "AA",
            "BB",
            "CC",
        ],
        [
            "AA",
            "BB",
            "CC",
        ],
    ]


def test_empty_file_returns_empty_without_writes(
    service,
    session_factory,
):
    entry, embedder = service

    result = entry.ingest_txt(
        "empty.txt",
        b"",
        chunk_size=2,
    )

    assert result.status == "empty"

    assert (
        isinstance(
            result.document_id,
            str,
        )
        and result.document_id
    )

    assert (
        result.file_name
        == "empty.txt"
    )

    assert (
        result.chunk_count
        == 0
    )

    assert embedder.batches == []

    assert get_counts(
        session_factory
    ) == (
        0,
        0,
        0,
    )


def test_invalid_utf8_propagates_without_writes(
    service,
    session_factory,
):
    entry, embedder = service

    with pytest.raises(
        UnicodeDecodeError
    ):
        entry.ingest_txt(
            "invalid.txt",
            b"\xff\xfe\xff",
            chunk_size=2,
        )

    assert embedder.batches == []

    assert get_counts(
        session_factory
    ) == (
        0,
        0,
        0,
    )


def test_embedding_insert_failure_rolls_back_all_rows(
    service,
    session_factory,
):
    entry, embedder = service

    failure = RuntimeError(
        "controlled file ingestion "
        "embedding insert failure"
    )

    counts_before_failure = []

    def fail_embedding_insert(
        mapper,
        connection,
        target,
    ):
        # Same translated connection:
        # Document and Chunk have already flushed.
        counts_before_failure.append(
            tuple(
                connection.scalar(
                    select(func.count())
                    .select_from(model)
                )
                for model in MODELS
            )
        )

        raise failure

    event.listen(
        ChunkEmbeddingORM,
        "before_insert",
        fail_embedding_insert,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "controlled file ingestion "
                "embedding insert failure"
            ),
        ) as raised:
            entry.ingest_txt(
                "failed.txt",
                b"AABBCC",
                chunk_size=2,
            )

        assert (
            raised.value
            is failure
        )

    finally:
        event.remove(
            ChunkEmbeddingORM,
            "before_insert",
            fail_embedding_insert,
        )

    assert not event.contains(
        ChunkEmbeddingORM,
        "before_insert",
        fail_embedding_insert,
    )

    assert counts_before_failure == [
        (
            1,
            3,
            0,
        )
    ]

    assert embedder.batches == [
        [
            "AA",
            "BB",
            "CC",
        ]
    ]

    assert get_counts(
        session_factory
    ) == (
        0,
        0,
        0,
    )


# =========================================================
# Phase 3.2 — Markdown MANUAL KEEP
# =========================================================


@pytest.mark.parametrize(
    "file_name",
    [
        "rag.md",
        "rag.markdown",
    ],
)
def test_markdown_upload_persists_original_text(
    service,
    session_factory,
    file_name,
):
    entry, embedder = service

    raw_text = (
        "# RAG\n"
        "- one\n"
        "* two"
    )

    # Independent literal expectations
    # for the existing six-character chunker.
    expected_contents = [
        "# RAG\n",
        "- one\n",
        "* two",
    ]

    assert (
        "".join(expected_contents)
        == raw_text
    )

    result = entry.ingest_file(
        file_name,
        raw_text.encode("utf-8"),
        chunk_size=6,
    )

    assert get_counts(
        session_factory
    ) == (
        1,
        3,
        3,
    )

    assert_saved_document(
        session_factory,
        result,
        file_name,
        file_type="markdown",
        expected_contents=expected_contents,
    )

    assert embedder.batches == [
        expected_contents
    ]


def test_markdown_duplicate_names_preserve_first_document(
    service,
    session_factory,
):
    entry, embedder = service

    first_text = (
        "# RAG\n"
        "- one\n"
        "* two"
    )

    second_text = (
        "# NEW\n"
        "- two\n"
        "* end"
    )

    first_contents = [
        "# RAG\n",
        "- one\n",
        "* two",
    ]

    second_contents = [
        "# NEW\n",
        "- two\n",
        "* end",
    ]

    assert (
        "".join(first_contents)
        == first_text
    )

    assert (
        "".join(second_contents)
        == second_text
    )

    first = entry.ingest_file(
        "rag.md",
        first_text.encode("utf-8"),
        chunk_size=6,
    )

    assert get_counts(
        session_factory
    ) == (
        1,
        3,
        3,
    )

    assert_saved_document(
        session_factory,
        first,
        "rag.md",
        file_type="markdown",
        expected_contents=first_contents,
    )

    second = entry.ingest_file(
        "rag.md",
        second_text.encode("utf-8"),
        chunk_size=6,
    )

    assert (
        first.document_id
        != second.document_id
    )

    assert [
        first.file_name,
        second.file_name,
    ] == [
        "rag.md",
        "rag(1).md",
    ]

    assert get_counts(
        session_factory
    ) == (
        2,
        6,
        6,
    )

    # Re-read the first document
    # after the second upload
    # to detect overwrites.
    assert_saved_document(
        session_factory,
        first,
        "rag.md",
        file_type="markdown",
        expected_contents=first_contents,
    )

    assert_saved_document(
        session_factory,
        second,
        "rag(1).md",
        file_type="markdown",
        expected_contents=second_contents,
    )

    assert embedder.batches == [
        first_contents,
        second_contents,
    ]


def test_invalid_markdown_utf8_propagates_without_writes(
    service,
    session_factory,
):
    entry, embedder = service

    with pytest.raises(
        UnicodeDecodeError
    ):
        entry.ingest_file(
            "invalid.md",
            b"\xff\xfe\xff",
            chunk_size=6,
        )

    assert embedder.batches == []

    assert get_counts(
        session_factory
    ) == (
        0,
        0,
        0,
    )


# =========================================================
# Phase 3.3 — PDF MANUAL KEEP
# =========================================================


def test_pdf_upload_persists_document_chunks_and_vectors(
    service,
    session_factory,
):
    entry, embedder = service

    file_bytes = build_text_pdf(
        "AABBCC"
    )

    result = entry.ingest_file(
        "rag.pdf",
        file_bytes,
        chunk_size=2,
    )

    assert get_counts(
        session_factory
    ) == (
        1,
        3,
        3,
    )

    assert_saved_document(
        session_factory,
        result,
        "rag.pdf",
        file_type="pdf",
        expected_contents=(
            "AA",
            "BB",
            "CC",
        ),
    )

    assert embedder.batches == [
        [
            "AA",
            "BB",
            "CC",
        ]
    ]


def test_pdf_duplicate_names_persist_distinct_documents(
    service,
    session_factory,
):
    entry, embedder = service

    file_bytes = build_text_pdf(
        "AABBCC"
    )

    first = entry.ingest_file(
        "rag.pdf",
        file_bytes,
        chunk_size=2,
    )

    assert get_counts(
        session_factory
    ) == (
        1,
        3,
        3,
    )

    second = entry.ingest_file(
        "rag.pdf",
        file_bytes,
        chunk_size=2,
    )

    assert (
        first.document_id
        != second.document_id
    )

    assert [
        first.file_name,
        second.file_name,
    ] == [
        "rag.pdf",
        "rag(1).pdf",
    ]

    assert get_counts(
        session_factory
    ) == (
        2,
        6,
        6,
    )

    # Re-read both documents to detect
    # accidental overwrite.
    assert_saved_document(
        session_factory,
        first,
        "rag.pdf",
        file_type="pdf",
        expected_contents=(
            "AA",
            "BB",
            "CC",
        ),
    )

    assert_saved_document(
        session_factory,
        second,
        "rag(1).pdf",
        file_type="pdf",
        expected_contents=(
            "AA",
            "BB",
            "CC",
        ),
    )

    assert embedder.batches == [
        [
            "AA",
            "BB",
            "CC",
        ],
        [
            "AA",
            "BB",
            "CC",
        ],
    ]


def test_blank_pdf_failure_propagates_without_writes(
    service,
    session_factory,
):
    entry, embedder = service

    file_bytes = build_blank_pdf()

    with pytest.raises(
        NoExtractableTextError
    ):
        entry.ingest_file(
            "blank.pdf",
            file_bytes,
            chunk_size=2,
        )

    assert embedder.batches == []

    assert get_counts(
        session_factory
    ) == (
        0,
        0,
        0,
    )