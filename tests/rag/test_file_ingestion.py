
import pytest

from sqlalchemy import create_engine, insert, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.models.rag_document import DocumentORM
from backend.rag.ingestion.file_ingestion import (
    FileIngestionService,
)
from backend.rag.ingestion.rag_ingestion import IngestionResult


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    DocumentORM.__table__.create(engine)
    factory = sessionmaker(bind=engine)

    yield factory
    engine.dispose()


class FakeIngestion:
    """模拟下游成功写入 Document，不执行 Embedding。"""

    def __init__(self, session_factory):
        self._session_factory = session_factory
        self.calls = []
        self.failure = None

    def ingest(self, document, raw_text, chunk_size):
        self.calls.append((document, raw_text, chunk_size))

        if self.failure is not None:
            raise self.failure

        if not raw_text.strip():
            return IngestionResult(
                document_id=document.document_id,
                status="empty",
                chunk_count=0,
            )

        with self._session_factory.begin() as session:
            session.execute(
                insert(DocumentORM.__table__).values(
                    document_id=document.document_id,
                    file_name=document.file_name,
                    file_type=document.file_type,
                )
            )

        return IngestionResult(
            document_id=document.document_id,
            status="success",
            chunk_count=(len(raw_text) + chunk_size - 1)
            // chunk_size,
        )


@pytest.fixture
def service(session_factory):
    downstream = FakeIngestion(session_factory)

    return (
        FileIngestionService(
            ingestion_service=downstream,
            session_factory=session_factory,
        ),
        downstream,
    )


def saved_names(session_factory):
    with session_factory() as session:
        return list(
            session.scalars(
                select(DocumentORM.file_name).order_by(
                    DocumentORM.file_name
                )
            ).all()
        )


def test_normal_upload(service, session_factory):
    entry, downstream = service

    result = entry.ingest_txt(
        file_name="rag.txt",
        file_bytes=b"ABCDEF",
        chunk_size=2,
    )

    assert result.status == "success"
    assert result.file_name == "rag.txt"
    assert result.chunk_count == 3
    assert result.document_id

    document, raw_text, size = downstream.calls[0]
    assert document.document_id == result.document_id
    assert document.file_type == "txt"
    assert raw_text == "ABCDEF"
    assert size == 2
    assert saved_names(session_factory) == ["rag.txt"]


def test_duplicate_names(service, session_factory):
    entry, _ = service

    results = [
        entry.ingest_txt("rag.txt", b"ABC", 2)
        for _ in range(3)
    ]

    assert [r.file_name for r in results] == [
        "rag.txt",
        "rag(1).txt",
        "rag(2).txt",
    ]

    assert len({r.document_id for r in results}) == 3
    assert len(saved_names(session_factory)) == 3


def test_same_content_different_names(service, session_factory):
    entry, _ = service

    a = entry.ingest_txt("a.txt", b"same", 2)
    b = entry.ingest_txt("b.txt", b"same", 2)

    assert a.document_id != b.document_id
    assert saved_names(session_factory) == ["a.txt", "b.txt"]


def test_existing_suffix_is_skipped(service, session_factory):
    entry, _ = service

    entry.ingest_txt("rag.txt", b"A", 2)
    entry.ingest_txt("rag(1).txt", b"B", 2)

    result = entry.ingest_txt("rag.txt", b"C", 2)

    assert result.file_name == "rag(2).txt"


def test_empty_does_not_reserve_name(service, session_factory):
    entry, _ = service

    empty = entry.ingest_txt("empty.txt", b"  \n ", 2)

    assert empty.status == "empty"
    assert empty.chunk_count == 0
    assert saved_names(session_factory) == []

    next_result = entry.ingest_txt("empty.txt", b"ABC", 2)

    assert next_result.file_name == "empty.txt"


def test_invalid_encoding_propagates(service, session_factory):
    entry, downstream = service

    with pytest.raises(UnicodeDecodeError):
        entry.ingest_txt("bad.txt", b"\xff\xfe\xff", 2)

    assert downstream.calls == []
    assert saved_names(session_factory) == []


def test_ingestion_failure_propagates(service, session_factory):
    entry, downstream = service
    downstream.failure = RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        entry.ingest_txt("failed.txt", b"ABC", 2)

    assert saved_names(session_factory) == []


@pytest.mark.parametrize(
    "file_name",
    ["", " bad.txt", "../bad.txt", r"C:\bad.txt", "a.pdf"],
)
def test_invalid_file_name(service, file_name):
    entry, downstream = service

    with pytest.raises(ValueError):
        entry.ingest_txt(file_name, b"ABC", 2)

    assert downstream.calls == []


def test_invalid_chunk_size(service):
    entry, downstream = service

    with pytest.raises(ValueError, match="chunk_size"):
        entry.ingest_txt("rag.txt", b"ABC", 0)

    assert downstream.calls == []
