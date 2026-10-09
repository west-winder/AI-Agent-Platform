
import pytest
from pydantic import ValidationError

from backend.rag.ingestion.txt_loader import TXTLoader
from backend.rag.ingestion.chunker import chunk_document


@pytest.fixture
def loader():
    return TXTLoader()


def test_normal_txt(loader):
    text = "RAG 是检索增强生成。\n第二行内容。"

    document, raw_text = loader.load(
        file_name="rag.txt",
        file_bytes=text.encode("utf-8"),
        document_id="doc_001",
    )

    assert document.document_id == "doc_001"
    assert document.file_name == "rag.txt"
    assert document.file_type == "txt"
    assert raw_text == text


def test_utf8_bom(loader):
    document, raw_text = loader.load(
        file_name="bom.txt",
        file_bytes=b"\xef\xbb\xbfHello",
        document_id="doc_002",
    )

    assert document.file_type == "txt"
    assert raw_text == "Hello"


@pytest.mark.parametrize("content", [b"", b"  \n\t  "])
def test_empty_or_whitespace_preserved(loader, content):
    _, raw_text = loader.load(
        file_name="empty.txt",
        file_bytes=content,
        document_id="doc_003",
    )

    assert raw_text == content.decode("utf-8")


def test_invalid_encoding_raises(loader):
    with pytest.raises(UnicodeDecodeError):
        loader.load(
            file_name="bad.txt",
            file_bytes=b"\xff\xfe\xff",
            document_id="doc_004",
        )


def test_unsupported_file_type(loader):
    with pytest.raises(ValueError, match="only accepts"):
        loader.load(
            file_name="report.pdf",
            file_bytes=b"hello",
            document_id="doc_005",
        )


def test_invalid_document_id(loader):
    with pytest.raises(ValidationError):
        loader.load(
            file_name="rag.txt",
            file_bytes=b"hello",
            document_id="  ",
        )


def test_loader_to_chunker_contract(loader):
    document, raw_text = loader.load(
        file_name="demo.txt",
        file_bytes=b"ABCDEF",
        document_id="doc_006",
    )

    chunks = chunk_document(
        document=document,
        raw_text=raw_text,
        chunk_size=3,
    )

    assert [chunk.content for chunk in chunks] == ["ABC", "DEF"]
    assert all(
        chunk.document_id == document.document_id
        for chunk in chunks
    )
