
import pytest
from pydantic import ValidationError

from backend.rag.ingestion.markdown_loader import MarkdownLoader
from backend.rag.ingestion.chunker import chunk_document


@pytest.fixture
def loader():
    return MarkdownLoader()


@pytest.mark.parametrize(
    "file_name",
    ["rag.md", "rag.markdown", "RAG.MD"],
)
def test_markdown_preserves_original_content(loader, file_name):
    text = (
        "# RAG\n\n"
        "## Retrieval\n"
        "- Dense Retrieval\n"
        "- Sparse Retrieval\n\n"
        "```python\n"
        "x = 1\n"
        "```\n"
    )

    document, raw_text = loader.load(
        file_name=file_name,
        file_bytes=text.encode("utf-8"),
        document_id="doc_001",
    )

    assert document.document_id == "doc_001"
    assert document.file_name == file_name
    assert document.file_type == "markdown"
    assert raw_text == text


def test_markdown_utf8_bom(loader):
    document, raw_text = loader.load(
        file_name="bom.md",
        file_bytes=b"\xef\xbb\xbf# Hello",
        document_id="doc_002",
    )

    assert document.file_type == "markdown"
    assert raw_text == "# Hello"


@pytest.mark.parametrize(
    "content",
    [b"", b"  \n\t  "],
)
def test_empty_markdown_preserved(loader, content):
    _, raw_text = loader.load(
        file_name="empty.md",
        file_bytes=content,
        document_id="doc_003",
    )

    assert raw_text == content.decode("utf-8")


@pytest.mark.parametrize(
    "file_name",
    ["rag.txt", "rag.pdf"],
)
def test_unsupported_file_type(loader, file_name):
    with pytest.raises(ValueError, match="only accepts"):
        loader.load(
            file_name=file_name,
            file_bytes=b"hello",
            document_id="doc_004",
        )


def test_invalid_utf8_raises(loader):
    with pytest.raises(UnicodeDecodeError):
        loader.load(
            file_name="bad.md",
            file_bytes=b"\xff\xfe\xff",
            document_id="doc_005",
        )


def test_invalid_document_id(loader):
    with pytest.raises(ValidationError):
        loader.load(
            file_name="rag.md",
            file_bytes=b"# RAG",
            document_id="  ",
        )


def test_markdown_to_chunker_contract(loader):
    text = "# RAG\n\nRetrieval"

    document, raw_text = loader.load(
        file_name="rag.md",
        file_bytes=text.encode("utf-8"),
        document_id="doc_006",
    )

    chunks = chunk_document(
        document=document,
        raw_text=raw_text,
        chunk_size=len(text),
    )

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].document_id == document.document_id
