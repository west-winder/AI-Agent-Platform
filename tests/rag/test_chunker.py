import pytest

from backend.rag.contracts import Document
from backend.rag.ingestion.chunker import chunk_document


def build_document() -> Document:
    return Document(
        document_id="doc_1",
        file_name="demo.txt",
        file_type="txt",
    )


def test_chunk_document_empty_text_returns_empty_list():
    """
    Empty Text 是合法输入。
    """

    result = chunk_document(
        document=build_document(),
        raw_text="",
        chunk_size=5,
    )

    assert result == []


def test_chunk_document_happy_path():
    """
    正常固定长度切分。
    同时验证：
    - content
    - chunk_id
    - document_id
    - ranking / 原始顺序
    """

    result = chunk_document(
        document=build_document(),
        raw_text="ABCDEFGHIJ",
        chunk_size=5,
    )

    assert len(result) == 2

    assert result[0].chunk_id == "doc_1_chunk_0"
    assert result[0].document_id == "doc_1"
    assert result[0].content == "ABCDE"

    assert result[1].chunk_id == "doc_1_chunk_1"
    assert result[1].document_id == "doc_1"
    assert result[1].content == "FGHIJ"


def test_chunk_document_keeps_remainder_chunk():
    """
    最后不足 chunk_size 的内容不能被丢弃。
    """

    result = chunk_document(
        document=build_document(),
        raw_text="ABCDEFGHIJKL",
        chunk_size=5,
    )

    assert len(result) == 3

    assert result[0].content == "ABCDE"
    assert result[1].content == "FGHIJ"
    assert result[2].content == "KL"

    assert result[2].chunk_id == "doc_1_chunk_2"


def test_chunk_document_raises_when_chunk_size_is_invalid():
    """
    chunk_size <= 0 属于 Invalid Configuration。
    """

    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than 0",
    ):
        chunk_document(
            document=build_document(),
            raw_text="ABCDEFGHIJ",
            chunk_size=0,
        )


def test_chunk_document_whitespace_only_returns_empty_list():
    """
    只有 whitespace 时没有有效知识内容。
    """

    result = chunk_document(
        document=build_document(),
        raw_text="   \n\t   ",
        chunk_size=5,
    )

    assert result == []