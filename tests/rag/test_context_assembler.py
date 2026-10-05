from backend.rag.context_assembler import ContextAssembler
from backend.rag.contracts import (
    AssembledContext,
    RetrievedChunk,
)


def test_assemble_returns_valid_context():
    """
    正常输入应该得到完整 AssembledContext。
    """

    assembler = ContextAssembler()

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content A",
            source="a.md",
            score=0.95,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            content="Content B",
            source="b.pdf",
            score=0.80,
        ),
    ]

    result = assembler.assemble(chunks)

    assert isinstance(result, AssembledContext)

    assert "Content A" in result.context_text
    assert "Content B" in result.context_text

    assert 'source="a.md"' in result.context_text
    assert 'source="b.pdf"' in result.context_text

    assert result.used_chunks == chunks


def test_assemble_empty_chunks_returns_empty_context():
    """
    Empty Retrieval 是合法状态，不应该抛异常。
    """

    assembler = ContextAssembler()

    result = assembler.assemble([])

    assert result == AssembledContext(
        context_text="",
        used_chunks=[],
    )


def test_assemble_preserves_chunk_order():
    """
    ContextAssembler 不拥有 Ranking 权限。

    输入 Chunk 的顺序必须同时保留在：
    1. used_chunks
    2. context_text
    """

    assembler = ContextAssembler()

    chunk_high = RetrievedChunk(
        chunk_id="chunk_high",
        document_id="doc_1",
        content="High relevance content",
        source="high.md",
        score=0.95,
    )

    chunk_low = RetrievedChunk(
        chunk_id="chunk_low",
        document_id="doc_2",
        content="Low relevance content",
        source="low.md",
        score=0.60,
    )

    result = assembler.assemble(
        [chunk_high, chunk_low]
    )

    # used_chunks 顺序不能变化
    assert result.used_chunks == [
        chunk_high,
        chunk_low,
    ]

    # LLM Context 中的顺序也不能变化
    high_position = result.context_text.index(
        "High relevance content"
    )

    low_position = result.context_text.index(
        "Low relevance content"
    )

    assert high_position < low_position


def test_assemble_uses_expected_context_format():
    """
    每个 RetrievedChunk 对应一个独立 Context Block。
    """

    assembler = ContextAssembler()

    chunks = [
        RetrievedChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content A",
            source="a.md",
            score=0.95,
        ),
        RetrievedChunk(
            chunk_id="chunk_2",
            document_id="doc_2",
            content="Content B",
            source="b.pdf",
            score=0.80,
        ),
    ]

    result = assembler.assemble(chunks)

    expected = (
        "<document_context>\n"
        '<chunk id="chunk_1" source="a.md">\n'
        "Content A\n"
        "</chunk>\n\n"
        '<chunk id="chunk_2" source="b.pdf">\n'
        "Content B\n"
        "</chunk>\n"
        "</document_context>"
    )

    assert result.context_text == expected


def test_assemble_exposes_citation_handle_but_not_internal_metadata():
    """
    Generation Context 只暴露 LLM 真正需要的信息：

    允许：
    - chunk_id
    - source
    - content

    不允许：
    - score
    - document_id
    """

    assembler = ContextAssembler()

    chunk = RetrievedChunk(
        chunk_id="internal_chunk_123",
        document_id="internal_doc_456",
        content="Visible knowledge",
        source="manual.pdf",
        score=0.876543,
    )

    result = assembler.assemble([chunk])

    context = result.context_text

    # LLM 需要的信息
    assert "Visible knowledge" in context
    assert "manual.pdf" in context
    # Citation Attribution 需要稳定 chunk_id
    assert "internal_chunk_123" in context

    # 系统内部 Metadata
    assert "internal_doc_456" not in context
    assert "0.876543" not in context
