import pytest

from backend.rag.contracts import (
    Chunk,
    FusedRetrieverResult,
    RerankDocument,
)
from backend.rag.retrieval.rerank_document_builder import (
    RerankDocumentBuilder,
)


CHUNK_STORE = {
    "chunk_1": Chunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        content="Content 1",
    ),
    "chunk_2": Chunk(
        chunk_id="chunk_2",
        document_id="doc_1",
        content="Content 2",
    ),
}


def fused(chunk_id, rrf_rank):
    return FusedRetrieverResult(
        chunk_id=chunk_id,
        rrf_score=1.0 / rrf_rank,
        rrf_rank=rrf_rank,
    )


# ==================================================
# 正常路径
# ==================================================

def test_build_returns_documents_in_fused_order():
    """
    RerankDocument 的顺序就是 previous_rank，
    因此必须保持 Fusion 的顺序。
    """

    documents = RerankDocumentBuilder().build(
        [
            fused("chunk_2", 1),
            fused("chunk_1", 2),
        ],
        CHUNK_STORE,
    )

    assert [
        document.chunk_id
        for document in documents
    ] == [
        "chunk_2",
        "chunk_1",
    ]

    assert [
        document.content
        for document in documents
    ] == [
        "Content 2",
        "Content 1",
    ]


def test_returns_rerank_document_contract():
    documents = RerankDocumentBuilder().build(
        [fused("chunk_1", 1)],
        CHUNK_STORE,
    )

    assert all(
        isinstance(document, RerankDocument)
        for document in documents
    )


def test_does_not_leak_hydration_fields():
    """
    Reranker 只需要 chunk_id + content，
    不允许携带 document_id / source 等 Hydration 数据。
    """

    documents = RerankDocumentBuilder().build(
        [fused("chunk_1", 1)],
        CHUNK_STORE,
    )

    document = documents[0]

    assert not hasattr(document, "document_id")
    assert not hasattr(document, "source")
    assert not hasattr(document, "retrieval_rank")


# ==================================================
# 边界情况
# ==================================================

def test_empty_fused_results_returns_empty_list():
    documents = RerankDocumentBuilder().build(
        [],
        CHUNK_STORE,
    )

    assert documents == []


def test_raises_when_chunk_is_missing():
    """
    Fusion 指向不存在的 Chunk：
    属于数据一致性错误，不能静默 skip。
    """

    with pytest.raises(KeyError):
        RerankDocumentBuilder().build(
            [fused("missing_chunk", 1)],
            CHUNK_STORE,
        )
