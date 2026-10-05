import numpy as np
import pytest

from backend.rag.ingestion.chunk_embedding import build_chunk_vectors
from backend.rag.contracts import Chunk


class FakeBatchEmbedder:
    """
    正常测试用 Embedder。

    输入多少条文本，
    就返回多少个确定向量。
    """

    def embed_batch(self, texts: list[str]):
        vectors = []

        for index, _ in enumerate(texts):
            vectors.append(
                np.array(
                    [float(index + 1), 1.0]
                )
            )

        return np.array(vectors)


class WrongCountEmbedder:
    """
    故意返回错误数量的 Embedding，
    用于验证数量一致性 Contract。
    """

    def embed_batch(self, texts: list[str]):
        return np.array([
            [1.0, 0.0]
        ])


def test_build_chunk_vectors_happy_path():
    """
    正常路径：

    Chunk[]
        ↓
    Batch Embedding
        ↓
    chunk_id -> vector
    """

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content A",
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Content B",
        ),
    ]

    result = build_chunk_vectors(
        chunks=chunks,
        embedder=FakeBatchEmbedder(),
    )

    assert set(result.keys()) == {
        "chunk_1",
        "chunk_2",
    }

    assert np.array_equal(
        result["chunk_1"],
        np.array([1.0, 1.0]),
    )

    assert np.array_equal(
        result["chunk_2"],
        np.array([2.0, 1.0]),
    )


def test_build_chunk_vectors_empty_chunks_returns_empty_dict():
    """
    Empty Corpus 是合法状态。
    """

    result = build_chunk_vectors(
        chunks=[],
        embedder=FakeBatchEmbedder(),
    )

    assert result == {}


def test_build_chunk_vectors_raises_when_embedding_count_mismatch():
    """
    Embedding 数量与 Chunk 数量不一致：

    属于数据一致性错误，
    不能依赖 zip 静默截断。
    """

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content A",
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            content="Content B",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="embedding result count does not match chunk count",
    ):
        build_chunk_vectors(
            chunks=chunks,
            embedder=WrongCountEmbedder(),
        )


def test_build_chunk_vectors_raises_when_chunk_id_is_duplicated():
    """
    Duplicate chunk_id 不能被 dict 静默覆盖。
    """

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content A",
        ),
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Content B",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate chunk_id found",
    ):
        build_chunk_vectors(
            chunks=chunks,
            embedder=FakeBatchEmbedder(),
        )