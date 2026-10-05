import numpy as np

from backend.rag.contracts import Chunk


def build_chunk_vectors(
    chunks: list[Chunk],
    embedder,
) -> dict[str, np.ndarray]:
    """
    将 Chunk[] 批量转换成：

        chunk_id -> embedding vector

    这是 RAG 的 Corpus-side Embedding。

    它发生在知识库准备阶段，
    而不是每次 Query Retrieval 时重新执行。
    """

    # --------------------------------------------------
    # 1. Empty Corpus 是合法状态
    # --------------------------------------------------

    if not chunks:
        return {}

    # --------------------------------------------------
    # 2. 检查 chunk_id 唯一性
    # --------------------------------------------------

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "duplicate chunk_id found"
        )

    # --------------------------------------------------
    # 3. 提取需要 Embedding 的正文
    # --------------------------------------------------

    texts = [
        chunk.content
        for chunk in chunks
    ]

    # --------------------------------------------------
    # 4. Batch Embedding
    # --------------------------------------------------

    vectors = embedder.embed_batch(
        texts
    )

    # --------------------------------------------------
    # 5. Embedding 数量必须和 Chunk 数量一致
    # --------------------------------------------------

    if len(vectors) != len(chunks):
        raise ValueError(
            "embedding result count does not match chunk count"
        )

    # --------------------------------------------------
    # 6. 显式建立 chunk_id ↔ vector 关系
    # --------------------------------------------------

    chunk_vectors = {
        chunk.chunk_id: np.asarray(
            vector,
            dtype=float,
        )
        for chunk, vector in zip(
            chunks,
            vectors,
        )
    }

    return chunk_vectors