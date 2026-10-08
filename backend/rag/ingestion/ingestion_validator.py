
import numpy as np

from backend.rag.contracts import Chunk


def validate_ingestion_data(
    chunks: list[Chunk],
    chunk_vectors: dict[str, np.ndarray],
    embedding_model: str,
    embedding_dimension: int,
) -> None:
    """
    校验准备持久化的 RAG 数据。

    只负责校验，不修改数据，不执行数据库操作。
    校验失败直接抛出异常。
    """

    # 1. 非空 Chunk 集合
    # Empty Document 应由 Orchestrator 提前处理
    if not chunks:
        raise ValueError(
            "ingestion validation requires non-empty chunks"
        )

    # 2. Embedding Model Identity
    if (
        not isinstance(embedding_model, str)
        or not embedding_model.strip()
    ):
        raise ValueError(
            "embedding_model must not be blank"
        )

    if len(embedding_model) > 255:
        raise ValueError(
            "embedding_model exceeds database column limit"
        )

    # 3. Embedding Dimension
    if (
        isinstance(embedding_dimension, bool)
        or not isinstance(embedding_dimension, int)
        or embedding_dimension <= 0
    ):
        raise ValueError(
            "embedding_dimension must be a positive integer"
        )

    # 4. Chunk ID 唯一性
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("duplicate chunk_id found")

    # 5. Chunk 与 Vector 必须一一对应
    chunk_id_set = set(chunk_ids)
    vector_id_set = set(chunk_vectors.keys())

    if chunk_id_set != vector_id_set:
        missing = chunk_id_set - vector_id_set
        extra = vector_id_set - chunk_id_set

        raise ValueError(
            "chunk/vector ID mismatch: "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    # 6. 检查每个 Vector
    for chunk_id in chunk_ids:
        vector = np.asarray(
            chunk_vectors[chunk_id],
            dtype=float,
        )

        # 必须是一维向量
        if vector.ndim != 1:
            raise ValueError(
                f"vector[{chunk_id}] must be 1-D"
            )

        # 维度必须匹配实际 Embedding Model
        if vector.size != embedding_dimension:
            raise ValueError(
                f"vector[{chunk_id}] dimension mismatch: "
                f"expected {embedding_dimension}, "
                f"got {vector.size}"
            )

        # 不允许 NaN / Inf
        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"vector[{chunk_id}] contains NaN or Inf"
            )

        # 不允许零向量
        if not np.any(vector):
            raise ValueError(
                f"vector[{chunk_id}] must not be zero"
            )
