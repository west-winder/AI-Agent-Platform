
import math

import numpy as np
from sqlalchemy import select

from backend.database.database import SessionLocal

# 加载相关 ORM，保证 Relationship 可以正常解析
from backend.models.rag_document import DocumentORM
from backend.models.rag_chunk import ChunkORM
from backend.models.rag_chunk_embedding import ChunkEmbeddingORM

from backend.rag.contracts import SearchResult


class EmbeddingCompatibilityError(RuntimeError):
    """知识库存在 Chunk，但当前 Embedding Model 无可用向量。"""


class PostgreSQLVectorSearch:
    """
    PostgreSQL + pgvector Exact Vector Search。

    职责：
    1. 校验 Query Vector 和检索参数
    2. 根据 embedding_model 过滤向量
    3. 使用 pgvector 计算 Cosine Distance
    4. 执行 Ranking + Top-K
    5. Distance -> Similarity
    6. 返回 SearchResult[]

    不负责：
    - Query Embedding
    - Chunk Embedding
    - Hydration
    - Context Assembly
    - LLM Generation
    """

    def __init__(
        self,
        session_factory=SessionLocal,
    ):
        self._session_factory = session_factory

    def search(
        self,
        query_vector: np.ndarray,
        embedding_model: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        # ==========================================
        # 1. 参数校验
        # ==========================================

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be int")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if (
            not isinstance(embedding_model, str)
            or not embedding_model.strip()
        ):
            raise ValueError(
                "embedding_model must not be blank"
            )

        embedding_model = embedding_model.strip()

        query_vector = np.asarray(
            query_vector,
            dtype=float,
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "query_vector must be 1-D"
            )

        if query_vector.size == 0:
            raise ValueError(
                "query_vector must not be empty"
            )

        if not np.all(np.isfinite(query_vector)):
            raise ValueError(
                "query_vector contains NaN or Inf"
            )

        if not np.any(query_vector):
            raise ValueError(
                "query_vector must not be zero"
            )

        # ==========================================
        # 2. 构造 pgvector Distance Expression
        # ==========================================

        distance = (
            ChunkEmbeddingORM.embedding.cosine_distance(
                query_vector.tolist()
            )
        )

        # ==========================================
        # 3. 构造 Exact Search SQL
        # ==========================================

        statement = (
            select(
                ChunkEmbeddingORM.chunk_id,
                (1.0 - distance).label("score"),
            )
            .where(
                ChunkEmbeddingORM.embedding_model
                == embedding_model
            )
            .order_by(
                distance.asc(),
                ChunkEmbeddingORM.chunk_id.asc(),
            )
            .limit(top_k)
        )

        # ==========================================
        # 4. 执行数据库查询
        # ==========================================

        with self._session_factory() as session:

            rows = session.execute(
                statement
            ).all()

            # --------------------------------------
            # 4.1 Empty / Model Compatibility
            # --------------------------------------

            if not rows:

                # 检查知识库是否存在 Chunk
                existing_chunk = session.execute(
                    select(ChunkORM.chunk_id).limit(1)
                ).first()

                if existing_chunk is not None:
                    raise EmbeddingCompatibilityError(
                        "Knowledge base contains chunks, "
                        "but no embeddings are available "
                        f"for model '{embedding_model}'"
                    )

                # 真正的 Empty Corpus
                return []

        # ==========================================
        # 5. 转换为 Application Contract
        # ==========================================

        results: list[SearchResult] = []

        for chunk_id, score in rows:

            if score is None or not math.isfinite(float(score)):
                raise ValueError(
                    "PostgreSQL returned invalid "
                    f"similarity score for '{chunk_id}'"
                )

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=float(score),
                )
            )

        return results
