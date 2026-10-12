from typing import Protocol

import numpy as np

from backend.rag.contracts import DenseRetrieverResult


class DenseRetriever(Protocol):
    """
    Dense Retrieval 的统一 Contract。

    RAGRetriever 只依赖本 Contract，不感知具体实现：

    - ExactDenseSearch          In-Memory Exact Search
    - PostgreSQLVectorSearch    PostgreSQL + pgvector

    实现方负责自己的 storage-specific logic：

    - 语料 / 向量空间的取得
    - Similarity 计算
    - Ranking（1-based dense_rank）

    Contract：

    search(query_vector, top_k)
        ↓
    list[DenseRetrieverResult]

    语料与 Embedding Model 身份在实现对象构造时绑定，
    因此 search() 只接收 Query Vector。
    """

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[DenseRetrieverResult]:
        ...
