from dataclasses import dataclass

import numpy as np

from backend.embedding.embedder import Embedder


@dataclass
class DenseRetrievedMemory:
    """
    Memory Read 阶段的 Dense Retrieval 候选。

    表示：
    原始 Memory Index
    +
    一条 Memory
    +
    它与 Query 的向量相似度

    index:
        当前 Memory 在传入 search() 的原始 memories 列表中的位置。

        注意：
        index != rank
        index != memory_id
    """

    index: int
    memory: object
    similarity: float


class DenseMemoryRetriever:
    """
    Memory Read 的 Dense Candidate Retrieval。

    当前实现：

    Query
        ↓
    Embedding
        ↓
    Query Vector
        ↓
    Memory Embeddings
        ↓
    Cosine Similarity
        ↓
    Ranking
        ↓
    Top-N Candidates

    当前仍然是：

    Brute-force Dense Retrieval

    本模块只负责 Dense Candidate Generation。

    不负责：

    1. 数据库查询
    2. BM25 Retrieval
    3. RRF
    4. Reranking
    5. LLM Judge
    6. Memory Injection
    7. Metadata Filtering
    """

    def __init__(
        self,
        embedder=None
    ):
        """
        初始化 Dense Retriever。

        参数：
            embedder:
                通用 Embedder。
                支持依赖注入，方便测试。

        如果没有传入，则在真正需要时自动创建 Embedder。
        """

        self._embedder = embedder

    # ==================================================
    # Embedder
    # ==================================================

    def _ensure_embedder(self):
        """
        确保 Embedder 已初始化。
        """

        if self._embedder is None:
            self._embedder = Embedder()

    # ==================================================
    # Cosine Similarity
    # ==================================================

    def _cosine(
        self,
        a,
        b
    ) -> float:
        """
        计算两个向量的 Cosine Similarity。

        参数：
            a:
                向量 A。

            b:
                向量 B。

        返回：
            float
        """

        a = np.asarray(
            a,
            dtype=float
        )

        b = np.asarray(
            b,
            dtype=float
        )

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        similarity = (
            np.dot(a, b)
            / denominator
        )

        return float(similarity)

    # ==================================================
    # Memory Retrieval
    # ==================================================

    def search(
        self,
        query: str,
        memories: list,
        top_n: int = 5
    ) -> list[DenseRetrievedMemory]:
        """
        根据 Query 从 Memory 集合中召回 Top-N Dense 候选。

        参数：
            query:
                用户当前查询。

            memories:
                Memory 对象列表。

                memories 中的位置 index
                会被保存在 DenseRetrievedMemory.index 中，
                用于后续 Hybrid Retrieval / RRF。

            top_n:
                最多返回多少条 Candidate。

        返回：
            list[DenseRetrievedMemory]

        流程：

            Query
              ↓
            Embedding
              ↓
            Query Vector
              ↓
            Memory Embeddings
              ↓
            Cosine Similarity
              ↓
            保留原始 Memory Index
              ↓
            Sort
              ↓
            Top-N
        """

        # --------------------------------------------------
        # 1. 参数校验
        # --------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query必须是str类型"
            )

        if not query.strip():
            raise ValueError(
                "query不能为空"
            )

        if not isinstance(memories, list):
            raise TypeError(
                "memories必须是list"
            )

        if not isinstance(top_n, int):
            raise TypeError(
                "top_n必须是int类型"
            )

        if top_n <= 0:
            raise ValueError(
                "top_n必须大于0"
            )

        # --------------------------------------------------
        # 2. 没有 Memory
        # --------------------------------------------------

        if not memories:
            return []

        # --------------------------------------------------
        # 3. 确保 Embedder
        # --------------------------------------------------

        self._ensure_embedder()

        # --------------------------------------------------
        # 4. Query → Embedding
        # --------------------------------------------------

        query_vector = self._embedder.embed(
            query
        )

        # --------------------------------------------------
        # 5. Memory → Embedding
        # --------------------------------------------------

        memory_texts = [
            memory.content
            for memory in memories
        ]

        memory_vectors = (
            self._embedder.embed_batch(
                memory_texts
            )
        )

        # --------------------------------------------------
        # 6. Similarity Calculation
        # --------------------------------------------------

        scored_memories = []

        for index, (memory, vector) in enumerate(
            zip(
                memories,
                memory_vectors
            )
        ):

            similarity = self._cosine(
                query_vector,
                vector
            )

            scored_memories.append(
                DenseRetrievedMemory(
                    index=index,
                    memory=memory,
                    similarity=similarity
                )
            )

        # --------------------------------------------------
        # 7. Similarity Ranking
        # --------------------------------------------------

        scored_memories.sort(
            key=lambda item: item.similarity,
            reverse=True
        )

        # --------------------------------------------------
        # 8. Top-N
        # --------------------------------------------------

        return scored_memories[:top_n]