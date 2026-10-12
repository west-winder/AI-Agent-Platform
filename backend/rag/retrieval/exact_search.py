import numpy as np

from backend.rag.contracts import DenseRetrieverResult


class ExactDenseSearch:
    """
    RAG 的 In-Memory Exact Dense Search。

    DenseRetriever Contract 的内存实现。

    职责：

    Query Vector
        ↓
    与所有 Chunk Vector 逐个计算 Cosine Similarity
        ↓
    按相似度排序
        ↓
    返回 Top-K DenseRetrieverResult

    语料在构造时绑定，
    因此 search() 只接收 Query Vector 与 top_k。

    本组件不负责：

    1. Query Embedding
    2. Chunk Embedding
    3. Chunk / Document Lookup
    4. Hydration
    5. Context Assembly
    6. LLM Generation
    """

    def __init__(
        self,
        chunk_vectors: dict[str, np.ndarray],
    ):
        """
        参数：
            chunk_vectors:
                chunk_id -> embedding vector 的显式映射，
                即本 Dense Retriever 检索的语料。
        """

        if not isinstance(chunk_vectors, dict):
            raise TypeError(
                "chunk_vectors must be dict[str, np.ndarray]"
            )

        self._chunk_vectors = chunk_vectors

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[DenseRetrieverResult]:
        """
        对 Query Vector 与所有 Chunk Vector 做 Exact Search。

        参数：
            query_vector:
                Query 的 Embedding Vector。

            top_k:
                最多返回多少个 DenseRetrieverResult。

        返回：
            按 dense_score 从高到低排列的 DenseRetrieverResult
            （含 1-based dense_rank）。
        """

        # --------------------------------------------------
        # 1. 参数校验
        # --------------------------------------------------

        if not isinstance(top_k, int):
            raise TypeError("top_k must be int")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # --------------------------------------------------
        # 2. Empty Corpus 是合法状态
        # --------------------------------------------------

        if not self._chunk_vectors:
            return []

        # --------------------------------------------------
        # 3. 校验 Query Vector
        # --------------------------------------------------

        query_vector = self._validate_vector(
            query_vector,
            vector_name="query_vector",
        )

        # --------------------------------------------------
        # 4. 逐个计算 Chunk Similarity
        # --------------------------------------------------

        scored: list[tuple[str, float]] = []

        for chunk_id, chunk_vector in self._chunk_vectors.items():

            if not isinstance(chunk_id, str):
                raise TypeError("chunk_id must be str")

            if not chunk_id.strip():
                raise ValueError("chunk_id must not be blank")

            chunk_vector = self._validate_vector(
                chunk_vector,
                vector_name=f"chunk_vector[{chunk_id}]",
            )

            if chunk_vector.shape != query_vector.shape:
                raise ValueError(
                    "query_vector and chunk_vector "
                    "must have the same dimension"
                )

            score = self._cosine(
                query_vector,
                chunk_vector,
            )

            scored.append((chunk_id, score))

        # --------------------------------------------------
        # 5. Ranking
        # --------------------------------------------------

        # score 降序；list.sort 稳定，保留原插入顺序作为 tie-break，
        # 与迁移前的排序行为一致。
        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # --------------------------------------------------
        # 6. Top-K + 1-based dense_rank
        # --------------------------------------------------

        return [
            DenseRetrieverResult(
                chunk_id=chunk_id,
                dense_score=score,
                dense_rank=rank,
            )
            for rank, (chunk_id, score) in enumerate(
                scored[:top_k],
                start=1,
            )
        ]

    @staticmethod
    def _validate_vector(
        vector: np.ndarray,
        *,
        vector_name: str,
    ) -> np.ndarray:
        """
        校验并统一 Vector。

        当前 Exact Search Contract 要求：
        - Vector 必须是一维
        - 不能包含 NaN / inf
        - 不能是零向量
        """

        vector = np.asarray(
            vector,
            dtype=float,
        )

        if vector.ndim != 1:
            raise ValueError(
                f"{vector_name} must be a 1-D vector"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"{vector_name} contains NaN or inf"
            )

        if np.linalg.norm(vector) == 0:
            raise ValueError(
                f"{vector_name} must not be a zero vector"
            )

        return vector

    @staticmethod
    def _cosine(
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        """
        计算两个合法向量的 Cosine Similarity。
        """

        score = (
            np.dot(vector_a, vector_b)
            /
            (
                np.linalg.norm(vector_a)
                * np.linalg.norm(vector_b)
            )
        )

        return float(score)
