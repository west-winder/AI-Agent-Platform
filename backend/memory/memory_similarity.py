from typing import List
import numpy as np

from backend.schemas.memory_similarity import (
    MemorySimilarityResult,
    MemorySimilaritySearchResult,
)


class MemorySimilarity:
    """
    MemorySimilarity 负责：
    - 使用 Embedder 生成 embedding（延迟导入以降低模块导入时的依赖）
    - 计算 Candidate 与已有 Memory 的相似度
    - 返回排序后的 top_k 匹配（并根据 threshold 进行候选筛选）

    注意：在单元测试中可以通过依赖注入替换为 FakeSimilarity，
    本实现尽量在导入时不触发 heavy 依赖（如 sentence_transformers）。
    """

    def __init__(self, embedder=None):
        # 延迟初始化 embedder，避免在模块导入时触发 heavy 导入
        self._embedder = embedder

    def _ensure_embedder(self):
        if self._embedder is None:
            # 延迟导入 Embedder
            from backend.embedding.embedder import Embedder

            self._embedder = Embedder()

    def _cosine(self, a, b) -> float:
        # a, b 可以是 list 或 numpy.ndarray
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def search(
        self,
        candidate,
        memories: List,
        top_k: int = 5,
        threshold: float = 0.70,
    ) -> MemorySimilaritySearchResult:
        """
        对 candidate 与 memories 进行相似度搜索。

        返回 MemorySimilaritySearchResult，其中 matches 是 MemorySimilarityResult 列表。
        """
        # 如果没有已有 memory，直接返回空匹配
        if not memories:
            return MemorySimilaritySearchResult(
                matches=[],
                threshold=threshold,
                top_k=top_k,
            )

        # 确保 embedder 已可用（延迟导入）
        self._ensure_embedder()

        # 生成 embeddings（为了效率对 memories 批量 embed）
        texts = [m.content for m in memories]

        try:
            candidate_vec = self._embedder.embed(candidate.content)
            mem_vecs = self._embedder.embed_batch(texts)
        except Exception as e:
            # 如果 embed 失败，返回空结果，交由上层决策（通常会直接保存）
            return MemorySimilaritySearchResult(
                matches=[],
                threshold=threshold,
                top_k=top_k,
            )

        # 计算相似度
        scored = []
        for mem, vec in zip(memories, mem_vecs):
            sim = self._cosine(candidate_vec, vec)
            scored.append((mem, sim))

        # 按相似度排序并取 top_k
        scored.sort(key=lambda x: x[1], reverse=True)

        matches = []
        for mem, sim in scored[:top_k]:
            if sim >= threshold:
                matches.append(
                    MemorySimilarityResult(
                        memory_id=getattr(mem, "id", None),
                        content=mem.content,
                        memory_type=mem.memory_type,
                        similarity=float(sim),
                    )
                )

        return MemorySimilaritySearchResult(
            matches=matches,
            threshold=threshold,
            top_k=top_k,
        )