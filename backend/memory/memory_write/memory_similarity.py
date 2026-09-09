from typing import List
import numpy as np

from backend.schemas.memory_similarity import (
    MemorySimilarityResult,
    MemorySimilaritySearchResult,
)


# ==================================================
# Related Retrieval Failure
# ==================================================

class MemorySimilarityError(Exception):
    """
    Related Retrieval 本身失败。

    例如：

    1. Embedding 模型加载失败
    2. Embedding 调用失败
    3. Similarity 计算运行时异常

    必须区分：

    真实搜索成功但没有 match
        → matches=[]
        → 可以继续

    Retrieval 本身失败
        → MemorySimilarityError
        → Fail Closed

    不允许把系统故障
    伪装成"没有相关 Memory"。
    """


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

            # ------------------------------------------
            # Embedding 模型不可用时，
            # 属于 Retrieval Failure，
            # 必须向上抛错，
            # 不能退化成"没有相关 Memory"。
            # ------------------------------------------

            try:

                self._embedder = Embedder()

            except MemorySimilarityError:
                raise

            except Exception as e:

                raise MemorySimilarityError(
                    "Embedding 模型初始化失败："
                    f"{e}"
                ) from e

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
        except MemorySimilarityError:
            raise

        except Exception as e:

            # ------------------------------------------
            # Embedding 失败属于 Retrieval Failure。
            #
            # 旧实现在这里返回 matches=[]，
            # 会把系统故障误判为
            # "没有相关 Memory"，
            # 导致 Candidate 被直接保存。
            #
            # Lifecycle V1 要求 Fail Closed。
            # ------------------------------------------

            raise MemorySimilarityError(
                "Embedding 执行失败："
                f"{e}"
            ) from e

        # ----------------------------------------------
        # Similarity 计算
        #
        # 这里同样属于 Retrieval 的一部分。
        # 运行时异常必须 Fail Closed。
        # ----------------------------------------------

        try:

            scored = []
            for mem, vec in zip(memories, mem_vecs):
                sim = self._cosine(candidate_vec, vec)
                scored.append((mem, sim))

        except MemorySimilarityError:
            raise

        except Exception as e:

            raise MemorySimilarityError(
                "Similarity 计算失败："
                f"{e}"
            ) from e

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