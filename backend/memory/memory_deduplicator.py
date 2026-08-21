from typing import List

from pydantic import BaseModel

from backend.schemas.memory_candidate import MemoryCandidate


class MemoryDeduplicationResult(BaseModel):
    """
    Memory 去重结果。

    duplicate:
        是否发现重复 Memory

    reason:
        判断原因，方便调试
    """

    duplicate: bool
    reason: str


def normalize_content(content: str) -> str:
    """
    对 Memory 内容进行基础标准化。

    当前只处理非常保守的文本格式问题：

    1. 去除首尾空白
    2. 去除常见中文/英文句末标点

    注意：

    这里不进行语义理解。

    例如：

    用户喜欢Python
    用户喜欢Python。

    经过标准化后：

    用户喜欢Python
    用户喜欢Python

    二者就可以进行 Exact Match。

    但是：

    用户喜欢Python
    用户喜欢使用Python进行后端开发

    仍然不会被认为是相同内容。

    因为这已经属于“语义重复”，
    后续需要 Embedding + Semantic Judge。
    """

    # 去除字符串首尾空白
    content = content.strip()

    # 去除常见句末标点
    content = content.rstrip("。！？!?")

    # 保持严格的 Exact Match 规范化：只做最保守的格式化处理，
    # 避免在去重阶段引入语义级别的归一化。
    # 这意味着不再移除“使用”等词语，也不删除中间空格。

    return content


def check_duplicate(
    candidate: MemoryCandidate,
    memories: List
) -> MemoryDeduplicationResult:
    """
    检查 Memory Candidate 是否与已有 Memory 重复。

    当前版本：

    Normalization
        ↓
    Exact Match

    后续会升级为：

    Normalization
        ↓
    Exact Match
        ↓
    Embedding
        ↓
    Similarity Search
        ↓
    LLM Semantic Judge
    """

    # 首先对 Candidate 进行标准化
    candidate_content = normalize_content(
        candidate.content
    )

    # 遍历当前用户已有的 Memory
    for memory in memories:

        # 对数据库中已有的 Memory 也进行标准化
        existing_content = normalize_content(
            memory.content
        )

        # 比较标准化后的内容
        if existing_content == candidate_content:

            return MemoryDeduplicationResult(
                duplicate=True,
                reason="存在内容完全相同的Memory"
            )

    # 遍历结束，没有发现重复
    return MemoryDeduplicationResult(
        duplicate=False,
        reason="没有发现内容完全相同的Memory"
    )