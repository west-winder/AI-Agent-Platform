import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.memory.memory_read.memory_reader import (
    MemoryReader,
)

from backend.memory.memory_read.memory_relevance_judge import (
    JudgeDecision,
)


# ==================================================
# Fake Memory
# ==================================================


@dataclass
class FakeMemory:
    content: str


# ==================================================
# Controlled Corpus
#
# 注意：
# Repository 是 Fake，
# 但 Retrieval / Reranker 都是真实组件。
# ==================================================


memories = [
    FakeMemory(
        content="用户最近开始学习 LangGraph"
    ),

    FakeMemory(
        content=(
            "Memory Read V1 对应 Git commit 为 22ecfec"
        )
    ),

    FakeMemory(
        content=(
            "用户当前使用 Qwen3-Embedding-0.6B "
            "作为 Embedding 模型"
        )
    ),

    FakeMemory(
        content=(
            "Memory Read V2 正在实现 "
            "Dense Retrieval、BM25 和 RRF"
        )
    ),

    FakeMemory(
        content=(
            "用户的 Reranker 使用 "
            "BAAI/bge-reranker-v2-m3"
        )
    ),

    FakeMemory(
        content="用户喜欢吃火锅"
    ),
]


# ==================================================
# Fake Repository
# ==================================================


def fake_repository(
    db,
    user_id
):
    return memories


# ==================================================
# Fake Judge
#
# 这一轮不测试真实 LLM Judge。
#
# 只选择包含 22ecfec 的 Memory，
# 方便验证最终 Injector。
# ==================================================


class FakeJudge:

    def judge(
        self,
        query,
        texts
    ):
        decisions = []

        for index, text in enumerate(texts):

            selected = (
                "22ecfec" in text
            )

            decisions.append(
                JudgeDecision(
                    index=index,
                    selected=selected,
                    reason=(
                        "contains target commit"
                        if selected
                        else "controlled reject"
                    )
                )
            )

        return decisions


# ==================================================
# Fake Injector
# ==================================================


class FakeInjector:

    def build_context(
        self,
        texts
    ):
        if not texts:
            return ""

        return (
            "<memory_context>"
            + "|".join(texts)
            + "</memory_context>"
        )


# ==================================================
# Helper
# ==================================================


def get_original_index(
    memory
):
    """
    根据对象身份找到其在原始 memories[]
    中的位置。

    这里只用于测试打印。
    """

    for index, original_memory in enumerate(
        memories
    ):
        if original_memory is memory:
            return index

    raise ValueError(
        "无法在原始 memories 中找到 Memory"
    )


def print_ranking(
    title,
    items,
    score_getter
):
    """
    根据指定 score 打印排名。

    这里只用于观察：
    Dense / BM25 / RRF 的排序。
    """

    ranked_items = sorted(
        items,
        key=score_getter,
        reverse=True
    )

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rank, item in enumerate(
        ranked_items,
        start=1
    ):
        score = score_getter(item)

        print(
            f"Rank {rank} | "
            f"Original Index "
            f"{get_original_index(item.memory)} | "
            f"Score {score} | "
            f"Text {item.memory.content}"
        )


# ==================================================
# Build MemoryReader
#
# 注意：
#
# 没有注入：
#
# dense_retriever
# bm25_retriever
# rrf_callable
# reranker
#
# 所以它们全部使用真实组件。
# ==================================================


reader = MemoryReader(
    repository_callable=fake_repository,
    judge=FakeJudge(),
    injector=FakeInjector(),
)


# ==================================================
# Query
# ==================================================


query = "22ecfec 是什么提交？"


# ==================================================
# Execute
#
# corpus 共 6 条，
# top_n=6：
# 让 Dense / BM25 都保留全部候选，
# 避免 Smoke Test 因 Top-N 截断变得脆弱。
#
# top_k=6：
# 让 Fake Judge 能看到全部候选。
# ==================================================


result = reader.read(
    db=None,
    user_id=1,
    query=query,
    top_n=6,
    top_k=6
)


# ==================================================
# Final Reranker Ranking
# ==================================================


print()
print("=" * 70)
print("FINAL RERANKER RANKING")
print("=" * 70)


for rank, item in enumerate(
    result.items,
    start=1
):
    print(
        f"Rank {rank} | "
        f"Original Index "
        f"{get_original_index(item.memory)} | "
        f"Dense {item.dense_score} | "
        f"BM25 {item.bm25_score} | "
        f"RRF {item.rrf_score} | "
        f"Rerank {item.rerank_score} | "
        f"Judge {item.judge_selected}"
    )

    print(
        f"         {item.memory.content}"
    )


# ==================================================
# Reconstruct / Observe Dense Ranking
# ==================================================


print_ranking(
    title="DENSE RANKING",
    items=result.items,
    score_getter=lambda item: (
        item.dense_score
        if item.dense_score is not None
        else float("-inf")
    )
)


# ==================================================
# Reconstruct / Observe BM25 Ranking
# ==================================================


print_ranking(
    title="BM25 RANKING",
    items=result.items,
    score_getter=lambda item: (
        item.bm25_score
        if item.bm25_score is not None
        else float("-inf")
    )
)


# ==================================================
# RRF Ranking
# ==================================================


print_ranking(
    title="RRF RANKING",
    items=result.items,
    score_getter=lambda item: (
        item.rrf_score
    )
)


# ==================================================
# Memory Context
# ==================================================


print()
print("=" * 70)
print("MEMORY CONTEXT")
print("=" * 70)

print(result.memory_context)


# ==================================================
# Assertions
# ==================================================


# --------------------------------------------------
# 1. 所有 6 条候选都应该走完整 Retrieval
# --------------------------------------------------

assert len(result.items) == 6


# --------------------------------------------------
# 2. 找到目标 Memory
# --------------------------------------------------

target_item = None

for item in result.items:

    if "22ecfec" in item.memory.content:
        target_item = item
        break


assert target_item is not None


# --------------------------------------------------
# 3. Dense 真组件确实产生分数
# --------------------------------------------------

assert target_item.dense_score is not None

assert isinstance(
    target_item.dense_score,
    float
)


# --------------------------------------------------
# 4. BM25 应该通过精确 token 命中 22ecfec
# --------------------------------------------------

assert target_item.bm25_score is not None

assert target_item.bm25_score > 0


# --------------------------------------------------
# 5. RRF 应该产生融合分数
# --------------------------------------------------

assert isinstance(
    target_item.rrf_score,
    float
)

assert target_item.rrf_score > 0


# --------------------------------------------------
# 6. 真实 Reranker 应该产生分数
# --------------------------------------------------

assert isinstance(
    target_item.rerank_score,
    float
)


# --------------------------------------------------
# 7. Fake Judge 应该只选择目标 Memory
# --------------------------------------------------

assert target_item.judge_selected is True


for item in result.items:

    if item is not target_item:
        assert item.judge_selected is False


# --------------------------------------------------
# 8. Injector 最终只应该注入目标 Memory
# --------------------------------------------------

assert "22ecfec" in result.memory_context

assert "用户喜欢吃火锅" not in (
    result.memory_context
)


# --------------------------------------------------
# 9. 打印目标最终排名
# --------------------------------------------------

target_rank = (
    result.items.index(target_item)
    + 1
)

print()
print("=" * 70)
print("TARGET MEMORY")
print("=" * 70)

print(
    f"Final Reranker Rank: "
    f"{target_rank}"
)

print(
    f"Dense Score: "
    f"{target_item.dense_score}"
)

print(
    f"BM25 Score: "
    f"{target_item.bm25_score}"
)

print(
    f"RRF Score: "
    f"{target_item.rrf_score}"
)

print(
    f"Rerank Score: "
    f"{target_item.rerank_score}"
)


# --------------------------------------------------
# 不强制 assert Rank 1。
#
# Smoke Test 的目标首先是确认：
# 真实组件链路正常工作。
#
# 但如果目标没有进入前 3，
# 打印提示，后续再分析 Ranking。
# --------------------------------------------------

if target_rank <= 3:
    print(
        "[GOOD] Target Memory "
        "进入最终 Top-3"
    )
else:
    print(
        "[WARNING] Target Memory "
        "没有进入最终 Top-3，"
        "需要检查 Ranking"
    )


print()
print("=" * 70)
print("REAL RETRIEVAL SMOKE TEST PASSED")
print("=" * 70)