from backend.schemas.memory_lifecycle import (
    MemoryLifecycleDecision,
)
from backend.schemas.memory_relationship import (
    MemoryRelationship,
    MemoryRelationshipResult,
)


# ==================================================
# Relationship → Lifecycle Action
# ==================================================

# duplicate
#     Candidate 与已有 Memory 重复。
#     不保存 Candidate。
#
# conflict
#     已有 Memory 不再代表当前状态。
#     该 Memory 应被置为 historical。
#
# related
#     不产生 Lifecycle Action。
#
# new
#     不产生 Lifecycle Action。

RELATIONSHIP_DUPLICATE = "duplicate"

RELATIONSHIP_CONFLICT = "conflict"


# ==================================================
# Lifecycle Decision
# ==================================================

def decide_memory_lifecycle(
    relationship_result
) -> MemoryLifecycleDecision:
    """
    根据 Relationship Judge 的结果，
    推导本次 Candidate 的 Lifecycle Action。

    这是一个纯确定性组件。

    本函数：

    1. 不调用 LLM
    2. 不访问数据库
    3. 不执行 persistence

    输入：

        MemoryRelationshipResult
        或者
        list[MemoryRelationship]

    输出：

        MemoryLifecycleDecision

    聚合规则：

    1. save_candidate

       只要存在任意 duplicate
           → False

       完全没有 duplicate
           → True

    2. historical_memory_ids

       所有 conflict 对应的 memory_id
           → historical_memory_ids

       related / new
           → 不产生 Lifecycle Action

    顺序无关性：

    输入 relationships 的排列顺序
    不得影响最终 Decision。

    这里通过 set 聚合 conflict IDs，
    并对最终结果排序，
    保证输出完全确定。
    """

    relationships = _normalize_relationships(
        relationship_result
    )

    # --------------------------------------------------
    # duplicate 检测
    # --------------------------------------------------

    has_duplicate = any(
        relationship.relationship == RELATIONSHIP_DUPLICATE
        for relationship in relationships
    )

    # --------------------------------------------------
    # conflict 聚合
    #
    # 使用 set 去重，避免同一个 memory_id
    # 因为重复输出而进入多次 Lifecycle Action。
    # --------------------------------------------------

    conflict_memory_ids = {
        relationship.memory_id
        for relationship in relationships
        if relationship.relationship == RELATIONSHIP_CONFLICT
    }

    return MemoryLifecycleDecision(
        save_candidate=(not has_duplicate),
        historical_memory_ids=sorted(
            conflict_memory_ids
        ),
    )


# ==================================================
# Input Normalization
# ==================================================

def _normalize_relationships(
    relationship_result
) -> list[MemoryRelationship]:
    """
    将输入统一为 list[MemoryRelationship]。

    支持：

    1. MemoryRelationshipResult
    2. list[MemoryRelationship]
    """

    if relationship_result is None:
        return []

    if isinstance(
        relationship_result,
        MemoryRelationshipResult
    ):
        return list(
            relationship_result.relationships
        )

    if isinstance(
        relationship_result,
        list
    ):
        return list(
            relationship_result
        )

    raise TypeError(
        "decide_memory_lifecycle "
        "只接受 MemoryRelationshipResult "
        "或 list[MemoryRelationship]"
    )
