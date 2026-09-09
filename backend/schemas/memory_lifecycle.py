from pydantic import BaseModel


class MemoryLifecycleDecision(BaseModel):
    """
    Memory Lifecycle V1 的最终决策结果。

    save_candidate:

        是否保存当前 Candidate。

        聚合规则：

        只要存在任意 duplicate
            → False

        完全没有 duplicate
            → True

    historical_memory_ids:

        本次需要被置为 historical 的
        已有 Memory ID。

        聚合规则：

        所有 conflict 对应的 memory_id
            → historical_memory_ids

        related / new
            → 不产生 Lifecycle Action

    注意：

    这是一个纯数据对象。

    它不访问数据库，
    不调用 LLM，
    不执行任何 persistence。
    """

    save_candidate: bool

    historical_memory_ids: list[int] = []
