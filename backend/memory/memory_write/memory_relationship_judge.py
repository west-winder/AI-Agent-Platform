import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.schemas.memory_candidate import MemoryCandidate
from backend.schemas.memory_relationship import (
    MemoryRelationship,
    MemoryRelationshipResult,
)
from backend.services.llm_service import call_llm_structured


# ==================================================
# Allowed Relationships
# ==================================================

ALLOWED_RELATIONSHIPS = {
    "duplicate",
    "conflict",
    "related",
    "new",
}


# ==================================================
# Relationship Judge LLM Boundary Contract
# ==================================================


class MemoryRelationshipLLMItem(BaseModel):
    """
    单条 Relationship Structured Output。

    这里故意先使用 Any 接住字段，
    再由 MemoryRelationshipJudge 按旧 Contract
    做逐条过滤。

    原实现的行为是：

    - 某一条 memory_id 非法 -> 跳过该条
    - 某一条 relationship 非法 -> 跳过该条
    - 某一条 reason 非法 -> 跳过该条
    - 同批次其他合法 Relationship 仍然保留

    如果直接把字段定义成 StrictInt / Literal / StrictStr，
    单条坏 item 可能导致整个 Structured Output
    Validation 失败，从而改变原来的逐条 skip 行为。
    """

    memory_id: Any = None
    relationship: Any = None
    reason: Any = None


class MemoryRelationshipLLMOutput(BaseModel):
    """
    Relationship Judge 的 Structured Output 顶层结构。

    顶层负责：

    - relationships 缺失 -> []
    - relationships=None -> []
    - relationships 非 list -> ValidationError
    - 非 dict / 非 MemoryRelationshipLLMItem item -> 跳过

    动态业务约束仍由 Judge / Coverage Validator 负责。
    """

    relationships: list[
        MemoryRelationshipLLMItem
    ] = Field(default_factory=list)

    @field_validator(
        "relationships",
        mode="before",
    )
    @classmethod
    def preserve_per_item_skip_contract(
        cls,
        value,
    ):
        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                "relationships 必须是 list"
            )

        return [
            item
            for item in value
            if isinstance(
                item,
                (
                    dict,
                    MemoryRelationshipLLMItem,
                ),
            )
        ]


# ==================================================
# Relationship Contract Failure
# ==================================================

class RelationshipContractError(Exception):
    """
    Relationship Judge Contract 校验失败。

    出现这个异常意味着：

    送入 Judge 的 Existing Memories
    没有被 Judge 完整、合法地判断。

    此时必须 Fail Closed：

    不得让 Pipeline
    继续保存 Candidate，
    也不得产生任何数据库 Lifecycle Mutation。
    """


# ==================================================
# Memory Relationship Judge
# ==================================================

class MemoryRelationshipJudge:
    """
    Memory Relationship Judge。

    负责判断：

    Candidate Memory
            ↓
    Existing Memories
            ↓
    LLM
            ↓
    Relationship

    Relationship包括：

    duplicate
    conflict
    related
    new

    注意：

    本模块不负责：

    1. Memory Extraction
    2. Memory Validation
    3. Embedding
    4. Similarity Search
    5. Database操作
    6. Memory Persistence

    它只负责：

        “Candidate 与 Existing Memory 之间是什么关系？”
    """

    # ==================================================
    # System Prompt
    # ==================================================

    SYSTEM_PROMPT = """
你是一个 AI Agent 的 Memory Relationship Judge。

你的任务是：

判断一条 Candidate Memory
与若干 Existing Memories 之间的关系。

你只能使用以下四种 Relationship：

1. duplicate
2. conflict
3. related
4. new


==================================================
一、duplicate
==================================================

表示：

Candidate 与 Existing Memory
表达的是同一个长期 Memory。

允许：

- 措辞不同
- 句式不同
- 表达方式不同
- 少量信息粒度差异

例如：

Candidate:
用户喜欢使用Python

Existing:
用户喜欢Python

Relationship:
duplicate


重要规则：

duplicate 原则上要求两条 Memory 的
memory_type 相同。

例如：

Candidate:
用户喜欢Python
type = preference

Existing:
用户正在学习Python
type = fact

即使两个内容语义相近，
也不要判断为 duplicate。


==================================================
二、conflict
==================================================

表示：

Candidate 与 Existing Memory
描述的是同一个主题、属性、状态或倾向，
但两者存在明显矛盾。

例如：

Candidate:
用户已经完成FastAPI学习

Existing:
用户正在学习FastAPI

Relationship:
conflict


重要规则：

conflict 不要求两个 Memory 的
memory_type 相同。

即使：

Candidate.type != Existing.type

只要两条 Memory 在语义上针对同一事实、
状态、倾向或目标，并且内容存在明显冲突，
仍然可以判断为 conflict。


==================================================
三、related
==================================================

表示：

Candidate 与 Existing Memory
存在有意义的语义或主题关联，

但：

- 不是同一个 Memory
- 不存在明显冲突

例如：

Candidate:
用户正在学习LangGraph

Existing:
用户正在学习FastAPI

Relationship:
related


related 不要求 memory_type 相同。


==================================================
四、new
==================================================

表示：

Candidate 与 Existing Memory
之间不存在足够强的：

- duplicate
- conflict
- related

关系。

也就是说：

Candidate 与该 Existing Memory
应当被视为两个独立的 Memory。


重要：

这里的 new 是针对“某一个 Existing Memory”
而言的关系。

如果 Candidate 与所有 Existing Memories
都是 new，
则 Candidate 才可以被视为真正的新 Memory。


==================================================
五、判断原则
==================================================

Embedding Similarity 只是候选召回依据。

Similarity 高：

不代表一定 duplicate。

Similarity 高：

也可能是 conflict。

例如：

用户喜欢Python
vs
用户不喜欢Python

可能具有很高的语义相似度，
但 Relationship 应该是 conflict。


请综合考虑：

1. content
2. memory_type
3. similarity

进行判断。


==================================================
六、输出语义要求
==================================================

你必须为每一个输入的 Existing Memory
返回一个 Relationship。

每条 Relationship 必须包含：

memory_id

必须对应输入 Existing Memory
中的 memory_id。

不得生成输入中不存在的 memory_id。


relationship

只能是：

duplicate
conflict
related
new


reason

必须简短说明为什么判断为该 Relationship。

输出结构由系统提供的
Structured Output Schema 约束。
"""

    # ==================================================
    # Judge
    # ==================================================

    async def judge(
        self,
        candidate: MemoryCandidate,
        existing_memories: list
    ) -> MemoryRelationshipResult:
        """
        判断 Candidate 与 Existing Memories 的关系。

        参数：

        candidate:
            当前待处理的 Candidate Memory。

        existing_memories:
            Similarity Search 返回的 Top-K Memory。

        返回：

        MemoryRelationshipResult
        """

        # --------------------------------------------------
        # 没有 Existing Memory
        # --------------------------------------------------

        if not existing_memories:

            return MemoryRelationshipResult(
                relationships=[]
            )

        # --------------------------------------------------
        # 构造合法 Memory ID 集合
        #
        # 用于防止 LLM 返回不存在的 memory_id。
        # --------------------------------------------------

        valid_memory_ids = {
            memory.memory_id
            for memory in existing_memories
        }

        # --------------------------------------------------
        # 构造 Existing Memory 数据
        # --------------------------------------------------

        memory_data = []

        for memory in existing_memories:

            memory_data.append(
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "memory_type": memory.memory_type,
                    "similarity": memory.similarity,
                }
            )

        # --------------------------------------------------
        # Candidate
        # --------------------------------------------------

        candidate_data = {
            "content": candidate.content,
            "memory_type": candidate.memory_type,
        }

        # --------------------------------------------------
        # User Prompt
        # --------------------------------------------------

        user_prompt = f"""
请判断 Candidate Memory 与 Existing Memories 的关系。

Candidate:

{json.dumps(
    candidate_data,
    ensure_ascii=False,
    indent=2
)}


Existing Memories:

{json.dumps(
    memory_data,
    ensure_ascii=False,
    indent=2
)}


请严格按照 System Prompt 中定义的规则判断。

必须为每一个 Existing Memory 返回一个 Relationship。

不得返回输入中不存在的 memory_id。
"""

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        # --------------------------------------------------
        # 调用 Structured Output LLM
        # --------------------------------------------------

        try:

            llm_output = await call_llm_structured(
                messages=messages,
                output_model=MemoryRelationshipLLMOutput,
            )

            # --------------------------------------------------
            # 构造 Relationship Result
            #
            # 这里继续保留旧的逐条 skip Contract：
            #
            # 单条坏 Relationship
            # 不影响同批次其他合法 Relationship。
            #
            # Pydantic 负责顶层 Structured Output，
            # Judge 继续负责当前业务上下文中的逐条校验。
            # --------------------------------------------------

            relationships = []

            for item in llm_output.relationships:

                memory_id = item.memory_id
                relationship = item.relationship
                reason = item.reason

                # ----------------------------------------------
                # memory_id 类型检查
                # ----------------------------------------------

                if not isinstance(
                    memory_id,
                    int
                ):
                    continue

                # ----------------------------------------------
                # memory_id 合法性检查
                #
                # valid_memory_ids 是本次运行时
                # 根据 Existing Memories 动态生成的集合。
                # 这属于 Runtime Business Validation，
                # 不能只靠静态 Schema 表达。
                # ----------------------------------------------

                if memory_id not in valid_memory_ids:
                    continue

                # ----------------------------------------------
                # relationship 检查
                # ----------------------------------------------

                if relationship not in ALLOWED_RELATIONSHIPS:
                    continue

                # ----------------------------------------------
                # reason 检查
                # ----------------------------------------------

                if not isinstance(
                    reason,
                    str
                ):
                    continue

                if not reason.strip():
                    continue

                # ----------------------------------------------
                # 创建 Business Relationship
                # ----------------------------------------------

                relationships.append(
                    MemoryRelationship(
                        memory_id=memory_id,
                        relationship=relationship,
                        reason=reason.strip()
                    )
                )

            return MemoryRelationshipResult(
                relationships=relationships
            )

        except Exception:
            # --------------------------------------------------
            # 保持旧 Failure Policy：
            #
            # Provider Error
            # Structured Output Validation Error
            # Runtime Processing Error
            #
            # 都先收敛为：
            #
            # MemoryRelationshipResult(relationships=[])
            #
            # 后续由 validate_relationship_coverage()
            # 发现覆盖不完整并触发 Fail Closed。
            # --------------------------------------------------

            return MemoryRelationshipResult(
                relationships=[]
            )


# ==================================================
# Relationship Judge Contract Validation
# ==================================================

def validate_relationship_coverage(
    existing_memories,
    relationship_result
):
    """
    校验 Judge 是否完整覆盖了
    所有送入 Judge 的 Existing Memory。

    合约：

    实际送入 Judge 的 memory IDs
        ==
    Judge 输出覆盖的 memory IDs

    注意：

    这里比较的不是配置 top_k，
    而是本次真实送入 Judge 的 ID 集合。

    检查项：

    1. 输入 ID 无重复
    2. 输出数量与输入数量一致
    3. 输出 ID 集合与输入 ID 集合完全一致
    4. 输出无重复 ID
    5. 输出无未知 ID
    6. relationship 合法
    7. reason 合法

    任何一项失败：

    raise RelationshipContractError

    本函数不修改任何数据。
    是否 Fail Closed 由调用方决定。
    """

    if relationship_result is None:

        raise RelationshipContractError(
            "Judge 没有返回任何结果"
        )

    relationships = relationship_result.relationships

    # --------------------------------------------------
    # 输入 ID
    # --------------------------------------------------

    expected_ids = []

    for memory in existing_memories:

        memory_id = getattr(
            memory,
            "memory_id",
            None
        )

        if not isinstance(
            memory_id,
            int
        ):

            raise RelationshipContractError(
                "Existing Memory 缺少合法的 "
                f"memory_id：{memory_id!r}"
            )

        expected_ids.append(
            memory_id
        )

    if len(
        set(expected_ids)
    ) != len(
        expected_ids
    ):

        raise RelationshipContractError(
            "送入 Judge 的 Existing Memory "
            f"存在重复 ID：{expected_ids}"
        )

    expected_id_set = set(
        expected_ids
    )

    # --------------------------------------------------
    # 输出数量
    # --------------------------------------------------

    if len(
        relationships
    ) != len(
        expected_ids
    ):

        raise RelationshipContractError(
            "Judge 输出数量 "
            f"{len(relationships)} "
            "与送入 Judge 的 Existing Memory 数量 "
            f"{len(expected_ids)} 不一致"
        )

    # --------------------------------------------------
    # 逐项校验
    # --------------------------------------------------

    seen_ids = set()

    for relationship in relationships:

        memory_id = relationship.memory_id

        # ----------------------------------------------
        # memory_id 类型
        # ----------------------------------------------

        if not isinstance(
            memory_id,
            int
        ):

            raise RelationshipContractError(
                "Judge 输出的 memory_id "
                f"不是 int：{memory_id!r}"
            )

        # ----------------------------------------------
        # 未知 ID
        # ----------------------------------------------

        if memory_id not in expected_id_set:

            raise RelationshipContractError(
                "Judge 输出了未知 memory_id："
                f"{memory_id}"
            )

        # ----------------------------------------------
        # 重复 ID
        # ----------------------------------------------

        if memory_id in seen_ids:

            raise RelationshipContractError(
                "Judge 输出了重复 memory_id："
                f"{memory_id}"
            )

        seen_ids.add(
            memory_id
        )

        # ----------------------------------------------
        # relationship 合法性
        # ----------------------------------------------

        if relationship.relationship not in (
            ALLOWED_RELATIONSHIPS
        ):

            raise RelationshipContractError(
                "Judge 输出了非法 relationship："
                f"{relationship.relationship!r}"
            )

        # ----------------------------------------------
        # reason 合法性
        # ----------------------------------------------

        if not isinstance(
            relationship.reason,
            str
        ):

            raise RelationshipContractError(
                "Judge 输出的 reason "
                "不是 str"
            )

        if not relationship.reason.strip():

            raise RelationshipContractError(
                f"Judge 输出的 reason 为空："
                f"memory_id={memory_id}"
            )

    # --------------------------------------------------
    # 完整覆盖
    # --------------------------------------------------

    if seen_ids != expected_id_set:

        raise RelationshipContractError(
            "Judge 没有完整覆盖所有 "
            "送入 Judge 的 Existing Memory"
        )