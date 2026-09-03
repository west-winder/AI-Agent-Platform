import json

from backend.schemas.memory_candidate import MemoryCandidate
from backend.schemas.memory_relationship import (
    MemoryRelationship,
    MemoryRelationshipResult,
)
from backend.services.llm_service import call_llm


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
六、输出要求
==================================================

必须严格返回 JSON。

不得输出任何 JSON 之外的内容。

返回格式：

{
    "relationships": [
        {
            "memory_id": 1,
            "relationship": "duplicate",
            "reason": "两条Memory表达的是同一个长期偏好。"
        }
    ]
}

relationship 只能是：

duplicate
conflict
related
new


memory_id 必须对应输入 Existing Memory
中的 memory_id。

不得生成输入中不存在的 memory_id。

必须为每一个输入的 Existing Memory
返回一个 Relationship。
"""

    # ==================================================
    # Judge
    # ==================================================

    def judge(
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
        # 调用 LLM
        # --------------------------------------------------

        try:

            response = call_llm(
                messages
            )

            if not response:

                return MemoryRelationshipResult(
                    relationships=[]
                )

            # --------------------------------------------------
            # JSON Parse
            # --------------------------------------------------

            data = json.loads(
                response
            )

            relationships_data = data.get(
                "relationships",
                []
            )

            if not isinstance(
                relationships_data,
                list
            ):

                return MemoryRelationshipResult(
                    relationships=[]
                )

            # --------------------------------------------------
            # 构造 Relationship Result
            # --------------------------------------------------

            relationships = []

            for item in relationships_data:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                memory_id = item.get(
                    "memory_id"
                )

                relationship = item.get(
                    "relationship"
                )

                reason = item.get(
                    "reason"
                )

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
                # 防止 LLM 返回不存在的 Memory ID。
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
                # 创建 Relationship
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

        except json.JSONDecodeError:

            return MemoryRelationshipResult(
                relationships=[]
            )

        except Exception:

            return MemoryRelationshipResult(
                relationships=[]
            )