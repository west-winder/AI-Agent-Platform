import json

from backend.schemas.memory_candidate import MemoryCandidate
from backend.services.llm_service import call_llm


class MemoryExtractor:
    """
    Memory提取器。

    负责：

    用户消息
        ↓
    调用LLM
        ↓
    提取候选Memory
        ↓
    返回MemoryCandidate

    注意：

    这个类不负责：
    - 数据库操作
    - Memory保存
    - Conversation管理
    - Chat业务流程
    """

    # ==================================================
    # 允许的Memory类型
    # ==================================================

    ALLOWED_MEMORY_TYPES = {
        "fact",
        "preference",
        "goal",
        "profile"
    }

    # ==================================================
    # Memory Extraction
    # ==================================================

    def extract(
        self,
        user_message: str
    ) -> list[MemoryCandidate]:
        """
        从用户消息中提取候选Memory。

        如果没有值得长期保存的信息，
        返回空列表。
        """

        system_prompt = """
你是一个 Memory Extraction Agent。

你的任务是：

从用户消息中识别“值得长期保存”的用户信息。

只有当信息在未来的其他对话中仍然可能有帮助时，
才应该提取为Memory。

允许的Memory类型只有以下四种：

1. fact
   用户明确提供的、未来可能有用的事实。

2. preference
   用户长期稳定的偏好、习惯或倾向。

3. goal
   用户长期目标、计划或愿望。

4. profile
   用户相对稳定的个人背景信息。

严格要求：

1. memory_type只能是：
   fact
   preference
   goal
   profile

2. 禁止使用任何其他memory_type。

3. 不要推测用户没有明确表达的信息。

4. 普通闲聊、一次性事件、临时状态通常不应该保存。

5. 如果没有值得长期保存的信息，返回空数组。

6. 必须严格返回JSON。

7. 不要输出JSON之外的任何内容。

返回格式：

{
    "memories": [
        {
            "content": "用户正在学习FastAPI",
            "memory_type": "fact"
        }
    ]
}

如果没有值得保存的信息：

{
    "memories": []
}
"""

        # ==================================================
        # 调用LLM
        # ==================================================

        response = call_llm(
            [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        # ==================================================
        # LLM没有返回内容
        # ==================================================

        if not response:
            return []

        # ==================================================
        # JSON解析
        # ==================================================

        try:

            data = json.loads(response)

        except json.JSONDecodeError:

            # 当前阶段：
            # 如果LLM没有返回合法JSON，
            # 则认为本次没有提取到Memory。
            return []

        # ==================================================
        # 获取Memory列表
        # ==================================================

        memories = data.get(
            "memories",
            []
        )

        if not isinstance(
            memories,
            list
        ):
            return []

        # ==================================================
        # 构造Candidate
        # ==================================================

        candidates = []

        for memory in memories:

            if not isinstance(
                memory,
                dict
            ):
                continue

            content = memory.get(
                "content"
            )

            memory_type = memory.get(
                "memory_type"
            )

            # ----------------------------------------------
            # content校验
            # ----------------------------------------------

            if not isinstance(
                content,
                str
            ):
                continue

            if not content.strip():
                continue

            # ----------------------------------------------
            # memory_type校验
            # ----------------------------------------------

            if memory_type not in self.ALLOWED_MEMORY_TYPES:
                continue

            # ----------------------------------------------
            # 创建Candidate
            # ----------------------------------------------

            candidate = MemoryCandidate(
                content=content.strip(),
                memory_type=memory_type
            )

            candidates.append(
                candidate
            )

        return candidates