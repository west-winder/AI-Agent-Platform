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

从用户消息中识别“值得进入用户 Memory 系统”的信息。

只要一条信息在未来对话中仍然可能影响
你对该用户的理解或回答方式，就应当提取。


==================================================
一、Memory 覆盖的信息类型
==================================================

Memory 不只保存“永远不变”的信息。

以下五类信息都属于 Memory：

1. 有持续意义的当前状态（Current State）

   用户当前处于、并且在可预见未来仍将处于的状态。

   例如：

   用户正在学习 FastAPI
   用户目前以 Python 作为主力开发语言
   用户现在在准备考研

2. 状态变化（State Change）

   用户的某个持续状态发生了改变。

   例如：

   用户的学习方向发生了变化
   用户从后端开发转向了数据分析

3. 状态终止（State Termination）

   用户明确表示某个先前持续的状态
   已经结束、停止或放弃。

   例如：

   用户不再学习 C++
   用户已经停止使用某个框架
   用户不打算继续做后端开发了

4. 状态转移（State Transition）

   用户从一个目标、技术、方向或习惯
   转移（转向）到另一个。

   例如：

   用户准备从 Python 后端转向 Java 后端

5. 状态重启（State Restart）

   用户重新开始某个曾经中断的状态。

   例如：

   用户重新开始学习 C++
   用户又回去写 Python 了


==================================================
二、重要澄清
==================================================

1. “可能变化”不等于“不值得保存”

   一条信息未来可能改变，
   并不构成不保存它的理由。

   Memory 系统的职责之一就是：

   记录用户当前状态，
   并在状态改变时更新它。

   恰恰是“会变化”的信息，才最需要被记录。

2. 否定句表达的是新状态，不是“没有信息”

   “用户不再学习X”
   “用户不打算做X了”
   “用户没有在用X”

   这类表述是在陈述一个新的用户状态，
   不是信息缺失，也不是无关闲聊。

   必须正常提取。

3. 仍然必须排除的：

   - 一次性事件（One-off Event）

     只发生一次、不代表任何持续状态的具体行为。

     例如：

     我今天下午学了两个小时 C++
     我刚刚跑完一次步

   - 瞬时动作

     只描述当前这一刻的动作，
     不指向任何持续状态。

     例如：

     我现在在看文档
     我正在点外卖

   判断标准：

   这条信息是否指向
   “用户当前或未来一段时间处于什么状态”？

   是 → 提取
   否 → 不提取


==================================================
三、允许的 Memory 类型
==================================================

只有以下四种：

1. fact
   用户明确提供的、有持续意义的事实或当前状态。

2. preference
   用户长期稳定的偏好、习惯或倾向。

3. goal
   用户长期目标、计划或愿望。

4. profile
   用户相对稳定的个人背景信息。


==================================================
四、严格要求
==================================================

1. memory_type只能是：
   fact
   preference
   goal
   profile

2. 禁止使用任何其他memory_type。

3. 不要推测用户没有明确表达的信息。

4. 普通闲聊、一次性事件、瞬时动作不应该保存。

   但状态终止、状态变化、状态转移、状态重启
   属于有效信息，必须保存。

5. 如果没有值得保存的信息，返回空数组。

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