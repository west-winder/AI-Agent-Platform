from pydantic import BaseModel, StrictBool, StrictStr

from backend.schemas.memory_candidate import MemoryCandidate
from backend.schemas.memory_validation import MemoryValidationResult
from backend.services.llm_service import call_llm_structured


# ==================================================
# Memory类型白名单
# ==================================================

ALLOWED_MEMORY_TYPES = {
    "fact",
    "preference",
    "goal",
    "profile",
}


# ==================================================
# Memory Validator LLM Boundary Contract
# ==================================================


class MemoryValidatorLLMOutput(BaseModel):
    """
    MemoryValidator 的 Structured Output 边界模型。

    保留旧 Contract：
    - valid 必须是 bool
    - reason 必须是 str
    - reason 允许为空字符串
    """

    valid: StrictBool
    reason: StrictStr


# ==================================================
# MemoryValidator
# ==================================================


class MemoryValidator:
    """
    Memory Validator。

    负责判断 MemoryCandidate 是否具备
    进入长期 Memory 的基本资格。

    注意：
    这里暂时不负责：

    1. Memory去重
    2. Memory冲突检测
    3. Memory更新
    4. Memory数据库保存

    这些属于后续阶段。
    """

    # ==================================================
    # 第一层：规则验证
    # ==================================================

    def validate_rules(
        self,
        candidate: MemoryCandidate
    ) -> MemoryValidationResult:
        """
        使用确定性规则进行基础验证。

        这一层不调用LLM。
        """

        content = candidate.content.strip()

        if not content:
            return MemoryValidationResult(
                valid=False,
                reason="Memory内容不能为空"
            )

        if len(content) < 5:
            return MemoryValidationResult(
                valid=False,
                reason="Memory内容过短，缺乏足够的信息量"
            )

        if candidate.memory_type not in ALLOWED_MEMORY_TYPES:
            return MemoryValidationResult(
                valid=False,
                reason=f"不支持的Memory类型：{candidate.memory_type}"
            )

        return MemoryValidationResult(
            valid=True,
            reason="通过基础规则检查"
        )

    # ==================================================
    # 第二层：LLM语义验证
    # ==================================================

    async def validate_with_llm(
        self,
        candidate: MemoryCandidate
    ) -> MemoryValidationResult:
        """
        使用LLM判断Memory是否具有长期记忆价值。

        注意：
        LLM只负责语义判断，
        不负责数据库操作。

        Structured Output 只替换旧的：
        str -> json.loads -> 手工字段校验。

        原有 fail-closed 策略保持不变：
        任意 LLM / Provider / Structured Validation 失败
        都返回 valid=False，不影响 Chat 主流程。
        """

        system_prompt = """
            你是一个AI Agent的Memory Validation模块。

            你的任务是判断一条Memory Candidate
            是否值得进入用户的 Memory 系统。


            ==================================================
            一、你的判断范围
            ==================================================

            你只需要判断 Candidate 自身
            是否具备进入 Memory 的资格。

            你不需要、也不应该判断：

            1. 它是否与已有 Memory 重复（duplicate）
            2. 它是否与已有 Memory 冲突（conflict）
            3. 它是否与已有 Memory 相关（related）
            4. 它是否是全新 Memory（new）

            这些属于后续 Relationship Judge 模块的职责。

            你没有、也无需已有 Memory 的信息。

            因此：

            即使一条 Candidate 看起来
            与某条已有 Memory 矛盾，
            也不得据此判 valid=false。


            ==================================================
            二、通常可以接受的情况
            ==================================================

            1. 有持续意义的当前状态（Current State）

            描述用户当前处于、
            并且在可预见未来仍将处于的状态。

            例如：

            用户正在学习 FastAPI
            用户目前以 Python 为主要开发语言
            用户现在在做后端开发

            2. 状态变化 / 状态终止 / 状态转移 / 状态重启

            用户明确陈述某个持续状态
            发生改变、结束、停止、转移或重新开始。

            例如：

            用户不再学习 C++
            用户已经停止使用某个框架
            用户从 Python 转向 Java
            用户重新开始健身

            3. 长期稳定的事实、偏好、目标或用户画像

            例如：

            用户长期稳定的偏好
            用户的个人事实
            用户的长期目标
            用户比较稳定的个人画像信息


            ==================================================
            三、重要澄清
            ==================================================

            1. “可能变化”不是自动拒绝理由

            一条信息未来可能改变，
            并不代表它现在没有 Memory 价值。

            Memory 系统本来就要记录用户当前状态，
            并在状态改变时更新它。

            仅仅因为
            “它可能变化”
            或
            “它描述的是当前状态”
            就判 valid=false，是错误的。

            2. 否定句表达的是新状态

            “不再 / 不打算 / 已经放弃 / 没有在 / 已停止”

            这类表述是在陈述一个新的用户状态，
            属于有效信息，
            不应仅因为是否定句而被拒绝。

            3. “当前正在发生的事情”必须区分两种情况

            应当拒绝：
            不指向任何持续状态的瞬时动作。
            例如：用户现在在看文档 / 用户正在点外卖

            不应拒绝：
            描述用户当前持续状态的陈述。
            例如：用户目前在做后端开发


            ==================================================
            四、通常应该拒绝的情况
            ==================================================

            - 一次性事件（One-off Event）

            只发生一次、不代表任何持续状态的具体行为。
            例如：用户今天下午学了两个小时 C++

            - 不指向持续状态的瞬时动作

            - 与用户本人无关的信息

            - 没有实际信息量的内容

            - 一次性的临时行为


            ==================================================
            五、输出语义要求
            ==================================================

            你必须给出：

            valid

            表示这条 Memory Candidate
            是否值得进入长期 Memory。

            reason

            简短说明判断原因。

            输出结构由系统提供的
            Structured Output Schema 约束。
        """

        user_prompt = f"""
            请判断下面这条Memory Candidate：

            content:
            {candidate.content}

            memory_type:
            {candidate.memory_type}
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        try:
            llm_output = await call_llm_structured(
                messages=messages,
                output_model=MemoryValidatorLLMOutput,
            )

            return MemoryValidationResult(
                valid=llm_output.valid,
                reason=llm_output.reason,
            )

        except Exception as e:
            return MemoryValidationResult(
                valid=False,
                reason=f"Memory Validation执行失败：{str(e)}"
            )

    # ==================================================
    # 完整Validation流程
    # ==================================================

    async def validate(
        self,
        candidate: MemoryCandidate
    ) -> MemoryValidationResult:
        """
        完整Validation流程：

        Rule Validation
                ↓
        LLM Validation
        """

        rule_result = self.validate_rules(candidate)

        if not rule_result.valid:
            return rule_result

        return await self.validate_with_llm(candidate)
