import json

from backend.schemas.memory_candidate import MemoryCandidate
from backend.schemas.memory_validation import MemoryValidationResult
from backend.services.llm_service import call_llm


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

        # --------------------------------------------------
        # 规则1：content不能为空
        # --------------------------------------------------

        content = candidate.content.strip()

        if not content:
            return MemoryValidationResult(
                valid=False,
                reason="Memory内容不能为空"
            )

        # --------------------------------------------------
        # 规则2：内容不能过短
        # --------------------------------------------------

        if len(content) < 5:
            return MemoryValidationResult(
                valid=False,
                reason="Memory内容过短，缺乏足够的信息量"
            )

        # --------------------------------------------------
        # 规则3：memory_type必须属于允许范围
        # --------------------------------------------------

        if candidate.memory_type not in ALLOWED_MEMORY_TYPES:
            return MemoryValidationResult(
                valid=False,
                reason=f"不支持的Memory类型：{candidate.memory_type}"
            )

        # --------------------------------------------------
        # 所有基础规则通过
        # --------------------------------------------------

        return MemoryValidationResult(
            valid=True,
            reason="通过基础规则检查"
        )

    # ==================================================
    # 第二层：LLM语义验证
    # ==================================================

    def validate_with_llm(
        self,
        candidate: MemoryCandidate
    ) -> MemoryValidationResult:
        """
        使用LLM判断Memory是否具有长期记忆价值。

        注意：
        LLM只负责语义判断，
        不负责数据库操作。
        """

        # --------------------------------------------------
        # System Prompt
        #
        # 定义LLM的角色和判断标准
        # --------------------------------------------------

        system_prompt = """
你是一个AI Agent的Memory Validation模块。

你的任务是判断一条Memory Candidate
是否值得作为“用户的长期Memory”保存。

你需要重点判断：

1. 是否与用户本人有关
2. 是否具有长期记忆价值
3. 是否只是短期、临时性的状态或行为
4. 是否属于稳定的事实、偏好、目标或用户画像
5. 是否具有未来帮助AI理解用户的价值

以下情况通常应该拒绝：

- 一次性的临时行为
- 当前正在发生的事情
- 没有长期价值的信息
- 与用户本人无关的信息
- 没有实际信息量的内容

以下情况通常可以接受：

- 用户长期稳定的偏好
- 用户的个人事实
- 用户的长期目标
- 用户比较稳定的个人画像信息

请严格返回JSON。
不要输出JSON之外的任何内容。

返回格式：

{
    "valid": true,
    "reason": "简短说明判断原因"
}

或者：

{
    "valid": false,
    "reason": "简短说明判断原因"
}
"""

        # --------------------------------------------------
        # User Prompt
        #
        # 提供本次需要判断的Candidate
        # --------------------------------------------------

        user_prompt = f"""
请判断下面这条Memory Candidate：

content:
{candidate.content}

memory_type:
{candidate.memory_type}
"""

        # --------------------------------------------------
        # 构造LLM标准消息格式
        #
        # 注意：
        # call_llm()要求的是：
        #
        # List[Dict[str, str]]
        #
        # 而不是单独的字符串。
        # --------------------------------------------------

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
            # --------------------------------------------------
            # 调用统一LLM Service
            # --------------------------------------------------

            response = call_llm(messages)

            # --------------------------------------------------
            # 解析LLM返回的JSON
            # --------------------------------------------------

            result = json.loads(response)

            valid = result.get("valid")
            reason = result.get("reason")

            # --------------------------------------------------
            # 校验LLM输出结构
            # --------------------------------------------------

            if not isinstance(valid, bool):
                return MemoryValidationResult(
                    valid=False,
                    reason="LLM返回的valid字段不是布尔值"
                )

            if not isinstance(reason, str):
                return MemoryValidationResult(
                    valid=False,
                    reason="LLM返回的reason字段不是字符串"
                )

            return MemoryValidationResult(
                valid=valid,
                reason=reason
            )

        except json.JSONDecodeError:
            return MemoryValidationResult(
                valid=False,
                reason="LLM返回的内容不是合法JSON"
            )

        except Exception as e:
            # --------------------------------------------------
            # Memory Validation失败不能影响Chat主流程。
            #
            # 因此：
            # Validation失败 → 不保存Memory
            # 但Chat仍然可以正常完成。
            # --------------------------------------------------

            return MemoryValidationResult(
                valid=False,
                reason=f"Memory Validation执行失败：{str(e)}"
            )

    # ==================================================
    # 完整Validation流程
    # ==================================================

    def validate(
        self,
        candidate: MemoryCandidate
    ) -> MemoryValidationResult:
        """
        完整Validation流程：

        Rule Validation
                ↓
        LLM Validation
        """

        # --------------------------------------------------
        # 第一阶段：规则验证
        # --------------------------------------------------

        rule_result = self.validate_rules(candidate)

        if not rule_result.valid:
            return rule_result

        # --------------------------------------------------
        # 第二阶段：LLM语义验证
        # --------------------------------------------------

        return self.validate_with_llm(candidate)