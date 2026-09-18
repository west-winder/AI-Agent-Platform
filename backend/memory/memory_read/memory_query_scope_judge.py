from typing import Literal

from pydantic import BaseModel, field_validator

from backend.services.llm_service import call_llm_structured

from backend.memory.memory_read.memory_query_scope import (
    MEMORY_QUERY_SCOPE_BOTH,
    MEMORY_QUERY_SCOPE_CURRENT,
    MEMORY_QUERY_SCOPE_HISTORICAL,
    MemoryQueryScopeDecision,
)


# ==================================================
# LLM Structured Output Model
# ==================================================

class MemoryQueryScopeLLMOutput(BaseModel):
    """
    LLM Scope Judge 的结构化输出。

    注意：

    这里只描述 LLM 应该返回的内容：

    scope
    reason

    source 不属于 LLM 的责任，
    而是 MemoryQueryScopeJudge 根据执行路径决定。
    """

    scope: Literal[
        "current",
        "historical",
        "both",
    ]

    reason: str | None = None

    @field_validator(
        "scope",
        mode="before",
    )
    @classmethod
    def normalize_scope(cls, value):
        """
        保留旧 Contract：

        " HISTORICAL "
        ↓
        historical
        """

        if isinstance(value, str):
            return value.strip().lower()

        return value

    @field_validator(
        "reason",
        mode="before",
    )
    @classmethod
    def normalize_reason(cls, value):
        """
        reason 只是 Runtime Trace。

        非字符串 / 空字符串
        ↓
        统一转成 None

        后续由 Judge 补默认 reason。
        """

        if not isinstance(value, str):
            return None

        value = value.strip()

        if not value:
            return None

        return value


# ==================================================
# Memory Query Scope Judge
# ==================================================

class MemoryQueryScopeJudge:
    """
    Memory Read 阶段的 Query Scope Judge。

    职责：

    Query
        ↓
    第一层：
    Cheap Lexical Rules
        ↓
    如果能够高置信度判断
        ↓
    current / historical / both

    如果第一层无法可靠判断
        ↓
    第二层：
    LLM Scope Judge
        ↓
    current / historical / both

    如果 LLM 失败
        ↓
    fallback current


    本模块只负责：

    Query
        ↓
    Retrieval Scope

    不负责：

    1. Database
    2. Memory Repository
    3. Dense Retrieval
    4. BM25
    5. RRF
    6. Reranking
    7. Memory Relevance Judge
    8. Memory Injection
    """


    # ==================================================
    # 第一层：
    # 高置信度 Historical Markers
    # ==================================================

    HISTORICAL_MARKERS = (
        "以前",
        "曾经",
        "过去",
        "之前",
        "当时",
        "最开始",
        "起初",
        "当初",
    )


    # ==================================================
    # 第一层：
    # 高置信度 Current Markers
    # ==================================================

    CURRENT_MARKERS = (
        "现在",
        "当前",
        "目前",
        "现阶段",
        "如今",
        "眼下",
    )


    # ==================================================
    # 第一层：
    # 高置信度 Both Patterns
    #
    # 注意：
    #
    # both 不采用：
    #
    # historical marker
    # +
    # current marker
    #
    # 的简单组合。
    #
    # 因为：
    #
    # “我现在想知道我以前学什么”
    #
    # 同时出现：
    #
    # 现在
    # 以前
    #
    # 但语义其实是 historical。
    #
    # 因此这里只匹配
    # 非常明确的时间比较 / 时间跨度短语。
    # ==================================================

    BOTH_PATTERNS = (
        "从以前到现在",
        "从过去到现在",
        "从之前到现在",
        "从曾经到现在",
        "从当时到现在",
        "从最开始到现在",
        "从起初到现在",

        "以前和现在",
        "现在和以前",
        "以前与现在",
        "现在与以前",

        "过去和现在",
        "现在和过去",
        "过去与现在",
        "现在与过去",

        "之前和现在",
        "现在和之前",
        "之前与现在",
        "现在与之前",

        "曾经和现在",
        "现在和曾经",
        "曾经与现在",
        "现在与曾经",

        "和以前相比",
        "与以前相比",
        "相比以前",

        "和过去相比",
        "与过去相比",
        "相比过去",

        "和之前相比",
        "与之前相比",
        "相比之前",
    )


    # ==================================================
    # Public API
    # ==================================================

    async def judge(
        self,
        query: str,
    ) -> MemoryQueryScopeDecision:
        """
        判断本次 Query 应该检索：

        current
        historical
        both
        """

        # --------------------------------------------------
        # 1. Input Validation
        # --------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query 必须是 str 类型"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query 不能为空"
            )


        # --------------------------------------------------
        # 2. 第一层：
        # Cheap Lexical Rule
        # --------------------------------------------------

        rule_decision = (
            self._judge_by_lexical_rules(
                query
            )
        )

        if rule_decision is not None:
            return rule_decision


        # --------------------------------------------------
        # 3. 第一层无法确定
        #
        # 交给第二层 LLM。
        # --------------------------------------------------

        return await self._judge_by_llm(
            query
        )


    # ==================================================
    # Layer 1
    # Lexical Rules
    # ==================================================

    def _judge_by_lexical_rules(
        self,
        query: str,
    ) -> MemoryQueryScopeDecision | None:
        """
        使用非常保守的词法规则判断 Query Scope。

        返回：

        MemoryQueryScopeDecision
            第一层能够可靠判断。

        None
            第一层拿不准，
            必须交给 LLM。
        """


        # --------------------------------------------------
        # 1. Explicit BOTH
        # --------------------------------------------------

        matched_both_patterns = [
            pattern
            for pattern in self.BOTH_PATTERNS
            if pattern in query
        ]

        if matched_both_patterns:

            return MemoryQueryScopeDecision(
                scope=MEMORY_QUERY_SCOPE_BOTH,
                reason=(
                    "规则层命中明确的 "
                    "current + historical "
                    "时间范围表达："
                    f"{matched_both_patterns}，"
                    "因此判定为 both"
                ),
                source="rule",
            )


        # --------------------------------------------------
        # 2. Historical Markers
        # --------------------------------------------------

        historical_hits = [
            marker
            for marker in self.HISTORICAL_MARKERS
            if marker in query
        ]


        # --------------------------------------------------
        # 3. Current Markers
        # --------------------------------------------------

        current_hits = [
            marker
            for marker in self.CURRENT_MARKERS
            if marker in query
        ]


        # --------------------------------------------------
        # 4. Historical + Current
        #
        # 没有命中明确 both pattern 时，
        # 第一层不强行判断。
        # --------------------------------------------------

        if historical_hits and current_hits:
            return None


        # --------------------------------------------------
        # 5. Historical
        # --------------------------------------------------

        if historical_hits:

            return MemoryQueryScopeDecision(
                scope=MEMORY_QUERY_SCOPE_HISTORICAL,
                reason=(
                    "规则层命中明确的历史时间词："
                    f"{historical_hits}，"
                    "且未命中当前时间词，"
                    "因此判定为 historical"
                ),
                source="rule",
            )


        # --------------------------------------------------
        # 6. Current
        # --------------------------------------------------

        if current_hits:

            return MemoryQueryScopeDecision(
                scope=MEMORY_QUERY_SCOPE_CURRENT,
                reason=(
                    "规则层命中明确的当前时间词："
                    f"{current_hits}，"
                    "且未命中历史时间词，"
                    "因此判定为 current"
                ),
                source="rule",
            )


        # --------------------------------------------------
        # 7. 没有高置信度信号
        #
        # 不默认 current。
        #
        # 留给第二层 LLM。
        # --------------------------------------------------

        return None


    # ==================================================
    # Layer 2
    # LLM Scope Judge
    # ==================================================

    async def _judge_by_llm(
        self,
        query: str,
    ) -> MemoryQueryScopeDecision:
        """
        第一层规则无法可靠判断时，
        使用 LLM 判断 Query Scope。
        """

        system_prompt = """
你是 AI Agent Memory Read 系统中的
Memory Query Scope Judge。

你的唯一任务是：

根据用户当前 Query，
判断本次 Memory Retrieval
应该使用哪个时间范围的 Memory。


只允许以下三种 scope：


1. current

用户主要询问：

当前
现在
目前
现阶段

的用户状态。

例如：

“我现在主要在学什么？”

→ current


2. historical

用户只需要知道过去的状态。

例如：

“我以前主要学什么？”

→ historical

“我曾经做过什么方向？”

→ historical

“我最开始学的是什么？”

→ historical


3. both

回答这个问题必须同时参考：

过去状态
+
当前状态

例如：

“我以前和现在有什么变化？”

→ both

“我的技术方向从最开始到现在
发生了什么变化？”

→ both

“我为什么从 Python 转向 Java？”

→ both


==================================================
重要判断规则
==================================================

不要只根据某一个时间词机械判断。

例如：

“我现在想知道我以前主要学什么？”

虽然同时出现：

“现在”
+
“以前”

但用户真正询问的是过去状态。

因此应该判断：

historical


如果 Query 涉及：

变化
对比
转变过程
从某个方向转向另一个方向
过去与现在之间的关系

通常需要：

both


如果 Query 的时间范围不明确，

优先选择：

current

不要因为不确定，
随意把 historical Memory
加入普通当前状态回答。
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ]


        # --------------------------------------------------
        # 1. Structured LLM Call
        # --------------------------------------------------

        try:

            llm_output = await call_llm_structured(
                messages=messages,
                output_model=MemoryQueryScopeLLMOutput,
            )

        except Exception as exc:

            return self._fallback_current(
                reason=(
                    "Scope Judge LLM 调用失败，"
                    "回退为 current。"
                    f"错误类型：{type(exc).__name__}"
                )
            )


        # --------------------------------------------------
        # 2. Reason Adaptation
        #
        # scope:
        #   已经由 Structured Output
        #   + Pydantic Model 验证。
        #
        # reason:
        #   只是 Runtime Trace。
        #   缺失不应该让合法 scope 失效。
        # --------------------------------------------------

        reason = llm_output.reason

        if reason is None:

            reason = (
                "LLM 返回了合法 scope，"
                "但没有提供有效 reason"
            )


        # --------------------------------------------------
        # 3. Final LLM Decision
        # --------------------------------------------------

        return MemoryQueryScopeDecision(
            scope=llm_output.scope,
            reason=reason,
            source="llm",
        )


    # ==================================================
    # Graceful Degradation
    # ==================================================

    def _fallback_current(
        self,
        reason: str,
    ) -> MemoryQueryScopeDecision:
        """
        Scope Judge 失败时：

        回退到 Memory Read 原本的行为：

        current only。
        """

        return MemoryQueryScopeDecision(
            scope=MEMORY_QUERY_SCOPE_CURRENT,
            reason=reason,
            source="fallback",
        )