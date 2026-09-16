import json

from backend.services.llm_service import call_llm

from backend.memory.memory_read.memory_query_scope import (
    ALLOWED_MEMORY_QUERY_SCOPES,
    MEMORY_QUERY_SCOPE_BOTH,
    MEMORY_QUERY_SCOPE_CURRENT,
    MEMORY_QUERY_SCOPE_HISTORICAL,
    MemoryQueryScopeDecision,
)


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
        #
        # 先判断最明确的 both 短语。
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
        # 4. 同时出现 Historical + Current
        #
        # 但又没有命中明确 both pattern。
        #
        # 例如：
        #
        # “我现在想知道我以前学什么？”
        #
        # 这种情况第一层不能乱判，
        # 留给 LLM。
        # --------------------------------------------------

        if historical_hits and current_hits:
            return None


        # --------------------------------------------------
        # 5. 明确 Historical
        # --------------------------------------------------

        if historical_hits:

            return MemoryQueryScopeDecision(
                scope=(
                    MEMORY_QUERY_SCOPE_HISTORICAL
                ),
                reason=(
                    "规则层命中明确的历史时间词："
                    f"{historical_hits}，"
                    "且未命中当前时间词，"
                    "因此判定为 historical"
                ),
                source="rule",
            )


        # --------------------------------------------------
        # 6. 明确 Current
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
        # 例如：
        #
        # “我为什么从 Python 转向 Java？”
        #
        # 没有明确：
        #
        # 以前 / 现在
        #
        # 但实际上需要 both。
        #
        # 因此交给 LLM。
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


==================================================
输出要求
==================================================

严格返回 JSON。

不要输出 JSON 之外的任何内容。

格式：

{
    "scope": "current",
    "reason": "简短说明为什么选择这个 scope"
}
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
        # 1. LLM Call
        # --------------------------------------------------

        try:

            response = await call_llm(
                messages
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
        # 2. Empty / Invalid Response Type
        # --------------------------------------------------

        if not isinstance(
            response,
            str
        ):

            return self._fallback_current(
                reason=(
                    "Scope Judge LLM "
                    "没有返回字符串，"
                    "回退为 current"
                )
            )


        response = response.strip()

        if not response:

            return self._fallback_current(
                reason=(
                    "Scope Judge LLM "
                    "返回空内容，"
                    "回退为 current"
                )
            )


        # --------------------------------------------------
        # 3. JSON Parse
        # --------------------------------------------------

        try:

            data = json.loads(
                response
            )

        except json.JSONDecodeError:

            return self._fallback_current(
                reason=(
                    "Scope Judge LLM "
                    "返回非法 JSON，"
                    "回退为 current"
                )
            )


        if not isinstance(
            data,
            dict
        ):

            return self._fallback_current(
                reason=(
                    "Scope Judge LLM "
                    "返回的 JSON 不是 object，"
                    "回退为 current"
                )
            )


        # --------------------------------------------------
        # 4. Scope Validation
        # --------------------------------------------------

        scope = data.get(
            "scope"
        )

        if isinstance(
            scope,
            str
        ):

            scope = (
                scope
                .strip()
                .lower()
            )


        if (
            scope
            not in ALLOWED_MEMORY_QUERY_SCOPES
        ):

            return self._fallback_current(
                reason=(
                    "Scope Judge LLM "
                    "返回非法 scope，"
                    "回退为 current"
                )
            )


        # --------------------------------------------------
        # 5. Reason
        #
        # reason 是 Runtime Trace 信息。
        #
        # scope 才是真正控制 Retrieval 的字段。
        #
        # 所以：
        #
        # scope 合法
        # 但 reason 缺失
        #
        # 不应该因此丢弃一个合法 scope。
        # --------------------------------------------------

        reason = data.get(
            "reason"
        )

        if (
            not isinstance(reason, str)
            or not reason.strip()
        ):

            reason = (
                "LLM 返回了合法 scope，"
                "但没有提供有效 reason"
            )

        else:

            reason = (
                reason.strip()
            )


        # --------------------------------------------------
        # 6. Final LLM Decision
        # --------------------------------------------------

        return MemoryQueryScopeDecision(
            scope=scope,
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