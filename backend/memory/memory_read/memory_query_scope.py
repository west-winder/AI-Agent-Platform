from dataclasses import dataclass


# ==================================================
# Memory Query Scope
# ==================================================

MEMORY_QUERY_SCOPE_CURRENT = "current"

MEMORY_QUERY_SCOPE_HISTORICAL = "historical"

MEMORY_QUERY_SCOPE_BOTH = "both"


ALLOWED_MEMORY_QUERY_SCOPES = {
    MEMORY_QUERY_SCOPE_CURRENT,
    MEMORY_QUERY_SCOPE_HISTORICAL,
    MEMORY_QUERY_SCOPE_BOTH,
}


# ==================================================
# Scope Decision
# ==================================================

@dataclass
class MemoryQueryScopeDecision:
    """
    一次 Memory Query Scope 判断结果。

    scope:

        current
        historical
        both

    reason:

        为什么做出这次判断。

    source:

        rule
            第一层词法规则直接判断。

        llm
            第一层无法确定，
            由 LLM 判断。

        fallback
            LLM 出错或返回非法结果，
            降级为 current。
    """

    scope: str

    reason: str

    source: str