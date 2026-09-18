import asyncio

import pytest
from pydantic import ValidationError

import backend.memory.memory_read.memory_query_scope_judge as scope_judge_module

from backend.memory.memory_read.memory_query_scope import (
    MEMORY_QUERY_SCOPE_BOTH,
    MEMORY_QUERY_SCOPE_CURRENT,
    MEMORY_QUERY_SCOPE_HISTORICAL,
)

from backend.memory.memory_read.memory_query_scope_judge import (
    MemoryQueryScopeJudge,
    MemoryQueryScopeLLMOutput,
)


# ==================================================
# [KEEP]
# Structured Output Model Contract
# ==================================================

def test_structured_output_normalizes_scope_and_reason():
    output = MemoryQueryScopeLLMOutput(
        scope=" HISTORICAL ",
        reason="  用户询问过去状态  ",
    )

    assert output.scope == MEMORY_QUERY_SCOPE_HISTORICAL
    assert output.reason == "用户询问过去状态"


def test_structured_output_rejects_invalid_scope():
    with pytest.raises(ValidationError):
        MemoryQueryScopeLLMOutput(
            scope="past",
            reason="非法 scope",
        )


# ==================================================
# [KEEP]
# Structured LLM Output -> Business Decision
# ==================================================

def test_structured_output_becomes_business_decision(
    monkeypatch,
):
    async def fake_call_llm_structured(
        *,
        messages,
        output_model,
        model=None,
    ):
        assert output_model is MemoryQueryScopeLLMOutput

        return MemoryQueryScopeLLMOutput(
            scope="both",
            reason="需要同时参考过去和当前状态",
        )

    monkeypatch.setattr(
        scope_judge_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    judge = MemoryQueryScopeJudge()

    result = asyncio.run(
        judge.judge(
            "我为什么从 Python 转向 Java？"
        )
    )

    assert result.scope == MEMORY_QUERY_SCOPE_BOTH
    assert result.reason == "需要同时参考过去和当前状态"
    assert result.source == "llm"


# ==================================================
# [KEEP]
# Missing Reason Compatibility
# ==================================================

def test_missing_reason_does_not_discard_valid_scope(
    monkeypatch,
):
    async def fake_call_llm_structured(
        *,
        messages,
        output_model,
        model=None,
    ):
        return MemoryQueryScopeLLMOutput(
            scope="historical",
            reason=None,
        )

    monkeypatch.setattr(
        scope_judge_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    judge = MemoryQueryScopeJudge()

    result = asyncio.run(
        judge.judge(
            "我为什么从 Python 转向 Java？"
        )
    )

    assert result.scope == MEMORY_QUERY_SCOPE_HISTORICAL
    assert result.source == "llm"

    assert result.reason == (
        "LLM 返回了合法 scope，"
        "但没有提供有效 reason"
    )


# ==================================================
# [KEEP]
# Structured LLM Failure -> Fallback
# ==================================================

def test_structured_llm_failure_falls_back_to_current(
    monkeypatch,
):
    async def fake_call_llm_structured(
        *,
        messages,
        output_model,
        model=None,
    ):
        raise RuntimeError(
            "fake provider failure"
        )

    monkeypatch.setattr(
        scope_judge_module,
        "call_llm_structured",
        fake_call_llm_structured,
    )

    judge = MemoryQueryScopeJudge()

    result = asyncio.run(
        judge.judge(
            "我为什么从 Python 转向 Java？"
        )
    )

    assert result.scope == MEMORY_QUERY_SCOPE_CURRENT
    assert result.source == "fallback"
    assert "RuntimeError" in result.reason


# ==================================================
# [KEEP]
# Layer 1 Rule Must Still Bypass LLM
# ==================================================

def test_lexical_rule_still_bypasses_structured_llm(
    monkeypatch,
):
    async def should_not_be_called(
        *,
        messages,
        output_model,
        model=None,
    ):
        raise AssertionError(
            "明确的 rule case 不应该调用 LLM"
        )

    monkeypatch.setattr(
        scope_judge_module,
        "call_llm_structured",
        should_not_be_called,
    )

    judge = MemoryQueryScopeJudge()

    result = asyncio.run(
        judge.judge(
            "我现在主要在学什么？"
        )
    )

    assert result.scope == MEMORY_QUERY_SCOPE_CURRENT
    assert result.source == "rule"