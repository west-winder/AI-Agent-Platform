"""验证默认组装、FakeLLM 注入和 Mock Client 接线。"""

import asyncio
import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from backend.agent_runtime.bootstrap import build_agent_runtime
from backend.agent_runtime.contracts import AgentRunStatus, LLMMessage, LLMResponse
from backend.agent_runtime.deepseek_adapter import DeepSeekAdapter
from backend.agent_runtime.context_builder import ContextBudget, ContextBuilder
from backend.agent_runtime.errors import ContextBudgetExceededError
from backend.tools.infrastructure.contracts import RegisteredTool, ToolContext
from backend.tools.infrastructure.dispatcher import ToolDispatcher
from backend.tools.infrastructure.registry import ToolRegistry


def test_fake_llm_path_does_not_import_legacy_service_or_load_credentials(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "backend.services" or name.startswith("backend.services."):
            raise AssertionError("FakeLLM must not import services")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    llm = SimpleNamespace(complete=AsyncMock(return_value=LLMResponse("FakeLLM 回答", "stop")))
    runtime = build_agent_runtime(llm=llm, max_iterations=2)
    assert runtime._llm is llm
    assert runtime._registry.get("get_weather").name == "get_weather"
    assert isinstance(runtime._dispatcher, ToolDispatcher)
    assert runtime._dispatcher._registry is runtime._registry
    result = asyncio.run(runtime.run([LLMMessage("user", "问题")], ToolContext()))
    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "FakeLLM 回答"


def test_injected_registry_used_for_definitions_and_dispatch() -> None:
    class Args(BaseModel):
        text: str

    context = ToolContext(user_id=42)

    async def handler(args, received_context):
        assert received_context is context
        return {"text": args.text}

    registry = ToolRegistry()
    registry.register(RegisteredTool("echo", "回声", Args, handler))
    from backend.agent_runtime.contracts import LLMToolCall

    llm = SimpleNamespace(complete=AsyncMock(side_effect=[
        LLMResponse(None, "tool_calls", [LLMToolCall("a", "echo", '{"text":"ok"}', {"text": "ok"})]),
        LLMResponse("完成", "stop"),
    ]))
    runtime = build_agent_runtime(llm=llm, registry=registry)
    result = asyncio.run(runtime.run([LLMMessage("user", "问题")], context))
    assert runtime._registry is registry
    assert llm.complete.await_args_list[0].args[0].tools[0].name == "echo"
    assert result.final_answer == "完成"


def test_complete_client_injection_avoids_legacy_import_and_does_not_close_client(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "backend.services" or name.startswith("backend.services."):
            raise AssertionError("Complete injection must not import services")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(role="assistant", content="mock", tool_calls=None),
    )]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock())
    translator = Mock()
    runtime = build_agent_runtime(client=client, model="injected-model", overall_timeout_seconds=2,
                                  translate_error=translator, registry=ToolRegistry())
    assert isinstance(runtime._llm, DeepSeekAdapter)
    result = asyncio.run(runtime.run([LLMMessage("user", "问题")], ToolContext()))
    assert result.final_answer == "mock"
    assert create.await_args.kwargs["model"] == "injected-model"
    assert "tools" not in create.await_args.kwargs
    client.close.assert_not_awaited()


def test_default_path_reuses_existing_client_timeout_model_and_error_translator(monkeypatch, legacy_service) -> None:
    llm_service = legacy_service
    client = SimpleNamespace()
    translator = Mock()
    monkeypatch.setattr(llm_service, "client", client)
    monkeypatch.setattr(llm_service, "_translate_external_error", translator)
    monkeypatch.setattr(llm_service, "LLM_CHAT_OVERALL_TIMEOUT_SECONDS", 17.0)
    monkeypatch.setenv("DEFAULT_MODEL", "configured-deepseek")
    runtime = build_agent_runtime()
    adapter = runtime._llm
    assert adapter._client is client
    assert adapter._translate_error is translator
    assert adapter._overall_timeout_seconds == 17.0
    assert adapter._model == "configured-deepseek"
    assert runtime._dispatcher._registry is runtime._registry


def test_partial_adapter_overrides_preserve_explicit_values(monkeypatch, legacy_service) -> None:
    llm_service = legacy_service
    client = SimpleNamespace()
    monkeypatch.setattr(llm_service, "LLM_CHAT_OVERALL_TIMEOUT_SECONDS", 9.0)
    runtime = build_agent_runtime(client=client, model="explicit-model", registry=ToolRegistry())
    assert runtime._llm._client is client
    assert runtime._llm._model == "explicit-model"
    assert runtime._llm._overall_timeout_seconds == 9.0
    assert runtime._llm._translate_error is llm_service._translate_external_error


@pytest.mark.parametrize("option", [{"client": SimpleNamespace()}, {"model": "model"}, {"overall_timeout_seconds": 1}, {"translate_error": Mock()}])
def test_conflicting_injected_llm_and_adapter_configuration_rejected(option) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        build_agent_runtime(llm=SimpleNamespace(), **option)


def test_context_builder_default_and_explicit_injection() -> None:
    """Bootstrap 创建默认实例，也允许注入指定构造器。"""
    llm = SimpleNamespace(complete=AsyncMock(return_value=LLMResponse("完成", "stop")))
    default = build_agent_runtime(llm=llm, registry=ToolRegistry())
    assert isinstance(default._context_builder, ContextBuilder)
    builder = ContextBuilder()
    runtime = build_agent_runtime(llm=llm, registry=ToolRegistry(), context_builder=builder)
    assert runtime._context_builder is builder
    result = asyncio.run(runtime.run([LLMMessage("user", "问题")], ToolContext()))
    assert result.final_answer == "完成"


def test_context_budget_configuration_applies_before_llm() -> None:
    """组装层预算配置生效，超额不进入模型。"""
    llm = SimpleNamespace(complete=AsyncMock())
    runtime = build_agent_runtime(llm=llm, registry=ToolRegistry(),
                                  context_budget=ContextBudget(1, 0, 0))
    with pytest.raises(ContextBudgetExceededError):
        asyncio.run(runtime.run([LLMMessage("user", "问题")], ToolContext()))
    llm.complete.assert_not_awaited()


def test_conflicting_context_configuration_rejected() -> None:
    """避免预算参数被已注入的构造器静默忽略。"""
    with pytest.raises(ValueError, match="cannot be combined"):
        build_agent_runtime(llm=SimpleNamespace(), context_builder=ContextBuilder(),
                            context_budget=ContextBudget())
