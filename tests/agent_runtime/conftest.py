"""仅作用于新模块的测试依赖，不要求真实 API Key。"""

from importlib import import_module
from types import ModuleType

import pytest


@pytest.fixture
def legacy_service(monkeypatch) -> ModuleType:
    # 旧模块在 import 时创建 Client；首次加载时用明确的测试配置。
    # 本 fixture 不发送请求，也不覆盖已经加载的服务对象。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid/v1")
    return import_module("backend.services.llm_service")
