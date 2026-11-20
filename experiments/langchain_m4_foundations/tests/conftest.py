import pytest
from langsmith import tracing_context


def pytest_addoption(parser):
    parser.addoption(
        "--deepseek-smoke",
        action="store_true",
        help="Allow the marked smoke tests to call the configured DeepSeek endpoint.",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("deepseek_smoke") and not item.config.getoption(
        "--deepseek-smoke"
    ):
        pytest.skip("DeepSeek smoke requires --deepseek-smoke")


@pytest.fixture(autouse=True)
def tracing_disabled():
    # 在当前调用上下文禁用上报，无需加载或修改环境变量。
    with tracing_context(enabled=False):
        yield
