import asyncio
from contextlib import contextmanager

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_deepseek import ChatDeepSeek

import e1_prompt_model_parser as e1
import e3_async_streaming as e3
import e4_tool_calling as e4
import e6_structured_output as e6
from deepseek_config import read_deepseek_config


pytestmark = pytest.mark.deepseek_smoke


@contextmanager
def redact_request_errors():
    # Provider 异常可能含请求 URL、响应或配置。仅保留异常类和 HTTP 状态。
    try:
        yield
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        detail = f" (HTTP {status})" if isinstance(status, int) else ""
        pytest.fail(f"DeepSeek request: {type(exc).__name__}{detail}", pytrace=False)


@pytest.fixture
def smoke_model():
    config = read_deepseek_config()
    if not config["DEEPSEEK_API_KEY"]:
        pytest.skip("DEEPSEEK_API_KEY is missing")
    if not config["DEFAULT_MODEL"]:
        pytest.skip("DEFAULT_MODEL is missing")
    with redact_request_errors():
        return ChatDeepSeek(
            model=config["DEFAULT_MODEL"],
            api_key=config["DEEPSEEK_API_KEY"],
            base_url=config["DEEPSEEK_BASE_URL"] or "https://api.deepseek.com",
            extra_body={"thinking": {"type": "disabled"}},
            timeout=20.0,
            max_retries=0,
            max_tokens=256,
        )


def test_e1_live_text(smoke_model):
    with redact_request_errors():
        text = e1.explain(smoke_model, e1.EXAMPLE)
    if not isinstance(text, str) or not text.strip():
        pytest.fail("E1 did not return nonempty text", pytrace=False)
    print(f"E1 text_type={type(text).__name__} characters={len(text.strip())}")


def test_e3_live_astream(smoke_model):
    async def collect():
        async with asyncio.timeout(25):
            return [chunk async for chunk in e3.stream_chunks_async(smoke_model, e3.QUESTION)]

    with redact_request_errors():
        chunks = asyncio.run(collect())
    if not chunks or not all(isinstance(chunk, AIMessageChunk) for chunk in chunks):
        pytest.fail("E3 did not yield AIMessageChunk objects", pytrace=False)
    if not e3.combine_chunks(chunks).text.strip():
        pytest.fail("E3 stream contained no nonempty text", pytrace=False)
    print(f"E3 chunks={len(chunks)} merged_text=yes")


def test_e4_live_tool_call(smoke_model):
    with redact_request_errors():
        result = e4.request_weather(smoke_model, e4.QUESTION)
    if not isinstance(result, AIMessage) or len(result.tool_calls) != 1:
        pytest.fail("E4 expected exactly one parsed Tool Call", pytrace=False)
    (call,) = result.tool_calls
    if call["name"] != e4.get_weather.name or call["args"] != {"city": "Beijing"} or not call["id"]:
        pytest.fail("E4 Tool Call name, arguments or ID did not match", pytrace=False)
    print("E4 tool_call=get_weather args_match=yes id_present=yes")


def test_e6_live_json_mode(smoke_model):
    with redact_request_errors():
        result = e6.review_paper(smoke_model, e6.QUESTION)
    if not isinstance(result["raw"], AIMessage):
        pytest.fail("E6 raw result was not an AIMessage", pytrace=False)
    if result["parsing_error"] is not None:
        error_name = type(result["parsing_error"]).__name__
        pytest.fail(f"E6 json_mode parsing_error={error_name}", pytrace=False)
    if not isinstance(result["parsed"], e6.PaperDecision):
        pytest.fail("E6 parsed result was not a PaperDecision", pytrace=False)
    if not result["parsed"].paper_title.strip() or not result["parsed"].reason.strip():
        pytest.fail("E6 returned an empty title or reason", pytrace=False)
    print("E6 method=json_mode raw=AIMessage parsed=PaperDecision parsing_error=None")
