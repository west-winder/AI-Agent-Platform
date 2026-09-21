from typing import List, Dict, Optional, TypeVar
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)

from backend.exceptions.external_exceptions import (
    ExternalServiceError,
    ExternalTimeoutError,
    ExternalRequestError,
)

from collections.abc import AsyncIterator
import asyncio
import httpx2


# 加载.env文件
load_dotenv()

# ============================================
# LLM Timeout Configuration
# ============================================

LLM_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "5")
)

LLM_READ_TIMEOUT_SECONDS = float(
    os.getenv("LLM_READ_TIMEOUT_SECONDS", "120")
)

LLM_WRITE_TIMEOUT_SECONDS = float(
    os.getenv("LLM_WRITE_TIMEOUT_SECONDS", "30")
)

LLM_POOL_TIMEOUT_SECONDS = float(
    os.getenv("LLM_POOL_TIMEOUT_SECONDS", "5")
)

LLM_CHAT_OVERALL_TIMEOUT_SECONDS = float(
    os.getenv("LLM_CHAT_OVERALL_TIMEOUT_SECONDS", "120")
)

LLM_STRUCTURED_OVERALL_TIMEOUT_SECONDS = float(
    os.getenv("LLM_STRUCTURED_OVERALL_TIMEOUT_SECONDS", "30")
)

LLM_HTTP_TIMEOUT = httpx2.Timeout(
    60.0,
    connect=LLM_CONNECT_TIMEOUT_SECONDS,
    read=LLM_READ_TIMEOUT_SECONDS,
    write=LLM_WRITE_TIMEOUT_SECONDS,
    pool=LLM_POOL_TIMEOUT_SECONDS,
)

LLM_MAX_RETRIES = int(
    os.getenv("LLM_MAX_RETRIES", "2")
)


# 初始化DeepSeek客户端
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    timeout=LLM_HTTP_TIMEOUT,
    max_retries=LLM_MAX_RETRIES,
)


StructuredOutputT = TypeVar(
    "StructuredOutputT",
    bound=BaseModel,
)

# ============================================
# External Error Translation
# ============================================
def _translate_external_error(
    exc: TimeoutError | APIError,
) -> ExternalServiceError:

    # --------------------------------------------------
    # 1. Timeout
    #
    # Python Overall Timeout
    # +
    # OpenAI SDK / HTTP Timeout
    # --------------------------------------------------

    if isinstance(
        exc,
        (TimeoutError, APITimeoutError),
    ):
        return ExternalTimeoutError(
            "外部服务调用超时"
        )

    # --------------------------------------------------
    # 2. Connection Failure
    # --------------------------------------------------

    if isinstance(
        exc,
        APIConnectionError,
    ):
        return ExternalServiceError(
            "外部服务当前不可用"
        )

    # --------------------------------------------------
    # 3. HTTP Status Error
    # --------------------------------------------------

    if isinstance(
        exc,
        APIStatusError,
    ):
        status_code = exc.status_code

        # Provider 明确返回 Request Timeout
        if status_code == 408:
            return ExternalTimeoutError(
                "外部服务调用超时"
            )

        # Conflict / Rate Limit / Server Error
        if (
            status_code in {409, 429}
            or status_code >= 500
        ):
            return ExternalServiceError(
                "外部服务当前不可用"
            )

        # 其他 4xx：
        # 请求 / 认证 / 权限 / 配置等问题
        if 400 <= status_code < 500:
            return ExternalRequestError(
                "外部服务请求或配置存在问题"
            )

    # --------------------------------------------------
    # 4. SDK Boundary Fallback
    #
    # 已确认属于 OpenAI SDK APIError，
    # 但 V1 没有进一步分类。
    # --------------------------------------------------

    return ExternalServiceError(
        "外部服务调用失败"
    )



async def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None
) -> str:
    """
    调用大语言模型生成回复

    Args:
        messages:
            LLM标准消息格式

        model:
            使用模型名称

    Returns:
        AI生成文本
    """


    # 如果没有指定模型
    # 使用.env中的默认模型
    if model is None:
        model = os.getenv(
            "DEFAULT_MODEL",
            "deepseek-flash"
        )


    # 调用DeepSeek API
    try:

        async with asyncio.timeout(
            LLM_CHAT_OVERALL_TIMEOUT_SECONDS
        ):
            response = await client.chat.completions.create(
                model=model,
                messages=messages
            )

    except (TimeoutError, APIError) as exc:

        external_error = (
            _translate_external_error(exc)
        )

        raise external_error from exc

    # 获取AI回复文本
    answer = response.choices[0].message.content


    return answer


async def call_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None
) -> str:
    """
    兼容旧调用接口
    """

    return await chat_completion(
        messages,
        model
    )


async def stream_chat_completion(
    messages: List[Dict[str, str]],
    model: str | None = None,
) -> AsyncIterator[str]:

    actual_model = (
        model
        or os.getenv("DEFAULT_MODEL")
    )

    # ============================================
    # 1. 建立 Stream
    # ============================================

    try:

        stream = await client.chat.completions.create(
            model=actual_model,
            messages=messages,
            stream=True
        )

    except (TimeoutError, APIError) as exc:

        external_error = (
            _translate_external_error(exc)
        )

        raise external_error from exc


    # ============================================
    # 2. 消费 Stream
    # ============================================

    try:

        async for chunk in stream:

            content = (
                chunk.choices[0].delta.content
            )

            if content:
                yield content

    except (TimeoutError, APIError) as exc:

        external_error = (
            _translate_external_error(exc)
        )

        raise external_error from exc


async def stream_llm(
    messages: List[Dict[str, str]],
    model: str | None = None,
) -> AsyncIterator[str]:

    async for chunk in stream_chat_completion(
        messages = messages,
        model = model
    ):
        yield chunk


async def structured_completion(
    messages: List[Dict[str, str]],
    output_model: type[StructuredOutputT],
    model: Optional[str] = None,
) -> StructuredOutputT:
    
    actual_model = model or os.getenv(
        "DEFAULT_MODEL",
        "deepseek-flash"
    )

    try:

        async with asyncio.timeout(
            LLM_STRUCTURED_OVERALL_TIMEOUT_SECONDS
        ):
            response = await client.responses.parse(
                model=actual_model,
                input=messages,
                text_format=output_model,
            )

    except (TimeoutError, APIError) as exc:

        external_error = (
            _translate_external_error(exc)
        )

        raise external_error from exc

    parsed = response.output_parsed

    if parsed is None:
        raise ValueError(
            "LLM structured output 没有返回可解析结果"
        )

    return parsed



async def call_llm_structured(
    messages: List[Dict[str, str]],
    output_model: type[StructuredOutputT],
    model: Optional[str] = None,
) -> StructuredOutputT:

    return await structured_completion(
        messages=messages,
        output_model=output_model,
        model=model,
    )