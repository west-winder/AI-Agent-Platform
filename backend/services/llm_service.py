from typing import List, Dict, Optional, TypeVar
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from collections.abc import AsyncIterator

# 加载.env文件
load_dotenv()


# 初始化DeepSeek客户端
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

StructuredOutputT = TypeVar(
    "StructuredOutputT",
    bound=BaseModel,
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
    response = await client.chat.completions.create(
        model=model,
        messages=messages
    )


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
    messages: List[Dict[str,str]],
    model: str | None = None,
) -> AsyncIterator[str]:

    actual_model = model or os.getenv("DEFAULT_MODEL")

    stream = await client.chat.completions.create(
        model=actual_model,
        messages=messages,
        stream=True
    )

    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


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
    messages: List[Dict[str,str]],
    output_model: type[StructuredOutputT],
    model: Optional[str] = None,
) -> StructuredOutputT:
    
    actual_model = model or os.getenv(
        "DEFAULT_MODEL",
        "deepseek-flash"
    )

    response = await client.responses.parse(
        model=actual_model,
        input=messages,
        text_format=output_model,
    )

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