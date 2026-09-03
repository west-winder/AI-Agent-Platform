from typing import List, Dict, Optional

import os
from dotenv import load_dotenv
from openai import OpenAI


# 加载.env文件
load_dotenv()


# 初始化DeepSeek客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


def chat_completion(
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
            "deepseek-v4-flash"
        )


    # 调用DeepSeek API
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )


    # 获取AI回复文本
    answer = response.choices[0].message.content


    return answer


def call_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None
) -> str:
    """
    兼容旧调用接口
    """

    return chat_completion(
        messages,
        model
    )