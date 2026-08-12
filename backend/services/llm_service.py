from typing import List, Dict


def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    调用大语言模型

    Args:
        messages:
            LLM需要的消息列表

            示例:
            [
                {
                    "role": "system",
                    "content": "你是科研助手"
                },
                {
                    "role": "user",
                    "content": "什么是RAG"
                }
            ]

    Returns:
        AI生成的文本
    """


    # TODO:
    # 后续这里替换成 DeepSeek/OpenAI API 调用


    return "这是模拟AI回复"