from pydantic import BaseModel


class MemoryCandidate(BaseModel):
    """
    LLM提取出来的候选Memory。

    注意：
    这还不是最终保存到数据库里的Memory，
    而是Memory Extraction阶段产生的临时对象。
    """

    content: str

    # 当前阶段限制Memory类型。
    # 虽然这里暂时不使用Enum，
    # 但Extractor的LLM Prompt会严格限制只能返回这些值。
    memory_type: str