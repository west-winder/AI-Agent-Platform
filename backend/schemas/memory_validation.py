from pydantic import BaseModel


class MemoryValidationResult(BaseModel):
    """
    Memory Validation 的结果对象。

    注意：
    这个对象不是数据库模型，
    只是 Memory Pipeline 中的临时数据对象。
    """

    valid: bool
    reason: str