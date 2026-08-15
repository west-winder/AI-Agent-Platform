from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MemoryCreate(BaseModel):
    """
    创建Memory时使用。

    user_id不由客户端传入，
    当前阶段由后端固定为1。
    """

    content: str
    memory_type: str


class MemoryUpdate(BaseModel):
    """
    更新Memory时使用。

    所有字段都是可选的，
    只更新客户端实际传入的字段。
    """

    content: str | None = None
    memory_type: str | None = None


class MemoryResponse(BaseModel):
    """
    返回Memory时使用。
    """

    id: int
    user_id: int
    content: str
    memory_type: str

    created_at: datetime
    updated_at: datetime

    # Pydantic 读取SQLAlchemy ORM对象
    model_config = ConfigDict(from_attributes=True)