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

    注意：

    这里刻意不包含：

    memory_status
    historical_at

    Lifecycle 状态只能由 Memory Lifecycle
    内部流程修改，不接受外部 Request 设置。
    """

    content: str | None = None
    memory_type: str | None = None


class MemoryResponse(BaseModel):
    """
    返回Memory时使用。

    memory_status / historical_at 属于
    Memory Lifecycle 系统内部管理状态。

    这里只作为只读输出暴露，
    便于 Swagger / Debug 检查 Lifecycle。

    它们不出现在：

    MemoryCreate
    MemoryUpdate

    因此普通 Create / Update Request
    无法直接设置生命周期字段。
    """

    id: int
    user_id: int
    content: str
    memory_type: str

    created_at: datetime
    updated_at: datetime

    memory_status: str

    historical_at: datetime | None = None

    # Pydantic 读取SQLAlchemy ORM对象
    model_config = ConfigDict(from_attributes=True)