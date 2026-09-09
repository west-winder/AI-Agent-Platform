from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func, text

from backend.database.database import Base


# ==================================================
# Memory Lifecycle 状态常量
# ==================================================
#
# Memory Lifecycle V1 只支持两种状态：
#
# current
#     当前仍有效。
#     普通 Memory Read 默认允许检索。
#     正常 Memory Write Lifecycle 的判断对象。
#
# historical
#     曾经成立，但现在不再代表当前状态。
#     普通 Memory Read 默认不检索。
#     不参与 Exact Dedup / Related Retrieval / Relationship Judge。
#
# Lifecycle V1 明确禁止：
#
# historical → current
#
# 因此这里不提供任何"恢复"入口。
# ==================================================

MEMORY_STATUS_CURRENT = "current"

MEMORY_STATUS_HISTORICAL = "historical"

ALLOWED_MEMORY_STATUSES = (
    MEMORY_STATUS_CURRENT,
    MEMORY_STATUS_HISTORICAL,
)


class Memory(Base):
    """
    Memory模型

    表示用户长期保存、未来可能继续使用的信息。

    当前v0.2第一版只实现User Memory。
    """

    __tablename__ = "memories"

    # ==================================================
    # 主键
    # ==================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # ==================================================
    # 所属用户
    # ==================================================

    # 一个User可以拥有多个Memory
    # 当前阶段Memory只属于User，不直接属于Agent或Conversation
    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # ==================================================
    # Memory内容
    # ==================================================

    # 保存自然语言形式的Memory
    #
    # 例如：
    # "用户喜欢使用中文交流。"
    # "用户正在学习FastAPI。"
    # "用户希望成为AI Agent开发工程师。"
    content = Column(
        Text,
        nullable=False
    )

    # ==================================================
    # Memory类型
    # ==================================================

    # 当前支持：
    #
    # profile
    # preference
    # goal
    # fact
    #
    # 第一版直接使用字符串保存。
    # 暂时不引入复杂Enum系统。
    memory_type = Column(
        String(50),
        nullable=False,
        index=True
    )

    # ==================================================
    # 创建时间
    # ==================================================

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # ==================================================
    # 更新时间
    # ==================================================

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # ==================================================
    # Memory Lifecycle 状态
    # ==================================================

    # current / historical
    #
    # Lifecycle V1 只支持这两个值。
    #
    # 该字段属于系统内部管理状态，
    # 不允许普通 Create / Update Request 直接设置。
    memory_status = Column(
        String(20),
        nullable=False,
        default=MEMORY_STATUS_CURRENT,
        server_default=text(f"'{MEMORY_STATUS_CURRENT}'"),
        index=True
    )

    # ==================================================
    # 变为 historical 的时间
    # ==================================================

    # current Memory 的 historical_at 永远为 NULL。
    #
    # historical Memory 的 historical_at
    # 记录其被判定为 historical 的时间。
    historical_at = Column(
        DateTime(timezone=True),
        nullable=True
    )