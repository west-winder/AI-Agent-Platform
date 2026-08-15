from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func

from backend.database.database import Base


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