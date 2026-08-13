from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.models.conversation import Conversation
from backend.models.agent import Agent

from backend.schemas.conversation import ConversationCreate



# 创建新的对话记录
def create_conversation(
    db: Session,
    user_id: int,
    conv_in: ConversationCreate
) -> Conversation:


    # 查询Agent
    agent = (
        db.query(Agent)
        .filter(
            Agent.id == conv_in.agent_id,
            Agent.deleted_at.is_(None)
        )
        .first()
    )


    if agent is None:
        raise ValueError(
            "Agent not found"
        )


    # 创建Conversation
    conv = Conversation(

        user_id=user_id,

        agent_id=agent.id,

        # 创建时不生成标题
        title=None,


        # 保存Agent快照
        agent_snapshot={
            "name": agent.name,
            "system_prompt": agent.system_prompt
        },


        # 默认状态
        status="active"
    )


    db.add(conv)

    db.commit()

    db.refresh(conv)

    return conv





# 根据用户ID获取对话列表
def get_conversations(
    db: Session,
    user_id: Optional[int] = None
) -> List[Conversation]:


    q = db.query(Conversation)


    if user_id is not None:

        q = q.filter(
            Conversation.user_id == user_id
        )


    # 不显示已删除Conversation
    q = q.filter(
        Conversation.deleted_at.is_(None)
    )


    return q.all()





# 根据ID获取单条Conversation
def get_conversation(
    db: Session,
    id: int
) -> Optional[Conversation]:


    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == id,
            Conversation.deleted_at.is_(None)
        )
        .first()
    )


    return conv





# 删除Conversation
# 使用软删除
def delete_conversation(
    db: Session,
    id: int
) -> bool:


    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == id,
            Conversation.deleted_at.is_(None)
        )
        .first()
    )


    if not conv:
        return False



    conv.deleted_at = datetime.utcnow()


    db.commit()


    db.refresh(conv)


    return True