from sqlalchemy.orm import Session

from backend.schemas.message import MessageCreate
from backend.schemas.chat import ChatResponse

from backend.services.conversation_service import (
    get_conversation
)

from backend.services.message_service import (
    create_message,
    get_messages_by_conversation
)

from backend.services.llm_service import (
    call_llm
)


def chat(
    db: Session,
    conversation_id: int,
    user_message: str
):
    """
    Chat核心业务流程

    1. 查询Conversation
    2. 获取agent_snapshot
    3. 保存用户消息
    4. 获取历史消息
    5. 转换LLM格式
    6. 调用LLM
    7. 保存AI回复
    8. 返回结果
    """


    conversation = get_conversation(
        db,
        conversation_id
    )


    if conversation is None:
        raise ValueError(
            "Conversation not found"
        )


    agent_snapshot = conversation.agent_snapshot


    if agent_snapshot is None:
        raise ValueError(
            "Agent snapshot not found"
        )


    user_message_data = MessageCreate(
        content=user_message
    )


    create_message(
        db,
        user_message_data,
        conversation_id,
        "user"
    )


    history_messages = get_messages_by_conversation(
        db,
        conversation_id
    )


    llm_messages = []


    llm_messages.append(
        {
            "role":"system",
            "content":agent_snapshot["system_prompt"]
        }
    )


    for message in history_messages:

        llm_messages.append(
            {
                "role":message.role,
                "content":message.content
            }
        )


    answer = call_llm(
        llm_messages
    )


    assistant_message_data = MessageCreate(
        content=answer
    )


    assistant_message = create_message(
        db,
        assistant_message_data,
        conversation_id,
        "assistant"
    )


    return ChatResponse(
        conversation_id=conversation_id,
        message_id=assistant_message.id,
        answer=answer
    )