from sqlalchemy.orm import Session

from backend.schemas.message import MessageCreate
from backend.schemas.chat import ChatResponse
from backend.schemas.memory import MemoryCreate

from backend.services.conversation_service import (
    get_conversation
)

from backend.services.message_service import (
    create_message,
    get_messages_by_conversation
)

from backend.services.memory_service import (
    create_memory
)

from backend.services.llm_service import (
    call_llm
)

from backend.memory.extractor import (
    MemoryExtractor
)


# ==================================================
# Memory Extractor
# ==================================================

memory_extractor = MemoryExtractor()


# ==================================================
# Chat核心业务
# ==================================================

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
    6. 调用Chat LLM
    7. 保存AI回复
    8. 提取Memory
    9. 保存Memory
    10. 返回结果
    """

    # ==================================================
    # 1. 查询Conversation
    # ==================================================

    conversation = get_conversation(
        db,
        conversation_id
    )

    if conversation is None:
        raise ValueError(
            "Conversation not found"
        )

    # ==================================================
    # 2. 获取Agent Snapshot
    # ==================================================

    agent_snapshot = conversation.agent_snapshot

    if agent_snapshot is None:
        raise ValueError(
            "Agent snapshot not found"
        )

    # ==================================================
    # 3. 保存用户消息
    # ==================================================

    user_message_data = MessageCreate(
        content=user_message
    )

    create_message(
        db,
        user_message_data,
        conversation_id,
        "user"
    )

    # ==================================================
    # 4. 获取历史消息
    # ==================================================

    history_messages = get_messages_by_conversation(
        db,
        conversation_id
    )

    # ==================================================
    # 5. 转换成LLM消息格式
    # ==================================================

    llm_messages = []

    llm_messages.append(
        {
            "role": "system",
            "content": agent_snapshot["system_prompt"]
        }
    )

    for message in history_messages:

        llm_messages.append(
            {
                "role": message.role,
                "content": message.content
            }
        )

    # ==================================================
    # 6. 调用Chat LLM
    # ==================================================

    answer = call_llm(
        llm_messages
    )

    # ==================================================
    # 7. 保存Assistant Message
    # ==================================================

    assistant_message_data = MessageCreate(
        content=answer
    )

    assistant_message = create_message(
        db,
        assistant_message_data,
        conversation_id,
        "assistant"
    )

    # ==================================================
    # 8. Memory Extraction
    # ==================================================

    try:

        candidates = memory_extractor.extract(
            user_message
        )

        # ==================================================
        # 9. 保存Memory
        # ==================================================

        for candidate in candidates:

            memory_data = MemoryCreate(
                content=candidate.content,
                memory_type=candidate.memory_type
            )

            create_memory(
                db,
                user_id=conversation.user_id,
                memory_data=memory_data
            )

    except Exception as e:

        # ==================================================
        # Memory属于辅助能力。
        #
        # 如果Memory Extraction失败，
        # 不应该影响正常Chat。
        # ==================================================

        print(
            f"Memory extraction failed: {e}"
        )

    # ==================================================
    # 10. 返回Chat结果
    # ==================================================

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=assistant_message.id,
        answer=answer
    )