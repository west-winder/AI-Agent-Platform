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

from backend.memory.memory_read.memory_reader import (
    MemoryReader
)


# ==================================================
# Memory Reader
# ==================================================

memory_reader = MemoryReader()


# ==================================================
# Chat 核心业务
# ==================================================

def chat(
    db: Session,
    conversation_id: int,
    user_message: str
):
    """
    Chat 核心业务流程。

    1. 查询 Conversation
    2. 获取 Agent Snapshot
    3. 保存 User Message
    4. Memory Read
    5. 获取历史消息
    6. 构造 LLM Messages
    7. 调用 Chat LLM
    8. 保存 Assistant Message
    9. Memory Write
    10. 返回结果
    """

    # ==================================================
    # 1. 查询 Conversation
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
    # 2. 获取 Agent Snapshot
    # ==================================================

    agent_snapshot = conversation.agent_snapshot

    if agent_snapshot is None:
        raise ValueError(
            "Agent snapshot not found"
        )

    # ==================================================
    # 3. 保存 User Message
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
    # 4. Memory Read Pipeline
    # ==================================================

    memory_context = ""

    try:

        memory_read_result = (
            memory_reader.read(
                db=db,
                user_id=conversation.user_id,
                query=user_message,
                top_n=20,
                top_k=5
            )
        )

        memory_context = (
            memory_read_result.memory_context
        )

    except Exception as e:

        # ==================================================
        # Memory Read 属于增强能力。
        #
        # 如果 Memory Read 失败，
        # 不应该阻断正常 Chat。
        #
        # 此时保持 memory_context = ""
        # 降级为普通 Chat。
        # ==================================================

        print(
            f"Memory read failed: {e}"
        )

    # ==================================================
    # 5. 获取历史消息
    # ==================================================

    history_messages = (
        get_messages_by_conversation(
            db,
            conversation_id
        )
    )

    # ==================================================
    # 6. 构造 LLM Messages
    # ==================================================

    llm_messages = []

    # --------------------------------------------------
    # 6.1 Agent System Prompt
    # --------------------------------------------------

    system_content = (
        agent_snapshot["system_prompt"]
    )

    # --------------------------------------------------
    # 6.2 Memory Context
    #
    # MemoryReader 负责：
    #
    # Repository
    # → Retrieval
    # → Reranker
    # → Judge
    # → Injector
    #
    # ChatService 负责决定：
    #
    # Memory Context 如何进入最终 LLM Context。
    # --------------------------------------------------

    if memory_context:

        system_content = (
            f"{system_content}\n\n"
            f"{memory_context}"
        )

    # --------------------------------------------------
    # 6.3 System Message
    # --------------------------------------------------

    llm_messages.append(
        {
            "role": "system",
            "content": system_content
        }
    )

    # --------------------------------------------------
    # 6.4 Conversation History
    # --------------------------------------------------

    for message in history_messages:

        llm_messages.append(
            {
                "role": message.role,
                "content": message.content
            }
        )

    # ==================================================
    # 7. 调用 Chat LLM
    # ==================================================

    answer = call_llm(
        llm_messages
    )

    # ==================================================
    # 8. 保存 Assistant Message
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
    # 9. Memory Write Pipeline
    # ==================================================

    try:

        from backend.memory.memory_write.memory_pipeline import (
            MemoryPipeline
        )

        pipeline = MemoryPipeline()

        pipeline.process(
            db=db,
            user_id=conversation.user_id,
            user_message=user_message,
        )

    except Exception as e:

        # ==================================================
        # Memory Write 同样属于增强能力。
        #
        # Memory Write 失败，
        # 不应该影响已经完成的正常 Chat。
        # ==================================================

        print(
            f"Memory write failed: {e}"
        )

    # ==================================================
    # 10. 返回 Chat 结果
    # ==================================================

    return ChatResponse(
        conversation_id=conversation_id,
        message_id=assistant_message.id,
        answer=answer
    )