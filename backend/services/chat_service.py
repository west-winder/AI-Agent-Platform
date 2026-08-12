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

from backend.services.agent_service import (
    AgentService
)

from backend.services.llm_service import (
    call_llm
)


# 创建AgentService实例
# 因为AgentService目前采用class形式
agent_service = AgentService()



def chat(
    db: Session,
    conversation_id: int,
    user_message: str
):
    """
    Chat核心业务流程

    完成一次完整聊天：

    1. 查询Conversation
    2. 获取对应Agent
    3. 保存用户消息
    4. 获取历史消息
    5. 转换成LLM格式
    6. 调用LLM
    7. 保存AI回复
    8. 返回结果

    """



    # ==================================================
    # 第一步：
    # 根据conversation_id查询当前聊天
    #
    # Conversation是一次具体聊天实例
    #
    # 它里面保存：
    # - user_id
    # - agent_id
    #
    # 所以后续需要通过它找到Agent
    # ==================================================

    conversation = get_conversation(
        db,
        conversation_id
    )


    if conversation is None:
        raise Exception(
            "Conversation not found"
        )



    # ==================================================
    # 第二步：
    # 根据Conversation找到对应Agent
    #
    # Agent保存：
    # - AI角色(system_prompt)
    # - 使用模型(model_id)
    #
    # 例如：
    #
    # 科研助手
    # 代码助手
    # 英语老师
    #
    # ==================================================

    agent = agent_service.get_agent(
        db,
        conversation.agent_id
    )


    if agent is None:
        raise Exception(
            "Agent not found"
        )



    # ==================================================
    # 第三步：
    # 保存用户当前发送的消息
    #
    # 注意：
    #
    # 这里先保存user消息，
    # 而不是等待LLM返回后再保存。
    #
    # 原因：
    # 即使LLM调用失败，
    # 用户的问题仍然应该被记录。
    #
    # ==================================================

    user_message_data = MessageCreate(
        role="user",
        content=user_message
    )


    create_message(
        db,
        user_message_data,
        conversation_id
    )



    # ==================================================
    # 第四步：
    # 查询历史聊天记录
    #
    # Message表保存所有聊天内容。
    #
    # 这里获取：
    #
    # user:
    # assistant:
    #
    # 之前所有交流
    #
    # ==================================================

    history_messages = get_messages_by_conversation(
        db,
        conversation_id
    )



    # ==================================================
    # 第五步：
    # 将数据库Message转换成LLM需要的格式
    #
    # 数据库:
    #
    # Message模型
    #
    # {
    #    role,
    #    content
    # }
    #
    #
    # LLM:
    #
    # [
    #    {
    #       "role":"user",
    #       "content":"你好"
    #    }
    # ]
    #
    # ==================================================

    llm_messages = []


    # 首先加入Agent的system_prompt
    #
    # 告诉模型：
    # 你是谁
    # 应该如何回答

    llm_messages.append(
        {
            "role": "system",
            "content": agent.system_prompt
        }
    )


    # 添加历史聊天
    for message in history_messages:

        llm_messages.append(
            {
                "role": message.role,
                "content": message.content
            }
        )



    # ==================================================
    # 第六步：
    # 调用LLM
    #
    # 当前llm_service还是模拟返回。
    #
    # 后续这里会替换为：
    #
    # DeepSeek API
    # OpenAI API
    #
    # ==================================================

    answer = call_llm(
        llm_messages
    )



    # ==================================================
    # 第七步：
    # 保存AI回复
    #
    # role:
    #
    # assistant
    #
    # 表示这是模型生成内容
    #
    # ==================================================

    assistant_message_data = MessageCreate(
        role="assistant",
        content=answer
    )


    create_message(
        db,
        assistant_message_data,
        conversation_id
    )



    # ==================================================
    # 第八步：
    # 返回聊天结果
    # ==================================================

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer
    )