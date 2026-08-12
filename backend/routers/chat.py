from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session


from backend.database.database import get_db

from backend.schemas.chat import (
    ChatRequest,
    ChatResponse
)

from backend.services.chat_service import (
    chat
)



# 创建Router对象
#
# prefix:
# 代表接口统一前缀
#
# tags:
# 方便Swagger文档分类

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)



@router.post(
    "",
    response_model=ChatResponse
)
def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    AI聊天接口

    请求:

    {
        "conversation_id":1,
        "message":"你好"
    }


    流程:

    Router

    ↓

    ChatService

    ↓

    LLM

    ↓

    返回答案

    """


    # 调用ChatService完成核心聊天流程
    #
    # Router不关心：
    # - 怎么找Agent
    # - 怎么查Message
    # - 怎么调用LLM
    #
    # 这些全部交给Service

    result = chat(
        db=db,
        conversation_id=request.conversation_id,
        user_message=request.message
    )


    return result