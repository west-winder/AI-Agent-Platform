from pydantic import BaseModel


class ChatRequest(BaseModel):
    """
    用户发送聊天请求
    """

    conversation_id: int

    content: str



class ChatResponse(BaseModel):
    """
    AI返回聊天结果
    """

    conversation_id: int

    message_id: int

    answer: str