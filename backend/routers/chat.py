from fastapi import APIRouter

from ..schemas.chat import ChatRequest, ChatResponse

from ..services.chat_service import build_chat_response


router = APIRouter(prefix="/chat", tags=["Chat"])


# 接收聊天请求并返回聊天响应
def chat(req:ChatRequest):

    resp = build_chat_response(req)

    return {
        "answer": resp.answer,
        "model": resp.model
    }