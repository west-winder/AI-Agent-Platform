from backend.schemas.chat import ChatRequest, ChatResponse


# 生成简单的 AI 回答文本
def generate_answer(question: str) -> str:

    answer = f"AI回答:{question}"

    return answer


# 将问答内容封装成聊天响应对象
def build_chat_response(req: ChatRequest) -> ChatResponse:

    answer = generate_answer(req.question)

    return ChatResponse(
        answer=answer,
        model=req.model
    )