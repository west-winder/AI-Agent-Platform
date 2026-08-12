from fastapi import FastAPI

from backend.database.database import Base, engine
from backend.models import agent as agent_model
from backend.models import message as message_model
from backend.models import model as model_model
from backend.models.conversation import Conversation as ConversationModel
from backend.models.user import User as UserModel
from backend.routers import agent as agent_router
from backend.routers import chat as chat_router
from backend.routers import conversation as conversation_router
from backend.routers import message as message_router
from backend.routers import model as model_router
from backend.routers import user as user_router

# 初始化数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Agent Backend")

# 注册路由
app.include_router(chat_router.router)
app.include_router(user_router.router)
app.include_router(conversation_router.router)
app.include_router(message_router.router)
app.include_router(model_router.router)
app.include_router(agent_router.router)
