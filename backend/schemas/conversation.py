from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class ConversationCreate(BaseModel):
    title: str
    agent_id: int


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    agent_id: int
    title: str
    summary: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
