from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, ConfigDict


class MessageCreate(BaseModel):

    # 用户发送内容
    content: str



class MessageResponse(BaseModel):

    id: int

    conversation_id: int

    role: str

    content: str

    token_count: Optional[int] = None

    message_metadata: Optional[Dict[str, Any]] = None

    created_at: datetime


    model_config = ConfigDict(
        from_attributes=True
    )