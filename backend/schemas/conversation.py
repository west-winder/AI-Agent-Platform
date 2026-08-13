from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime
from typing import Optional, Dict, Any

class ConversationCreate(BaseModel):

    agent_id: int



class ConversationResponse(BaseModel):

    id: int

    user_id: int

    agent_id: int

    title: Optional[str] = None

    summary: Optional[str] = None

    status: str

    agent_snapshot: Optional[Dict[str, Any]] = None

    updated_at: datetime

    last_message_time: Optional[datetime] = None

    created_at: datetime

    deleted_at: Optional[datetime] = None


    model_config = ConfigDict(
        from_attributes=True
    )