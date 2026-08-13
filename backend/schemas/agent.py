from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AgentBase(BaseModel):

    name: str

    system_prompt: str



class AgentCreate(AgentBase):
    pass



class AgentUpdate(BaseModel):

    name: Optional[str] = None

    system_prompt: Optional[str] = None



class AgentResponse(AgentBase):

    id: int

    user_id: int

    created_at: datetime

    updated_at: datetime

    deleted_at: Optional[datetime] = None


    model_config = ConfigDict(
        from_attributes=True
    )