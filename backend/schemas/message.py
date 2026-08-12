from datetime import datetime
from pydantic import BaseModel


class MessageBase(BaseModel):

    role: str

    content: str



class MessageCreate(MessageBase):

    pass



class MessageResponse(MessageBase):

    id: int

    conversation_id: int

    token_count: int | None = None

    created_at: datetime


    class Config:

        from_attributes = True