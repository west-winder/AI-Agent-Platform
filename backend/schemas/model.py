from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelBase(BaseModel):

    provider: str

    model_name: str

    display_name: str | None = None

    api_endpoint: str | None = None

    context_window: int | None = None



class ModelCreate(ModelBase):
    pass



class ModelResponse(ModelBase):

    id: int

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)