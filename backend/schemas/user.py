from pydantic import BaseModel, ConfigDict


class UserCreate(BaseModel):

    username:str

    email:str

class UserUpdate(BaseModel):

    username:str | None = None
    email:str | None = None



class UserResponse(BaseModel):

    id:int

    username:str

    email:str

    model_config = ConfigDict(from_attributes=True)



class MessageResponse(BaseModel):

    message:str