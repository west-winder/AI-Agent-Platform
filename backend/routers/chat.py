from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db

from backend.schemas.chat import (
    ChatRequest,
    ChatResponse
)

from backend.services.chat_service import (
    chat
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post(
    "",
    response_model=ChatResponse
)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):

    try:

        result = await chat(
            db=db,
            conversation_id=request.conversation_id,
            user_message=request.content
        )

        return result


    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )