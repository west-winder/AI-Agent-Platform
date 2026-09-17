from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db

from backend.schemas.chat import (
    ChatRequest,
    ChatResponse
)

from backend.services.chat_service import (
    chat,
    stream_chat
)

import json

from fastapi.responses import StreamingResponse


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


@router.post(
    "/stream"
)
async def stream_chat_endpoint(
    request : ChatRequest,
    db: Session = Depends(get_db)
):
    async def event_stream():
        async for chunk in stream_chat(
            db=db,
            conversation_id=request.conversation_id,
            user_message=request.content
        ):
            payload = json.dumps(
                {
                    "delta": chunk
                },
                ensure_ascii=False,
            )
            yield f"data: {payload}\n\n"

    response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
    return response