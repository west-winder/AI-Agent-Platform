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

from backend.exceptions.external_exceptions import (
    ExternalServiceError,
    ExternalTimeoutError,
    ExternalRequestError,
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
    request : ChatRequest,
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

    except ExternalTimeoutError as e:

        raise HTTPException(
            status_code=504,
            detail="外部服务响应超时"
        )

    except ExternalRequestError as e:

        raise HTTPException(
            status_code=500,
            detail="外部服务请求失败"
        )

    except ExternalServiceError as e:

        raise HTTPException(
            status_code=503,
            detail="外部服务暂时不可用"
        )


@router.post(
    "/stream"
)
async def stream_chat_endpoint(
    request : ChatRequest,
    db: Session = Depends(get_db)
):
    async def event_stream():

        try:

            async for chunk in stream_chat(
                db=db,
                conversation_id=request.conversation_id,
                user_message=request.content
            ):
                payload = json.dumps(
                    {
                        "type": "delta",
                        "delta": chunk,
                    },
                    ensure_ascii=False,
                )

                yield f"data: {payload}\n\n"

        except ExternalTimeoutError:

            payload = json.dumps(
                {
                    "type": "error",
                    "code": "timeout",
                    "message": "外部服务响应超时",
                },
                ensure_ascii=False,
            )

            yield (
                "event: error\n"
                f"data: {payload}\n\n"
            )

        except ExternalRequestError:

            payload = json.dumps(
                {
                    "type": "error",
                    "code": "request_error",
                    "message": "外部服务请求失败",
                },
                ensure_ascii=False,
            )

            yield (
                "event: error\n"
                f"data: {payload}\n\n"
            )

        except ExternalServiceError:

            payload = json.dumps(
                {
                    "type": "error",
                    "code": "service_unavailable",
                    "message": "外部服务暂时不可用",
                },
                ensure_ascii=False,
            )

            yield (
                "event: error\n"
                f"data: {payload}\n\n"
            )

    response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
    return response