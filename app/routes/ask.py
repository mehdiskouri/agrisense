"""Natural language query route."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.auth.models import User
from app.database import get_db
from app.models.enums import UserRoleEnum
from app.schemas.ask import AskRequest, AskResponse
from app.services.llm_service import LLMService

router = APIRouter(prefix="/ask", tags=["ask"])


def _as_sse(event_name: str, payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=True)
    return f"event: {event_name}\ndata: {body}\n\n".encode()


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ask failure")


@router.post(
    "/{farm_id}",
    response_model=AskResponse,
)
async def ask_farm(
    farm_id: uuid.UUID,
    payload: AskRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
) -> AskResponse:
    service = LLMService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.ask(
            farm_id=farm_id,
            question=payload.question,
            language=payload.language,
            user_id=_user.id,
            conversation_id=payload.conversation_id,
        )
    except Exception as exc:
        raise _map_error(exc) from exc


@router.delete("/{farm_id}/conversation", status_code=status.HTTP_204_NO_CONTENT)
async def clear_farm_conversation(
    farm_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
) -> None:
    service = LLMService(db, getattr(request.app.state, "redis", None))
    try:
        await service.clear_conversation(farm_id=farm_id, user_id=_user.id)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/{farm_id}/stream",
    response_class=StreamingResponse,
)
async def stream_ask_farm(
    farm_id: uuid.UUID,
    payload: AskRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
) -> StreamingResponse:
    service = LLMService(db, getattr(request.app.state, "redis", None))

    async def event_iter() -> AsyncIterator[bytes]:
        try:
            async for event in service.ask_stream(
                farm_id=farm_id,
                question=payload.question,
                language=payload.language,
                user_id=_user.id,
                conversation_id=payload.conversation_id,
            ):
                event_name = str(event.get("type") or "message")
                yield _as_sse(event_name, event)
        except Exception as exc:
            error_payload: dict[str, Any] = {
                "type": "error",
                "conversation_id": payload.conversation_id or "",
                "data": {"detail": str(_map_error(exc).detail)},
            }
            yield _as_sse("error", error_payload)

    return StreamingResponse(event_iter(), media_type="text/event-stream")
