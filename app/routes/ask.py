"""Natural language query route."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.auth.models import User
from app.database import get_db
from app.models.enums import UserRoleEnum
from app.schemas.ask import AskRequest, AskResponse
from app.services.llm_service import LLMService

router = APIRouter(prefix="/ask", tags=["ask"])


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
		return await service.ask(farm_id=farm_id, question=payload.question, language=payload.language)
	except Exception as exc:
		raise _map_error(exc) from exc
