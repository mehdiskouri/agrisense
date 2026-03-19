"""Anomaly thresholds and webhook subscription routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.database import get_db
from app.models.enums import UserRoleEnum
from app.schemas.anomalies import (
    ThresholdCreate,
    ThresholdListResponse,
    ThresholdRead,
    ThresholdUpdate,
    WebhookCreate,
    WebhookListResponse,
    WebhookRead,
    WebhookTestResponse,
    WebhookUpdate,
)
from app.services.anomaly_service import AnomalyService
from app.services.webhook_service import WebhookService

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


def _map_error(exc: Exception) -> HTTPException:
    message = str(exc)
    if isinstance(exc, LookupError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    if isinstance(exc, ValueError):
        if "exists" in message or "conflict" in message:
            return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message)
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="anomaly failure"
    )


@router.get("/{farm_id}/thresholds", response_model=ThresholdListResponse)
async def list_thresholds(
    farm_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
) -> ThresholdListResponse:
    service = AnomalyService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.get_thresholds(farm_id)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/{farm_id}/thresholds", response_model=ThresholdRead, status_code=status.HTTP_201_CREATED
)
async def create_threshold(
    farm_id: uuid.UUID,
    payload: ThresholdCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> ThresholdRead:
    service = AnomalyService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.create_threshold(farm_id, payload)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.put("/{farm_id}/thresholds/{threshold_id}", response_model=ThresholdRead)
async def update_threshold(
    farm_id: uuid.UUID,
    threshold_id: uuid.UUID,
    payload: ThresholdUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> ThresholdRead:
    service = AnomalyService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.update_threshold(farm_id, threshold_id, payload)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.delete("/{farm_id}/thresholds/{threshold_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_threshold(
    farm_id: uuid.UUID,
    threshold_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin)),
) -> None:
    service = AnomalyService(db, getattr(request.app.state, "redis", None))
    try:
        await service.delete_threshold(farm_id, threshold_id)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.get("/{farm_id}/webhooks", response_model=WebhookListResponse)
async def list_webhooks(
    farm_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> WebhookListResponse:
    service = WebhookService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.list_subscriptions(farm_id)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post("/{farm_id}/webhooks", response_model=WebhookRead, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    farm_id: uuid.UUID,
    payload: WebhookCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin)),
) -> WebhookRead:
    service = WebhookService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.create_subscription(farm_id, payload)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.put("/{farm_id}/webhooks/{webhook_id}", response_model=WebhookRead)
async def update_webhook(
    farm_id: uuid.UUID,
    webhook_id: uuid.UUID,
    payload: WebhookUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin)),
) -> WebhookRead:
    service = WebhookService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.update_subscription(farm_id, webhook_id, payload)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.delete("/{farm_id}/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    farm_id: uuid.UUID,
    webhook_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin)),
) -> None:
    service = WebhookService(db, getattr(request.app.state, "redis", None))
    try:
        await service.delete_subscription(farm_id, webhook_id)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/{farm_id}/webhooks/{webhook_id}/test",
    response_model=WebhookTestResponse,
)
async def test_webhook(
    farm_id: uuid.UUID,
    webhook_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin)),
) -> WebhookTestResponse:
    service = WebhookService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.test_subscription(farm_id, webhook_id)
    except Exception as exc:
        raise _map_error(exc) from exc
