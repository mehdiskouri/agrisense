"""Analytics & prediction routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.analytics import (
	AlertsResponse,
	FarmStatusResponse,
	IrrigationScheduleResponse,
	NutrientReportResponse,
	YieldForecastResponse,
	ZoneDetailQuery,
	ZoneDetailResponse,
)
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _map_error(exc: Exception) -> HTTPException:
	if isinstance(exc, LookupError):
		return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
	if isinstance(exc, ValueError):
		return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
	return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="analytics failure")


@router.get("/{farm_id}/status", response_model=FarmStatusResponse)
async def get_farm_status(
	farm_id: uuid.UUID,
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> FarmStatusResponse:
	service = AnalyticsService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.get_farm_status(farm_id)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.get("/{farm_id}/zones/{zone_id}", response_model=ZoneDetailResponse)
async def get_zone_detail(
	farm_id: uuid.UUID,
	zone_id: uuid.UUID,
	request: Request,
	vertex_id: uuid.UUID | None = Query(default=None),
	db: AsyncSession = Depends(get_db),
) -> ZoneDetailResponse:
	service = AnalyticsService(db, getattr(request.app.state, "redis", None))
	query = ZoneDetailQuery(zone_id=zone_id, vertex_id=vertex_id)
	try:
		return await service.get_zone_detail(farm_id, query)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.get("/{farm_id}/vertices/{vertex_id}", response_model=ZoneDetailResponse)
async def get_vertex_detail(
	farm_id: uuid.UUID,
	vertex_id: uuid.UUID,
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> ZoneDetailResponse:
	service = AnalyticsService(db, getattr(request.app.state, "redis", None))
	query = ZoneDetailQuery(vertex_id=vertex_id)
	try:
		return await service.get_zone_detail(farm_id, query)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.get("/{farm_id}/irrigation/schedule", response_model=IrrigationScheduleResponse)
async def get_irrigation_schedule(
	farm_id: uuid.UUID,
	request: Request,
	horizon_days: int = Query(default=7, ge=1, le=30),
	db: AsyncSession = Depends(get_db),
) -> IrrigationScheduleResponse:
	service = AnalyticsService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.get_irrigation_schedule(farm_id, horizon_days=horizon_days)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.get("/{farm_id}/nutrients/report", response_model=NutrientReportResponse)
async def get_nutrient_report(
	farm_id: uuid.UUID,
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> NutrientReportResponse:
	service = AnalyticsService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.get_nutrient_report(farm_id)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.get("/{farm_id}/yield/forecast", response_model=YieldForecastResponse)
async def get_yield_forecast(
	farm_id: uuid.UUID,
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> YieldForecastResponse:
	service = AnalyticsService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.get_yield_forecast(farm_id)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.get("/{farm_id}/alerts", response_model=AlertsResponse)
async def get_active_alerts(
	farm_id: uuid.UUID,
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> AlertsResponse:
	service = AnalyticsService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.get_active_alerts(farm_id)
	except Exception as exc:
		raise _map_error(exc) from exc
