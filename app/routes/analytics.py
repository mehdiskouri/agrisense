"""Analytics & prediction routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.database import async_session_factory, get_db
from app.models.enums import UserRoleEnum
from app.schemas.analytics import (
    AlertsResponse,
    BacktestJobCreateResponse,
    BacktestJobStatusResponse,
    BacktestResponse,
    EnsembleYieldForecastResponse,
    FarmStatusResponse,
    IrrigationScheduleResponse,
    NutrientReportResponse,
    YieldForecastResponse,
    ZoneDetailQuery,
    ZoneDetailResponse,
)
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


async def _run_yield_backtest_job(job_id: uuid.UUID, redis_client: object | None) -> None:
    async with async_session_factory() as session:
        service = AnalyticsService(session, redis_client)  # type: ignore[arg-type]
        try:
            await service.execute_yield_backtest_job(job_id)
            await session.commit()
        except Exception:
            await session.rollback()


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LookupError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="analytics failure"
    )


@router.get("/{farm_id}/status", response_model=FarmStatusResponse)
async def get_farm_status(
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
    _user: object = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
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
    _user: object = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
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
    _user: object = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
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
    _user: object = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
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
    _user: object = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
) -> YieldForecastResponse:
    service = AnalyticsService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.get_yield_forecast(farm_id)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.get("/{farm_id}/yield/forecast/ensemble", response_model=EnsembleYieldForecastResponse)
async def get_ensemble_yield_forecast(
    farm_id: uuid.UUID,
    request: Request,
    include_members: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(
        require_role(
            UserRoleEnum.admin,
            UserRoleEnum.agronomist,
            UserRoleEnum.field_operator,
            UserRoleEnum.readonly,
        )
    ),
) -> EnsembleYieldForecastResponse:
    service = AnalyticsService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.get_ensemble_yield_forecast(farm_id, include_members=include_members)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post("/{farm_id}/yield/backtest", response_model=BacktestResponse)
async def run_yield_backtest(
    farm_id: uuid.UUID,
    request: Request,
    n_folds: int = Query(default=5, ge=2, le=20),
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> BacktestResponse:
    service = AnalyticsService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.run_yield_backtest(farm_id, n_folds=n_folds)
    except Exception as exc:
        raise _map_error(exc) from exc


@router.post(
    "/{farm_id}/yield/backtest/async",
    response_model=BacktestJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def enqueue_yield_backtest(
    farm_id: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    n_folds: int = Query(default=5, ge=2, le=20),
    min_history: int = Query(default=24, ge=8, le=192),
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> BacktestJobCreateResponse:
    redis_client = getattr(request.app.state, "redis", None)
    service = AnalyticsService(db, redis_client)
    try:
        response = await service.create_yield_backtest_job(
            farm_id,
            n_folds=n_folds,
            min_history=min_history,
        )
    except Exception as exc:
        raise _map_error(exc) from exc

    background_tasks.add_task(_run_yield_backtest_job, response.job_id, redis_client)
    return response


@router.get("/{farm_id}/yield/backtest/jobs/{job_id}", response_model=BacktestJobStatusResponse)
async def get_yield_backtest_job_status(
    farm_id: uuid.UUID,
    job_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> BacktestJobStatusResponse:
    service = AnalyticsService(db, getattr(request.app.state, "redis", None))
    try:
        status_payload = await service.get_yield_backtest_job_status(job_id)
    except Exception as exc:
        raise _map_error(exc) from exc

    if status_payload.farm_id != farm_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job does not belong to farm",
        )
    return status_payload


@router.get("/{farm_id}/alerts", response_model=AlertsResponse)
async def get_active_alerts(
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
) -> AlertsResponse:
    service = AnalyticsService(db, getattr(request.app.state, "redis", None))
    try:
        return await service.get_active_alerts(farm_id)
    except Exception as exc:
        raise _map_error(exc) from exc
