"""Data ingestion routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_machine_scope
from app.database import get_db
from app.schemas.ingest import (
	BulkIngestReceipt,
	BulkIngestRequest,
	IngestReceipt,
	IrrigationIngestRequest,
	NpkIngestRequest,
	SoilIngestRequest,
	VisionIngestRequest,
	WeatherIngestRequest,
)
from app.services.ingest_service import IngestService

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _map_error(exc: Exception) -> HTTPException:
	if isinstance(exc, LookupError):
		return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
	if isinstance(exc, ValueError):
		return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
	return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ingest failure")


@router.post("/soil", response_model=IngestReceipt)
async def ingest_soil(
	payload: SoilIngestRequest,
	request: Request,
	db: AsyncSession = Depends(get_db),
	_principal: object = Depends(require_machine_scope("ingest")),
) -> IngestReceipt:
	service = IngestService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.ingest_soil(payload.farm_id, payload.readings)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.post("/weather", response_model=IngestReceipt)
async def ingest_weather(
	payload: WeatherIngestRequest,
	request: Request,
	db: AsyncSession = Depends(get_db),
	_principal: object = Depends(require_machine_scope("ingest")),
) -> IngestReceipt:
	service = IngestService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.ingest_weather(payload.farm_id, payload.readings)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.post("/irrigation", response_model=IngestReceipt)
async def ingest_irrigation(
	payload: IrrigationIngestRequest,
	request: Request,
	db: AsyncSession = Depends(get_db),
	_principal: object = Depends(require_machine_scope("ingest")),
) -> IngestReceipt:
	service = IngestService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.ingest_irrigation(payload.farm_id, payload.events)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.post("/npk", response_model=IngestReceipt)
async def ingest_npk(
	payload: NpkIngestRequest,
	request: Request,
	db: AsyncSession = Depends(get_db),
	_principal: object = Depends(require_machine_scope("ingest")),
) -> IngestReceipt:
	service = IngestService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.ingest_npk(payload.farm_id, payload.samples)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.post("/vision", response_model=IngestReceipt)
async def ingest_vision(
	payload: VisionIngestRequest,
	request: Request,
	db: AsyncSession = Depends(get_db),
	_principal: object = Depends(require_machine_scope("ingest")),
) -> IngestReceipt:
	service = IngestService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.ingest_vision(payload.farm_id, payload.events)
	except Exception as exc:
		raise _map_error(exc) from exc


@router.post("/bulk", response_model=BulkIngestReceipt)
async def ingest_bulk(
	payload: BulkIngestRequest,
	request: Request,
	db: AsyncSession = Depends(get_db),
	_principal: object = Depends(require_machine_scope("ingest")),
) -> BulkIngestReceipt:
	service = IngestService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.ingest_bulk(
			payload.farm_id,
			soil=payload.soil,
			weather=payload.weather,
			irrigation=payload.irrigation,
			npk=payload.npk,
			vision=payload.vision,
			lighting=payload.lighting,
		)
	except Exception as exc:
		raise _map_error(exc) from exc
