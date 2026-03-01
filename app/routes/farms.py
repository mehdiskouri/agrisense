"""Farm & topology CRUD routes."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.database import get_db
from app.models.enums import UserRoleEnum
from app.schemas.farm import (
	FarmCreate,
	FarmGraphRead,
	FarmListRead,
	FarmRead,
	VertexCreate,
	VertexRead,
	ZoneCreate,
	ZoneRead,
)
from app.services.farm_service import FarmService

router = APIRouter(prefix="/farms", tags=["farms"])


def _map_error(exc: Exception) -> HTTPException:
	if isinstance(exc, LookupError):
		return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
	if isinstance(exc, ValueError):
		return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
	return HTTPException(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
		detail="Unexpected farm service failure",
	)


def _to_zone_read(zone: Any) -> ZoneRead:
	return ZoneRead(
		id=zone.id,
		farm_id=zone.farm_id,
		name=zone.name,
		zone_type=zone.zone_type,
		area_m2=zone.area_m2,
		soil_type=zone.soil_type,
		metadata=zone.metadata_,
		created_at=zone.created_at,
		updated_at=zone.updated_at,
	)


def _to_vertex_read(vertex: Any) -> VertexRead:
	return VertexRead(
		id=vertex.id,
		farm_id=vertex.farm_id,
		zone_id=vertex.zone_id,
		vertex_type=vertex.vertex_type,
		config=vertex.config,
		installed_at=vertex.installed_at,
		last_seen_at=vertex.last_seen_at,
		created_at=vertex.created_at,
		updated_at=vertex.updated_at,
	)


def _to_farm_read(farm: Any, service: FarmService) -> FarmRead:
	return FarmRead(
		id=farm.id,
		name=farm.name,
		farm_type=farm.farm_type,
		timezone=farm.timezone,
		model_overrides=farm.model_overrides,
		created_at=farm.created_at,
		updated_at=farm.updated_at,
		active_layers=service.active_layers_for_farm(farm.farm_type),
		zones=[_to_zone_read(zone) for zone in farm.zones],
		vertices=[_to_vertex_read(vertex) for vertex in farm.vertices],
	)


@router.post("", response_model=FarmRead, status_code=status.HTTP_201_CREATED)
async def create_farm(
	payload: FarmCreate,
	db: AsyncSession = Depends(get_db),
	_user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> FarmRead:
	service = FarmService(db)
	try:
		farm = await service.create_farm(payload)
		farm = await service.get_farm(farm.id)
	except Exception as exc:
		raise _map_error(exc) from exc
	return _to_farm_read(farm, service)


@router.get("", response_model=FarmListRead)
async def list_farms(
	db: AsyncSession = Depends(get_db),
	_user: object = Depends(
		require_role(
			UserRoleEnum.admin,
			UserRoleEnum.agronomist,
			UserRoleEnum.field_operator,
			UserRoleEnum.readonly,
		)
	),
) -> FarmListRead:
	service = FarmService(db)
	try:
		farms = await service.list_farms()
	except Exception as exc:
		raise _map_error(exc) from exc
	return FarmListRead(items=[_to_farm_read(farm, service) for farm in farms])


@router.get("/{farm_id}", response_model=FarmRead)
async def get_farm(
	farm_id: uuid.UUID,
	db: AsyncSession = Depends(get_db),
	_user: object = Depends(
		require_role(
			UserRoleEnum.admin,
			UserRoleEnum.agronomist,
			UserRoleEnum.field_operator,
			UserRoleEnum.readonly,
		)
	),
) -> FarmRead:
	service = FarmService(db)
	try:
		farm = await service.get_farm(farm_id)
	except Exception as exc:
		raise _map_error(exc) from exc
	return _to_farm_read(farm, service)


@router.post("/{farm_id}/zones", response_model=ZoneRead, status_code=status.HTTP_201_CREATED)
async def add_zone(
	farm_id: uuid.UUID,
	payload: ZoneCreate,
	db: AsyncSession = Depends(get_db),
	_user: object = Depends(require_role(UserRoleEnum.admin, UserRoleEnum.agronomist)),
) -> ZoneRead:
	service = FarmService(db)
	try:
		zone = await service.add_zone(farm_id, payload)
	except Exception as exc:
		raise _map_error(exc) from exc
	return _to_zone_read(zone)


@router.post("/{farm_id}/sensors", response_model=VertexRead, status_code=status.HTTP_201_CREATED)
async def register_vertex(
	farm_id: uuid.UUID,
	payload: VertexCreate,
	db: AsyncSession = Depends(get_db),
	_user: object = Depends(
		require_role(UserRoleEnum.admin, UserRoleEnum.agronomist, UserRoleEnum.field_operator)
	),
) -> VertexRead:
	service = FarmService(db)
	try:
		vertex = await service.register_vertex(farm_id, payload)
	except Exception as exc:
		raise _map_error(exc) from exc
	return _to_vertex_read(vertex)


@router.get("/{farm_id}/graph", response_model=FarmGraphRead)
async def get_graph(
	farm_id: uuid.UUID,
	db: AsyncSession = Depends(get_db),
	_user: object = Depends(
		require_role(
			UserRoleEnum.admin,
			UserRoleEnum.agronomist,
			UserRoleEnum.field_operator,
			UserRoleEnum.readonly,
		)
	),
) -> FarmGraphRead:
	service = FarmService(db)
	try:
		graph_state = await service.get_graph(farm_id)
	except Exception as exc:
		raise _map_error(exc) from exc
	return FarmGraphRead(**graph_state)
