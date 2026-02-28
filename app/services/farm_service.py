"""Farm CRUD and topology management service."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.enums import FarmTypeEnum, VertexTypeEnum, ZoneTypeEnum
from app.models.farm import Farm, HyperEdge, Vertex, Zone
from app.schemas.farm import FarmCreate, VertexCreate, ZoneCreate
from app.services import julia_bridge

_ACTIVE_LAYERS: dict[FarmTypeEnum, list[str]] = {
	FarmTypeEnum.open_field: [
		"soil",
		"irrigation",
		"solar",
		"weather",
		"crop_requirements",
		"npk",
	],
	FarmTypeEnum.greenhouse: [
		"soil",
		"irrigation",
		"lighting",
		"weather",
		"crop_requirements",
		"npk",
		"vision",
	],
	FarmTypeEnum.hybrid: [
		"soil",
		"irrigation",
		"lighting",
		"solar",
		"weather",
		"crop_requirements",
		"npk",
		"vision",
	],
}

_MODEL_DEFAULTS: dict[str, bool] = {
	"irrigation": True,
	"nutrients": True,
	"yield_forecast": True,
	"anomaly_detection": True,
}


class FarmService:
	"""Service for farm CRUD, topology assembly, and graph bridge calls."""

	def __init__(self, db: AsyncSession):
		self.db = db

	async def create_farm(self, payload: FarmCreate) -> Farm:
		farm = Farm(
			name=payload.name,
			farm_type=payload.farm_type,
			timezone=payload.timezone,
			model_overrides=payload.model_overrides,
		)
		self.db.add(farm)
		await self.db.flush()
		await self.db.refresh(farm)
		return farm

	async def list_farms(self) -> list[Farm]:
		stmt = (
			select(Farm)
			.options(selectinload(Farm.zones), selectinload(Farm.vertices))
			.order_by(Farm.created_at.desc())
		)
		rows = await self.db.execute(stmt)
		return list(rows.scalars().all())

	async def get_farm(self, farm_id: uuid.UUID) -> Farm:
		stmt = (
			select(Farm)
			.where(Farm.id == farm_id)
			.options(selectinload(Farm.zones), selectinload(Farm.vertices))
		)
		row = await self.db.execute(stmt)
		farm = row.scalar_one_or_none()
		if farm is None:
			raise LookupError(f"Farm {farm_id} not found")
		return farm

	async def add_zone(self, farm_id: uuid.UUID, payload: ZoneCreate) -> Zone:
		farm = await self.get_farm(farm_id)
		zone_type = self._resolve_zone_type(farm.farm_type, payload.zone_type)
		zone = Zone(
			farm_id=farm.id,
			name=payload.name,
			zone_type=zone_type,
			area_m2=payload.area_m2,
			soil_type=payload.soil_type,
			metadata_=payload.metadata,
		)
		self.db.add(zone)
		await self.db.flush()
		await self.db.refresh(zone)
		return zone

	async def register_vertex(self, farm_id: uuid.UUID, payload: VertexCreate) -> Vertex:
		farm = await self.get_farm(farm_id)
		if payload.zone_id is not None:
			zone = await self._get_zone(payload.zone_id)
			if zone.farm_id != farm.id:
				raise ValueError("zone_id does not belong to the target farm")
			self._validate_vertex_for_zone(farm.farm_type, payload.vertex_type, zone.zone_type)
		else:
			self._validate_farm_level_vertex(payload.vertex_type)

		vertex = Vertex(
			farm_id=farm.id,
			zone_id=payload.zone_id,
			vertex_type=payload.vertex_type,
			config=payload.config,
			installed_at=payload.installed_at,
			last_seen_at=payload.last_seen_at,
		)
		self.db.add(vertex)
		await self.db.flush()
		await self.db.refresh(vertex)
		return vertex

	async def get_graph(self, farm_id: uuid.UUID) -> dict[str, Any]:
		config = await self.build_farm_graph_config(farm_id)
		return julia_bridge.build_graph(config)

	async def resolve_zone_query_vertex_id(
		self,
		farm_id: uuid.UUID,
		zone_id: uuid.UUID,
	) -> str:
		zone = await self._get_zone(zone_id)
		if zone.farm_id != farm_id:
			raise ValueError("zone_id does not belong to the target farm")

		stmt = (
			select(Vertex)
			.where(Vertex.farm_id == farm_id, Vertex.zone_id == zone_id)
			.order_by(Vertex.created_at.asc())
		)
		rows = await self.db.execute(stmt)
		vertices = list(rows.scalars().all())
		if not vertices:
			raise LookupError("zone_id has no registered vertices for graph status query")
		return str(vertices[0].id)

	async def query_zone_status(
		self,
		farm_id: uuid.UUID,
		zone_id: uuid.UUID,
	) -> dict[str, Any]:
		graph_state = await self.get_graph(farm_id)
		query_vertex_id = await self.resolve_zone_query_vertex_id(farm_id, zone_id)
		return julia_bridge.query_farm_status(graph_state, query_vertex_id)

	async def build_farm_graph_config(self, farm_id: uuid.UUID) -> dict[str, Any]:
		farm = await self.get_farm(farm_id)

		edges_stmt = select(HyperEdge).where(HyperEdge.farm_id == farm.id)
		edges_rows = await self.db.execute(edges_stmt)
		edges = list(edges_rows.scalars().all())

		return {
			"farm_id": str(farm.id),
			"farm_type": farm.farm_type.value,
			"active_layers": self.active_layers_for_farm(farm.farm_type),
			"zones": [self._zone_to_config(zone) for zone in farm.zones],
			"models": self._model_config(farm.model_overrides),
			"vertices": [self._vertex_to_config(vertex) for vertex in farm.vertices],
			"edges": [self._edge_to_config(edge) for edge in edges],
		}

	@staticmethod
	def active_layers_for_farm(farm_type: FarmTypeEnum) -> list[str]:
		return list(_ACTIVE_LAYERS[farm_type])

	async def _get_zone(self, zone_id: uuid.UUID) -> Zone:
		row = await self.db.execute(select(Zone).where(Zone.id == zone_id))
		zone = row.scalar_one_or_none()
		if zone is None:
			raise LookupError(f"Zone {zone_id} not found")
		return zone

	@staticmethod
	def _resolve_zone_type(
		farm_type: FarmTypeEnum,
		requested: ZoneTypeEnum | None,
	) -> ZoneTypeEnum:
		if farm_type == FarmTypeEnum.open_field:
			if requested in (None, ZoneTypeEnum.open_field):
				return ZoneTypeEnum.open_field
			raise ValueError("open_field farms can only contain open_field zones")

		if farm_type == FarmTypeEnum.greenhouse:
			if requested in (None, ZoneTypeEnum.greenhouse):
				return ZoneTypeEnum.greenhouse
			raise ValueError("greenhouse farms can only contain greenhouse zones")

		if requested is None:
			raise ValueError("hybrid farms require explicit zone_type")
		return requested

	@staticmethod
	def _validate_vertex_for_zone(
		farm_type: FarmTypeEnum,
		vertex_type: VertexTypeEnum,
		zone_type: ZoneTypeEnum,
	) -> None:
		if zone_type == ZoneTypeEnum.open_field and vertex_type in {
			VertexTypeEnum.camera,
			VertexTypeEnum.light_fixture,
			VertexTypeEnum.climate_controller,
		}:
			raise ValueError("selected vertex_type requires a greenhouse zone")

		if farm_type == FarmTypeEnum.open_field and vertex_type in {
			VertexTypeEnum.camera,
			VertexTypeEnum.light_fixture,
			VertexTypeEnum.climate_controller,
		}:
			raise ValueError("open_field farms cannot register greenhouse-only vertex types")

	@staticmethod
	def _validate_farm_level_vertex(vertex_type: VertexTypeEnum) -> None:
		if vertex_type != VertexTypeEnum.weather_station:
			raise ValueError("zone_id is required for non-weather-station vertices")

	@staticmethod
	def _model_config(overrides: dict[str, Any] | None) -> dict[str, bool]:
		merged = dict(_MODEL_DEFAULTS)
		if overrides:
			for key, value in overrides.items():
				if key in merged:
					merged[key] = bool(value)
		return merged

	@staticmethod
	def _zone_to_config(zone: Zone) -> dict[str, Any]:
		return {
			"id": str(zone.id),
			"name": zone.name,
			"zone_type": zone.zone_type.value,
			"area_m2": float(zone.area_m2),
			"soil_type": zone.soil_type,
		}

	@staticmethod
	def _vertex_to_config(vertex: Vertex) -> dict[str, Any]:
		payload: dict[str, Any] = {
			"id": str(vertex.id),
			"type": vertex.vertex_type.value,
			"config": vertex.config or {},
		}
		if vertex.zone_id is not None:
			payload["zone_id"] = str(vertex.zone_id)
		return payload

	@staticmethod
	def _edge_to_config(edge: HyperEdge) -> dict[str, Any]:
		return {
			"id": str(edge.id),
			"layer": edge.layer.value,
			"vertex_ids": [str(vertex_id) for vertex_id in edge.vertex_ids],
			"metadata": edge.metadata_ or {},
		}
