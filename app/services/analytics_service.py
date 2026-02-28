"""Analytics aggregation service — delegates computation to Julia core."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import FarmTypeEnum
from app.models.farm import Vertex, Zone
from app.schemas.analytics import (
	AlertItem,
	AlertsResponse,
	CrossLayerLink,
	FarmStatusResponse,
	IrrigationScheduleResponse,
	NutrientReportResponse,
	YieldForecastResponse,
	ZoneAlerts,
	ZoneDetailQuery,
	ZoneDetailResponse,
	ZoneStatus,
)
from app.services import julia_bridge
from app.services.farm_service import FarmService


class AnalyticsService:
	def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
		self.db = db
		self.redis_client = redis_client

	async def get_farm_status(self, farm_id: uuid.UUID) -> FarmStatusResponse:
		farm_service = FarmService(self.db)
		farm = await farm_service.get_farm(farm_id)
		graph_state = await farm_service.get_graph(farm_id)

		zones: list[ZoneStatus] = []
		for zone in farm.zones:
			query_vertex = await farm_service.resolve_zone_query_vertex_id(farm_id, zone.id)
			status = julia_bridge.query_farm_status(graph_state, query_vertex)
			zones.append(
				ZoneStatus(
					zone_id=zone.id,
					query_vertex_id=uuid.UUID(query_vertex),
					status=status,
				)
			)

		return FarmStatusResponse(
			farm_id=farm_id,
			generated_at=datetime.now(UTC),
			zones=zones,
		)

	async def get_zone_detail(self, farm_id: uuid.UUID, query: ZoneDetailQuery) -> ZoneDetailResponse:
		farm_service = FarmService(self.db)
		await farm_service.get_farm(farm_id)
		graph_state = await farm_service.get_graph(farm_id)

		zone_id = query.zone_id
		if query.vertex_id is not None:
			vertex = await self._require_vertex(query.vertex_id, farm_id)
			query_vertex_id = vertex.id
			if zone_id is None:
				zone_id = vertex.zone_id
		else:
			assert zone_id is not None
			query_vertex_id = uuid.UUID(await farm_service.resolve_zone_query_vertex_id(farm_id, zone_id))

		layers = julia_bridge.query_farm_status(graph_state, str(query_vertex_id))
		cross_layer = await self._build_cross_layer(graph_state, farm_id, query_vertex_id)

		return ZoneDetailResponse(
			farm_id=farm_id,
			zone_id=zone_id,
			query_vertex_id=query_vertex_id,
			layers=layers,
			cross_layer=cross_layer,
		)

	async def get_irrigation_schedule(self, farm_id: uuid.UUID, horizon_days: int = 7) -> IrrigationScheduleResponse:
		farm_service = FarmService(self.db)
		await farm_service.get_farm(farm_id)
		cache_key = f"farm:{farm_id}:analytics:irrigation:{horizon_days}"

		if self.redis_client is not None:
			cached = await self.redis_client.get(cache_key)
			if cached is not None:
				payload = json.loads(cached)
				return IrrigationScheduleResponse(
					farm_id=farm_id,
					horizon_days=horizon_days,
					cached=True,
					generated_at=datetime.now(UTC),
					items=payload,
				)

		graph_state = await farm_service.get_graph(farm_id)
		items = julia_bridge.irrigation_schedule(graph_state, horizon_days)

		if self.redis_client is not None:
			await self.redis_client.setex(cache_key, 900, json.dumps(items))

		return IrrigationScheduleResponse(
			farm_id=farm_id,
			horizon_days=horizon_days,
			cached=False,
			generated_at=datetime.now(UTC),
			items=items,
		)

	async def get_nutrient_report(self, farm_id: uuid.UUID) -> NutrientReportResponse:
		farm_service = FarmService(self.db)
		await farm_service.get_farm(farm_id)
		graph_state = await farm_service.get_graph(farm_id)
		items = julia_bridge.nutrient_report(graph_state)
		return NutrientReportResponse(farm_id=farm_id, generated_at=datetime.now(UTC), items=items)

	async def get_yield_forecast(self, farm_id: uuid.UUID) -> YieldForecastResponse:
		farm_service = FarmService(self.db)
		await farm_service.get_farm(farm_id)
		graph_state = await farm_service.get_graph(farm_id)
		items = julia_bridge.yield_forecast(graph_state)
		return YieldForecastResponse(farm_id=farm_id, generated_at=datetime.now(UTC), items=items)

	async def get_active_alerts(self, farm_id: uuid.UUID) -> AlertsResponse:
		farm_service = FarmService(self.db)
		farm = await farm_service.get_farm(farm_id)
		graph_state = await farm_service.get_graph(farm_id)

		anomalies = julia_bridge.detect_anomalies(graph_state)
		nutrients = julia_bridge.nutrient_report(graph_state)

		zone_index = await self._zone_vertex_index(farm_id)
		zone_alert_map: dict[uuid.UUID | None, list[AlertItem]] = defaultdict(list)

		for item in anomalies:
			zone_key = self._resolve_alert_zone(item, zone_index)
			severity = str(item.get("severity") or item.get("urgency") or "warning")
			zone_alert_map[zone_key].append(
				AlertItem(source="anomaly", severity=severity, payload=item)
			)

		for item in nutrients:
			zone_key = self._resolve_alert_zone(item, zone_index)
			severity = str(item.get("urgency") or item.get("severity") or "info")
			zone_alert_map[zone_key].append(
				AlertItem(source="nutrients", severity=severity, payload=item)
			)

		if FarmTypeEnum(farm.farm_type) in {FarmTypeEnum.greenhouse, FarmTypeEnum.hybrid}:
			for item in anomalies:
				layer = str(item.get("layer") or "")
				if layer != "vision":
					continue
				zone_key = self._resolve_alert_zone(item, zone_index)
				zone_alert_map[zone_key].append(
					AlertItem(source="vision", severity=str(item.get("severity") or "warning"), payload=item)
				)

		zone_payload = [
			ZoneAlerts(zone_id=zone_id, alerts=alerts)
			for zone_id, alerts in zone_alert_map.items()
		]

		zone_payload.sort(key=lambda item: str(item.zone_id) if item.zone_id is not None else "zzzz")

		return AlertsResponse(
			farm_id=farm_id,
			generated_at=datetime.now(UTC),
			zones=zone_payload,
		)

	async def _build_cross_layer(
		self,
		graph_state: dict[str, Any],
		farm_id: uuid.UUID,
		query_vertex_id: uuid.UUID,
	) -> list[CrossLayerLink]:
		farm = await FarmService(self.db).get_farm(farm_id)
		active_layers = FarmService.active_layers_for_farm(FarmTypeEnum(farm.farm_type))
		normalized_layers = [
			"lighting" if token in {"lighting", "solar"} else token
			for token in active_layers
			if token in {"soil", "irrigation", "lighting", "weather", "npk", "vision", "solar"}
		]
		ordered_layers = sorted(set(normalized_layers))

		result: list[CrossLayerLink] = []
		for idx, layer_a in enumerate(ordered_layers):
			for layer_b in ordered_layers[idx + 1 :]:
				data = julia_bridge.cross_layer_query(graph_state, layer_a, layer_b)
				result.append(CrossLayerLink(layer_a=layer_a, layer_b=layer_b, data={"query_vertex_id": str(query_vertex_id), "result": data}))
		return result

	async def _require_vertex(self, vertex_id: uuid.UUID, farm_id: uuid.UUID) -> Vertex:
		row = await self.db.execute(select(Vertex).where(Vertex.id == vertex_id))
		vertex = row.scalar_one_or_none()
		if vertex is None:
			raise LookupError(f"Vertex {vertex_id} not found")
		if vertex.farm_id != farm_id:
			raise ValueError("vertex does not belong to the target farm")
		return vertex

	async def _zone_vertex_index(self, farm_id: uuid.UUID) -> dict[uuid.UUID, uuid.UUID | None]:
		rows = await self.db.execute(select(Vertex.id, Vertex.zone_id).where(Vertex.farm_id == farm_id))
		return {vertex_id: zone_id for vertex_id, zone_id in rows.all()}

	@staticmethod
	def _resolve_alert_zone(item: dict[str, Any], zone_index: dict[uuid.UUID, uuid.UUID | None]) -> uuid.UUID | None:
		zone_token = item.get("zone_id")
		if zone_token is not None:
			try:
				return uuid.UUID(str(zone_token))
			except ValueError:
				return None

		vertex_token = item.get("vertex_id")
		if vertex_token is None:
			return None
		try:
			vertex_id = uuid.UUID(str(vertex_token))
		except ValueError:
			return None
		return zone_index.get(vertex_id)
