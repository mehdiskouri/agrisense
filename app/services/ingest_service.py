"""Data ingestion validation, storage, and graph notification service."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import FarmTypeEnum, VertexTypeEnum
from app.models.farm import Farm, Vertex, Zone
from app.models.sensors import (
	IrrigationEvent,
	LightingReading,
	NpkSample,
	SoilReading,
	VisionEvent,
	WeatherReading,
)
from app.schemas.ingest import (
	BulkIngestReceipt,
	IngestReceipt,
	IngestWarning,
	IrrigationEventIn,
	LightingReadingIn,
	NpkSampleIn,
	SoilReadingIn,
	VisionEventIn,
	WeatherReadingIn,
)
from app.services import julia_bridge
from app.services.farm_service import FarmService

LAYER_ALIASES = {
	"solar": "lighting",
	"lighting": "lighting",
	"soil": "soil",
	"weather": "weather",
	"irrigation": "irrigation",
	"npk": "npk",
	"vision": "vision",
}

ENDPOINT_LAYER_TOKEN = {
	"soil": "soil",
	"weather": "weather",
	"irrigation": "irrigation",
	"npk": "npk",
	"vision": "vision",
	"lighting": "lighting",
}


class IngestService:
	def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
		self.db = db
		self.redis_client = redis_client
		self._graph_state_cache: dict[str, Any] | None = None

	@staticmethod
	def normalize_layer(layer: str) -> str:
		normalized = LAYER_ALIASES.get(layer.strip().lower())
		if normalized is None:
			raise ValueError(f"unsupported layer token: {layer}")
		return normalized

	async def ingest_soil(self, farm_id: uuid.UUID, readings: Sequence[SoilReadingIn]) -> IngestReceipt:
		await self._validate_active_layer(farm_id, ENDPOINT_LAYER_TOKEN["soil"])
		rows: list[SoilReading] = []
		warnings: list[IngestWarning] = []
		graph_state = await self._get_graph_state(farm_id)

		for idx, item in enumerate(readings):
			vertex = await self._require_vertex(item.sensor_id, farm_id, {VertexTypeEnum.sensor})
			if vertex.config and vertex.config.get("sensor_type") not in (None, "soil"):
				warnings.append(IngestWarning(index=idx, message="sensor_type mismatch for soil endpoint"))

			row = SoilReading(
				sensor_id=item.sensor_id,
				timestamp=item.timestamp,
				moisture=item.moisture,
				temperature=item.temperature,
				conductivity=item.conductivity,
				ph=item.ph,
			)
			rows.append(row)

			features = [
				float(item.moisture),
				float(item.temperature),
				float(item.conductivity or 0.0),
				float(item.ph or 0.0),
			]
			graph_state = self._safe_graph_update(
				graph_state,
				"soil",
				str(item.sensor_id),
				features,
				idx,
				warnings,
			)

		self.db.add_all(rows)
		await self.db.flush()
		event_ids = [int(row.id) for row in rows]
		await self._publish_events(farm_id, "soil", rows)
		self._graph_state_cache = graph_state
		return IngestReceipt(
			farm_id=farm_id,
			layer="soil",
			status="ok" if not warnings else "partial",
			inserted_count=len(rows),
			failed_count=0,
			event_ids=event_ids,
			warnings=warnings,
		)

	async def ingest_weather(self, farm_id: uuid.UUID, readings: Sequence[WeatherReadingIn]) -> IngestReceipt:
		await self._validate_active_layer(farm_id, ENDPOINT_LAYER_TOKEN["weather"])
		rows: list[WeatherReading] = []
		warnings: list[IngestWarning] = []
		graph_state = await self._get_graph_state(farm_id)

		for idx, item in enumerate(readings):
			await self._require_vertex(item.station_id, farm_id, {VertexTypeEnum.weather_station})
			row = WeatherReading(
				station_id=item.station_id,
				timestamp=item.timestamp,
				temperature=item.temperature,
				humidity=item.humidity,
				precipitation_mm=item.precipitation_mm,
				wind_speed=item.wind_speed,
				wind_direction=item.wind_direction,
				pressure_hpa=item.pressure_hpa,
				et0=item.et0,
			)
			rows.append(row)

			features = [
				float(item.temperature),
				float(item.humidity),
				float(item.precipitation_mm),
				float(item.wind_speed or 0.0),
				float(item.et0 or 0.0),
			]
			graph_state = self._safe_graph_update(
				graph_state,
				"weather",
				str(item.station_id),
				features,
				idx,
				warnings,
			)

		self.db.add_all(rows)
		await self.db.flush()
		event_ids = [int(row.id) for row in rows]
		await self._publish_events(farm_id, "weather", rows)
		self._graph_state_cache = graph_state
		return IngestReceipt(
			farm_id=farm_id,
			layer="weather",
			status="ok" if not warnings else "partial",
			inserted_count=len(rows),
			failed_count=0,
			event_ids=event_ids,
			warnings=warnings,
		)

	async def ingest_irrigation(
		self,
		farm_id: uuid.UUID,
		events: Sequence[IrrigationEventIn],
	) -> IngestReceipt:
		await self._validate_active_layer(farm_id, ENDPOINT_LAYER_TOKEN["irrigation"])
		rows: list[IrrigationEvent] = []
		warnings: list[IngestWarning] = []
		graph_state = await self._get_graph_state(farm_id)

		for idx, item in enumerate(events):
			await self._require_vertex(item.valve_id, farm_id, {VertexTypeEnum.valve})
			row = IrrigationEvent(
				valve_id=item.valve_id,
				timestamp_start=item.timestamp_start,
				timestamp_end=item.timestamp_end,
				volume_liters=item.volume_liters,
				trigger=item.trigger,
			)
			rows.append(row)

			duration_minutes = 0.0
			if item.timestamp_end is not None:
				duration_minutes = max(
					0.0,
					(item.timestamp_end - item.timestamp_start).total_seconds() / 60.0,
				)
			flow_rate = float(item.volume_liters or 0.0) / duration_minutes if duration_minutes > 0 else float(item.volume_liters or 0.0)
			valve_state = 1.0 if item.timestamp_end is None else 0.0
			features = [flow_rate, 0.0, valve_state]

			graph_state = self._safe_graph_update(
				graph_state,
				"irrigation",
				str(item.valve_id),
				features,
				idx,
				warnings,
			)

		self.db.add_all(rows)
		await self.db.flush()
		event_ids = [int(row.id) for row in rows]
		await self._publish_events(farm_id, "irrigation", rows)
		self._graph_state_cache = graph_state
		return IngestReceipt(
			farm_id=farm_id,
			layer="irrigation",
			status="ok" if not warnings else "partial",
			inserted_count=len(rows),
			failed_count=0,
			event_ids=event_ids,
			warnings=warnings,
		)

	async def ingest_npk(self, farm_id: uuid.UUID, samples: Sequence[NpkSampleIn]) -> IngestReceipt:
		await self._validate_active_layer(farm_id, ENDPOINT_LAYER_TOKEN["npk"])
		rows: list[NpkSample] = []
		warnings: list[IngestWarning] = []
		graph_state = await self._get_graph_state(farm_id)

		for idx, item in enumerate(samples):
			zone = await self._require_zone(item.zone_id, farm_id)
			row = NpkSample(
				zone_id=item.zone_id,
				timestamp=item.timestamp,
				nitrogen_mg_kg=item.nitrogen_mg_kg,
				phosphorus_mg_kg=item.phosphorus_mg_kg,
				potassium_mg_kg=item.potassium_mg_kg,
				organic_matter_pct=item.organic_matter_pct,
				source=item.source,
			)
			rows.append(row)

			target_vertex = await self._resolve_zone_sensor_vertex(zone, farm_id)
			if target_vertex is None:
				warnings.append(IngestWarning(index=idx, message="no sensor vertex found in zone for npk graph update"))
				continue

			features = [
				float(item.nitrogen_mg_kg),
				float(item.phosphorus_mg_kg),
				float(item.potassium_mg_kg),
			]
			graph_state = self._safe_graph_update(
				graph_state,
				"npk",
				str(target_vertex.id),
				features,
				idx,
				warnings,
			)

		self.db.add_all(rows)
		await self.db.flush()
		event_ids = [int(row.id) for row in rows]
		await self._publish_events(farm_id, "npk", rows)
		self._graph_state_cache = graph_state
		return IngestReceipt(
			farm_id=farm_id,
			layer="npk",
			status="ok" if not warnings else "partial",
			inserted_count=len(rows),
			failed_count=0,
			event_ids=event_ids,
			warnings=warnings,
		)

	async def ingest_vision(self, farm_id: uuid.UUID, events: Sequence[VisionEventIn]) -> IngestReceipt:
		await self._validate_active_layer(farm_id, ENDPOINT_LAYER_TOKEN["vision"])
		rows: list[VisionEvent] = []
		warnings: list[IngestWarning] = []
		graph_state = await self._get_graph_state(farm_id)

		for idx, item in enumerate(events):
			await self._require_vertex(item.camera_id, farm_id, {VertexTypeEnum.camera})
			await self._require_vertex(item.crop_bed_id, farm_id, {VertexTypeEnum.crop_bed})
			row = VisionEvent(
				camera_id=item.camera_id,
				crop_bed_id=item.crop_bed_id,
				timestamp=item.timestamp,
				anomaly_type=item.anomaly_type,
				confidence=item.confidence,
				canopy_coverage_pct=item.canopy_coverage_pct,
				metadata_=item.metadata,
			)
			rows.append(row)

			anomaly_score = float(item.confidence) if item.anomaly_type.value != "none" else 0.0
			features = [
				float(item.canopy_coverage_pct or 0.0),
				0.0,
				anomaly_score,
				0.0,
			]
			graph_state = self._safe_graph_update(
				graph_state,
				"vision",
				str(item.camera_id),
				features,
				idx,
				warnings,
			)

		self.db.add_all(rows)
		await self.db.flush()
		event_ids = [int(row.id) for row in rows]
		await self._publish_events(farm_id, "vision", rows)
		self._graph_state_cache = graph_state
		return IngestReceipt(
			farm_id=farm_id,
			layer="vision",
			status="ok" if not warnings else "partial",
			inserted_count=len(rows),
			failed_count=0,
			event_ids=event_ids,
			warnings=warnings,
		)

	async def ingest_lighting(
		self,
		farm_id: uuid.UUID,
		readings: Sequence[LightingReadingIn],
	) -> IngestReceipt:
		await self._validate_active_layer(farm_id, ENDPOINT_LAYER_TOKEN["lighting"])
		rows: list[LightingReading] = []
		warnings: list[IngestWarning] = []
		graph_state = await self._get_graph_state(farm_id)

		for idx, item in enumerate(readings):
			normalized = self.normalize_layer(item.layer)
			if normalized != "lighting":
				warnings.append(IngestWarning(index=idx, message="lighting payload received unsupported layer alias"))

			await self._require_vertex(item.fixture_id, farm_id, {VertexTypeEnum.light_fixture})
			row = LightingReading(
				fixture_id=item.fixture_id,
				timestamp=item.timestamp,
				par_umol=item.par_umol,
				dli_cumulative=item.dli_cumulative,
				duty_cycle_pct=item.duty_cycle_pct,
				spectrum_profile=item.spectrum_profile,
			)
			rows.append(row)

			features = [float(item.par_umol), float(item.dli_cumulative), 0.0]
			graph_state = self._safe_graph_update(
				graph_state,
				"lighting",
				str(item.fixture_id),
				features,
				idx,
				warnings,
			)

		self.db.add_all(rows)
		await self.db.flush()
		event_ids = [int(row.id) for row in rows]
		await self._publish_events(farm_id, "lighting", rows)
		self._graph_state_cache = graph_state
		return IngestReceipt(
			farm_id=farm_id,
			layer="lighting",
			status="ok" if not warnings else "partial",
			inserted_count=len(rows),
			failed_count=0,
			event_ids=event_ids,
			warnings=warnings,
		)

	async def ingest_bulk(
		self,
		farm_id: uuid.UUID,
		*,
		soil: Sequence[SoilReadingIn],
		weather: Sequence[WeatherReadingIn],
		irrigation: Sequence[IrrigationEventIn],
		npk: Sequence[NpkSampleIn],
		vision: Sequence[VisionEventIn],
		lighting: Sequence[LightingReadingIn],
	) -> BulkIngestReceipt:
		layer_receipts: dict[str, IngestReceipt] = {}
		aggregate_warnings: list[IngestWarning] = []
		total_inserted = 0
		total_failed = 0

		tasks: list[tuple[str, Any, Sequence[Any]]] = [
			("soil", self.ingest_soil, soil),
			("weather", self.ingest_weather, weather),
			("irrigation", self.ingest_irrigation, irrigation),
			("npk", self.ingest_npk, npk),
			("vision", self.ingest_vision, vision),
			("lighting", self.ingest_lighting, lighting),
		]

		for layer, handler, records in tasks:
			if not records:
				continue

			async with self.db.begin_nested():
				try:
					receipt = await handler(farm_id, records)
					layer_receipts[layer] = receipt
					total_inserted += receipt.inserted_count
					total_failed += receipt.failed_count
					aggregate_warnings.extend(receipt.warnings)
				except Exception as exc:
					failed_count = len(records)
					total_failed += failed_count
					warning = IngestWarning(index=0, message=f"{layer} ingest failed: {exc}")
					aggregate_warnings.append(warning)
					layer_receipts[layer] = IngestReceipt(
						farm_id=farm_id,
						layer=layer,
						status="failed",
						inserted_count=0,
						failed_count=failed_count,
						event_ids=[],
						warnings=[warning],
					)

		overall_status = "ok"
		if total_failed > 0 and total_inserted > 0:
			overall_status = "partial"
		elif total_failed > 0 and total_inserted == 0:
			overall_status = "failed"

		return BulkIngestReceipt(
			farm_id=farm_id,
			status=overall_status,
			inserted_count=total_inserted,
			failed_count=total_failed,
			warnings=aggregate_warnings,
			layers=layer_receipts,
		)

	async def _get_graph_state(self, farm_id: uuid.UUID) -> dict[str, Any]:
		if self._graph_state_cache is not None:
			return self._graph_state_cache
		self._graph_state_cache = await FarmService(self.db).get_graph(farm_id)
		return self._graph_state_cache

	def _safe_graph_update(
		self,
		graph_state: dict[str, Any],
		layer: str,
		vertex_id: str,
		features: list[float],
		index: int,
		warnings: list[IngestWarning],
	) -> dict[str, Any]:
		try:
			return julia_bridge.update_features(graph_state, layer, vertex_id, features)
		except Exception as exc:
			warnings.append(IngestWarning(index=index, message=f"graph update failed for {layer}: {exc}"))
			return graph_state

	async def _require_farm(self, farm_id: uuid.UUID) -> Farm:
		row = await self.db.execute(select(Farm).where(Farm.id == farm_id))
		farm = row.scalar_one_or_none()
		if farm is None:
			raise LookupError(f"Farm {farm_id} not found")
		return farm

	async def _require_zone(self, zone_id: uuid.UUID, farm_id: uuid.UUID) -> Zone:
		row = await self.db.execute(select(Zone).where(Zone.id == zone_id))
		zone = row.scalar_one_or_none()
		if zone is None:
			raise LookupError(f"Zone {zone_id} not found")
		if zone.farm_id != farm_id:
			raise ValueError("zone does not belong to the target farm")
		return zone

	async def _require_vertex(
		self,
		vertex_id: uuid.UUID,
		farm_id: uuid.UUID,
		allowed_types: set[VertexTypeEnum],
	) -> Vertex:
		row = await self.db.execute(select(Vertex).where(Vertex.id == vertex_id))
		vertex = row.scalar_one_or_none()
		if vertex is None:
			raise LookupError(f"Vertex {vertex_id} not found")
		if vertex.farm_id != farm_id:
			raise ValueError("vertex does not belong to the target farm")
		if vertex.vertex_type not in allowed_types:
			allowed_str = ",".join(sorted(value.value for value in allowed_types))
			raise ValueError(f"vertex {vertex_id} has invalid type {vertex.vertex_type}; expected one of {allowed_str}")
		return vertex

	async def _validate_active_layer(self, farm_id: uuid.UUID, layer: str) -> None:
		farm = await self._require_farm(farm_id)
		active_layers = FarmService.active_layers_for_farm(FarmTypeEnum(farm.farm_type))
		normalized_active = {self.normalize_layer(token) for token in active_layers if token in LAYER_ALIASES}
		normalized_requested = self.normalize_layer(layer)
		if normalized_requested not in normalized_active:
			raise ValueError(f"layer '{layer}' is not active for farm type '{farm.farm_type}'")

	async def _resolve_zone_sensor_vertex(self, zone: Zone, farm_id: uuid.UUID) -> Vertex | None:
		stmt = (
			select(Vertex)
			.where(
				Vertex.farm_id == farm_id,
				Vertex.zone_id == zone.id,
				Vertex.vertex_type == VertexTypeEnum.sensor,
			)
			.order_by(Vertex.created_at.asc())
		)
		row = await self.db.execute(stmt)
		return row.scalar_one_or_none()

	async def _publish_events(self, farm_id: uuid.UUID, layer: str, rows: Sequence[Any]) -> None:
		if self.redis_client is None:
			return
		channel = f"farm:{farm_id}:live"
		for row in rows:
			payload = {
				"event_type": "ingest",
				"layer": layer,
				"farm_id": str(farm_id),
				"record_id": int(row.id),
				"ingested_at": datetime.utcnow().isoformat(),
			}
			await self.redis_client.publish(channel, json.dumps(payload))
