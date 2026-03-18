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
from app.models.farm import Vertex
from app.schemas.analytics import (
    AlertItem,
    AlertsResponse,
    BacktestJobCreateResponse,
    BacktestJobStatusResponse,
    BacktestResponse,
    CrossLayerLink,
    EnsembleYieldForecastResponse,
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
    BACKTEST_JOB_TTL_SECONDS = 60 * 60 * 24

    def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
        self.db = db
        self.redis_client = redis_client

    @staticmethod
    def _backtest_job_key(job_id: uuid.UUID) -> str:
        return f"analytics:yield_backtest:job:{job_id}"

    async def _ensure_graph(self, farm_id: uuid.UUID) -> None:
        """Ensure graph is built and cached in Julia for this farm."""
        try:
            julia_bridge.ensure_graph_cached(str(farm_id))
        except Exception:
            await FarmService(self.db).get_graph(farm_id)

    async def get_farm_status(self, farm_id: uuid.UUID) -> FarmStatusResponse:
        farm_service = FarmService(self.db)
        farm = await farm_service.get_farm(farm_id)
        await self._ensure_graph(farm_id)

        zones: list[ZoneStatus] = []
        for zone in farm.zones:
            query_vertex = await farm_service.resolve_zone_query_vertex_id(farm_id, zone.id)
            status = julia_bridge.query_farm_status(str(farm_id), query_vertex)
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

    async def get_zone_detail(
        self, farm_id: uuid.UUID, query: ZoneDetailQuery
    ) -> ZoneDetailResponse:
        farm_service = FarmService(self.db)
        await farm_service.get_farm(farm_id)
        await self._ensure_graph(farm_id)
        zone_id = query.zone_id
        if query.vertex_id is not None:
            vertex = await self._require_vertex(query.vertex_id, farm_id)
            query_vertex_id = vertex.id
            if zone_id is None:
                zone_id = vertex.zone_id
        else:
            assert zone_id is not None
            query_vertex_id = uuid.UUID(
                await farm_service.resolve_zone_query_vertex_id(farm_id, zone_id)
            )

        layers = julia_bridge.query_farm_status(str(farm_id), str(query_vertex_id))
        cross_layer = await self._build_cross_layer(farm_id, query_vertex_id)

        return ZoneDetailResponse(
            farm_id=farm_id,
            zone_id=zone_id,
            query_vertex_id=query_vertex_id,
            layers=layers,
            cross_layer=cross_layer,
        )

    async def get_irrigation_schedule(
        self, farm_id: uuid.UUID, horizon_days: int = 7
    ) -> IrrigationScheduleResponse:
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

        await self._ensure_graph(farm_id)
        items = julia_bridge.irrigation_schedule(str(farm_id), horizon_days)

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
        await self._ensure_graph(farm_id)
        items = julia_bridge.nutrient_report(str(farm_id))
        return NutrientReportResponse(farm_id=farm_id, generated_at=datetime.now(UTC), items=items)

    async def get_yield_forecast(self, farm_id: uuid.UUID) -> YieldForecastResponse:
        farm_service = FarmService(self.db)
        await farm_service.get_farm(farm_id)
        await self._ensure_graph(farm_id)
        items = julia_bridge.yield_forecast(str(farm_id))
        return YieldForecastResponse(farm_id=farm_id, generated_at=datetime.now(UTC), items=items)

    async def get_ensemble_yield_forecast(
        self,
        farm_id: uuid.UUID,
        include_members: bool,
    ) -> EnsembleYieldForecastResponse:
        farm_service = FarmService(self.db)
        await farm_service.get_farm(farm_id)
        await self._ensure_graph(farm_id)
        items = julia_bridge.yield_forecast_ensemble(str(farm_id), include_members=include_members)

        aggregate_weights: dict[str, float] = {}
        for item in items:
            raw = item.get("ensemble_weights")
            if isinstance(raw, dict):
                aggregate_weights = {
                    str(key): float(value)
                    for key, value in raw.items()
                    if isinstance(value, (int, float))
                }
                break

        return EnsembleYieldForecastResponse(
            farm_id=farm_id,
            generated_at=datetime.now(UTC),
            include_members=include_members,
            ensemble_weights=aggregate_weights,
            items=items,
        )

    async def run_yield_backtest(self, farm_id: uuid.UUID, n_folds: int) -> BacktestResponse:
        farm_service = FarmService(self.db)
        await farm_service.get_farm(farm_id)
        await self._ensure_graph(farm_id)
        payload = julia_bridge.backtest_yield(str(farm_id), n_folds=n_folds)

        weights_raw = payload.get("weights")
        weights: dict[str, float] = {}
        if isinstance(weights_raw, dict):
            weights = {
                str(key): float(value)
                for key, value in weights_raw.items()
                if isinstance(value, (int, float))
            }

        folds_raw = payload.get("per_fold_metrics")
        per_fold_metrics: list[dict[str, Any]] = []
        if isinstance(folds_raw, list):
            per_fold_metrics = [item for item in folds_raw if isinstance(item, dict)]

        aggregate_raw = payload.get("aggregate_metrics")
        aggregate_metrics = aggregate_raw if isinstance(aggregate_raw, dict) else {}

        temporal_split_raw = payload.get("temporal_split")
        temporal_split: dict[str, dict[str, int]] = {}
        if isinstance(temporal_split_raw, dict):
            for segment, window in temporal_split_raw.items():
                if isinstance(window, dict):
                    start = window.get("start")
                    end = window.get("end")
                    if isinstance(start, int) and isinstance(end, int):
                        temporal_split[str(segment)] = {"start": start, "end": end}

        oracle_raw = payload.get("oracle_provenance")
        oracle_provenance = oracle_raw if isinstance(oracle_raw, dict) else {}

        status = str(payload.get("status") or "ok")
        n_folds_raw = payload.get("n_folds")
        folds_value = int(n_folds_raw) if isinstance(n_folds_raw, int) else n_folds

        return BacktestResponse(
            farm_id=farm_id,
            generated_at=datetime.now(UTC),
            n_folds=folds_value,
            status=status,
            per_fold_metrics=per_fold_metrics,
            aggregate_metrics=aggregate_metrics,
            weights=weights,
            hyperparameters={
                str(k): float(v)
                for k, v in (payload.get("hyperparameters") or {}).items()
                if isinstance(v, (int, float))
            }
            if isinstance(payload.get("hyperparameters"), dict)
            else {},
            temporal_split=temporal_split,
            oracle_provenance=oracle_provenance,
        )

    async def create_yield_backtest_job(
        self,
        farm_id: uuid.UUID,
        n_folds: int,
        min_history: int,
    ) -> BacktestJobCreateResponse:
        await FarmService(self.db).get_farm(farm_id)
        if self.redis_client is None:
            raise RuntimeError("redis is required for async backtest jobs")

        now = datetime.now(UTC)
        job_id = uuid.uuid4()
        payload = {
            "job_id": str(job_id),
            "farm_id": str(farm_id),
            "n_folds": int(n_folds),
            "min_history": int(min_history),
            "status": "queued",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "completed_at": None,
            "error": None,
            "result": None,
        }
        await self.redis_client.setex(
            self._backtest_job_key(job_id),
            self.BACKTEST_JOB_TTL_SECONDS,
            json.dumps(payload),
        )
        return BacktestJobCreateResponse(
            job_id=job_id,
            farm_id=farm_id,
            status="queued",
            created_at=now,
        )

    async def execute_yield_backtest_job(self, job_id: uuid.UUID) -> BacktestJobStatusResponse:
        if self.redis_client is None:
            raise RuntimeError("redis is required for async backtest jobs")

        key = self._backtest_job_key(job_id)
        raw = await self.redis_client.get(key)
        if raw is None:
            raise LookupError(f"Backtest job {job_id} not found")

        payload = json.loads(raw)
        farm_id = uuid.UUID(str(payload["farm_id"]))
        n_folds = int(payload.get("n_folds", 5))
        min_history = int(payload.get("min_history", 24))

        now = datetime.now(UTC)
        payload["status"] = "running"
        payload["updated_at"] = now.isoformat()
        await self.redis_client.setex(key, self.BACKTEST_JOB_TTL_SECONDS, json.dumps(payload))

        try:
            result = await self.run_yield_backtest(farm_id, n_folds=n_folds)
            payload["status"] = "succeeded"
            payload["result"] = result.model_dump(mode="json")
            payload["error"] = None
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = str(exc)
            payload["result"] = None

        completed = datetime.now(UTC)
        payload["updated_at"] = completed.isoformat()
        payload["completed_at"] = completed.isoformat()
        payload["min_history"] = min_history
        await self.redis_client.setex(key, self.BACKTEST_JOB_TTL_SECONDS, json.dumps(payload))

        return await self.get_yield_backtest_job_status(job_id)

    async def get_yield_backtest_job_status(self, job_id: uuid.UUID) -> BacktestJobStatusResponse:
        if self.redis_client is None:
            raise RuntimeError("redis is required for async backtest jobs")

        raw = await self.redis_client.get(self._backtest_job_key(job_id))
        if raw is None:
            raise LookupError(f"Backtest job {job_id} not found")
        payload = json.loads(raw)

        completed_at_raw = payload.get("completed_at")
        completed_at = (
            datetime.fromisoformat(completed_at_raw) if isinstance(completed_at_raw, str) else None
        )
        result_payload = payload.get("result")
        result_dict = result_payload if isinstance(result_payload, dict) else None

        return BacktestJobStatusResponse(
            job_id=uuid.UUID(str(payload["job_id"])),
            farm_id=uuid.UUID(str(payload["farm_id"])),
            status=str(payload.get("status") or "queued"),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            updated_at=datetime.fromisoformat(str(payload["updated_at"])),
            completed_at=completed_at,
            error=str(payload["error"]) if payload.get("error") is not None else None,
            result=result_dict,
        )

    async def get_active_alerts(self, farm_id: uuid.UUID) -> AlertsResponse:
        farm_service = FarmService(self.db)
        farm = await farm_service.get_farm(farm_id)
        await self._ensure_graph(farm_id)

        anomalies = julia_bridge.detect_anomalies(str(farm_id))
        nutrients = julia_bridge.nutrient_report(str(farm_id))

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
                    AlertItem(
                        source="vision",
                        severity=str(item.get("severity") or "warning"),
                        payload=item,
                    )
                )

        zone_payload = [
            ZoneAlerts(zone_id=zone_id, alerts=alerts) for zone_id, alerts in zone_alert_map.items()
        ]

        zone_payload.sort(
            key=lambda item: str(item.zone_id) if item.zone_id is not None else "zzzz"
        )

        return AlertsResponse(
            farm_id=farm_id,
            generated_at=datetime.now(UTC),
            zones=zone_payload,
        )

    async def _build_cross_layer(
        self,
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
                data = julia_bridge.cross_layer_query(str(farm_id), layer_a, layer_b)
                result.append(
                    CrossLayerLink(
                        layer_a=layer_a,
                        layer_b=layer_b,
                        data={"query_vertex_id": str(query_vertex_id), "result": data},
                    )
                )
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
        rows = await self.db.execute(
            select(Vertex.id, Vertex.zone_id).where(Vertex.farm_id == farm_id)
        )
        return {vertex_id: zone_id for vertex_id, zone_id in rows.all()}

    @staticmethod
    def _resolve_alert_zone(
        item: dict[str, Any], zone_index: dict[uuid.UUID, uuid.UUID | None]
    ) -> uuid.UUID | None:
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
