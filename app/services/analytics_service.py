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

from app.models.enums import FarmTypeEnum, JobStatusEnum
from app.models.farm import Vertex
from app.models.jobs import BacktestJob
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
from app.schemas.anomalies import AnomalyHistoryQuery, AnomalyHistoryResponse
from app.schemas.farm import (
    CrossLayerSummary,
    VisualizationAlertItem,
    VisualizationLayerMeta,
    VisualizationLink,
    VisualizationNode,
    VisualizationResponse,
)
from app.services import julia_bridge
from app.services.anomaly_service import AnomalyService
from app.services.farm_service import FarmService

LAYER_VIS_META: dict[str, dict[str, Any]] = {
    "soil": {
        "color": "#8B4513",
        "feature_names": ["moisture", "temperature", "ph", "conductivity"],
    },
    "irrigation": {
        "color": "#1E90FF",
        "feature_names": ["flow_rate", "valve_state", "pressure"],
    },
    "lighting": {
        "color": "#FFD700",
        "feature_names": ["intensity", "spectrum", "duration"],
    },
    "weather": {
        "color": "#87CEEB",
        "feature_names": [
            "temperature",
            "humidity",
            "wind_speed",
            "precipitation",
            "solar_radiation",
        ],
    },
    "crop_requirements": {
        "color": "#228B22",
        "feature_names": ["water_need", "npk_n", "npk_p", "npk_k", "growth_stage"],
    },
    "npk": {
        "color": "#FF6347",
        "feature_names": ["nitrogen", "phosphorus", "potassium"],
    },
    "vision": {
        "color": "#9370DB",
        "feature_names": ["health_index", "canopy_cover", "stress_score", "growth_rate"],
    },
}


class AnalyticsService:
    BACKTEST_JOB_TTL_SECONDS = 60 * 60 * 24

    def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
        self.db = db
        self.redis_client = redis_client

    @staticmethod
    def _backtest_job_key(job_id: uuid.UUID) -> str:
        return f"analytics:yield_backtest:job:{job_id}"

    @staticmethod
    def _serialize_backtest_job(job: BacktestJob) -> dict[str, Any]:
        return {
            "job_id": str(job.id),
            "farm_id": str(job.farm_id),
            "n_folds": int(job.n_folds),
            "min_history": int(job.min_history),
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at is not None else None,
            "error": job.error,
            "result": job.result,
        }

    @staticmethod
    def _status_from_payload(payload: dict[str, Any]) -> BacktestJobStatusResponse:
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

    async def _persist_backtest_cache(self, job: BacktestJob) -> None:
        if self.redis_client is None:
            return
        await self.redis_client.setex(
            self._backtest_job_key(job.id),
            self.BACKTEST_JOB_TTL_SECONDS,
            json.dumps(self._serialize_backtest_job(job)),
        )

    async def _read_backtest_cache(self, job_id: uuid.UUID) -> BacktestJobStatusResponse | None:
        if self.redis_client is None:
            return None
        raw = await self.redis_client.get(self._backtest_job_key(job_id))
        if raw is None:
            return None
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        return self._status_from_payload(payload)

    async def _require_backtest_job(self, job_id: uuid.UUID) -> BacktestJob:
        row = await self.db.execute(select(BacktestJob).where(BacktestJob.id == job_id))
        job = row.scalar_one_or_none()
        if job is None:
            raise LookupError(f"Backtest job {job_id} not found")
        return job

    @staticmethod
    def _to_backtest_status(job: BacktestJob) -> BacktestJobStatusResponse:
        return BacktestJobStatusResponse(
            job_id=job.id,
            farm_id=job.farm_id,
            status=job.status.value,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            error=job.error,
            result=job.result,
        )

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

        job = BacktestJob(
            farm_id=farm_id,
            status=JobStatusEnum.queued,
            n_folds=int(n_folds),
            min_history=int(min_history),
            error=None,
            result=None,
        )
        self.db.add(job)
        await self.db.flush()
        await self.db.refresh(job)
        await self._persist_backtest_cache(job)

        return BacktestJobCreateResponse(
            job_id=job.id,
            farm_id=farm_id,
            status="queued",
            created_at=job.created_at,
        )

    async def execute_yield_backtest_job(self, job_id: uuid.UUID) -> BacktestJobStatusResponse:
        job = await self._require_backtest_job(job_id)
        job.status = JobStatusEnum.running
        job.error = None
        await self.db.flush()
        await self._persist_backtest_cache(job)

        try:
            result = await self.run_yield_backtest(job.farm_id, n_folds=job.n_folds)
            job.status = JobStatusEnum.succeeded
            job.result = result.model_dump(mode="json")
            job.error = None
        except Exception as exc:
            job.status = JobStatusEnum.failed
            job.error = str(exc)
            job.result = None

        job.completed_at = datetime.now(UTC)
        await self.db.flush()
        await self._persist_backtest_cache(job)
        return self._to_backtest_status(job)

    async def get_yield_backtest_job_status(self, job_id: uuid.UUID) -> BacktestJobStatusResponse:
        cached = await self._read_backtest_cache(job_id)
        if cached is not None:
            return cached

        job = await self._require_backtest_job(job_id)
        await self._persist_backtest_cache(job)
        return self._to_backtest_status(job)

    async def get_active_alerts(self, farm_id: uuid.UUID) -> AlertsResponse:
        farm_service = FarmService(self.db)
        farm = await farm_service.get_farm(farm_id)
        await self._ensure_graph(farm_id)

        anomaly_service = AnomalyService(self.db, self.redis_client)
        persisted_anomalies = await anomaly_service.detect_and_persist(farm_id)
        nutrients = julia_bridge.nutrient_report(str(farm_id))

        zone_index = await self._zone_vertex_index(farm_id)
        zone_alert_map: dict[uuid.UUID | None, list[AlertItem]] = defaultdict(list)

        for event in persisted_anomalies:
            item = event.payload if isinstance(event.payload, dict) else {}
            zone_key = self._resolve_alert_zone(item, zone_index)
            severity = event.severity.value
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
            for event in persisted_anomalies:
                item = event.payload if isinstance(event.payload, dict) else {}
                layer = str(item.get("layer") or "")
                if layer != "vision":
                    continue
                zone_key = self._resolve_alert_zone(item, zone_index)
                zone_alert_map[zone_key].append(
                    AlertItem(
                        source="vision",
                        severity=event.severity.value,
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

    async def get_anomaly_history(
        self,
        farm_id: uuid.UUID,
        query: AnomalyHistoryQuery,
    ) -> AnomalyHistoryResponse:
        service = AnomalyService(self.db, self.redis_client)
        return await service.get_history(farm_id, query)

    async def get_visualization(self, farm_id: uuid.UUID) -> VisualizationResponse:
        farm_service = FarmService(self.db)
        farm = await farm_service.get_farm(farm_id)
        graph_state = await farm_service.get_graph(farm_id)

        vertex_rows = {str(vertex.id): vertex for vertex in farm.vertices}
        zone_names = {str(zone.id): zone.name for zone in farm.zones}
        layer_memberships: dict[str, set[str]] = defaultdict(set)
        vertex_features: dict[str, dict[str, list[float]]] = defaultdict(dict)
        layer_vertex_presence: dict[str, set[str]] = defaultdict(set)

        links: list[VisualizationLink] = []
        nodes: list[VisualizationNode] = []
        layer_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n_edges": 0, "n_vertices": 0}
        )

        layers_payload = graph_state.get("layers")
        layers_data = layers_payload if isinstance(layers_payload, dict) else {}

        for layer_name, layer_raw in layers_data.items():
            layer = layer_raw if isinstance(layer_raw, dict) else {}
            edge_ids = [str(item) for item in (layer.get("edge_ids") or [])]
            edge_meta = layer.get("edge_metadata") or []
            incidence_rows = [int(item) for item in (layer.get("incidence_rows") or [])]
            incidence_cols = [int(item) for item in (layer.get("incidence_cols") or [])]
            vertex_ids = [str(item) for item in (layer.get("vertex_ids") or [])]
            feature_matrix = layer.get("vertex_features") or []

            row_to_vertex_id = {idx + 1: vertex_id for idx, vertex_id in enumerate(vertex_ids)}

            # Build per-layer current feature vectors for each vertex.
            for row_idx, row in enumerate(feature_matrix):
                if not isinstance(row, list):
                    continue
                vertex_id = row_to_vertex_id.get(row_idx + 1)
                if vertex_id is None:
                    continue
                vector = [float(value) for value in row if isinstance(value, (int, float))]
                if vector:
                    vertex_features[vertex_id][str(layer_name)] = vector

            edge_members: dict[int, set[str]] = defaultdict(set)
            for inc_row, inc_col in zip(incidence_rows, incidence_cols, strict=False):
                vertex_id = row_to_vertex_id.get(inc_row)
                if vertex_id is None:
                    continue
                edge_members[inc_col].add(vertex_id)
                layer_memberships[vertex_id].add(str(layer_name))
                layer_vertex_presence[str(layer_name)].add(vertex_id)

            for edge_index, edge_id in enumerate(edge_ids, start=1):
                hub_id = f"he:{layer_name}:{edge_id}"
                metadata = (
                    edge_meta[edge_index - 1]
                    if edge_index - 1 < len(edge_meta)
                    and isinstance(edge_meta[edge_index - 1], dict)
                    else {}
                )
                members = sorted(edge_members.get(edge_index, set()))
                nodes.append(
                    VisualizationNode(
                        id=hub_id,
                        type="hyperedge",
                        layer=str(layer_name),
                        metadata=metadata,
                        member_count=len(members),
                        layer_memberships=[str(layer_name)],
                    )
                )
                for vertex_id in members:
                    links.append(
                        VisualizationLink(source=hub_id, target=vertex_id, layer=str(layer_name))
                    )

            layer_stats[str(layer_name)] = {
                "n_edges": len(edge_ids),
                "n_vertices": len(layer_vertex_presence[str(layer_name)]),
            }

        known_vertex_ids = set(vertex_rows.keys()) | set(vertex_features.keys())
        for vertex_id in sorted(known_vertex_ids):
            vertex = vertex_rows.get(vertex_id)
            zone_id = (
                str(vertex.zone_id) if vertex is not None and vertex.zone_id is not None else None
            )
            nodes.append(
                VisualizationNode(
                    id=vertex_id,
                    type="vertex",
                    vertex_type=(vertex.vertex_type.value if vertex is not None else None),
                    zone_id=uuid.UUID(zone_id) if zone_id is not None else None,
                    zone_name=zone_names.get(zone_id) if zone_id is not None else None,
                    features=vertex_features.get(vertex_id, {}),
                    layer_memberships=sorted(layer_memberships.get(vertex_id, set())),
                )
            )

        layer_metas: list[VisualizationLayerMeta] = []
        for layer_name in sorted(layers_data.keys()):
            meta = LAYER_VIS_META.get(str(layer_name), {"color": "#6B7280", "feature_names": []})
            stats = layer_stats.get(str(layer_name), {"n_edges": 0, "n_vertices": 0})
            layer_metas.append(
                VisualizationLayerMeta(
                    name=str(layer_name),
                    color=str(meta["color"]),
                    feature_names=[str(item) for item in meta.get("feature_names", [])],
                    n_edges=int(stats["n_edges"]),
                    n_vertices=int(stats["n_vertices"]),
                )
            )

        alerts_response = await self.get_active_alerts(farm_id)
        flat_alerts: list[VisualizationAlertItem] = []
        for zone_alerts in alerts_response.zones:
            for alert in zone_alerts.alerts:
                payload = alert.payload if isinstance(alert.payload, dict) else {}
                vertex_token = payload.get("vertex_id")
                flat_alerts.append(
                    VisualizationAlertItem(
                        vertex_id=str(vertex_token) if vertex_token is not None else None,
                        source=alert.source,
                        severity=alert.severity,
                        payload=payload,
                    )
                )

        cross_layer: list[CrossLayerSummary] = []
        ordered_layers = sorted(layer_vertex_presence.keys())
        for index, layer_a in enumerate(ordered_layers):
            for layer_b in ordered_layers[index + 1 :]:
                shared = len(layer_vertex_presence[layer_a] & layer_vertex_presence[layer_b])
                cross_layer.append(
                    CrossLayerSummary(
                        layer_a=layer_a,
                        layer_b=layer_b,
                        shared_vertices=shared,
                    )
                )

        return VisualizationResponse(
            farm_id=farm.id,
            farm_name=farm.name,
            farm_type=farm.farm_type,
            generated_at=datetime.now(UTC),
            layers=layer_metas,
            nodes=nodes,
            links=links,
            alerts=flat_alerts,
            cross_layer_summary=cross_layer,
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
