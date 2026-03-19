"""Anomaly threshold management, persistence, and history querying."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.anomalies import AnomalyEvent, AnomalyThreshold
from app.models.enums import AnomalySeverityEnum, HyperEdgeLayerEnum, VertexTypeEnum
from app.models.farm import Vertex
from app.schemas.anomalies import (
    AnomalyEventRead,
    AnomalyHistoryQuery,
    AnomalyHistoryResponse,
    ThresholdCreate,
    ThresholdListResponse,
    ThresholdRead,
    ThresholdUpdate,
)
from app.services import julia_bridge
from app.services.farm_service import FarmService
from app.services.webhook_service import WebhookService

logger = logging.getLogger("agrisense.anomaly")


class AnomalyService:
    """Coordinates anomaly filtering, persistence, and threshold configuration."""

    def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
        self.db = db
        self.redis_client = redis_client

    async def get_thresholds(self, farm_id: uuid.UUID) -> ThresholdListResponse:
        await FarmService(self.db).get_farm(farm_id)
        rows = await self.db.execute(
            select(AnomalyThreshold)
            .where(AnomalyThreshold.farm_id == farm_id)
            .order_by(
                AnomalyThreshold.vertex_type.asc(),
                AnomalyThreshold.layer.asc().nullsfirst(),
            )
        )
        items = [self._to_threshold_read(item) for item in rows.scalars().all()]
        return ThresholdListResponse(farm_id=farm_id, items=items)

    async def create_threshold(self, farm_id: uuid.UUID, payload: ThresholdCreate) -> ThresholdRead:
        await FarmService(self.db).get_farm(farm_id)
        threshold = AnomalyThreshold(
            farm_id=farm_id,
            vertex_type=payload.vertex_type,
            layer=payload.layer,
            sigma1=payload.sigma1,
            sigma2=payload.sigma2,
            sigma3=payload.sigma3,
            min_history=payload.min_history,
            min_nan_run_outage=payload.min_nan_run_outage,
            vision_anomaly_score_threshold=payload.vision_anomaly_score_threshold,
            suppress_rule3_only=payload.suppress_rule3_only,
            enabled=payload.enabled,
        )
        self.db.add(threshold)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise ValueError("threshold already exists for farm/vertex_type/layer") from exc
        await self.db.refresh(threshold)
        return self._to_threshold_read(threshold)

    async def update_threshold(
        self,
        farm_id: uuid.UUID,
        threshold_id: uuid.UUID,
        payload: ThresholdUpdate,
    ) -> ThresholdRead:
        threshold = await self._require_threshold(farm_id, threshold_id)
        updates = payload.model_dump(exclude_none=True)
        for key, value in updates.items():
            setattr(threshold, key, value)

        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise ValueError("threshold update conflicts with existing configuration") from exc

        await self.db.refresh(threshold)
        return self._to_threshold_read(threshold)

    async def delete_threshold(self, farm_id: uuid.UUID, threshold_id: uuid.UUID) -> None:
        threshold = await self._require_threshold(farm_id, threshold_id)
        await self.db.delete(threshold)
        await self.db.flush()

    async def detect_and_persist(self, farm_id: uuid.UUID) -> list[AnomalyEvent]:
        await FarmService(self.db).get_farm(farm_id)

        thresholds = await self._effective_thresholds(farm_id)
        julia_thresholds = await self._build_julia_threshold_payload(farm_id, thresholds)
        if julia_thresholds:
            raw_items = julia_bridge.detect_anomalies(
                str(farm_id),
                thresholds=julia_thresholds,
            )
        else:
            raw_items = julia_bridge.detect_anomalies(str(farm_id))
        vertex_map = await self._vertex_map(farm_id, raw_items)
        events: list[AnomalyEvent] = []

        for raw in raw_items:
            filtered = self._normalize_and_filter(raw, vertex_map, thresholds)
            if filtered is None:
                continue
            events.append(
                AnomalyEvent(
                    farm_id=farm_id,
                    vertex_id=filtered["vertex_id"],
                    zone_id=filtered["zone_id"],
                    layer=filtered["layer"],
                    anomaly_type=filtered["anomaly_type"],
                    severity=filtered["severity"],
                    feature=filtered["feature"],
                    current_value=filtered["current_value"],
                    rolling_mean=filtered["rolling_mean"],
                    rolling_std=filtered["rolling_std"],
                    sigma_deviation=filtered["sigma_deviation"],
                    anomaly_rules=filtered["anomaly_rules"],
                    cross_layer_confirmed=filtered["cross_layer_confirmed"],
                    payload=filtered["payload"],
                    detected_at=filtered["detected_at"],
                )
            )

        if events:
            self.db.add_all(events)
            await self.db.flush()

        await self._reconcile_resolved_events(farm_id, events)

        if not events:
            return []

        event_ids = [event.id for event in events]
        dispatch_task = asyncio.create_task(self._dispatch_webhooks_background(farm_id, event_ids))
        dispatch_task.add_done_callback(lambda _task: None)
        return events

    async def _reconcile_resolved_events(
        self,
        farm_id: uuid.UUID,
        active_events: list[AnomalyEvent],
    ) -> None:
        # Resolution key follows the design note: vertex + layer + feature.
        active_latest_by_key: dict[
            tuple[uuid.UUID | None, str, str | None],
            tuple[uuid.UUID, datetime],
        ] = {}
        for event in active_events:
            key = (event.vertex_id, event.layer, event.feature)
            previous = active_latest_by_key.get(key)
            if previous is None:
                active_latest_by_key[key] = (event.id, event.detected_at)
                continue
            if event.detected_at >= previous[1]:
                active_latest_by_key[key] = (event.id, event.detected_at)

        rows = await self.db.execute(
            select(AnomalyEvent).where(
                AnomalyEvent.farm_id == farm_id,
                AnomalyEvent.resolved_at.is_(None),
            )
        )
        unresolved = list(rows.scalars().all())
        if not unresolved:
            return

        now = datetime.now(UTC)
        for event in unresolved:
            key = (event.vertex_id, event.layer, event.feature)
            active = active_latest_by_key.get(key)
            if active is None:
                event.resolved_at = now
                continue
            if event.id != active[0]:
                event.resolved_at = now

        await self.db.flush()

    async def get_history(
        self,
        farm_id: uuid.UUID,
        query: AnomalyHistoryQuery,
    ) -> AnomalyHistoryResponse:
        await FarmService(self.db).get_farm(farm_id)

        filters = [AnomalyEvent.farm_id == farm_id]
        if query.severity is not None:
            filters.append(AnomalyEvent.severity == query.severity)
        if query.layer is not None:
            filters.append(AnomalyEvent.layer == query.layer.value)
        if query.vertex_id is not None:
            filters.append(AnomalyEvent.vertex_id == query.vertex_id)
        if query.anomaly_type is not None:
            filters.append(AnomalyEvent.anomaly_type == query.anomaly_type)
        if query.since is not None:
            filters.append(AnomalyEvent.detected_at >= query.since)
        if query.until is not None:
            filters.append(AnomalyEvent.detected_at <= query.until)

        where_clause = and_(*filters)
        count_stmt = select(func.count()).select_from(AnomalyEvent).where(where_clause)
        total_count = int((await self.db.execute(count_stmt)).scalar_one())

        rows = await self.db.execute(
            select(AnomalyEvent)
            .where(where_clause)
            .order_by(AnomalyEvent.detected_at.desc())
            .offset(query.offset)
            .limit(query.limit)
        )
        items = [self._to_anomaly_read(item) for item in rows.scalars().all()]

        filters_applied = {
            key: value for key, value in query.model_dump(mode="json").items() if value is not None
        }
        return AnomalyHistoryResponse(
            farm_id=farm_id,
            total_count=total_count,
            items=items,
            filters_applied=filters_applied,
        )

    async def _effective_thresholds(
        self,
        farm_id: uuid.UUID,
    ) -> dict[tuple[VertexTypeEnum, HyperEdgeLayerEnum | None], AnomalyThreshold]:
        rows = await self.db.execute(
            select(AnomalyThreshold).where(AnomalyThreshold.farm_id == farm_id)
        )
        result: dict[tuple[VertexTypeEnum, HyperEdgeLayerEnum | None], AnomalyThreshold] = {}
        for threshold in rows.scalars().all():
            key = (threshold.vertex_type, threshold.layer)
            result[key] = threshold
        return result

    async def _build_julia_threshold_payload(
        self,
        farm_id: uuid.UUID,
        thresholds: dict[tuple[VertexTypeEnum, HyperEdgeLayerEnum | None], AnomalyThreshold],
    ) -> dict[str, Any]:
        if not thresholds:
            return {}

        payload: dict[str, Any] = {
            "default": {
                "sigma1": 1.0,
                "sigma2": 2.0,
                "sigma3": 3.0,
                "min_history": 8,
                "min_nan_run_outage": 4,
                "vision_anomaly_score_threshold": 0.7,
                "suppress_rule3_only": True,
                "enabled": True,
            },
            "by_vertex_layer": {},
            "by_layer": {},
        }

        rows = await self.db.execute(
            select(Vertex.id, Vertex.vertex_type).where(Vertex.farm_id == farm_id)
        )
        vertex_rows = list(rows.all())

        for vertex_id, vertex_type in vertex_rows:
            wildcard_cfg = thresholds.get((vertex_type, None))
            if wildcard_cfg is not None:
                payload["by_vertex_layer"][f"{vertex_id}|*"] = self._threshold_payload(wildcard_cfg)

            for layer in HyperEdgeLayerEnum:
                layer_cfg = thresholds.get((vertex_type, layer))
                if layer_cfg is None:
                    continue
                payload["by_vertex_layer"][f"{vertex_id}|{layer.value}"] = self._threshold_payload(
                    layer_cfg
                )

        return payload

    def _normalize_and_filter(
        self,
        item: dict[str, Any],
        vertex_map: dict[uuid.UUID, tuple[VertexTypeEnum, uuid.UUID | None]],
        thresholds: dict[tuple[VertexTypeEnum, HyperEdgeLayerEnum | None], AnomalyThreshold],
    ) -> dict[str, Any] | None:
        vertex_id = self._parse_uuid(item.get("vertex_id"))
        layer_token = str(item.get("layer") or "unknown")
        layer_enum = self._to_layer_enum(layer_token)
        anomaly_rules = self._extract_rules(item)

        zone_id = self._parse_uuid(item.get("zone_id"))
        vertex_type: VertexTypeEnum | None = None
        if vertex_id is not None and vertex_id in vertex_map:
            vertex_type, resolved_zone = vertex_map[vertex_id]
            if zone_id is None:
                zone_id = resolved_zone

        threshold = None
        if vertex_type is not None:
            threshold = thresholds.get((vertex_type, layer_enum)) or thresholds.get(
                (vertex_type, None)
            )

        if threshold is not None and not threshold.enabled:
            return None

        sigma_deviation = self._to_float(
            item.get("sigma_deviation")
            or item.get("sigma")
            or item.get("z_score")
            or item.get("deviation_sigma")
        )
        history_count = self._to_int(item.get("history_count") or item.get("n_history"))
        outage_run = self._to_int(item.get("nan_run") or item.get("nan_run_length"))
        anomaly_score = self._to_float(item.get("anomaly_score") or item.get("score"))

        if threshold is not None:
            if history_count is not None and history_count < threshold.min_history:
                return None
            if (
                layer_token == "vision"
                and anomaly_score is not None
                and anomaly_score < threshold.vision_anomaly_score_threshold
            ):
                return None
            if threshold.suppress_rule3_only and self._is_rule3_only(anomaly_rules):
                return None
            if sigma_deviation is not None and sigma_deviation < threshold.sigma1:
                return None

        severity = self._classify_severity(item, threshold, sigma_deviation, outage_run)

        anomaly_type = str(item.get("anomaly_type") or item.get("type") or "unknown")
        feature = item.get("feature")
        detected_at = self._parse_datetime(item.get("detected_at") or item.get("timestamp"))

        return {
            "vertex_id": vertex_id,
            "zone_id": zone_id,
            "layer": layer_token,
            "anomaly_type": anomaly_type,
            "severity": severity,
            "feature": str(feature) if feature is not None else None,
            "current_value": self._to_float(item.get("current_value") or item.get("value")),
            "rolling_mean": self._to_float(item.get("rolling_mean") or item.get("mean")),
            "rolling_std": self._to_float(item.get("rolling_std") or item.get("std")),
            "sigma_deviation": sigma_deviation,
            "anomaly_rules": anomaly_rules,
            "cross_layer_confirmed": bool(item.get("cross_layer_confirmed") or False),
            "payload": item,
            "detected_at": detected_at,
        }

    def _classify_severity(
        self,
        item: dict[str, Any],
        threshold: AnomalyThreshold | None,
        sigma_deviation: float | None,
        outage_run: int | None,
    ) -> AnomalySeverityEnum:
        raw_severity = str(item.get("severity") or item.get("urgency") or "").lower()
        cross_layer_confirmed = bool(item.get("cross_layer_confirmed") or False)

        if threshold is not None and sigma_deviation is not None:
            if sigma_deviation >= threshold.sigma3:
                return AnomalySeverityEnum.critical
            if sigma_deviation >= threshold.sigma2:
                return AnomalySeverityEnum.warning
            return AnomalySeverityEnum.info

        if raw_severity == "alarm":
            if cross_layer_confirmed:
                return AnomalySeverityEnum.critical
            if (
                threshold is not None
                and outage_run is not None
                and outage_run >= threshold.min_nan_run_outage * 2
            ):
                return AnomalySeverityEnum.critical
            return AnomalySeverityEnum.critical

        if raw_severity == "warning":
            return AnomalySeverityEnum.warning

        anomaly_type = str(item.get("anomaly_type") or "").lower()
        if anomaly_type in {"nutrient_deficiency", "low_signal"}:
            return AnomalySeverityEnum.info

        return AnomalySeverityEnum.info

    async def _vertex_map(
        self,
        farm_id: uuid.UUID,
        items: list[dict[str, Any]],
    ) -> dict[uuid.UUID, tuple[VertexTypeEnum, uuid.UUID | None]]:
        vertex_ids: set[uuid.UUID] = set()
        for item in items:
            parsed = self._parse_uuid(item.get("vertex_id"))
            if parsed is not None:
                vertex_ids.add(parsed)

        if not vertex_ids:
            return {}

        rows = await self.db.execute(
            select(Vertex.id, Vertex.vertex_type, Vertex.zone_id).where(
                Vertex.farm_id == farm_id,
                Vertex.id.in_(vertex_ids),
            )
        )
        return {vertex_id: (vertex_type, zone_id) for vertex_id, vertex_type, zone_id in rows.all()}

    async def _require_threshold(
        self,
        farm_id: uuid.UUID,
        threshold_id: uuid.UUID,
    ) -> AnomalyThreshold:
        await FarmService(self.db).get_farm(farm_id)
        rows = await self.db.execute(
            select(AnomalyThreshold).where(AnomalyThreshold.id == threshold_id)
        )
        threshold = rows.scalar_one_or_none()
        if threshold is None or threshold.farm_id != farm_id:
            raise LookupError(f"Threshold {threshold_id} not found")
        return threshold

    @staticmethod
    def _parse_uuid(value: Any) -> uuid.UUID | None:
        if value is None:
            return None
        try:
            return uuid.UUID(str(value))
        except ValueError:
            return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
            except ValueError:
                pass
        return datetime.now(UTC)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_layer_enum(value: str) -> HyperEdgeLayerEnum | None:
        try:
            return HyperEdgeLayerEnum(value)
        except ValueError:
            return None

    @staticmethod
    def _extract_rules(item: dict[str, Any]) -> list[str]:
        raw = item.get("anomaly_rules")
        if raw is None:
            raw = item.get("rules")
        if not isinstance(raw, list):
            return []
        return [str(value) for value in raw]

    @staticmethod
    def _is_rule3_only(rules: list[str]) -> bool:
        if len(rules) != 1:
            return False
        token = rules[0].lower()
        return token in {"rule3", "rule_3", "3sigma", "rule_3_sigma"}

    @staticmethod
    def _threshold_payload(threshold: AnomalyThreshold) -> dict[str, Any]:
        return {
            "sigma1": float(threshold.sigma1),
            "sigma2": float(threshold.sigma2),
            "sigma3": float(threshold.sigma3),
            "min_history": int(threshold.min_history),
            "min_nan_run_outage": int(threshold.min_nan_run_outage),
            "vision_anomaly_score_threshold": float(threshold.vision_anomaly_score_threshold),
            "suppress_rule3_only": bool(threshold.suppress_rule3_only),
            "enabled": bool(threshold.enabled),
        }

    @staticmethod
    def _to_threshold_read(threshold: AnomalyThreshold) -> ThresholdRead:
        return ThresholdRead(
            id=threshold.id,
            farm_id=threshold.farm_id,
            vertex_type=threshold.vertex_type,
            layer=threshold.layer,
            sigma1=threshold.sigma1,
            sigma2=threshold.sigma2,
            sigma3=threshold.sigma3,
            min_history=threshold.min_history,
            min_nan_run_outage=threshold.min_nan_run_outage,
            vision_anomaly_score_threshold=threshold.vision_anomaly_score_threshold,
            suppress_rule3_only=threshold.suppress_rule3_only,
            enabled=threshold.enabled,
            created_at=threshold.created_at,
            updated_at=threshold.updated_at,
        )

    @staticmethod
    def _to_anomaly_read(event: AnomalyEvent) -> AnomalyEventRead:
        return AnomalyEventRead(
            id=event.id,
            farm_id=event.farm_id,
            vertex_id=event.vertex_id,
            zone_id=event.zone_id,
            layer=event.layer,
            anomaly_type=event.anomaly_type,
            severity=event.severity,
            feature=event.feature,
            current_value=event.current_value,
            rolling_mean=event.rolling_mean,
            rolling_std=event.rolling_std,
            sigma_deviation=event.sigma_deviation,
            anomaly_rules=list(event.anomaly_rules),
            cross_layer_confirmed=event.cross_layer_confirmed,
            payload=dict(event.payload),
            detected_at=event.detected_at,
            resolved_at=event.resolved_at,
            webhook_notified=event.webhook_notified,
            created_at=event.created_at,
            updated_at=event.updated_at,
        )

    async def _dispatch_webhooks_background(
        self,
        farm_id: uuid.UUID,
        event_ids: list[uuid.UUID],
    ) -> None:
        if not event_ids:
            return

        try:
            async with async_session_factory() as session:
                webhook_service = WebhookService(session, self.redis_client)

                # Queue-backed dispatch is preferred when Redis is available.
                if self.redis_client is not None:
                    queued = await webhook_service.enqueue_dispatch_job(farm_id, event_ids)
                    if queued:
                        await session.commit()
                        return

                rows = await session.execute(
                    select(AnomalyEvent).where(
                        and_(
                            AnomalyEvent.farm_id == farm_id,
                            AnomalyEvent.id.in_(event_ids),
                        )
                    )
                )
                events = list(rows.scalars().all())
                await webhook_service.dispatch_anomaly_events(farm_id, events)
                await session.commit()
        except Exception as exc:
            logger.warning(
                "anomaly webhook dispatch failed",
                extra={"farm_id": str(farm_id), "error": str(exc)},
            )
