"""Anomaly detection persistence and webhook subscription models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import AnomalySeverityEnum, HyperEdgeLayerEnum, VertexTypeEnum


class AnomalyEvent(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Durable anomaly history records emitted by analytics detection runs."""

    __tablename__ = "anomaly_events"
    __table_args__ = (
        Index(
            "ix_anomaly_events_farm_detected_at",
            "farm_id",
            text("detected_at DESC"),
        ),
        Index("ix_anomaly_events_farm_severity", "farm_id", "severity"),
        Index("ix_anomaly_events_vertex_layer", "vertex_id", "layer"),
    )

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    vertex_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vertices.id", ondelete="SET NULL"),
        nullable=True,
    )
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("zones.id", ondelete="SET NULL"),
        nullable=True,
    )
    layer: Mapped[str] = mapped_column(String(64), nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(128), nullable=False)
    severity: Mapped[AnomalySeverityEnum] = mapped_column(
        Enum(
            AnomalySeverityEnum,
            name="anomaly_severity",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )
    feature: Mapped[str | None] = mapped_column(String(128), nullable=True)
    current_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    rolling_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    sigma_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    anomaly_rules: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    cross_layer_confirmed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    webhook_notified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )


class AnomalyThreshold(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Per-farm sensitivity configuration for anomaly detection post-filtering."""

    __tablename__ = "anomaly_thresholds"
    __table_args__ = (
        UniqueConstraint(
            "farm_id",
            "vertex_type",
            "layer",
            name="uq_anomaly_thresholds_farm_vertex_layer",
        ),
    )

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    vertex_type: Mapped[VertexTypeEnum] = mapped_column(
        Enum(
            VertexTypeEnum,
            name="vertex_type",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )
    layer: Mapped[HyperEdgeLayerEnum | None] = mapped_column(
        Enum(
            HyperEdgeLayerEnum,
            name="hyperedge_layer",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=True,
    )
    sigma1: Mapped[float] = mapped_column(Float, nullable=False, default=1.0, server_default="1")
    sigma2: Mapped[float] = mapped_column(Float, nullable=False, default=2.0, server_default="2")
    sigma3: Mapped[float] = mapped_column(Float, nullable=False, default=3.0, server_default="3")
    min_history: Mapped[int] = mapped_column(Integer, nullable=False, default=8, server_default="8")
    min_nan_run_outage: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=4,
        server_default="4",
    )
    vision_anomaly_score_threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.7,
        server_default="0.7",
    )
    suppress_rule3_only: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )


class WebhookSubscription(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Webhook endpoints subscribed to farm anomaly events."""

    __tablename__ = "webhook_subscriptions"
    __table_args__ = (Index("ix_webhook_subscriptions_farm_active", "farm_id", "is_active"),)

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    secret: Mapped[str] = mapped_column(String(256), nullable=False)
    event_types: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    retry_max: Mapped[int] = mapped_column(Integer, nullable=False, default=3, server_default="3")
    last_triggered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
