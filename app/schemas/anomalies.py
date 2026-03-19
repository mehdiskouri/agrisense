"""Pydantic schemas for anomaly history, thresholds, and webhooks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl

from app.models.enums import AnomalySeverityEnum, HyperEdgeLayerEnum, VertexTypeEnum


class AnomalyEventRead(BaseModel):
    id: uuid.UUID
    farm_id: uuid.UUID
    vertex_id: uuid.UUID | None = None
    zone_id: uuid.UUID | None = None
    layer: str
    anomaly_type: str
    severity: AnomalySeverityEnum
    feature: str | None = None
    current_value: float | None = None
    rolling_mean: float | None = None
    rolling_std: float | None = None
    sigma_deviation: float | None = None
    anomaly_rules: list[str] = Field(default_factory=list)
    cross_layer_confirmed: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime
    resolved_at: datetime | None = None
    webhook_notified: bool
    created_at: datetime
    updated_at: datetime


class AnomalyHistoryQuery(BaseModel):
    severity: AnomalySeverityEnum | None = None
    layer: HyperEdgeLayerEnum | None = None
    vertex_id: uuid.UUID | None = None
    anomaly_type: str | None = Field(default=None, max_length=128)
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class AnomalyHistoryResponse(BaseModel):
    farm_id: uuid.UUID
    total_count: int
    items: list[AnomalyEventRead] = Field(default_factory=list)
    filters_applied: dict[str, Any] = Field(default_factory=dict)


class ThresholdBase(BaseModel):
    vertex_type: VertexTypeEnum
    layer: HyperEdgeLayerEnum | None = None
    sigma1: float = 1.0
    sigma2: float = 2.0
    sigma3: float = 3.0
    min_history: int = Field(default=8, ge=1)
    min_nan_run_outage: int = Field(default=4, ge=1)
    vision_anomaly_score_threshold: float = Field(default=0.7, ge=0.0)
    suppress_rule3_only: bool = True
    enabled: bool = True


class ThresholdCreate(ThresholdBase):
    pass


class ThresholdUpdate(BaseModel):
    vertex_type: VertexTypeEnum | None = None
    layer: HyperEdgeLayerEnum | None = None
    sigma1: float | None = None
    sigma2: float | None = None
    sigma3: float | None = None
    min_history: int | None = Field(default=None, ge=1)
    min_nan_run_outage: int | None = Field(default=None, ge=1)
    vision_anomaly_score_threshold: float | None = Field(default=None, ge=0.0)
    suppress_rule3_only: bool | None = None
    enabled: bool | None = None


class ThresholdRead(ThresholdBase):
    id: uuid.UUID
    farm_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class ThresholdListResponse(BaseModel):
    farm_id: uuid.UUID
    items: list[ThresholdRead] = Field(default_factory=list)


class WebhookCreate(BaseModel):
    url: HttpUrl
    secret: str = Field(min_length=16, max_length=256)
    event_types: list[str] = Field(default_factory=lambda: ["anomaly.*"])
    is_active: bool = True
    retry_max: int = Field(default=3, ge=1, le=10)


class WebhookUpdate(BaseModel):
    url: HttpUrl | None = None
    secret: str | None = Field(default=None, min_length=16, max_length=256)
    event_types: list[str] | None = None
    is_active: bool | None = None
    retry_max: int | None = Field(default=None, ge=1, le=10)


class WebhookRead(BaseModel):
    id: uuid.UUID
    farm_id: uuid.UUID
    url: str
    secret: str
    event_types: list[str] = Field(default_factory=list)
    is_active: bool
    retry_max: int
    last_triggered_at: datetime | None = None
    last_status_code: int | None = None
    failure_count: int
    created_at: datetime
    updated_at: datetime


class WebhookListResponse(BaseModel):
    farm_id: uuid.UUID
    items: list[WebhookRead] = Field(default_factory=list)


class WebhookTestResponse(BaseModel):
    webhook_id: uuid.UUID
    delivered: bool
    status_code: int | None = None
    detail: str
