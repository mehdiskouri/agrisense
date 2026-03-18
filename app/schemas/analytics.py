"""Pydantic schemas for analytics endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ZoneStatus(BaseModel):
    zone_id: uuid.UUID
    query_vertex_id: uuid.UUID
    status: dict[str, Any] = Field(default_factory=dict)


class FarmStatusResponse(BaseModel):
    farm_id: uuid.UUID
    generated_at: datetime
    zones: list[ZoneStatus] = Field(default_factory=list)


class ZoneDetailQuery(BaseModel):
    zone_id: uuid.UUID | None = None
    vertex_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _validate_identifier(self) -> ZoneDetailQuery:
        if self.zone_id is None and self.vertex_id is None:
            raise ValueError("provide either zone_id or vertex_id")
        return self


class CrossLayerLink(BaseModel):
    layer_a: str
    layer_b: str
    data: dict[str, Any] = Field(default_factory=dict)


class ZoneDetailResponse(BaseModel):
    farm_id: uuid.UUID
    zone_id: uuid.UUID | None = None
    query_vertex_id: uuid.UUID
    layers: dict[str, Any] = Field(default_factory=dict)
    cross_layer: list[CrossLayerLink] = Field(default_factory=list)


class IrrigationScheduleResponse(BaseModel):
    farm_id: uuid.UUID
    horizon_days: int
    cached: bool
    generated_at: datetime
    items: list[dict[str, Any]] = Field(default_factory=list)


class NutrientReportResponse(BaseModel):
    farm_id: uuid.UUID
    generated_at: datetime
    items: list[dict[str, Any]] = Field(default_factory=list)


class YieldForecastResponse(BaseModel):
    farm_id: uuid.UUID
    generated_at: datetime
    items: list[dict[str, Any]] = Field(default_factory=list)


class EnsembleMember(BaseModel):
    model_name: str
    yield_estimate: float
    lower: float
    upper: float
    weight: float


class EnsembleYieldItem(BaseModel):
    crop_bed_id: str
    yield_estimate_kg_m2: float
    yield_lower: float
    yield_upper: float
    confidence: float
    stress_factors: dict[str, Any] = Field(default_factory=dict)
    model_layer: str
    ensemble_weights: dict[str, float] = Field(default_factory=dict)
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    ensemble_members: list[EnsembleMember] = Field(default_factory=list)


class EnsembleYieldForecastResponse(BaseModel):
    farm_id: uuid.UUID
    generated_at: datetime
    include_members: bool
    ensemble_weights: dict[str, float] = Field(default_factory=dict)
    items: list[EnsembleYieldItem] = Field(default_factory=list)


class BacktestResponse(BaseModel):
    farm_id: uuid.UUID
    generated_at: datetime
    n_folds: int
    status: str
    per_fold_metrics: list[dict[str, Any]] = Field(default_factory=list)
    aggregate_metrics: dict[str, Any] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    temporal_split: dict[str, dict[str, int]] = Field(default_factory=dict)
    oracle_provenance: dict[str, Any] = Field(default_factory=dict)


class BacktestJobCreateResponse(BaseModel):
    job_id: uuid.UUID
    farm_id: uuid.UUID
    status: str
    created_at: datetime


class BacktestJobStatusResponse(BaseModel):
    job_id: uuid.UUID
    farm_id: uuid.UUID
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class AlertItem(BaseModel):
    source: str
    severity: str = "info"
    payload: dict[str, Any] = Field(default_factory=dict)


class ZoneAlerts(BaseModel):
    zone_id: uuid.UUID | None = None
    alerts: list[AlertItem] = Field(default_factory=list)


class AlertsResponse(BaseModel):
    farm_id: uuid.UUID
    generated_at: datetime
    zones: list[ZoneAlerts] = Field(default_factory=list)
