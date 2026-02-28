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
	def _validate_identifier(self) -> "ZoneDetailQuery":
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
