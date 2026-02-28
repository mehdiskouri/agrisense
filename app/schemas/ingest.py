"""Pydantic schemas for data ingestion payloads."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import AnomalyTypeEnum, IrrigationTriggerEnum, NpkSourceEnum


class IngestWarning(BaseModel):
	index: int
	message: str


class IngestReceipt(BaseModel):
	farm_id: uuid.UUID
	layer: str
	status: str
	inserted_count: int = 0
	failed_count: int = 0
	event_ids: list[int] = Field(default_factory=list)
	timestamp_start: datetime | None = None
	timestamp_end: datetime | None = None
	warnings: list[IngestWarning] = Field(default_factory=list)


class SoilReadingIn(BaseModel):
	sensor_id: uuid.UUID
	timestamp: datetime
	moisture: float
	temperature: float
	conductivity: float | None = None
	ph: float | None = None


class WeatherReadingIn(BaseModel):
	station_id: uuid.UUID
	timestamp: datetime
	temperature: float
	humidity: float
	precipitation_mm: float
	wind_speed: float | None = None
	wind_direction: float | None = None
	pressure_hpa: float | None = None
	et0: float | None = None


class IrrigationEventIn(BaseModel):
	valve_id: uuid.UUID
	timestamp_start: datetime
	timestamp_end: datetime | None = None
	volume_liters: float | None = None
	trigger: IrrigationTriggerEnum


class NpkSampleIn(BaseModel):
	zone_id: uuid.UUID
	timestamp: datetime
	nitrogen_mg_kg: float
	phosphorus_mg_kg: float
	potassium_mg_kg: float
	organic_matter_pct: float | None = None
	source: NpkSourceEnum


class VisionEventIn(BaseModel):
	camera_id: uuid.UUID
	crop_bed_id: uuid.UUID
	timestamp: datetime
	anomaly_type: AnomalyTypeEnum
	confidence: float
	canopy_coverage_pct: float | None = None
	metadata: dict[str, Any] | None = None


class LightingReadingIn(BaseModel):
	fixture_id: uuid.UUID
	timestamp: datetime
	par_umol: float
	dli_cumulative: float
	duty_cycle_pct: float
	spectrum_profile: dict[str, Any] | None = None
	layer: str = "lighting"


class SoilIngestRequest(BaseModel):
	farm_id: uuid.UUID
	readings: list[SoilReadingIn] = Field(min_length=1)


class WeatherIngestRequest(BaseModel):
	farm_id: uuid.UUID
	readings: list[WeatherReadingIn] = Field(min_length=1)


class IrrigationIngestRequest(BaseModel):
	farm_id: uuid.UUID
	events: list[IrrigationEventIn] = Field(min_length=1)


class NpkIngestRequest(BaseModel):
	farm_id: uuid.UUID
	samples: list[NpkSampleIn] = Field(min_length=1)


class VisionIngestRequest(BaseModel):
	farm_id: uuid.UUID
	events: list[VisionEventIn] = Field(min_length=1)


class BulkIngestRequest(BaseModel):
	farm_id: uuid.UUID
	soil: list[SoilReadingIn] = Field(default_factory=list)
	weather: list[WeatherReadingIn] = Field(default_factory=list)
	irrigation: list[IrrigationEventIn] = Field(default_factory=list)
	npk: list[NpkSampleIn] = Field(default_factory=list)
	vision: list[VisionEventIn] = Field(default_factory=list)
	lighting: list[LightingReadingIn] = Field(default_factory=list)


class BulkIngestReceipt(BaseModel):
	farm_id: uuid.UUID
	status: str
	inserted_count: int = 0
	failed_count: int = 0
	timestamp_start: datetime | None = None
	timestamp_end: datetime | None = None
	warnings: list[IngestWarning] = Field(default_factory=list)
	layers: dict[str, IngestReceipt] = Field(default_factory=dict)
