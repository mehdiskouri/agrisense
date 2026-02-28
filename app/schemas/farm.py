"""Pydantic request/response schemas for farm objects."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import FarmTypeEnum, VertexTypeEnum, ZoneTypeEnum


class FarmCreate(BaseModel):
	name: str = Field(min_length=1, max_length=255)
	farm_type: FarmTypeEnum
	timezone: str = Field(default="UTC", min_length=1, max_length=64)
	model_overrides: dict[str, Any] | None = None


class ZoneCreate(BaseModel):
	name: str = Field(min_length=1, max_length=255)
	zone_type: ZoneTypeEnum | None = None
	area_m2: float = Field(gt=0)
	soil_type: str = Field(default="unknown", min_length=1, max_length=100)
	metadata: dict[str, Any] | None = None


class VertexCreate(BaseModel):
	vertex_type: VertexTypeEnum
	zone_id: uuid.UUID | None = None
	config: dict[str, Any] | None = None
	installed_at: datetime | None = None
	last_seen_at: datetime | None = None


class ZoneRead(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: uuid.UUID
	farm_id: uuid.UUID
	name: str
	zone_type: ZoneTypeEnum
	area_m2: float
	soil_type: str
	metadata: dict[str, Any] | None = None
	created_at: datetime
	updated_at: datetime


class VertexRead(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: uuid.UUID
	farm_id: uuid.UUID
	zone_id: uuid.UUID | None
	vertex_type: VertexTypeEnum
	config: dict[str, Any] | None = None
	installed_at: datetime | None
	last_seen_at: datetime | None
	created_at: datetime
	updated_at: datetime


class FarmRead(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: uuid.UUID
	name: str
	farm_type: FarmTypeEnum
	timezone: str
	model_overrides: dict[str, Any] | None = None
	created_at: datetime
	updated_at: datetime
	active_layers: list[str]
	zones: list[ZoneRead] = Field(default_factory=list)
	vertices: list[VertexRead] = Field(default_factory=list)


class FarmGraphRead(BaseModel):
	farm_id: str
	n_vertices: int
	vertex_index: dict[str, int]
	layers: dict[str, Any]


class FarmListRead(BaseModel):
	items: list[FarmRead]
