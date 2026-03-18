"""Pydantic schemas for spreadsheet report generation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    irrigation_horizon_days: int = Field(default=7, ge=1, le=30)
    include_members: bool = True
    include_history_charts: bool = True


class ReportMeta(BaseModel):
    farm_id: uuid.UUID
    farm_name: str
    farm_type: str
    generated_at: datetime
    sheets: list[str] = Field(default_factory=list)


class ReportJobCreateResponse(BaseModel):
    job_id: uuid.UUID
    farm_id: uuid.UUID
    status: str
    created_at: datetime


class ReportJobStatusResponse(BaseModel):
    job_id: uuid.UUID
    farm_id: uuid.UUID
    status: str
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    filename: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
