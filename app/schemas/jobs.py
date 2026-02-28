"""Pydantic schemas for recompute job endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class JobCreateResponse(BaseModel):
	job_id: uuid.UUID
	farm_id: uuid.UUID
	status: str
	created_at: datetime


class JobStatusResponse(BaseModel):
	job_id: uuid.UUID
	farm_id: uuid.UUID
	status: str
	created_at: datetime
	started_at: datetime | None = None
	completed_at: datetime | None = None
	error: str | None = None
	updated_at: datetime
