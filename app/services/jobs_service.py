"""Durable recompute job orchestration service."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import JobStatusEnum
from app.models.jobs import RecomputeJob
from app.schemas.jobs import JobCreateResponse, JobStatusResponse
from app.services.farm_service import FarmService

JOB_STATUS_TTL_SECONDS = 60 * 60 * 24


class JobsService:
	def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
		self.db = db
		self.redis_client = redis_client

	async def create_recompute_job(self, farm_id: uuid.UUID) -> JobCreateResponse:
		await FarmService(self.db).get_farm(farm_id)
		job = RecomputeJob(farm_id=farm_id, status=JobStatusEnum.queued)
		self.db.add(job)
		await self.db.flush()
		await self.db.refresh(job)
		await self._persist_job_status(job)
		return JobCreateResponse(
			job_id=job.id,
			farm_id=job.farm_id,
			status=job.status.value,
			created_at=job.created_at,
		)

	async def get_job_status(self, job_id: uuid.UUID) -> JobStatusResponse:
		cached = await self._read_cached_status(job_id)
		if cached is not None:
			return cached

		job = await self._require_job(job_id)
		payload = self._to_status_payload(job)
		await self._persist_job_status(job)
		return payload

	async def execute_recompute(self, job_id: uuid.UUID) -> JobStatusResponse:
		job = await self._require_job(job_id)
		job.status = JobStatusEnum.running
		job.started_at = datetime.now(UTC)
		job.error = None
		await self.db.flush()
		await self._persist_job_status(job)

		try:
			await FarmService(self.db).get_graph(job.farm_id)
			job.status = JobStatusEnum.succeeded
			job.completed_at = datetime.now(UTC)
		except Exception as exc:
			job.status = JobStatusEnum.failed
			job.completed_at = datetime.now(UTC)
			job.error = str(exc)
			await self.db.flush()
			await self._persist_job_status(job)
			raise

		await self.db.flush()
		await self._persist_job_status(job)
		return self._to_status_payload(job)

	async def _require_job(self, job_id: uuid.UUID) -> RecomputeJob:
		row = await self.db.execute(select(RecomputeJob).where(RecomputeJob.id == job_id))
		job = row.scalar_one_or_none()
		if job is None:
			raise LookupError(f"Job {job_id} not found")
		return job

	async def _persist_job_status(self, job: RecomputeJob) -> None:
		if self.redis_client is None:
			return
		key = f"job:{job.id}:status"
		payload = self._to_status_payload(job).model_dump(mode="json")
		await self.redis_client.setex(key, JOB_STATUS_TTL_SECONDS, json.dumps(payload))

	async def _read_cached_status(self, job_id: uuid.UUID) -> JobStatusResponse | None:
		if self.redis_client is None:
			return None
		value = await self.redis_client.get(f"job:{job_id}:status")
		if value is None:
			return None
		return JobStatusResponse(**json.loads(value))

	@staticmethod
	def _to_status_payload(job: RecomputeJob) -> JobStatusResponse:
		return JobStatusResponse(
			job_id=job.id,
			farm_id=job.farm_id,
			status=job.status.value,
			created_at=job.created_at,
			started_at=job.started_at,
			completed_at=job.completed_at,
			error=job.error,
			updated_at=job.updated_at,
		)
