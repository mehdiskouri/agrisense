from __future__ import annotations

from datetime import UTC, datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.schemas.jobs import JobCreateResponse, JobStatusResponse
from app.routes import jobs as jobs_routes
from app.models.enums import JobStatusEnum
from app.models.jobs import RecomputeJob
from app.services.jobs_service import JobsService
from app.services.farm_service import FarmService


@pytest.mark.asyncio
async def test_create_recompute_job_endpoint(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
	farm_id = uuid4()
	job_id = uuid4()

	async def fake_create(self: JobsService, _farm_id: object) -> JobCreateResponse:
		return JobCreateResponse(
			job_id=job_id,
			farm_id=farm_id,
			status="queued",
			created_at=datetime.now(UTC),
		)

	monkeypatch.setattr(JobsService, "create_recompute_job", fake_create)

	response = await client.post(f"/api/v1/jobs/{farm_id}/recompute")
	assert response.status_code == 202
	body = response.json()
	assert body["job_id"] == str(job_id)
	assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_get_job_status_endpoint(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
	farm_id = uuid4()
	job_id = uuid4()

	async def fake_status(self: JobsService, _job_id: object) -> JobStatusResponse:
		now = datetime.now(UTC)
		return JobStatusResponse(
			job_id=job_id,
			farm_id=farm_id,
			status="succeeded",
			created_at=now,
			started_at=now,
			completed_at=now,
			error=None,
			updated_at=now,
		)

	monkeypatch.setattr(JobsService, "get_job_status", fake_status)

	response = await client.get(f"/api/v1/jobs/{job_id}/status")
	assert response.status_code == 200
	assert response.json()["status"] == "succeeded"


@pytest.mark.asyncio
async def test_jobs_openapi_contract(client: AsyncClient) -> None:
	response = await client.get("/openapi.json")
	assert response.status_code == 200
	paths = response.json()["paths"]
	assert "/api/v1/jobs/{farm_id}/recompute" in paths
	assert "/api/v1/jobs/{job_id}/status" in paths


@pytest.mark.asyncio
async def test_background_recompute_failure_commits_status(monkeypatch: pytest.MonkeyPatch) -> None:
	class _FakeSession:
		def __init__(self) -> None:
			self.commit = AsyncMock()
			self.rollback = AsyncMock()

	fake_session = _FakeSession()

	@asynccontextmanager
	async def _fake_factory():
		yield fake_session

	async def _failing_execute(self: JobsService, _job_id: object) -> object:
		raise RuntimeError("recompute failed")

	monkeypatch.setattr(jobs_routes, "async_session_factory", _fake_factory)
	monkeypatch.setattr(JobsService, "execute_recompute", _failing_execute)

	await jobs_routes._run_recompute_job(uuid4(), None)
	assert fake_session.commit.await_count == 1
	assert fake_session.rollback.await_count == 0


@pytest.mark.asyncio
async def test_jobs_service_execute_recompute_failure_marks_failed(monkeypatch: pytest.MonkeyPatch) -> None:
	class _Row:
		def __init__(self, job: RecomputeJob) -> None:
			self._job = job

		def scalar_one_or_none(self) -> RecomputeJob:
			return self._job

	class _FakeDb:
		def __init__(self, job: RecomputeJob) -> None:
			self._job = job
			self.flush = AsyncMock()
			self.execute = AsyncMock(return_value=_Row(job))

	now = datetime.now(UTC)
	job = RecomputeJob(farm_id=uuid4(), status=JobStatusEnum.queued)
	job.id = uuid4()
	job.created_at = now
	job.updated_at = now

	fake_db = _FakeDb(job)
	service = JobsService(fake_db, None)  # type: ignore[arg-type]

	async def _failing_get_graph(self: FarmService, _farm_id: object) -> object:
		raise RuntimeError("graph rebuild failed")

	monkeypatch.setattr(FarmService, "get_graph", _failing_get_graph)

	with pytest.raises(RuntimeError):
		await service.execute_recompute(job.id)

	assert job.status == JobStatusEnum.failed
	assert job.completed_at is not None
	assert job.error == "graph rebuild failed"
