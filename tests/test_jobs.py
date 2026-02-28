from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.schemas.jobs import JobCreateResponse, JobStatusResponse
from app.services.jobs_service import JobsService


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
