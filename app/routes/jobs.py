"""Background recompute job routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.schemas.jobs import JobCreateResponse, JobStatusResponse
from app.services.jobs_service import JobsService

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _map_error(exc: Exception) -> HTTPException:
	if isinstance(exc, LookupError):
		return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
	if isinstance(exc, ValueError):
		return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
	return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="job failure")


async def _run_recompute_job(job_id: uuid.UUID, redis_client: object | None) -> None:
	async with async_session_factory() as session:
		service = JobsService(session, redis_client)  # type: ignore[arg-type]
		try:
			await service.execute_recompute(job_id)
			await session.commit()
		except Exception:
			try:
				await session.rollback()
			except Exception:
				return


@router.post("/{farm_id}/recompute", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_recompute_job(
	farm_id: uuid.UUID,
	request: Request,
	background_tasks: BackgroundTasks,
	db: AsyncSession = Depends(get_db),
) -> JobCreateResponse:
	service = JobsService(db, getattr(request.app.state, "redis", None))
	try:
		response = await service.create_recompute_job(farm_id)
	except Exception as exc:
		raise _map_error(exc) from exc

	background_tasks.add_task(_run_recompute_job, response.job_id, getattr(request.app.state, "redis", None))
	return response


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
	job_id: uuid.UUID,
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
	service = JobsService(db, getattr(request.app.state, "redis", None))
	try:
		return await service.get_job_status(job_id)
	except Exception as exc:
		raise _map_error(exc) from exc
