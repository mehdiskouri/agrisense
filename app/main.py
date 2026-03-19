"""FastAPI application entrypoint — lifespan, routers, middleware."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from sqlalchemy import select, text

from app.config import get_settings, parse_cors_origins
from app.database import async_session_factory, engine
from app.middleware.logging import (
    RequestLoggingMiddleware,
    configure_structured_logging,
)
from app.middleware.rate_limit import RateLimitMiddleware
from app.models.farm import Farm
from app.routes import analytics, anomalies, ask, farms, ingest, jobs, ws
from app.services import julia_bridge
from app.services.farm_service import FarmService
from app.services.webhook_service import run_dispatch_queue_worker

logger = logging.getLogger("agrisense")


async def _ping_redis(redis_client: Redis) -> bool:
    result = redis_client.ping()
    if isinstance(result, bool):
        return result
    return await result


async def _bootstrap_graph_cache() -> int:
    """Build and cache graph state for all farms present in the database."""
    async with async_session_factory() as session:
        rows = await session.execute(select(Farm.id))
        farm_ids = list(rows.scalars().all())
        if not farm_ids:
            return 0

        service = FarmService(session)
        count = 0
        for farm_id in farm_ids:
            await service.get_graph(farm_id)
            count += 1
        return count


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown lifecycle.

    Startup:
      1. Initialize structured logging
      2. Initialize Julia runtime + AgriSenseCore (GPU kernel warm-up)
      3. Connect to Redis
      4. Build in-memory hypergraph state from DB

    Shutdown:
      1. Close Redis connection pool
      2. Dispose SQLAlchemy engine
    """
    settings = get_settings()
    configure_structured_logging()
    logger.info(
        "AgriSense starting",
        extra={
            "log_level": settings.log_level,
            "farm_default_type": settings.farm_default_type.value,
        },
    )

    redis: Redis | None = None
    webhook_worker_stop: asyncio.Event | None = None
    webhook_worker_task: asyncio.Task[None] | None = None
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        await _ping_redis(redis)
        app.state.redis = redis

        webhook_worker_stop = asyncio.Event()
        webhook_worker_task = asyncio.create_task(
            run_dispatch_queue_worker(redis, webhook_worker_stop)
        )
        app.state.webhook_worker_task = webhook_worker_task

        julia_bridge.initialize_julia()
        julia_bridge.generate_synthetic(
            farm_type=settings.farm_default_type.value,
            days=1,
            seed=1,
        )
        app.state.julia_ready = True

        if settings.bootstrap_graph_cache_on_startup:
            bootstrap_count = await _bootstrap_graph_cache()
            app.state.graph_bootstrap_count = bootstrap_count
        else:
            app.state.graph_bootstrap_count = 0
    except Exception as exc:
        if webhook_worker_task is not None:
            webhook_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await webhook_worker_task
        logger.exception("startup failure", extra={"error": str(exc)})
        raise

    yield

    logger.info("AgriSense shutting down")
    if webhook_worker_stop is not None:
        webhook_worker_stop.set()
    if webhook_worker_task is not None:
        try:
            await asyncio.wait_for(webhook_worker_task, timeout=5.0)
        except TimeoutError:
            webhook_worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await webhook_worker_task

    if redis is not None:
        await redis.aclose()
    await engine.dispose()


app = FastAPI(
    title="AgriSense API",
    description=(
        "Agricultural hypergraph API — models farms as layered hypergraphs "
        "with GPU-accelerated Julia computational core for irrigation scheduling, "
        "nutrient management, yield forecasting, and natural language queries."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────────────────────
settings = get_settings()
cors_origins = parse_cors_origins(settings.cors_allowed_origins)
allow_credentials = settings.cors_allow_credentials and cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)


# ── Health check ────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Basic health check — verifies the API process is alive."""
    return {
        "status": "ok",
        "service": "agrisense",
        "version": "0.1.0",
    }


async def _run_readiness_checks(app_instance: FastAPI) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {
        "database": {"ok": True, "message": "ok"},
        "redis": {"ok": True, "message": "ok"},
        "julia": {"ok": True, "message": "ok"},
    }

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:
        checks["database"] = {"ok": False, "message": str(exc)}

    redis_client = getattr(app_instance.state, "redis", None)
    if redis_client is None:
        checks["redis"] = {"ok": False, "message": "redis not initialized"}
    else:
        try:
            await _ping_redis(redis_client)
        except Exception as exc:
            checks["redis"] = {"ok": False, "message": str(exc)}

    try:
        julia_bridge.initialize_julia()
    except Exception as exc:
        checks["julia"] = {"ok": False, "message": str(exc)}

    return checks


@app.get("/health/ready", tags=["system"])
async def readiness_check(request: Request) -> JSONResponse:
    checks = await _run_readiness_checks(request.app)
    ok = all(item["ok"] for item in checks.values())
    status_code = 200 if ok else 503
    payload = {
        "status": "ok" if ok else "degraded",
        "service": "agrisense",
        "version": "0.1.0",
        "checks": checks,
    }
    return JSONResponse(status_code=status_code, content=payload)


# ── Router registration ────────────────────────────────────────────────────
app.include_router(farms.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(anomalies.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(ask.router, prefix="/api/v1")
# Auth provisioning endpoints are intentionally external for this portfolio service.
# Users/API keys are currently managed via seed/admin tooling, not public API routes.
app.include_router(ws.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
