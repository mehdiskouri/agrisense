"""FastAPI application entrypoint — lifespan, routers, middleware."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import select, text

from app.config import get_settings
from app.database import async_session_factory, engine
from app.middleware.rate_limit import RateLimitMiddleware
from app.models.farm import Farm
from app.routes import analytics, ask, farms, ingest, jobs, ws
from app.services import julia_bridge
from app.services.farm_service import FarmService

logger = logging.getLogger("agrisense")


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
    logger.info(
        "AgriSense starting",
        extra={
            "log_level": settings.log_level,
            "farm_default_type": settings.farm_default_type.value,
        },
    )

    redis: Redis | None = None
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
        app.state.redis = redis

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
        logger.exception("startup failure", extra={"error": str(exc)})
        raise

    yield

    logger.info("AgriSense shutting down")
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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


# ── Router registration ────────────────────────────────────────────────────
app.include_router(farms.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(ask.router, prefix="/api/v1")
app.include_router(ws.router)
