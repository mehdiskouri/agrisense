"""FastAPI application entrypoint — lifespan, routers, middleware."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import text

from app.config import get_settings
from app.database import engine
from app.routes import farms
from app.services import julia_bridge

logger = logging.getLogger("agrisense")


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
