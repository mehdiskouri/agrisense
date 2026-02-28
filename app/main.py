"""FastAPI application entrypoint — lifespan, routers, middleware."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

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

    # TODO (Phase 5): Initialize Julia bridge
    # TODO (Phase 5): Build hypergraph state from DB

    yield

    logger.info("AgriSense shutting down")
    # TODO: Cleanup resources


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
# TODO (Phase 5+): Register routers as they are implemented
# from app.routes import farms, ingest, analytics, ask, ws
# app.include_router(farms.router, prefix="/api/v1", tags=["farms"])
# app.include_router(ingest.router, prefix="/api/v1", tags=["ingest"])
# app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
# app.include_router(ask.router, prefix="/api/v1", tags=["ask"])
# app.include_router(ws.router, tags=["websocket"])
