"""ORM model registry — importing this module registers every table on Base.metadata.

Alembic ``env.py`` imports ``Base`` from here (not from ``base.py``) so that
autogenerate sees all tables.  Application code can also do::

    from app.models import Farm, Zone, Vertex, ...
"""

# ── Base & Mixins ───────────────────────────────────────────────────────────
from app.models.base import (
    Base,
    TimeSeriesMixin,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

# ── Crop reference ──────────────────────────────────────────────────────────
from app.models.crops import CropProfile

# ── Enums ───────────────────────────────────────────────────────────────────
from app.models.enums import (
    AnomalyTypeEnum,
    FarmTypeEnum,
    HyperEdgeLayerEnum,
    IrrigationTriggerEnum,
    JobStatusEnum,
    NpkSourceEnum,
    UserRoleEnum,
    VertexTypeEnum,
    ZoneTypeEnum,
)

# ── Core topology models ────────────────────────────────────────────────────
from app.models.farm import Farm, HyperEdge, Vertex, Zone
from app.models.jobs import BacktestJob, RecomputeJob

# ── Time-series sensor models ──────────────────────────────────────────────
from app.models.sensors import (
    IrrigationEvent,
    LightingReading,
    NpkSample,
    SoilReading,
    VisionEvent,
    WeatherReading,
)

__all__ = [
    "AnomalyTypeEnum",
    "BacktestJob",
    "Base",
    "CropProfile",
    "Farm",
    "FarmTypeEnum",
    "HyperEdge",
    "HyperEdgeLayerEnum",
    "IrrigationEvent",
    "IrrigationTriggerEnum",
    "JobStatusEnum",
    "LightingReading",
    "NpkSample",
    "NpkSourceEnum",
    "RecomputeJob",
    "SoilReading",
    "TimeSeriesMixin",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "UserRoleEnum",
    "Vertex",
    "VertexTypeEnum",
    "VisionEvent",
    "WeatherReading",
    "Zone",
    "ZoneTypeEnum",
]
