"""ORM model registry — importing this module registers every table on Base.metadata.

Alembic ``env.py`` imports ``Base`` from here (not from ``base.py``) so that
autogenerate sees all tables.  Application code can also do::

    from app.models import Farm, Zone, Vertex, ...
"""

# ── Base & Mixins ───────────────────────────────────────────────────────────
# ── Auth models ─────────────────────────────────────────────────────────────
from app.auth.models import APIKey, User
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
    NpkSourceEnum,
    UserRoleEnum,
    VertexTypeEnum,
    ZoneTypeEnum,
)

# ── Core topology models ────────────────────────────────────────────────────
from app.models.farm import Farm, HyperEdge, Vertex, Zone

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
    "APIKey",
    "AnomalyTypeEnum",
    # Base & mixins
    "Base",
    # Crop reference
    "CropProfile",
    # Core topology
    "Farm",
    # Enums
    "FarmTypeEnum",
    "HyperEdge",
    "HyperEdgeLayerEnum",
    "IrrigationEvent",
    "IrrigationTriggerEnum",
    "LightingReading",
    "NpkSample",
    "NpkSourceEnum",
    # Time-series
    "SoilReading",
    "TimeSeriesMixin",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    # Auth
    "User",
    "UserRoleEnum",
    "Vertex",
    "VertexTypeEnum",
    "VisionEvent",
    "WeatherReading",
    "Zone",
    "ZoneTypeEnum",
]
