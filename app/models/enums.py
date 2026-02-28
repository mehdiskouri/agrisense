"""PostgreSQL-backed enum types for all ORM models.

Each StrEnum maps 1:1 to a PostgreSQL CREATE TYPE ... AS ENUM.
These are separate from the Pydantic StrEnum in app/config.py —
config enums validate settings, ORM enums type database columns.
"""

from enum import StrEnum

# ── Core topology enums ─────────────────────────────────────────────────────


class FarmTypeEnum(StrEnum):
    """Farm deployment configuration (sets active hypergraph layers)."""

    open_field = "open_field"
    greenhouse = "greenhouse"
    hybrid = "hybrid"


class ZoneTypeEnum(StrEnum):
    """Zone classification within a farm."""

    open_field = "open_field"
    greenhouse = "greenhouse"


class VertexTypeEnum(StrEnum):
    """Physical entity types in the hypergraph."""

    sensor = "sensor"
    valve = "valve"
    crop_bed = "crop_bed"
    weather_station = "weather_station"
    camera = "camera"
    light_fixture = "light_fixture"
    climate_controller = "climate_controller"


class HyperEdgeLayerEnum(StrEnum):
    """Hypergraph layer classification (maps to Julia Symbol set)."""

    soil = "soil"
    irrigation = "irrigation"
    lighting = "lighting"
    weather = "weather"
    crop_requirements = "crop_requirements"
    npk = "npk"
    vision = "vision"


# ── Time-series enums ───────────────────────────────────────────────────────


class IrrigationTriggerEnum(StrEnum):
    """What triggered an irrigation event."""

    manual = "manual"
    scheduled = "scheduled"
    auto = "auto"
    emergency = "emergency"


class NpkSourceEnum(StrEnum):
    """Source of NPK sample data."""

    lab = "lab"
    inline_sensor = "inline_sensor"


class AnomalyTypeEnum(StrEnum):
    """Computer vision anomaly classification."""

    none = "none"
    pest = "pest"
    disease = "disease"
    nutrient_deficiency = "nutrient_deficiency"
    wilting = "wilting"
    other = "other"


# ── Auth enums ──────────────────────────────────────────────────────────────


class UserRoleEnum(StrEnum):
    """User authorization roles for RBAC."""

    admin = "admin"
    manager = "manager"
    viewer = "viewer"
