"""TypedDict contracts for JSONB-backed columns.

These contracts are introduced as shared type boundaries before model/schema
migration. They intentionally encode the canonical keys used by AgriSense while
remaining permissive through optional fields.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class FarmModelOverrides(TypedDict, total=False):
    """Optional per-farm model toggles and threshold tuning."""

    irrigation: bool
    nutrients: bool
    yield_forecast: bool
    anomaly_detection: bool
    thresholds: dict[str, float | int]
    notes: str


class ZoneMetadata(TypedDict, total=False):
    """Free-form zone-level metadata stored in JSONB."""

    tags: list[str]
    notes: str
    attributes: dict[str, Any]


class VertexConfig(TypedDict, total=False):
    """Type-specific vertex configuration payload."""

    calibration_date: str
    flow_rate_max: float
    manufacturer: str
    model: str
    firmware_version: str
    spectrum_profile: dict[str, float]
    attributes: dict[str, Any]


class HyperEdgeMetadata(TypedDict, total=False):
    """Additional relationship metadata for hyperedges."""

    relation_type: str
    weight: float
    confidence: float
    context: dict[str, Any]


class VisionBoundingBox(TypedDict):
    """Normalized bounding-box shape from vision models."""

    x: float
    y: float
    width: float
    height: float


class VisionDetection(TypedDict, total=False):
    """Single detection artifact in a vision event payload."""

    label: str
    confidence: float
    bbox: VisionBoundingBox


class VisionEventMetadata(TypedDict, total=False):
    """Computer-vision inference artifacts for an event."""

    image_uri: str
    model_version: str
    labels: list[str]
    detections: list[VisionDetection]
    attributes: dict[str, Any]


class LightingSpectrumProfile(TypedDict, total=False):
    """Lighting channel intensities or share percentages."""

    red: float
    green: float
    blue: float
    white: float
    far_red: float
    uv: float


class CropNpkDemand(TypedDict):
    """Per-stage nutrient demand profile."""

    n: float
    p: float
    k: float


class CropGrowthStage(TypedDict):
    """Single crop growth-stage requirement entry."""

    name: str
    duration_days: int
    optimal_moisture_range: list[float]
    optimal_temp_range: list[float]
    water_demand_mm_day: float
    light_requirement_dli: float
    npk_demand: CropNpkDemand
    notes: NotRequired[str]


APIKeyScopes = TypedDict(
    "APIKeyScopes",
    {
        "read:farms": bool,
        "read:analytics": bool,
        "read:jobs": bool,
        # Canonical machine scopes used by runtime enforcement.
        "ingest": bool,
        "jobs": bool,
        # Legacy write-prefixed aliases accepted for backward compatibility.
        "write:ingest": bool,
        "write:jobs": bool,
        "admin": bool,
    },
    total=False,
)
