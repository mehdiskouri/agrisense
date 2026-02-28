"""Time-series sensor ORM models.

All six tables use BIGSERIAL primary keys (via ``TimeSeriesMixin``) rather
than UUIDs — auto-increment is more efficient for append-heavy, high-volume
sensor data.  Each includes a composite index on (fk, timestamp) for fast
range queries and an ``ingested_at`` column for ingestion-lag monitoring.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimeSeriesMixin
from app.models.enums import (
    AnomalyTypeEnum,
    IrrigationTriggerEnum,
    NpkSourceEnum,
)

# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Soil Substrate
# ═══════════════════════════════════════════════════════════════════════════


class SoilReading(Base, TimeSeriesMixin):
    """Soil sensor reading — moisture, temperature, EC, pH."""

    __tablename__ = "soil_readings"
    __table_args__ = (
        Index("ix_soil_readings_sensor_ts", "sensor_id", "timestamp"),
    )

    sensor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vertices.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    moisture: Mapped[float] = mapped_column(Float, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    conductivity: Mapped[float | None] = mapped_column(Float, nullable=True)
    ph: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<SoilReading id={self.id} sensor={self.sensor_id} "
            f"ts={self.timestamp}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4 — Rainfall & Weather
# ═══════════════════════════════════════════════════════════════════════════


class WeatherReading(Base, TimeSeriesMixin):
    """Weather station reading — temp, humidity, rain, wind, pressure, ET₀."""

    __tablename__ = "weather_readings"
    __table_args__ = (
        Index("ix_weather_readings_station_ts", "station_id", "timestamp"),
    )

    station_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vertices.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    humidity: Mapped[float] = mapped_column(Float, nullable=False)
    precipitation_mm: Mapped[float] = mapped_column(Float, nullable=False)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_hpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    et0: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<WeatherReading id={self.id} station={self.station_id} "
            f"ts={self.timestamp}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Irrigation System
# ═══════════════════════════════════════════════════════════════════════════


class IrrigationEvent(Base, TimeSeriesMixin):
    """Irrigation event — tracks valve open/close, volume, trigger type."""

    __tablename__ = "irrigation_events"
    __table_args__ = (
        Index(
            "ix_irrigation_events_valve_ts",
            "valve_id",
            "timestamp_start",
        ),
    )

    valve_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vertices.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    timestamp_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    volume_liters: Mapped[float | None] = mapped_column(Float, nullable=True)
    trigger: Mapped[IrrigationTriggerEnum] = mapped_column(
        Enum(
            IrrigationTriggerEnum,
            name="irrigation_trigger",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<IrrigationEvent id={self.id} valve={self.valve_id} "
            f"trigger={self.trigger}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6 — NPK & Nutrient
# ═══════════════════════════════════════════════════════════════════════════


class NpkSample(Base, TimeSeriesMixin):
    """NPK nutrient sample — lab results or inline sensor readings."""

    __tablename__ = "npk_samples"
    __table_args__ = (
        Index("ix_npk_samples_zone_ts", "zone_id", "timestamp"),
    )

    zone_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("zones.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    nitrogen_mg_kg: Mapped[float] = mapped_column(Float, nullable=False)
    phosphorus_mg_kg: Mapped[float] = mapped_column(Float, nullable=False)
    potassium_mg_kg: Mapped[float] = mapped_column(Float, nullable=False)
    organic_matter_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    source: Mapped[NpkSourceEnum] = mapped_column(
        Enum(
            NpkSourceEnum,
            name="npk_source",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<NpkSample id={self.id} zone={self.zone_id} "
            f"source={self.source}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 7 — Computer Vision (Greenhouse)
# ═══════════════════════════════════════════════════════════════════════════


class VisionEvent(Base, TimeSeriesMixin):
    """Vision inference result — anomaly detection + canopy metrics.

    ``metadata_`` (JSONB, DB column "metadata") stores bounding boxes,
    class labels, and other inference artifacts.
    """

    __tablename__ = "vision_events"
    __table_args__ = (
        Index("ix_vision_events_camera_ts", "camera_id", "timestamp"),
    )

    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vertices.id", ondelete="CASCADE"),
        nullable=False,
    )
    crop_bed_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vertices.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    anomaly_type: Mapped[AnomalyTypeEnum] = mapped_column(
        Enum(
            AnomalyTypeEnum,
            name="anomaly_type",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    canopy_coverage_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<VisionEvent id={self.id} camera={self.camera_id} "
            f"anomaly={self.anomaly_type}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — Solar / Lighting (Greenhouse)
# ═══════════════════════════════════════════════════════════════════════════


class LightingReading(Base, TimeSeriesMixin):
    """Lighting fixture reading — PAR, DLI, duty cycle, spectrum."""

    __tablename__ = "lighting_readings"
    __table_args__ = (
        Index(
            "ix_lighting_readings_fixture_ts", "fixture_id", "timestamp"
        ),
    )

    fixture_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vertices.id", ondelete="CASCADE"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    par_umol: Mapped[float] = mapped_column(Float, nullable=False)
    dli_cumulative: Mapped[float] = mapped_column(Float, nullable=False)
    duty_cycle_pct: Mapped[float] = mapped_column(Float, nullable=False)
    spectrum_profile: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<LightingReading id={self.id} fixture={self.fixture_id} "
            f"par={self.par_umol}>"
        )

