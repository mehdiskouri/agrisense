"""Farm, Zone, Vertex, HyperEdge ORM models — core hypergraph topology.

These four tables define the persistent representation of the layered
hypergraph.  Vertex/edge data stored here is loaded into the Julia
in-memory ``LayeredHyperGraph`` via the bridge at runtime.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geography
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    FarmTypeEnum,
    HyperEdgeLayerEnum,
    VertexTypeEnum,
    ZoneTypeEnum,
)

# ═══════════════════════════════════════════════════════════════════════════
# Farm
# ═══════════════════════════════════════════════════════════════════════════


class Farm(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A physical farm — root entity of the hypergraph.

    ``model_overrides`` stores optional per-farm overrides for
    ``ModelConfig`` (irrigation, nutrients, yield_forecast,
    anomaly_detection bool flags + threshold tuning).
    When NULL the service layer derives defaults from ``farm_type``.
    """

    __tablename__ = "farms"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    farm_type: Mapped[FarmTypeEnum] = mapped_column(
        Enum(
            FarmTypeEnum,
            name="farm_type",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )
    location: Mapped[Any] = mapped_column(
        Geography(geometry_type="POINT", srid=4326),
        nullable=True,
    )
    timezone: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="UTC",
        server_default=text("'UTC'"),
    )
    model_overrides: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Relationships ────────────────────────────────────────────────────
    zones: Mapped[list[Zone]] = relationship(
        back_populates="farm",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    vertices: Mapped[list[Vertex]] = relationship(
        back_populates="farm",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    hyperedges: Mapped[list[HyperEdge]] = relationship(
        back_populates="farm",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Farm id={self.id} name={self.name!r} type={self.farm_type}>"


# ═══════════════════════════════════════════════════════════════════════════
# Zone
# ═══════════════════════════════════════════════════════════════════════════


class Zone(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A bounded sub-region within a farm.

    ``boundary`` is nullable because greenhouse zones are structurally
    bounded (bay, bench, row) rather than GPS-bounded.
    """

    __tablename__ = "zones"
    __table_args__ = (Index("ix_zones_farm_id", "farm_id"),)

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    zone_type: Mapped[ZoneTypeEnum] = mapped_column(
        Enum(
            ZoneTypeEnum,
            name="zone_type",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )
    boundary: Mapped[Any] = mapped_column(
        Geography(geometry_type="POLYGON", srid=4326),
        nullable=True,
    )
    area_m2: Mapped[float] = mapped_column(Float, nullable=False)
    soil_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="unknown",
        server_default=text("'unknown'"),
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    # ── Relationships ────────────────────────────────────────────────────
    farm: Mapped[Farm] = relationship(back_populates="zones")
    vertices: Mapped[list[Vertex]] = relationship(
        back_populates="zone",
        passive_deletes=True,
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Zone id={self.id} name={self.name!r} farm={self.farm_id}>"


# ═══════════════════════════════════════════════════════════════════════════
# Vertex
# ═══════════════════════════════════════════════════════════════════════════


class Vertex(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A physical entity in the hypergraph (sensor, valve, crop bed, …).

    ``zone_id`` is nullable — weather stations may be farm-level with no
    specific zone assignment.  ``config`` (JSONB) stores type-specific
    attributes (calibration_date, flow_rate_max, spectrum_profile, etc.).
    """

    __tablename__ = "vertices"
    __table_args__ = (
        Index("ix_vertices_farm_id_type", "farm_id", "vertex_type"),
    )

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    zone_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("zones.id", ondelete="SET NULL"),
        nullable=True,
    )
    vertex_type: Mapped[VertexTypeEnum] = mapped_column(
        Enum(
            VertexTypeEnum,
            name="vertex_type",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    installed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ────────────────────────────────────────────────────
    farm: Mapped[Farm] = relationship(back_populates="vertices")
    zone: Mapped[Zone | None] = relationship(back_populates="vertices")

    def __repr__(self) -> str:
        return (
            f"<Vertex id={self.id} type={self.vertex_type} "
            f"zone={self.zone_id}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
# HyperEdge
# ═══════════════════════════════════════════════════════════════════════════


class HyperEdge(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """A hyperedge connecting an arbitrary subset of vertices within a layer.

    ``vertex_ids`` is a PostgreSQL UUID[] column — deliberate
    denormalization that mirrors the sparse incidence matrix layout in the
    Julia ``HyperGraphLayer``.  A GIN index enables efficient containment
    queries (``@>``, ``<@``, ``&&``).
    """

    __tablename__ = "hyperedges"
    __table_args__ = (
        Index("ix_hyperedges_farm_layer", "farm_id", "layer"),
        Index(
            "ix_hyperedges_vertex_ids_gin",
            "vertex_ids",
            postgresql_using="gin",
        ),
    )

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    layer: Mapped[HyperEdgeLayerEnum] = mapped_column(
        Enum(
            HyperEdgeLayerEnum,
            name="hyperedge_layer",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
    )
    vertex_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
    )
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    # ── Relationships ────────────────────────────────────────────────────
    farm: Mapped[Farm] = relationship(back_populates="hyperedges")

    def __repr__(self) -> str:
        n = len(self.vertex_ids) if self.vertex_ids else 0
        return f"<HyperEdge id={self.id} layer={self.layer} vertices={n}>"

