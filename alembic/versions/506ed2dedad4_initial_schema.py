"""initial_schema

Revision ID: 506ed2dedad4
Revises:
Create Date: 2026-02-28 00:00:00.000000

Creates all 13 tables, 8 PostgreSQL enum types, and all indexes for the
AgriSense schema.  Designed to run against a PostgreSQL 16 + PostGIS 3.4
database with uuid-ossp and postgis extensions already enabled
(via scripts/init_db.sql).
"""

from collections.abc import Sequence

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "506ed2dedad4"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# ── Enum type names (PostgreSQL CREATE TYPE) ────────────────────────────────
ENUM_FARM_TYPE = postgresql.ENUM(
    "open_field", "greenhouse", "hybrid", name="farm_type", create_type=False
)
ENUM_ZONE_TYPE = postgresql.ENUM(
    "open_field", "greenhouse", name="zone_type", create_type=False
)
ENUM_VERTEX_TYPE = postgresql.ENUM(
    "sensor",
    "valve",
    "crop_bed",
    "weather_station",
    "camera",
    "light_fixture",
    "climate_controller",
    name="vertex_type",
    create_type=False,
)
ENUM_HYPEREDGE_LAYER = postgresql.ENUM(
    "soil",
    "irrigation",
    "lighting",
    "weather",
    "crop_requirements",
    "npk",
    "vision",
    name="hyperedge_layer",
    create_type=False,
)
ENUM_IRRIGATION_TRIGGER = postgresql.ENUM(
    "manual",
    "scheduled",
    "auto",
    "emergency",
    name="irrigation_trigger",
    create_type=False,
)
ENUM_NPK_SOURCE = postgresql.ENUM(
    "lab", "inline_sensor", name="npk_source", create_type=False
)
ENUM_ANOMALY_TYPE = postgresql.ENUM(
    "none",
    "pest",
    "disease",
    "nutrient_deficiency",
    "wilting",
    "other",
    name="anomaly_type",
    create_type=False,
)
ENUM_USER_ROLE = postgresql.ENUM(
    "admin", "agronomist", "field_operator", "readonly", name="user_role", create_type=False
)


def upgrade() -> None:
    # ── 1. Create enum types ────────────────────────────────────────────
    ENUM_FARM_TYPE.create(op.get_bind(), checkfirst=True)
    ENUM_ZONE_TYPE.create(op.get_bind(), checkfirst=True)
    ENUM_VERTEX_TYPE.create(op.get_bind(), checkfirst=True)
    ENUM_HYPEREDGE_LAYER.create(op.get_bind(), checkfirst=True)
    ENUM_IRRIGATION_TRIGGER.create(op.get_bind(), checkfirst=True)
    ENUM_NPK_SOURCE.create(op.get_bind(), checkfirst=True)
    ENUM_ANOMALY_TYPE.create(op.get_bind(), checkfirst=True)
    ENUM_USER_ROLE.create(op.get_bind(), checkfirst=True)

    # ── 2. Core tables ──────────────────────────────────────────────────

    # farms
    op.create_table(
        "farms",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("farm_type", ENUM_FARM_TYPE, nullable=False),
        sa.Column(
            "location",
            geoalchemy2.types.Geography(
                geometry_type="POINT", srid=4326, from_text="ST_GeogFromText"
            ),
            nullable=True,
        ),
        sa.Column(
            "timezone",
            sa.String(64),
            server_default=sa.text("'UTC'"),
            nullable=False,
        ),
        sa.Column("model_overrides", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # zones
    op.create_table(
        "zones",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "farm_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("zone_type", ENUM_ZONE_TYPE, nullable=False),
        sa.Column(
            "boundary",
            geoalchemy2.types.Geography(
                geometry_type="POLYGON",
                srid=4326,
                from_text="ST_GeogFromText",
            ),
            nullable=True,
        ),
        sa.Column("area_m2", sa.Float(), nullable=False),
        sa.Column(
            "soil_type",
            sa.String(100),
            server_default=sa.text("'unknown'"),
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["farm_id"], ["farms.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_zones_farm_id", "zones", ["farm_id"])

    # vertices
    op.create_table(
        "vertices",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "farm_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "zone_id", postgresql.UUID(as_uuid=True), nullable=True
        ),
        sa.Column("vertex_type", ENUM_VERTEX_TYPE, nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column(
            "installed_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "last_seen_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["farm_id"], ["farms.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["zone_id"], ["zones.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_vertices_farm_id_type", "vertices", ["farm_id", "vertex_type"]
    )

    # hyperedges
    op.create_table(
        "hyperedges",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "farm_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("layer", ENUM_HYPEREDGE_LAYER, nullable=False),
        sa.Column(
            "vertex_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
        ),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["farm_id"], ["farms.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_hyperedges_farm_layer", "hyperedges", ["farm_id", "layer"]
    )
    op.create_index(
        "ix_hyperedges_vertex_ids_gin",
        "hyperedges",
        ["vertex_ids"],
        postgresql_using="gin",
    )

    # ── 3. Time-series tables ───────────────────────────────────────────

    # soil_readings
    op.create_table(
        "soil_readings",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "sensor_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "timestamp", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("moisture", sa.Float(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("conductivity", sa.Float(), nullable=True),
        sa.Column("ph", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["sensor_id"], ["vertices.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_soil_readings_sensor_ts",
        "soil_readings",
        ["sensor_id", sa.text("timestamp DESC")],
    )

    # weather_readings
    op.create_table(
        "weather_readings",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "station_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "timestamp", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("humidity", sa.Float(), nullable=False),
        sa.Column("precipitation_mm", sa.Float(), nullable=False),
        sa.Column("wind_speed", sa.Float(), nullable=True),
        sa.Column("wind_direction", sa.Float(), nullable=True),
        sa.Column("pressure_hpa", sa.Float(), nullable=True),
        sa.Column("et0", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["station_id"], ["vertices.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_weather_readings_station_ts",
        "weather_readings",
        ["station_id", sa.text("timestamp DESC")],
    )

    # irrigation_events
    op.create_table(
        "irrigation_events",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "valve_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "timestamp_start", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "timestamp_end", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("volume_liters", sa.Float(), nullable=True),
        sa.Column("trigger", ENUM_IRRIGATION_TRIGGER, nullable=False),
        sa.ForeignKeyConstraint(
            ["valve_id"], ["vertices.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_irrigation_events_valve_ts",
        "irrigation_events",
        ["valve_id", sa.text("timestamp_start DESC")],
    )

    # npk_samples
    op.create_table(
        "npk_samples",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "zone_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "timestamp", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("nitrogen_mg_kg", sa.Float(), nullable=False),
        sa.Column("phosphorus_mg_kg", sa.Float(), nullable=False),
        sa.Column("potassium_mg_kg", sa.Float(), nullable=False),
        sa.Column("organic_matter_pct", sa.Float(), nullable=True),
        sa.Column("source", ENUM_NPK_SOURCE, nullable=False),
        sa.ForeignKeyConstraint(
            ["zone_id"], ["zones.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_npk_samples_zone_ts",
        "npk_samples",
        ["zone_id", sa.text("timestamp DESC")],
    )

    # vision_events
    op.create_table(
        "vision_events",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "camera_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "crop_bed_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "timestamp", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("anomaly_type", ENUM_ANOMALY_TYPE, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("canopy_coverage_pct", sa.Float(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(
            ["camera_id"], ["vertices.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["crop_bed_id"], ["vertices.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_vision_events_camera_ts",
        "vision_events",
        ["camera_id", sa.text("timestamp DESC")],
    )

    # lighting_readings
    op.create_table(
        "lighting_readings",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "fixture_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "timestamp", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("par_umol", sa.Float(), nullable=False),
        sa.Column("dli_cumulative", sa.Float(), nullable=False),
        sa.Column("duty_cycle_pct", sa.Float(), nullable=False),
        sa.Column("spectrum_profile", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(
            ["fixture_id"], ["vertices.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_lighting_readings_fixture_ts",
        "lighting_readings",
        ["fixture_id", sa.text("timestamp DESC")],
    )

    # ── 4. Crop reference table ─────────────────────────────────────────

    op.create_table(
        "crop_profiles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("crop_type", sa.String(100), nullable=False),
        sa.Column("growth_stages", postgresql.JSONB(), nullable=False),
        sa.Column(
            "source",
            sa.String(100),
            server_default="FAO",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("crop_type"),
    )

    # ── 5. Auth tables ──────────────────────────────────────────────────

    # users
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(128), nullable=False),
        sa.Column(
            "role",
            ENUM_USER_ROLE,
            server_default="readonly",
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # api_keys
    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "user_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("key_hash", sa.String(128), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("scopes", postgresql.JSONB(), nullable=True),
        sa.Column(
            "expires_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])


def downgrade() -> None:
    # ── Drop tables in reverse dependency order ─────────────────────────
    op.drop_table("api_keys")
    op.drop_table("users")
    op.drop_table("crop_profiles")
    op.drop_table("lighting_readings")
    op.drop_table("vision_events")
    op.drop_table("npk_samples")
    op.drop_table("irrigation_events")
    op.drop_table("weather_readings")
    op.drop_table("soil_readings")
    op.drop_table("hyperedges")
    op.drop_table("vertices")
    op.drop_table("zones")
    op.drop_table("farms")

    # ── Drop enum types ─────────────────────────────────────────────────
    ENUM_USER_ROLE.drop(op.get_bind(), checkfirst=True)
    ENUM_ANOMALY_TYPE.drop(op.get_bind(), checkfirst=True)
    ENUM_NPK_SOURCE.drop(op.get_bind(), checkfirst=True)
    ENUM_IRRIGATION_TRIGGER.drop(op.get_bind(), checkfirst=True)
    ENUM_HYPEREDGE_LAYER.drop(op.get_bind(), checkfirst=True)
    ENUM_VERTEX_TYPE.drop(op.get_bind(), checkfirst=True)
    ENUM_ZONE_TYPE.drop(op.get_bind(), checkfirst=True)
    ENUM_FARM_TYPE.drop(op.get_bind(), checkfirst=True)
