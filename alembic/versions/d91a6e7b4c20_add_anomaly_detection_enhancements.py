"""add_anomaly_detection_enhancements

Revision ID: d91a6e7b4c20
Revises: c3a1b9d2e4f6
Create Date: 2026-03-18 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d91a6e7b4c20"
down_revision: str | None = "c3a1b9d2e4f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ENUM_ANOMALY_SEVERITY = postgresql.ENUM(
    "info",
    "warning",
    "critical",
    name="anomaly_severity",
    create_type=False,
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


def upgrade() -> None:
    ENUM_ANOMALY_SEVERITY.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "anomaly_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("farm_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("vertex_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("layer", sa.String(length=64), nullable=False),
        sa.Column("anomaly_type", sa.String(length=128), nullable=False),
        sa.Column("severity", ENUM_ANOMALY_SEVERITY, nullable=False),
        sa.Column("feature", sa.String(length=128), nullable=True),
        sa.Column("current_value", sa.Float(), nullable=True),
        sa.Column("rolling_mean", sa.Float(), nullable=True),
        sa.Column("rolling_std", sa.Float(), nullable=True),
        sa.Column("sigma_deviation", sa.Float(), nullable=True),
        sa.Column(
            "anomaly_rules",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "cross_layer_confirmed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "webhook_notified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
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
        sa.ForeignKeyConstraint(["farm_id"], ["farms.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["vertex_id"], ["vertices.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["zone_id"], ["zones.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_anomaly_events_farm_detected_at",
        "anomaly_events",
        ["farm_id", sa.text("detected_at DESC")],
    )
    op.create_index(
        "ix_anomaly_events_farm_severity",
        "anomaly_events",
        ["farm_id", "severity"],
    )
    op.create_index(
        "ix_anomaly_events_vertex_layer",
        "anomaly_events",
        ["vertex_id", "layer"],
    )

    op.create_table(
        "anomaly_thresholds",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("farm_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("vertex_type", ENUM_VERTEX_TYPE, nullable=False),
        sa.Column("layer", ENUM_HYPEREDGE_LAYER, nullable=True),
        sa.Column("sigma1", sa.Float(), nullable=False, server_default=sa.text("1")),
        sa.Column("sigma2", sa.Float(), nullable=False, server_default=sa.text("2")),
        sa.Column("sigma3", sa.Float(), nullable=False, server_default=sa.text("3")),
        sa.Column("min_history", sa.Integer(), nullable=False, server_default=sa.text("8")),
        sa.Column(
            "min_nan_run_outage",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("4"),
        ),
        sa.Column(
            "vision_anomaly_score_threshold",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.7"),
        ),
        sa.Column(
            "suppress_rule3_only",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
        sa.ForeignKeyConstraint(["farm_id"], ["farms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "farm_id",
            "vertex_type",
            "layer",
            name="uq_anomaly_thresholds_farm_vertex_layer",
        ),
    )

    op.create_table(
        "webhook_subscriptions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("farm_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("secret", sa.String(length=256), nullable=False),
        sa.Column(
            "event_types",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("retry_max", sa.Integer(), nullable=False, server_default=sa.text("3")),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status_code", sa.Integer(), nullable=True),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
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
        sa.ForeignKeyConstraint(["farm_id"], ["farms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_webhook_subscriptions_farm_active",
        "webhook_subscriptions",
        ["farm_id", "is_active"],
    )


def downgrade() -> None:
    op.drop_index("ix_webhook_subscriptions_farm_active", table_name="webhook_subscriptions")
    op.drop_table("webhook_subscriptions")

    op.drop_table("anomaly_thresholds")

    op.drop_index("ix_anomaly_events_vertex_layer", table_name="anomaly_events")
    op.drop_index("ix_anomaly_events_farm_severity", table_name="anomaly_events")
    op.drop_index("ix_anomaly_events_farm_detected_at", table_name="anomaly_events")
    op.drop_table("anomaly_events")

    ENUM_ANOMALY_SEVERITY.drop(op.get_bind(), checkfirst=True)
