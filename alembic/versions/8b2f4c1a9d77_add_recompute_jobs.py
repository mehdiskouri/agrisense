"""add_recompute_jobs

Revision ID: 8b2f4c1a9d77
Revises: 506ed2dedad4
Create Date: 2026-02-28 23:45:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "8b2f4c1a9d77"
down_revision: str | None = "506ed2dedad4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ENUM_JOB_STATUS = postgresql.ENUM(
	"queued",
	"running",
	"succeeded",
	"failed",
	name="job_status",
	create_type=False,
)


def upgrade() -> None:
	ENUM_JOB_STATUS.create(op.get_bind(), checkfirst=True)

	op.create_table(
		"recompute_jobs",
		sa.Column(
			"id",
			postgresql.UUID(as_uuid=True),
			server_default=sa.text("uuid_generate_v4()"),
			nullable=False,
		),
		sa.Column("farm_id", postgresql.UUID(as_uuid=True), nullable=False),
		sa.Column("status", ENUM_JOB_STATUS, nullable=False, server_default=sa.text("'queued'")),
		sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
		sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
		sa.Column("error", sa.String(length=2048), nullable=True),
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
	op.create_index("ix_recompute_jobs_farm_status", "recompute_jobs", ["farm_id", "status"])


def downgrade() -> None:
	op.drop_index("ix_recompute_jobs_farm_status", table_name="recompute_jobs")
	op.drop_table("recompute_jobs")
	ENUM_JOB_STATUS.drop(op.get_bind(), checkfirst=True)
