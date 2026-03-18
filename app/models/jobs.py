"""Durable recompute job model for analytics refresh tasks."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import JobStatusEnum


class RecomputeJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Tracks lifecycle of background farm graph recomputation jobs."""

    __tablename__ = "recompute_jobs"
    __table_args__ = (Index("ix_recompute_jobs_farm_status", "farm_id", "status"),)

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[JobStatusEnum] = mapped_column(
        Enum(
            JobStatusEnum,
            name="job_status",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
        default=JobStatusEnum.queued,
        server_default=JobStatusEnum.queued.value,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    def __repr__(self) -> str:
        return f"<RecomputeJob id={self.id} farm={self.farm_id} status={self.status}>"


class BacktestJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Tracks async yield-backtest jobs with durable status and result payloads."""

    __tablename__ = "backtest_jobs"
    __table_args__ = (Index("ix_backtest_jobs_farm_status", "farm_id", "status"),)

    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("farms.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[JobStatusEnum] = mapped_column(
        Enum(
            JobStatusEnum,
            name="job_status",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
        default=JobStatusEnum.queued,
        server_default=JobStatusEnum.queued.value,
    )
    n_folds: Mapped[int] = mapped_column(Integer, nullable=False, default=5, server_default="5")
    min_history: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=24,
        server_default="24",
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    result: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<BacktestJob id={self.id} farm={self.farm_id} status={self.status}>"
