"""ORM base class and mixins — all models inherit from Base."""

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base — shared MetaData registry for all models."""

    pass


class TimestampMixin:
    """Adds created_at / updated_at audit columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """Adds a UUID primary key with both Python and server-side defaults."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )


class TimeSeriesMixin:
    """BIGSERIAL PK + ingestion timestamp for high-volume sensor tables.

    Time-series tables do NOT use UUIDPrimaryKeyMixin (auto-increment is
    more efficient for append-heavy workloads) and do NOT use TimestampMixin
    (the sensor ``timestamp`` column IS the record time).  A single
    ``ingested_at`` column tracks ingestion lag.
    """

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
