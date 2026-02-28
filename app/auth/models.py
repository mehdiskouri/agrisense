"""User and APIKey ORM models for JWT + API-key authentication.

Users authenticate via email/password (JWT), or via hashed API keys for
machine-to-machine access with optional scopes and expiration.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import UserRoleEnum


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Application user — authenticates via email/password (JWT)."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True
    )
    hashed_password: Mapped[str] = mapped_column(
        String(128), nullable=False
    )
    role: Mapped[UserRoleEnum] = mapped_column(
        Enum(
            UserRoleEnum,
            name="user_role",
            create_constraint=False,
            native_enum=True,
        ),
        nullable=False,
        default=UserRoleEnum.readonly,
        server_default="readonly",
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        server_default=text("true"),
        nullable=False,
    )

    # ── Relationships ────────────────────────────────────────────────────
    api_keys: Mapped[list[APIKey]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role}>"


class APIKey(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Hashed API key for machine-to-machine access.

    ``key_hash`` stores bcrypt/argon2 hash — plaintext is shown only once
    at creation time.  ``scopes`` (JSONB) allows fine-grained permission
    control (e.g., {"read:farms": true, "write:ingest": true}).
    """

    __tablename__ = "api_keys"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    key_hash: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    scopes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        server_default=text("true"),
        nullable=False,
    )

    # ── Relationships ────────────────────────────────────────────────────
    user: Mapped[User] = relationship(back_populates="api_keys")

    def __repr__(self) -> str:
        return f"<APIKey id={self.id} name={self.name!r} user={self.user_id}>"
