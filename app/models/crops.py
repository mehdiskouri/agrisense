"""CropProfile ORM model — agronomic reference table.

``growth_stages`` (JSONB) encodes per-stage requirements used by the Julia
crop requirements layer (Layer 5):

    [
        {
            "name": "seedling",
            "duration_days": 14,
            "optimal_moisture_range": [0.25, 0.35],
            "optimal_temp_range": [18, 26],
            "water_demand_mm_day": 3.5,
            "light_requirement_dli": 12,
            "npk_demand": {"n": 40, "p": 20, "k": 30}
        },
        ...
    ]
"""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CropProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Agronomic reference: crop type + growth-stage requirements."""

    __tablename__ = "crop_profiles"

    crop_type: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False
    )
    growth_stages: Mapped[dict] = mapped_column(JSONB, nullable=False)
    source: Mapped[str] = mapped_column(
        String(100), nullable=False, default="FAO", server_default="FAO"
    )

    def __repr__(self) -> str:
        return (
            f"<CropProfile id={self.id} crop={self.crop_type!r} "
            f"source={self.source!r}>"
        )
