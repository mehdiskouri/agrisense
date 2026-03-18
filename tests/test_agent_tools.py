from __future__ import annotations

import uuid
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.services.agent_tools import build_tools


class _FakeAnalytics:
    async def get_farm_status(self, _farm_id: uuid.UUID) -> dict[str, Any]:
        return {"ok": True}

    async def get_irrigation_schedule(
        self,
        _farm_id: uuid.UUID,
        horizon_days: int = 7,
    ) -> dict[str, Any]:
        return {"horizon_days": horizon_days}

    async def get_nutrient_report(self, _farm_id: uuid.UUID) -> dict[str, Any]:
        return {"npk": {"n": 1, "p": 1, "k": 1}}

    async def get_ensemble_yield_forecast(
        self,
        _farm_id: uuid.UUID,
        include_members: bool = False,
    ) -> dict[str, Any]:
        return {"include_members": include_members}

    async def get_active_alerts(self, _farm_id: uuid.UUID) -> list[dict[str, Any]]:
        return []

    async def get_zone_detail(self, _farm_id: uuid.UUID, _query: object) -> dict[str, Any]:
        return {"zone": "ok"}

    async def run_yield_backtest(self, _farm_id: uuid.UUID, n_folds: int = 5) -> dict[str, Any]:
        return {"n_folds": n_folds}


def _tool_names(settings: Settings) -> list[str]:
    tools = build_tools(uuid4(), _FakeAnalytics(), settings)  # type: ignore[arg-type]
    return [tool.name for tool in tools]


def test_backtest_tool_disabled_by_default() -> None:
    settings = Settings()
    names = _tool_names(settings)
    assert "run_yield_backtest" not in names


def test_backtest_tool_enabled_when_flag_on() -> None:
    settings = Settings(ask_enable_backtest_tool=True)
    names = _tool_names(settings)
    assert "run_yield_backtest" in names


def test_zone_detail_tool_can_be_disabled() -> None:
    settings = Settings(ask_enable_zone_detail_tool=False)
    names = _tool_names(settings)
    assert "get_zone_detail" not in names
