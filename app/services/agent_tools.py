"""LangChain tool definitions for farm analytics."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool, tool

from app.schemas.analytics import ZoneDetailQuery
from app.services.analytics_service import AnalyticsService


def _json_payload(payload: Any) -> str:
    """Serialize pydantic payloads into JSON for tool responses."""
    data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    return json.dumps(data, ensure_ascii=True)


def build_tools(farm_id: uuid.UUID, analytics: AnalyticsService) -> list[BaseTool]:
    """Create tool list with farm_id bound so the model cannot change target farm."""

    async def _get_farm_status() -> str:
        return _json_payload(await analytics.get_farm_status(farm_id))

    async def _get_irrigation_schedule(horizon_days: int = 7) -> str:
        return _json_payload(
            await analytics.get_irrigation_schedule(farm_id, horizon_days=horizon_days)
        )

    async def _get_nutrient_report() -> str:
        return _json_payload(await analytics.get_nutrient_report(farm_id))

    async def _get_yield_forecast(include_members: bool = False) -> str:
        return _json_payload(
            await analytics.get_ensemble_yield_forecast(
                farm_id,
                include_members=include_members,
            )
        )

    async def _get_active_alerts() -> str:
        return _json_payload(await analytics.get_active_alerts(farm_id))

    async def _get_zone_detail(zone_id: str) -> str:
        zone_uuid = uuid.UUID(zone_id)
        query = ZoneDetailQuery(zone_id=zone_uuid)
        return _json_payload(await analytics.get_zone_detail(farm_id, query))

    async def _run_yield_backtest(n_folds: int = 5) -> str:
        return _json_payload(await analytics.run_yield_backtest(farm_id, n_folds=n_folds))

    def _named_tool(name: str, description: str, coro: Callable[..., Awaitable[str]]) -> BaseTool:
        coro.__name__ = name
        coro.__doc__ = description
        return tool(name, return_direct=False)(coro)

    return [
        _named_tool(
            "get_farm_status",
            "Get overall farm status across all zones.",
            _get_farm_status,
        ),
        _named_tool(
            "get_irrigation_schedule",
            "Get irrigation schedule and water recommendations. Optional horizon_days.",
            _get_irrigation_schedule,
        ),
        _named_tool(
            "get_nutrient_report",
            "Get nutrient (NPK) report for the farm.",
            _get_nutrient_report,
        ),
        _named_tool(
            "get_yield_forecast",
            "Get ensemble yield forecast with optional ensemble members.",
            _get_yield_forecast,
        ),
        _named_tool(
            "get_active_alerts",
            "Get active alerts, anomalies, and risk indicators.",
            _get_active_alerts,
        ),
        _named_tool(
            "get_zone_detail",
            "Get detailed zone status and cross-layer links for a zone_id.",
            _get_zone_detail,
        ),
        _named_tool(
            "run_yield_backtest",
            "Run ensemble yield model backtest and return accuracy metrics.",
            _run_yield_backtest,
        ),
    ]
