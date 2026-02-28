from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.schemas.analytics import (
	AlertsResponse,
	FarmStatusResponse,
	IrrigationScheduleResponse,
	ZoneAlerts,
	ZoneDetailQuery,
	ZoneDetailResponse,
	ZoneStatus,
)
from app.services import julia_bridge
from app.services.analytics_service import AnalyticsService
from app.services.farm_service import FarmService


@pytest.mark.asyncio
async def test_analytics_status_endpoint_success(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
	farm_id = uuid4()
	zone_id = uuid4()
	vertex_id = uuid4()

	async def fake_status(self: AnalyticsService, _farm_id: object) -> FarmStatusResponse:
		return FarmStatusResponse(
			farm_id=farm_id,
			generated_at=datetime.now(UTC),
			zones=[ZoneStatus(zone_id=zone_id, query_vertex_id=vertex_id, status={"soil": {"ok": True}})],
		)

	monkeypatch.setattr(AnalyticsService, "get_farm_status", fake_status)

	response = await client.get(f"/api/v1/analytics/{farm_id}/status")
	assert response.status_code == 200
	body = response.json()
	assert body["farm_id"] == str(farm_id)
	assert body["zones"][0]["zone_id"] == str(zone_id)


@pytest.mark.asyncio
async def test_analytics_zone_detail_endpoint_supports_vertex(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
	farm_id = uuid4()
	zone_id = uuid4()
	vertex_id = uuid4()

	async def fake_detail(self: AnalyticsService, _farm_id: object, _query: ZoneDetailQuery) -> ZoneDetailResponse:
		return ZoneDetailResponse(
			farm_id=farm_id,
			zone_id=zone_id,
			query_vertex_id=vertex_id,
			layers={"soil": {"moisture": 0.2}},
			cross_layer=[],
		)

	monkeypatch.setattr(AnalyticsService, "get_zone_detail", fake_detail)

	response = await client.get(f"/api/v1/analytics/{farm_id}/zones/{zone_id}?vertex_id={vertex_id}")
	assert response.status_code == 200
	body = response.json()
	assert body["query_vertex_id"] == str(vertex_id)
	assert "soil" in body["layers"]


@pytest.mark.asyncio
async def test_analytics_vertex_detail_endpoint(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
	farm_id = uuid4()
	vertex_id = uuid4()

	async def fake_detail(self: AnalyticsService, _farm_id: object, query: ZoneDetailQuery) -> ZoneDetailResponse:
		assert query.vertex_id == vertex_id
		return ZoneDetailResponse(
			farm_id=farm_id,
			zone_id=None,
			query_vertex_id=vertex_id,
			layers={"weather": {"ok": True}},
			cross_layer=[],
		)

	monkeypatch.setattr(AnalyticsService, "get_zone_detail", fake_detail)

	response = await client.get(f"/api/v1/analytics/{farm_id}/vertices/{vertex_id}")
	assert response.status_code == 200
	assert response.json()["query_vertex_id"] == str(vertex_id)


@pytest.mark.asyncio
async def test_irrigation_schedule_cache_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
	farm_id = uuid4()
	fake_db = object()
	fake_redis = SimpleNamespace(get=AsyncMock(return_value=None), setex=AsyncMock())
	service = AnalyticsService(fake_db, fake_redis)

	farm = SimpleNamespace(id=farm_id)
	graph = {"farm_id": str(farm_id)}

	async def fake_get_farm(self: FarmService, _farm_id: object) -> object:
		return farm

	async def fake_get_graph(self: FarmService, _farm_id: object) -> dict[str, object]:
		return graph

	calls = {"count": 0}

	def fake_schedule(_graph: dict[str, object], _horizon: int) -> list[dict[str, object]]:
		calls["count"] += 1
		return [{"zone_id": str(uuid4()), "volume_liters": 120.0}]

	monkeypatch.setattr(FarmService, "get_farm", fake_get_farm)
	monkeypatch.setattr(FarmService, "get_graph", fake_get_graph)
	monkeypatch.setattr(julia_bridge, "irrigation_schedule", fake_schedule)

	first = await service.get_irrigation_schedule(farm_id, horizon_days=7)
	assert isinstance(first, IrrigationScheduleResponse)
	assert first.cached is False
	assert calls["count"] == 1
	assert fake_redis.setex.await_count == 1

	fake_redis.get.return_value = '[{"zone_id":"z1","volume_liters":80.0}]'
	second = await service.get_irrigation_schedule(farm_id, horizon_days=7)
	assert second.cached is True
	assert calls["count"] == 1


@pytest.mark.asyncio
async def test_alerts_endpoint_and_openapi_contract(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
	farm_id = uuid4()
	zone_id = uuid4()

	async def fake_alerts(self: AnalyticsService, _farm_id: object) -> AlertsResponse:
		return AlertsResponse(
			farm_id=farm_id,
			generated_at=datetime.now(UTC),
			zones=[ZoneAlerts(zone_id=zone_id, alerts=[])],
		)

	monkeypatch.setattr(AnalyticsService, "get_active_alerts", fake_alerts)

	response = await client.get(f"/api/v1/analytics/{farm_id}/alerts")
	assert response.status_code == 200
	assert response.json()["zones"][0]["zone_id"] == str(zone_id)

	openapi = await client.get("/openapi.json")
	assert openapi.status_code == 200
	paths = openapi.json()["paths"]
	assert "/api/v1/analytics/{farm_id}/status" in paths
	assert "/api/v1/analytics/{farm_id}/zones/{zone_id}" in paths
	assert "/api/v1/analytics/{farm_id}/vertices/{vertex_id}" in paths
	assert "/api/v1/analytics/{farm_id}/irrigation/schedule" in paths
	assert "/api/v1/analytics/{farm_id}/nutrients/report" in paths
	assert "/api/v1/analytics/{farm_id}/yield/forecast" in paths
	assert "/api/v1/analytics/{farm_id}/alerts" in paths
