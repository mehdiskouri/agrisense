from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.models.enums import FarmTypeEnum, VertexTypeEnum, ZoneTypeEnum
from app.services.farm_service import FarmService
from app.services import julia_bridge


def _farm_obj() -> SimpleNamespace:
    now = datetime.now(UTC)
    zone = SimpleNamespace(
        id=uuid4(),
        farm_id=uuid4(),
        name="Zone A",
        zone_type=ZoneTypeEnum.greenhouse,
        area_m2=100.0,
        soil_type="loam",
        metadata_={"x": 1},
        created_at=now,
        updated_at=now,
    )
    vertex = SimpleNamespace(
        id=uuid4(),
        farm_id=zone.farm_id,
        zone_id=zone.id,
        vertex_type=VertexTypeEnum.sensor,
        config={"sensor_type": "soil"},
        installed_at=now,
        last_seen_at=now,
        created_at=now,
        updated_at=now,
    )
    return SimpleNamespace(
        id=zone.farm_id,
        name="Farm 1",
        farm_type=FarmTypeEnum.greenhouse,
        timezone="UTC",
        model_overrides=None,
        created_at=now,
        updated_at=now,
        zones=[zone],
        vertices=[vertex],
    )


@pytest.mark.asyncio
async def test_create_and_get_farm(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    farm = _farm_obj()

    async def fake_create(self: FarmService, payload: object) -> object:
        return farm

    async def fake_get(self: FarmService, farm_id: object) -> object:
        return farm

    monkeypatch.setattr(FarmService, "create_farm", fake_create)
    monkeypatch.setattr(FarmService, "get_farm", fake_get)

    response = await client.post(
        "/api/v1/farms",
        json={"name": "Farm 1", "farm_type": "greenhouse", "timezone": "UTC"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Farm 1"
    assert body["farm_type"] == "greenhouse"
    assert "vision" in body["active_layers"]


@pytest.mark.asyncio
async def test_list_farms(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    farm = _farm_obj()

    async def fake_list(self: FarmService) -> list[object]:
        return [farm]

    monkeypatch.setattr(FarmService, "list_farms", fake_list)

    response = await client.get("/api/v1/farms")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["farm_type"] == "greenhouse"


@pytest.mark.asyncio
async def test_get_graph_endpoint(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_state = {
        "farm_id": str(uuid4()),
        "n_vertices": 2,
        "vertex_index": {"v1": 1, "v2": 2},
        "layers": {},
    }

    async def fake_get_graph(self: FarmService, farm_id: object) -> dict[str, object]:
        return graph_state

    monkeypatch.setattr(FarmService, "get_graph", fake_get_graph)

    response = await client.get(f"/api/v1/farms/{uuid4()}/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["n_vertices"] == 2
    assert body["vertex_index"]["v1"] == 1


def test_zone_resolution_rules() -> None:
    assert (
        FarmService._resolve_zone_type(FarmTypeEnum.open_field, None)
        == ZoneTypeEnum.open_field
    )
    assert (
        FarmService._resolve_zone_type(FarmTypeEnum.greenhouse, None)
        == ZoneTypeEnum.greenhouse
    )
    with pytest.raises(ValueError):
        FarmService._resolve_zone_type(FarmTypeEnum.hybrid, None)


@pytest.mark.asyncio
async def test_add_zone_endpoint(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    farm_id = uuid4()
    zone = SimpleNamespace(
        id=uuid4(),
        farm_id=farm_id,
        name="Zone B",
        zone_type=ZoneTypeEnum.open_field,
        area_m2=80.0,
        soil_type="clay",
        metadata_={"irrigated": True},
        created_at=now,
        updated_at=now,
    )

    async def fake_add_zone(self: FarmService, _farm_id: object, _payload: object) -> object:
        return zone

    monkeypatch.setattr(FarmService, "add_zone", fake_add_zone)

    response = await client.post(
        f"/api/v1/farms/{farm_id}/zones",
        json={"name": "Zone B", "zone_type": "open_field", "area_m2": 80.0},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Zone B"
    assert body["zone_type"] == "open_field"


@pytest.mark.asyncio
async def test_register_sensor_endpoint(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    farm_id = uuid4()
    vertex = SimpleNamespace(
        id=uuid4(),
        farm_id=farm_id,
        zone_id=uuid4(),
        vertex_type=VertexTypeEnum.sensor,
        config={"sensor_type": "soil"},
        installed_at=now,
        last_seen_at=now,
        created_at=now,
        updated_at=now,
    )

    async def fake_register(self: FarmService, _farm_id: object, _payload: object) -> object:
        return vertex

    monkeypatch.setattr(FarmService, "register_vertex", fake_register)

    response = await client.post(
        f"/api/v1/farms/{farm_id}/sensors",
        json={
            "vertex_type": "sensor",
            "zone_id": str(vertex.zone_id),
            "config": {"sensor_type": "soil"},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["vertex_type"] == "sensor"


@pytest.mark.asyncio
async def test_register_sensor_validation_error(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_register(self: FarmService, _farm_id: object, _payload: object) -> object:
        raise ValueError("invalid vertex")

    monkeypatch.setattr(FarmService, "register_vertex", fake_register)

    response = await client.post(
        f"/api/v1/farms/{uuid4()}/sensors",
        json={"vertex_type": "sensor", "config": {}},
    )

    assert response.status_code == 400
    assert "invalid vertex" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_farm_not_found_maps_to_404(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(self: FarmService, farm_id: object) -> object:
        raise LookupError(f"Farm {farm_id} not found")

    monkeypatch.setattr(FarmService, "get_farm", fake_get)

    response = await client.get(f"/api/v1/farms/{uuid4()}")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_query_zone_status_uses_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    service = FarmService(SimpleNamespace())
    farm_id = uuid4()
    zone_id = uuid4()

    async def fake_get_graph(_farm_id: object) -> dict[str, object]:
        return {"farm_id": str(farm_id)}

    async def fake_resolve(_farm_id: object, _zone_id: object) -> str:
        return "resolved-vertex"

    monkeypatch.setattr(service, "get_graph", fake_get_graph)
    monkeypatch.setattr(service, "resolve_zone_query_vertex_id", fake_resolve)
    monkeypatch.setattr(
        julia_bridge,
        "query_farm_status",
        lambda graph, zone: {"graph": graph, "zone": zone},
    )

    status = await service.query_zone_status(farm_id, zone_id)

    assert status["zone"] == "resolved-vertex"
