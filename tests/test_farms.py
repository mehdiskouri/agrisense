from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import FarmTypeEnum, VertexTypeEnum, ZoneTypeEnum
from app.schemas.farm import VisualizationResponse
from app.services import julia_bridge
from app.services.analytics_service import AnalyticsService
from app.services.farm_service import FarmService


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
    assert FarmService._resolve_zone_type(FarmTypeEnum.open_field, None) == ZoneTypeEnum.open_field
    assert FarmService._resolve_zone_type(FarmTypeEnum.greenhouse, None) == ZoneTypeEnum.greenhouse
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
async def test_register_sensor_endpoint(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    service = FarmService(cast(AsyncSession, SimpleNamespace()))
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


@pytest.mark.asyncio
async def test_visualization_endpoint_returns_200(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    vertex_id = str(uuid4())
    payload = VisualizationResponse(
        farm_id=farm_id,
        farm_name="Farm Viz",
        farm_type=FarmTypeEnum.greenhouse,
        generated_at=datetime.now(UTC),
        layers=[
            {
                "name": "soil",
                "color": "#8B4513",
                "feature_names": ["moisture"],
                "n_edges": 1,
                "n_vertices": 1,
            }
        ],
        nodes=[
            {
                "id": vertex_id,
                "type": "vertex",
                "vertex_type": "sensor",
                "features": {"soil": [0.1]},
                "layer_memberships": ["soil"],
            }
        ],
        links=[],
        alerts=[],
        cross_layer_summary=[],
    )

    async def fake_visualization(self: AnalyticsService, _farm_id: object) -> VisualizationResponse:
        return payload

    monkeypatch.setattr(AnalyticsService, "get_visualization", fake_visualization)

    response = await client.get(f"/api/v1/farms/{farm_id}/visualization")

    assert response.status_code == 200
    body = response.json()
    assert body["farm_name"] == "Farm Viz"
    assert body["nodes"][0]["id"] == vertex_id
    assert body["layers"][0]["name"] == "soil"


@pytest.mark.asyncio
async def test_visualization_node_structure(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    vertex_id = str(uuid4())
    hyperedge_id = "he:soil:e-1"

    async def fake_visualization(self: AnalyticsService, _farm_id: object) -> VisualizationResponse:
        return VisualizationResponse(
            farm_id=farm_id,
            farm_name="Farm Viz",
            farm_type=FarmTypeEnum.greenhouse,
            generated_at=datetime.now(UTC),
            layers=[],
            nodes=[
                {
                    "id": vertex_id,
                    "type": "vertex",
                    "vertex_type": "sensor",
                    "features": {"soil": [0.2, 0.3]},
                    "layer_memberships": ["soil"],
                },
                {
                    "id": hyperedge_id,
                    "type": "hyperedge",
                    "layer": "soil",
                    "metadata": {"kind": "zone"},
                    "member_count": 1,
                    "layer_memberships": ["soil"],
                },
            ],
            links=[{"source": hyperedge_id, "target": vertex_id, "layer": "soil"}],
            alerts=[],
            cross_layer_summary=[],
        )

    monkeypatch.setattr(AnalyticsService, "get_visualization", fake_visualization)
    response = await client.get(f"/api/v1/farms/{farm_id}/visualization")

    assert response.status_code == 200
    nodes = response.json()["nodes"]
    vertex = next(item for item in nodes if item["type"] == "vertex")
    hub = next(item for item in nodes if item["type"] == "hyperedge")
    assert "soil" in vertex["features"]
    assert hub["layer"] == "soil"


@pytest.mark.asyncio
async def test_visualization_link_integrity(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_visualization(self: AnalyticsService, _farm_id: object) -> VisualizationResponse:
        return VisualizationResponse(
            farm_id=farm_id,
            farm_name="Farm Viz",
            farm_type=FarmTypeEnum.greenhouse,
            generated_at=datetime.now(UTC),
            layers=[],
            nodes=[
                {
                    "id": "vertex-1",
                    "type": "vertex",
                    "vertex_type": "sensor",
                    "features": {},
                    "layer_memberships": ["soil"],
                },
                {
                    "id": "he:soil:e-1",
                    "type": "hyperedge",
                    "layer": "soil",
                    "metadata": {},
                    "member_count": 1,
                    "layer_memberships": ["soil"],
                },
            ],
            links=[{"source": "he:soil:e-1", "target": "vertex-1", "layer": "soil"}],
            alerts=[],
            cross_layer_summary=[],
        )

    monkeypatch.setattr(AnalyticsService, "get_visualization", fake_visualization)
    response = await client.get(f"/api/v1/farms/{farm_id}/visualization")

    assert response.status_code == 200
    body = response.json()
    node_ids = {node["id"] for node in body["nodes"]}
    for link in body["links"]:
        assert link["source"] in node_ids
        assert link["target"] in node_ids


@pytest.mark.asyncio
async def test_visualization_layer_metadata(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_visualization(self: AnalyticsService, _farm_id: object) -> VisualizationResponse:
        return VisualizationResponse(
            farm_id=farm_id,
            farm_name="Farm Viz",
            farm_type=FarmTypeEnum.greenhouse,
            generated_at=datetime.now(UTC),
            layers=[
                {
                    "name": "soil",
                    "color": "#8B4513",
                    "feature_names": ["moisture", "temperature"],
                    "n_edges": 2,
                    "n_vertices": 3,
                },
                {
                    "name": "weather",
                    "color": "#87CEEB",
                    "feature_names": ["temperature"],
                    "n_edges": 1,
                    "n_vertices": 1,
                },
            ],
            nodes=[],
            links=[],
            alerts=[],
            cross_layer_summary=[],
        )

    monkeypatch.setattr(AnalyticsService, "get_visualization", fake_visualization)
    response = await client.get(f"/api/v1/farms/{farm_id}/visualization")

    assert response.status_code == 200
    layers = {item["name"]: item for item in response.json()["layers"]}
    assert layers["soil"]["color"] == "#8B4513"
    assert layers["soil"]["n_edges"] == 2
    assert layers["weather"]["feature_names"] == ["temperature"]


@pytest.mark.asyncio
async def test_visualization_missing_farm_maps_to_404(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_visualization(self: AnalyticsService, _farm_id: object) -> VisualizationResponse:
        raise LookupError("Farm not found")

    monkeypatch.setattr(AnalyticsService, "get_visualization", fake_visualization)
    response = await client.get(f"/api/v1/farms/{uuid4()}/visualization")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_visualization_openapi_contract(client: AsyncClient) -> None:
    openapi = await client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/api/v1/farms/{farm_id}/visualization" in openapi.json()["paths"]


@pytest.mark.asyncio
async def test_static_dashboard_html_served(client: AsyncClient) -> None:

    page = await client.get("/static/dashboard.html")
    assert page.status_code == 200
    assert "text/html" in page.headers.get("content-type", "")
