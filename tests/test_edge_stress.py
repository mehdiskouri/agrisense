"""Comprehensive edge-case and stress tests across all API endpoints.

Covers: Pydantic 422 validation, error-path mapping (400/404/500),
boundary values, RBAC per-endpoint, auth edge cases, schema contracts,
cross-cutting middleware, and response format consistency.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import AsyncClient

from app.auth.dependencies import AuthPrincipal, get_current_user
from app.auth.jwt import AuthError, create_access_token, create_refresh_token, decode_token
from app.main import app
from app.models.enums import (
    FarmTypeEnum,
    UserRoleEnum,
    VertexTypeEnum,
    ZoneTypeEnum,
)
from app.schemas.analytics import (
    AlertsResponse,
    FarmStatusResponse,
    IrrigationScheduleResponse,
    NutrientReportResponse,
    YieldForecastResponse,
    ZoneAlerts,
)
from app.schemas.ask import AskLanguage, AskResponse
from app.schemas.ingest import BulkIngestReceipt, IngestReceipt
from app.schemas.jobs import JobCreateResponse, JobStatusResponse
from app.services.analytics_service import AnalyticsService
from app.services.farm_service import FarmService
from app.services.ingest_service import IngestService
from app.services.jobs_service import JobsService
from app.services.llm_service import LLMService

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

_NOW = datetime.now(UTC)


def _farm_obj(**overrides: Any) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        name="Edge Farm",
        farm_type=FarmTypeEnum.greenhouse,
        timezone="UTC",
        model_overrides=None,
        created_at=_NOW,
        updated_at=_NOW,
        zones=[],
        vertices=[],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _zone_obj(farm_id: uuid.UUID | None = None, **kw: Any) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        farm_id=farm_id or uuid.uuid4(),
        name="Zone Edge",
        zone_type=ZoneTypeEnum.greenhouse,
        area_m2=50.0,
        soil_type="loam",
        metadata_=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _vertex_obj(farm_id: uuid.UUID | None = None, **kw: Any) -> SimpleNamespace:
    defaults = dict(
        id=uuid.uuid4(),
        farm_id=farm_id or uuid.uuid4(),
        zone_id=uuid.uuid4(),
        vertex_type=VertexTypeEnum.sensor,
        config={"sensor_type": "soil"},
        installed_at=_NOW,
        last_seen_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# ──────────────────────────────────────────────────────────────────────────
# 1. FARMS — Pydantic 422 validation
# ──────────────────────────────────────────────────────────────────────────


class TestFarmsValidation:
    """Pydantic schema validation edge cases for /farms endpoints."""

    @pytest.mark.asyncio
    async def test_create_farm_empty_name_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "", "farm_type": "greenhouse", "timezone": "UTC"},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body

    @pytest.mark.asyncio
    async def test_create_farm_name_too_long_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "x" * 256, "farm_type": "greenhouse", "timezone": "UTC"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_farm_invalid_farm_type_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "Test", "farm_type": "underwater", "timezone": "UTC"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_farm_missing_name_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/farms",
            json={"farm_type": "greenhouse"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_farm_missing_farm_type_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "Test"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_farm_empty_body_422(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/farms", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_farm_no_body_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/farms", content=b"", headers={"content-type": "application/json"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_farm_empty_timezone_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "Test", "farm_type": "greenhouse", "timezone": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_add_zone_area_zero_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "Z", "area_m2": 0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_add_zone_area_negative_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "Z", "area_m2": -5.0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_add_zone_empty_name_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "", "area_m2": 10.0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_add_zone_name_too_long_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "z" * 256, "area_m2": 10.0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_register_sensor_invalid_vertex_type_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/sensors",
            json={"vertex_type": "teleporter"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_register_sensor_missing_vertex_type_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/sensors",
            json={},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_farm_non_uuid_path_422(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/farms/not-a-uuid")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_graph_non_uuid_422(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/farms/not-a-uuid/graph")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_add_zone_invalid_zone_type_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "Z", "area_m2": 10.0, "zone_type": "swamp"},
        )
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────
# 2. FARMS — Error path mapping (400/404/500)
# ──────────────────────────────────────────────────────────────────────────


class TestFarmsErrorMapping:
    """Verify _map_error correctly routes ValueError→400, LookupError→404, other→500."""

    @pytest.mark.asyncio
    async def test_create_farm_value_error_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, payload: object) -> object:
            raise ValueError("bad payload")

        monkeypatch.setattr(FarmService, "create_farm", boom)
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "F", "farm_type": "greenhouse"},
        )
        assert resp.status_code == 400
        assert "bad payload" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_farm_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, payload: object) -> object:
            raise LookupError("farm config not found")

        monkeypatch.setattr(FarmService, "create_farm", boom)
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "F", "farm_type": "greenhouse"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_farm_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, payload: object) -> object:
            raise RuntimeError("DB on fire")

        monkeypatch.setattr(FarmService, "create_farm", boom)
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "F", "farm_type": "greenhouse"},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Unexpected farm service failure"

    @pytest.mark.asyncio
    async def test_list_farms_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService) -> object:
            raise RuntimeError("list exploded")

        monkeypatch.setattr(FarmService, "list_farms", boom)
        resp = await client.get("/api/v1/farms")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_add_zone_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, _fid: object, _p: object) -> object:
            raise LookupError("farm not found")

        monkeypatch.setattr(FarmService, "add_zone", boom)
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "Z", "area_m2": 10.0},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_zone_value_error_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, _fid: object, _p: object) -> object:
            raise ValueError("zone conflict")

        monkeypatch.setattr(FarmService, "add_zone", boom)
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "Z", "area_m2": 10.0},
        )
        assert resp.status_code == 400
        assert "zone conflict" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_graph_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, _fid: object) -> object:
            raise LookupError("graph not found")

        monkeypatch.setattr(FarmService, "get_graph", boom)
        resp = await client.get(f"/api/v1/farms/{uuid.uuid4()}/graph")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_graph_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, _fid: object) -> object:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(FarmService, "get_graph", boom)
        resp = await client.get(f"/api/v1/farms/{uuid.uuid4()}/graph")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_register_vertex_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, _fid: object, _p: object) -> object:
            raise LookupError("farm not found")

        monkeypatch.setattr(FarmService, "register_vertex", boom)
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/sensors",
            json={"vertex_type": "sensor"},
        )
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────────
# 3. FARMS — Successful response shape assertions
# ──────────────────────────────────────────────────────────────────────────


class TestFarmsResponseShape:
    """Verify response JSON has all expected fields."""

    @pytest.mark.asyncio
    async def test_create_farm_response_has_all_fields(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm = _farm_obj()

        async def fake_create(self: FarmService, p: object) -> object:
            return farm

        async def fake_get(self: FarmService, fid: object) -> object:
            return farm

        monkeypatch.setattr(FarmService, "create_farm", fake_create)
        monkeypatch.setattr(FarmService, "get_farm", fake_get)

        resp = await client.post(
            "/api/v1/farms",
            json={"name": "F", "farm_type": "greenhouse"},
        )
        assert resp.status_code == 201
        body = resp.json()
        required_keys = {
            "id",
            "name",
            "farm_type",
            "timezone",
            "created_at",
            "updated_at",
            "active_layers",
            "zones",
            "vertices",
        }
        assert required_keys.issubset(set(body.keys()))

    @pytest.mark.asyncio
    async def test_list_farms_empty(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_list(self: FarmService) -> list[object]:
            return []

        monkeypatch.setattr(FarmService, "list_farms", fake_list)

        resp = await client.get("/api/v1/farms")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []

    @pytest.mark.asyncio
    async def test_graph_response_shape(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_graph(self: FarmService, fid: object) -> dict[str, object]:
            return {
                "farm_id": str(uuid.uuid4()),
                "n_vertices": 0,
                "vertex_index": {},
                "layers": {},
            }

        monkeypatch.setattr(FarmService, "get_graph", fake_graph)
        resp = await client.get(f"/api/v1/farms/{uuid.uuid4()}/graph")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"farm_id", "n_vertices", "vertex_index", "layers"}


# ──────────────────────────────────────────────────────────────────────────
# 4. INGEST — All endpoints + validation + error paths
# ──────────────────────────────────────────────────────────────────────────


class TestIngestEndpoints:
    """Test all ingest endpoints — happy path, validation, and error mapping."""

    @pytest.mark.asyncio
    async def test_ingest_irrigation_success(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(self: IngestService, _fid: object, _events: object) -> IngestReceipt:
            return IngestReceipt(
                farm_id=farm_id,
                layer="irrigation",
                status="ok",
                inserted_count=1,
            )

        monkeypatch.setattr(IngestService, "ingest_irrigation", fake)
        resp = await client.post(
            "/api/v1/ingest/irrigation",
            json={
                "farm_id": str(farm_id),
                "events": [
                    {
                        "valve_id": str(uuid.uuid4()),
                        "timestamp_start": _NOW.isoformat(),
                        "trigger": "manual",
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["layer"] == "irrigation"

    @pytest.mark.asyncio
    async def test_ingest_npk_success(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(self: IngestService, _fid: object, _samples: object) -> IngestReceipt:
            return IngestReceipt(farm_id=farm_id, layer="npk", status="ok", inserted_count=2)

        monkeypatch.setattr(IngestService, "ingest_npk", fake)
        resp = await client.post(
            "/api/v1/ingest/npk",
            json={
                "farm_id": str(farm_id),
                "samples": [
                    {
                        "zone_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "nitrogen_mg_kg": 14.2,
                        "phosphorus_mg_kg": 8.1,
                        "potassium_mg_kg": 21.0,
                        "source": "lab",
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["layer"] == "npk"
        assert resp.json()["inserted_count"] == 2

    @pytest.mark.asyncio
    async def test_ingest_vision_success(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(self: IngestService, _fid: object, _events: object) -> IngestReceipt:
            return IngestReceipt(farm_id=farm_id, layer="vision", status="ok", inserted_count=1)

        monkeypatch.setattr(IngestService, "ingest_vision", fake)
        resp = await client.post(
            "/api/v1/ingest/vision",
            json={
                "farm_id": str(farm_id),
                "events": [
                    {
                        "camera_id": str(uuid.uuid4()),
                        "crop_bed_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "anomaly_type": "pest",
                        "confidence": 0.95,
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["layer"] == "vision"

    @pytest.mark.asyncio
    async def test_ingest_soil_empty_readings_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={"farm_id": str(uuid.uuid4()), "readings": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_weather_empty_readings_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/weather",
            json={"farm_id": str(uuid.uuid4()), "readings": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_irrigation_empty_events_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/irrigation",
            json={"farm_id": str(uuid.uuid4()), "events": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_npk_empty_samples_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/npk",
            json={"farm_id": str(uuid.uuid4()), "samples": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_vision_empty_events_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/vision",
            json={"farm_id": str(uuid.uuid4()), "events": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_soil_missing_farm_id_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={
                "readings": [
                    {
                        "sensor_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "moisture": 0.3,
                        "temperature": 24.0,
                    }
                ]
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_soil_malformed_farm_id_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={
                "farm_id": "not-a-uuid",
                "readings": [
                    {
                        "sensor_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "moisture": 0.3,
                        "temperature": 24.0,
                    }
                ],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_soil_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, _r: object) -> object:
            raise LookupError("sensor not found")

        monkeypatch.setattr(IngestService, "ingest_soil", boom)
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={
                "farm_id": str(uuid.uuid4()),
                "readings": [
                    {
                        "sensor_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "moisture": 0.3,
                        "temperature": 24.0,
                    }
                ],
            },
        )
        assert resp.status_code == 404
        assert "sensor not found" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_ingest_soil_value_error_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, _r: object) -> object:
            raise ValueError("layer not active")

        monkeypatch.setattr(IngestService, "ingest_soil", boom)
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={
                "farm_id": str(uuid.uuid4()),
                "readings": [
                    {
                        "sensor_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "moisture": 0.3,
                        "temperature": 24.0,
                    }
                ],
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_soil_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, _r: object) -> object:
            raise RuntimeError("DB exploded")

        monkeypatch.setattr(IngestService, "ingest_soil", boom)
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={
                "farm_id": str(uuid.uuid4()),
                "readings": [
                    {
                        "sensor_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "moisture": 0.3,
                        "temperature": 24.0,
                    }
                ],
            },
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "ingest failure"

    @pytest.mark.asyncio
    async def test_ingest_weather_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, _r: object) -> object:
            raise LookupError("station not found")

        monkeypatch.setattr(IngestService, "ingest_weather", boom)
        resp = await client.post(
            "/api/v1/ingest/weather",
            json={
                "farm_id": str(uuid.uuid4()),
                "readings": [
                    {
                        "station_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "temperature": 20.0,
                        "humidity": 55.0,
                        "precipitation_mm": 0.0,
                    }
                ],
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ingest_irrigation_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, _ev: object) -> object:
            raise RuntimeError("pipe burst")

        monkeypatch.setattr(IngestService, "ingest_irrigation", boom)
        resp = await client.post(
            "/api/v1/ingest/irrigation",
            json={
                "farm_id": str(uuid.uuid4()),
                "events": [
                    {
                        "valve_id": str(uuid.uuid4()),
                        "timestamp_start": _NOW.isoformat(),
                        "trigger": "manual",
                    }
                ],
            },
        )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_ingest_npk_value_error_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, _s: object) -> object:
            raise ValueError("duplicate zone sample")

        monkeypatch.setattr(IngestService, "ingest_npk", boom)
        resp = await client.post(
            "/api/v1/ingest/npk",
            json={
                "farm_id": str(uuid.uuid4()),
                "samples": [
                    {
                        "zone_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "nitrogen_mg_kg": 10.0,
                        "phosphorus_mg_kg": 5.0,
                        "potassium_mg_kg": 20.0,
                        "source": "lab",
                    }
                ],
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_ingest_vision_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, _ev: object) -> object:
            raise LookupError("camera not found")

        monkeypatch.setattr(IngestService, "ingest_vision", boom)
        resp = await client.post(
            "/api/v1/ingest/vision",
            json={
                "farm_id": str(uuid.uuid4()),
                "events": [
                    {
                        "camera_id": str(uuid.uuid4()),
                        "crop_bed_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "anomaly_type": "none",
                        "confidence": 0.1,
                    }
                ],
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ingest_bulk_all_empty_ok(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(self: IngestService, _fid: object, **kw: object) -> BulkIngestReceipt:
            return BulkIngestReceipt(farm_id=farm_id, status="ok", inserted_count=0, failed_count=0)

        monkeypatch.setattr(IngestService, "ingest_bulk", fake)
        resp = await client.post(
            "/api/v1/ingest/bulk",
            json={"farm_id": str(farm_id)},
        )
        assert resp.status_code == 200
        assert resp.json()["inserted_count"] == 0

    @pytest.mark.asyncio
    async def test_ingest_bulk_missing_farm_id_422(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/ingest/bulk", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_bulk_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: IngestService, _fid: object, **kw: object) -> object:
            raise RuntimeError("bulk boom")

        monkeypatch.setattr(IngestService, "ingest_bulk", boom)
        resp = await client.post(
            "/api/v1/ingest/bulk",
            json={"farm_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_ingest_soil_invalid_reading_fields_422(self, client: AsyncClient) -> None:
        """Missing required fields inside a reading."""
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={
                "farm_id": str(uuid.uuid4()),
                "readings": [{"sensor_id": str(uuid.uuid4())}],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_npk_invalid_source_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/npk",
            json={
                "farm_id": str(uuid.uuid4()),
                "samples": [
                    {
                        "zone_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "nitrogen_mg_kg": 10.0,
                        "phosphorus_mg_kg": 5.0,
                        "potassium_mg_kg": 20.0,
                        "source": "magic",
                    }
                ],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_irrigation_invalid_trigger_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/irrigation",
            json={
                "farm_id": str(uuid.uuid4()),
                "events": [
                    {
                        "valve_id": str(uuid.uuid4()),
                        "timestamp_start": _NOW.isoformat(),
                        "trigger": "telepathy",
                    }
                ],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_vision_invalid_anomaly_type_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ingest/vision",
            json={
                "farm_id": str(uuid.uuid4()),
                "events": [
                    {
                        "camera_id": str(uuid.uuid4()),
                        "crop_bed_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "anomaly_type": "alien_invasion",
                        "confidence": 0.5,
                    }
                ],
            },
        )
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────
# 5. ANALYTICS — All endpoints + error paths + boundary values
# ──────────────────────────────────────────────────────────────────────────


class TestAnalyticsEndpoints:
    """All analytics endpoints: happy path, error mapping, boundary conditions."""

    @pytest.mark.asyncio
    async def test_irrigation_schedule_success(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(
            self: AnalyticsService, _fid: object, horizon_days: int = 7
        ) -> IrrigationScheduleResponse:
            return IrrigationScheduleResponse(
                farm_id=farm_id,
                horizon_days=horizon_days,
                cached=False,
                generated_at=_NOW,
                items=[{"zone_id": str(uuid.uuid4()), "volume_liters": 100.0}],
            )

        monkeypatch.setattr(AnalyticsService, "get_irrigation_schedule", fake)
        resp = await client.get(f"/api/v1/analytics/{farm_id}/irrigation/schedule")
        assert resp.status_code == 200
        body = resp.json()
        assert body["horizon_days"] == 7
        assert body["cached"] is False
        assert len(body["items"]) == 1

    @pytest.mark.asyncio
    async def test_nutrient_report_success(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(self: AnalyticsService, _fid: object) -> NutrientReportResponse:
            return NutrientReportResponse(
                farm_id=farm_id,
                generated_at=_NOW,
                items=[{"zone_id": str(uuid.uuid4()), "nitrogen_deficit": 3.2}],
            )

        monkeypatch.setattr(AnalyticsService, "get_nutrient_report", fake)
        resp = await client.get(f"/api/v1/analytics/{farm_id}/nutrients/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["farm_id"] == str(farm_id)
        assert len(body["items"]) == 1

    @pytest.mark.asyncio
    async def test_yield_forecast_success(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(self: AnalyticsService, _fid: object) -> YieldForecastResponse:
            return YieldForecastResponse(
                farm_id=farm_id,
                generated_at=_NOW,
                items=[{"zone_id": str(uuid.uuid4()), "expected_kg": 450.0}],
            )

        monkeypatch.setattr(AnalyticsService, "get_yield_forecast", fake)
        resp = await client.get(f"/api/v1/analytics/{farm_id}/yield/forecast")
        assert resp.status_code == 200
        assert resp.json()["items"][0]["expected_kg"] == 450.0

    @pytest.mark.asyncio
    async def test_irrigation_schedule_horizon_min_boundary(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(
            self: AnalyticsService, _fid: object, horizon_days: int = 7
        ) -> IrrigationScheduleResponse:
            return IrrigationScheduleResponse(
                farm_id=farm_id,
                horizon_days=horizon_days,
                cached=False,
                generated_at=_NOW,
                items=[],
            )

        monkeypatch.setattr(AnalyticsService, "get_irrigation_schedule", fake)
        resp = await client.get(f"/api/v1/analytics/{farm_id}/irrigation/schedule?horizon_days=1")
        assert resp.status_code == 200
        assert resp.json()["horizon_days"] == 1

    @pytest.mark.asyncio
    async def test_irrigation_schedule_horizon_max_boundary(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(
            self: AnalyticsService, _fid: object, horizon_days: int = 7
        ) -> IrrigationScheduleResponse:
            return IrrigationScheduleResponse(
                farm_id=farm_id,
                horizon_days=horizon_days,
                cached=False,
                generated_at=_NOW,
                items=[],
            )

        monkeypatch.setattr(AnalyticsService, "get_irrigation_schedule", fake)
        resp = await client.get(f"/api/v1/analytics/{farm_id}/irrigation/schedule?horizon_days=30")
        assert resp.status_code == 200
        assert resp.json()["horizon_days"] == 30

    @pytest.mark.asyncio
    async def test_irrigation_schedule_horizon_below_min_422(self, client: AsyncClient) -> None:
        resp = await client.get(
            f"/api/v1/analytics/{uuid.uuid4()}/irrigation/schedule?horizon_days=0"
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_irrigation_schedule_horizon_above_max_422(self, client: AsyncClient) -> None:
        resp = await client.get(
            f"/api/v1/analytics/{uuid.uuid4()}/irrigation/schedule?horizon_days=31"
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_irrigation_schedule_horizon_negative_422(self, client: AsyncClient) -> None:
        resp = await client.get(
            f"/api/v1/analytics/{uuid.uuid4()}/irrigation/schedule?horizon_days=-1"
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_analytics_status_non_uuid_422(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/analytics/not-a-uuid/status")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_analytics_zone_detail_non_uuid_422(self, client: AsyncClient) -> None:
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/zones/not-a-uuid")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_analytics_status_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object) -> object:
            raise LookupError("farm not found")

        monkeypatch.setattr(AnalyticsService, "get_farm_status", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/status")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_analytics_status_value_error_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object) -> object:
            raise ValueError("bad farm state")

        monkeypatch.setattr(AnalyticsService, "get_farm_status", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/status")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_analytics_status_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object) -> object:
            raise RuntimeError("analytics meltdown")

        monkeypatch.setattr(AnalyticsService, "get_farm_status", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/status")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "analytics failure"

    @pytest.mark.asyncio
    async def test_irrigation_schedule_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object, **kw: object) -> object:
            raise LookupError("farm gone")

        monkeypatch.setattr(AnalyticsService, "get_irrigation_schedule", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/irrigation/schedule")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_nutrient_report_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object) -> object:
            raise RuntimeError("nutrient boom")

        monkeypatch.setattr(AnalyticsService, "get_nutrient_report", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/nutrients/report")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_yield_forecast_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object) -> object:
            raise LookupError("farm not found")

        monkeypatch.setattr(AnalyticsService, "get_yield_forecast", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/yield/forecast")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_alerts_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object) -> object:
            raise LookupError("farm not found")

        monkeypatch.setattr(AnalyticsService, "get_active_alerts", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/alerts")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_alerts_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object) -> object:
            raise RuntimeError("alert crash")

        monkeypatch.setattr(AnalyticsService, "get_active_alerts", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/alerts")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_zone_detail_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object, _q: object) -> object:
            raise LookupError("zone not found")

        monkeypatch.setattr(AnalyticsService, "get_zone_detail", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/zones/{uuid.uuid4()}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_vertex_detail_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: AnalyticsService, _fid: object, _q: object) -> object:
            raise LookupError("vertex not found")

        monkeypatch.setattr(AnalyticsService, "get_zone_detail", boom)
        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/vertices/{uuid.uuid4()}")
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────────
# 6. ASK — Validation + error paths
# ──────────────────────────────────────────────────────────────────────────


class TestAskEndpoint:
    """Natural language query endpoint edge cases."""

    @pytest.mark.asyncio
    async def test_ask_question_too_short_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"question": "ab", "language": "en"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_question_too_long_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"question": "x" * 2001, "language": "en"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_invalid_language_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"question": "When to irrigate?", "language": "klingon"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_missing_question_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"language": "en"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_empty_body_422(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/ask/{uuid.uuid4()}", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_non_uuid_farm_id_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/ask/not-a-uuid",
            json={"question": "What is the soil?", "language": "en"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(
            self: LLMService,
            *,
            farm_id: object,
            question: str,
            language: object,
            user_id: object,
            conversation_id: str | None,
        ) -> object:
            raise LookupError("farm not found")

        monkeypatch.setattr(LLMService, "ask", boom)
        resp = await client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"question": "What is the status?", "language": "en"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ask_value_error_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(
            self: LLMService,
            *,
            farm_id: object,
            question: str,
            language: object,
            user_id: object,
            conversation_id: str | None,
        ) -> object:
            raise ValueError("bad question")

        monkeypatch.setattr(LLMService, "ask", boom)
        resp = await client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"question": "What is the status?", "language": "en"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_ask_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(
            self: LLMService,
            *,
            farm_id: object,
            question: str,
            language: object,
            user_id: object,
            conversation_id: str | None,
        ) -> object:
            raise RuntimeError("LLM died")

        monkeypatch.setattr(LLMService, "ask", boom)
        resp = await client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"question": "What is the status?", "language": "en"},
        )
        assert resp.status_code == 500
        assert resp.json()["detail"] == "ask failure"

    @pytest.mark.asyncio
    async def test_ask_boundary_question_3_chars(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exactly 3 chars — should pass validation."""
        farm_id = uuid.uuid4()

        async def fake(
            self: LLMService,
            *,
            farm_id: object,
            question: str,
            language: object,
            user_id: object,
            conversation_id: str | None,
        ) -> AskResponse:
            return AskResponse(
                farm_id=str(farm_id),
                question=question,
                language=AskLanguage.en,
                intent="status",
                answer="OK",
                confidence=0.5,
                conversation_id=conversation_id or "conversation:test",
                tools_called=[],
            )

        monkeypatch.setattr(LLMService, "ask", fake)
        resp = await client.post(
            f"/api/v1/ask/{farm_id}",
            json={"question": "why", "language": "en"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ask_boundary_question_2000_chars(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exactly 2000 chars — should pass validation."""
        farm_id = uuid.uuid4()

        async def fake(
            self: LLMService,
            *,
            farm_id: object,
            question: str,
            language: object,
            user_id: object,
            conversation_id: str | None,
        ) -> AskResponse:
            return AskResponse(
                farm_id=str(farm_id),
                question=question,
                language=AskLanguage.en,
                intent="status",
                answer="OK",
                confidence=0.5,
                conversation_id=conversation_id or "conversation:test",
                tools_called=[],
            )

        monkeypatch.setattr(LLMService, "ask", fake)
        resp = await client.post(
            f"/api/v1/ask/{farm_id}",
            json={"question": "q" * 2000, "language": "en"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ask_all_supported_languages(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(
            self: LLMService,
            *,
            farm_id: object,
            question: str,
            language: object,
            user_id: object,
            conversation_id: str | None,
        ) -> AskResponse:
            return AskResponse(
                farm_id=str(farm_id),
                question=question,
                language=language,
                intent="status",
                answer="OK",
                confidence=0.5,
                conversation_id=conversation_id or "conversation:test",
                tools_called=[],
            )

        monkeypatch.setattr(LLMService, "ask", fake)
        for lang in ["en", "fr", "ar"]:
            resp = await client.post(
                f"/api/v1/ask/{farm_id}",
                json={"question": "What is the status?", "language": lang},
            )
            assert resp.status_code == 200, f"Failed for language={lang}"
            assert resp.json()["language"] == lang


# ──────────────────────────────────────────────────────────────────────────
# 7. JOBS — Validation + error paths
# ──────────────────────────────────────────────────────────────────────────


class TestJobsEdgeCases:
    """Jobs endpoint edge cases and error mapping."""

    @pytest.mark.asyncio
    async def test_create_job_non_uuid_422(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/jobs/not-a-uuid/recompute")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_job_status_non_uuid_422(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/jobs/not-a-uuid/status")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_job_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: JobsService, _fid: object) -> object:
            raise LookupError("farm not found")

        monkeypatch.setattr(JobsService, "create_recompute_job", boom)
        resp = await client.post(f"/api/v1/jobs/{uuid.uuid4()}/recompute")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_job_value_error_400(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: JobsService, _fid: object) -> object:
            raise ValueError("already queued")

        monkeypatch.setattr(JobsService, "create_recompute_job", boom)
        resp = await client.post(f"/api/v1/jobs/{uuid.uuid4()}/recompute")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_job_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: JobsService, _fid: object) -> object:
            raise RuntimeError("scheduler died")

        monkeypatch.setattr(JobsService, "create_recompute_job", boom)
        resp = await client.post(f"/api/v1/jobs/{uuid.uuid4()}/recompute")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "job failure"

    @pytest.mark.asyncio
    async def test_get_job_status_lookup_error_404(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: JobsService, _jid: object) -> object:
            raise LookupError("job not found")

        monkeypatch.setattr(JobsService, "get_job_status", boom)
        resp = await client.get(f"/api/v1/jobs/{uuid.uuid4()}/status")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_job_status_unexpected_error_500(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: JobsService, _jid: object) -> object:
            raise RuntimeError("status check died")

        monkeypatch.setattr(JobsService, "get_job_status", boom)
        resp = await client.get(f"/api/v1/jobs/{uuid.uuid4()}/status")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_job_create_response_shape(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()
        job_id = uuid.uuid4()

        async def fake(self: JobsService, _fid: object) -> JobCreateResponse:
            return JobCreateResponse(
                job_id=job_id, farm_id=farm_id, status="queued", created_at=_NOW
            )

        monkeypatch.setattr(JobsService, "create_recompute_job", fake)
        resp = await client.post(f"/api/v1/jobs/{farm_id}/recompute")
        assert resp.status_code == 202
        body = resp.json()
        assert set(body.keys()) == {"job_id", "farm_id", "status", "created_at"}

    @pytest.mark.asyncio
    async def test_job_status_response_shape(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job_id = uuid.uuid4()

        async def fake(self: JobsService, _jid: object) -> JobStatusResponse:
            return JobStatusResponse(
                job_id=job_id,
                farm_id=uuid.uuid4(),
                status="succeeded",
                created_at=_NOW,
                started_at=_NOW,
                completed_at=_NOW,
                error=None,
                updated_at=_NOW,
            )

        monkeypatch.setattr(JobsService, "get_job_status", fake)
        resp = await client.get(f"/api/v1/jobs/{job_id}/status")
        assert resp.status_code == 200
        body = resp.json()
        expected_keys = {
            "job_id",
            "farm_id",
            "status",
            "created_at",
            "started_at",
            "completed_at",
            "error",
            "updated_at",
        }
        assert set(body.keys()) == expected_keys


# ──────────────────────────────────────────────────────────────────────────
# 8. AUTH — JWT edge cases + RBAC per-endpoint matrix
# ──────────────────────────────────────────────────────────────────────────


class TestAuthEdgeCases:
    """JWT token edge cases and RBAC enforcement across endpoints."""

    def test_expired_jwt_raises_auth_error(self) -> None:
        token = create_access_token("test-user", expires_minutes=-1)
        with pytest.raises(AuthError) as exc_info:
            decode_token(token, expected_type="access")
        # jose catches expiry at decode time → token_invalid
        assert exc_info.value.code in {"token_invalid", "token_expired"}

    def test_refresh_token_rejected_as_access(self) -> None:
        token = create_refresh_token("test-user", expires_minutes=30)
        with pytest.raises(AuthError) as exc_info:
            decode_token(token, expected_type="access")
        assert exc_info.value.code == "token_type_invalid"

    def test_access_token_rejected_as_refresh(self) -> None:
        token = create_access_token("test-user", expires_minutes=30)
        with pytest.raises(AuthError) as exc_info:
            decode_token(token, expected_type="refresh")
        assert exc_info.value.code == "token_type_invalid"

    def test_decode_empty_string_raises_auth_error(self) -> None:
        with pytest.raises(AuthError):
            decode_token("", expected_type="access")

    def test_decode_garbage_raises_auth_error(self) -> None:
        with pytest.raises(AuthError):
            decode_token("aaa.bbb.ccc", expected_type="access")

    @pytest.mark.asyncio
    async def test_missing_auth_on_farms_create_401(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post(
            "/api/v1/farms",
            json={"name": "F", "farm_type": "greenhouse"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_on_ingest_401(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post(
            "/api/v1/ingest/soil",
            json={
                "farm_id": str(uuid.uuid4()),
                "readings": [
                    {
                        "sensor_id": str(uuid.uuid4()),
                        "timestamp": _NOW.isoformat(),
                        "moisture": 0.3,
                        "temperature": 24.0,
                    }
                ],
            },
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_on_jobs_401(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post(f"/api/v1/jobs/{uuid.uuid4()}/recompute")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_on_ask_401(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post(
            f"/api/v1/ask/{uuid.uuid4()}",
            json={"question": "What is soil?", "language": "en"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_readonly_cannot_create_zone_403(self, client: AsyncClient) -> None:
        async def _readonly_user() -> object:
            return SimpleNamespace(
                id=uuid.uuid4(),
                role=UserRoleEnum.readonly,
                is_active=True,
                email="readonly@test.local",
            )

        app.dependency_overrides[get_current_user] = _readonly_user
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/zones",
            json={"name": "Z", "area_m2": 10.0},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_readonly_cannot_register_sensor_403(self, client: AsyncClient) -> None:
        async def _readonly_user() -> object:
            return SimpleNamespace(
                id=uuid.uuid4(),
                role=UserRoleEnum.readonly,
                is_active=True,
                email="readonly@test.local",
            )

        app.dependency_overrides[get_current_user] = _readonly_user
        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/sensors",
            json={"vertex_type": "sensor"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_field_operator_cannot_create_farm_403(self, client: AsyncClient) -> None:
        async def _field_op() -> object:
            return SimpleNamespace(
                id=uuid.uuid4(),
                role=UserRoleEnum.field_operator,
                is_active=True,
                email="field@test.local",
            )

        app.dependency_overrides[get_current_user] = _field_op
        resp = await client.post(
            "/api/v1/farms",
            json={"name": "F", "farm_type": "greenhouse"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_field_operator_can_register_sensor(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _field_op() -> object:
            return SimpleNamespace(
                id=uuid.uuid4(),
                role=UserRoleEnum.field_operator,
                is_active=True,
                email="field@test.local",
            )

        app.dependency_overrides[get_current_user] = _field_op

        vertex = _vertex_obj()

        async def fake_register(self: FarmService, _fid: object, _p: object) -> object:
            return vertex

        monkeypatch.setattr(FarmService, "register_vertex", fake_register)

        resp = await client.post(
            f"/api/v1/farms/{uuid.uuid4()}/sensors",
            json={"vertex_type": "sensor"},
        )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_readonly_can_list_farms(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _readonly_user() -> object:
            return SimpleNamespace(
                id=uuid.uuid4(),
                role=UserRoleEnum.readonly,
                is_active=True,
                email="readonly@test.local",
            )

        app.dependency_overrides[get_current_user] = _readonly_user

        async def fake_list(self: FarmService) -> list[object]:
            return []

        monkeypatch.setattr(FarmService, "list_farms", fake_list)

        resp = await client.get("/api/v1/farms")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_readonly_can_view_analytics(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _readonly_user() -> object:
            return SimpleNamespace(
                id=uuid.uuid4(),
                role=UserRoleEnum.readonly,
                is_active=True,
                email="readonly@test.local",
            )

        app.dependency_overrides[get_current_user] = _readonly_user

        async def fake_status(self: AnalyticsService, _fid: object) -> FarmStatusResponse:
            return FarmStatusResponse(farm_id=uuid.uuid4(), generated_at=_NOW, zones=[])

        monkeypatch.setattr(AnalyticsService, "get_farm_status", fake_status)

        resp = await client.get(f"/api/v1/analytics/{uuid.uuid4()}/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_machine_scope_jwt_user_rejected(self) -> None:
        """Machine-scoped endpoints require API-key principals."""
        from fastapi import HTTPException

        from app.auth.dependencies import require_machine_scope

        dep = require_machine_scope("ingest")
        jwt_principal = AuthPrincipal(
            auth_type="jwt",
            subject_id=uuid.uuid4(),
            role=UserRoleEnum.admin,
            scopes=set(),
        )
        with pytest.raises(HTTPException) as exc_info:
            await dep(jwt_principal)
        assert exc_info.value.status_code == 403
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error"] == "api_key_required"

    @pytest.mark.asyncio
    async def test_machine_scope_invalid_scope_403(self) -> None:
        from fastapi import HTTPException

        from app.auth.dependencies import require_machine_scope

        dep = require_machine_scope("admin_override")
        api_principal = AuthPrincipal(
            auth_type="api_key",
            subject_id=uuid.uuid4(),
            role=UserRoleEnum.admin,
            scopes={"admin_override"},
            api_key_id=uuid.uuid4(),
        )
        with pytest.raises(HTTPException) as exc_info:
            await dep(api_principal)
        assert exc_info.value.status_code == 403
        detail = exc_info.value.detail
        assert isinstance(detail, dict)
        assert detail["error"] == "scope_invalid"


# ──────────────────────────────────────────────────────────────────────────
# 9. WEBSOCKET — Additional edge cases
# ──────────────────────────────────────────────────────────────────────────


class TestWebSocketEdgeCases:
    """WebSocket edge cases beyond the existing test_websocket.py suite."""

    def test_whitespace_only_token_rejected(self, fake_redis: Any, monkeypatch: Any) -> None:
        from fastapi.testclient import TestClient

        from app.routes import ws

        async def fake_farm_exists(_fid: Any) -> bool:
            return True

        monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
        monkeypatch.setattr(ws, "AUTH_MESSAGE_TIMEOUT_SECONDS", 0.01)
        app.state.redis = fake_redis

        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def noop_lifespan(_: Any) -> AsyncIterator[None]:
            yield

        app.router.lifespan_context = noop_lifespan
        with (
            TestClient(app) as tc,
            tc.websocket_connect(f"/ws/{uuid.uuid4()}/live") as ws_conn,
        ):
            ws_conn.send_json({"type": "auth", "token": "  "})
            payload = ws_conn.receive_json()
            assert payload["error"] == "auth_required"
        app.router.lifespan_context = original_lifespan

    def test_multiple_messages_forwarded(self, fake_redis: Any, monkeypatch: Any) -> None:
        from fastapi.testclient import TestClient

        from app.routes import ws

        farm_id = str(uuid.uuid4())

        async def fake_farm_exists(_fid: Any) -> bool:
            return True

        async def fake_auth(_token: str) -> bool:
            return True

        monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
        monkeypatch.setattr(ws, "_authenticate_token", fake_auth)

        messages = [
            {
                "type": "message",
                "data": json.dumps({"event_type": "ingest", "layer": "soil", "seq": i}),
            }
            for i in range(3)
        ]
        fake_redis.payloads = messages
        app.state.redis = fake_redis

        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def noop_lifespan(_: Any) -> AsyncIterator[None]:
            yield

        app.router.lifespan_context = noop_lifespan
        with (
            TestClient(app) as tc,
            tc.websocket_connect(f"/ws/{farm_id}/live") as ws_conn,
        ):
            ws_conn.send_json({"type": "auth", "token": "test-token"})
            received = []
            for _ in range(3):
                received.append(ws_conn.receive_json())
            assert len(received) == 3
            seqs = [m["seq"] for m in received]
            assert seqs == [0, 1, 2]
        app.router.lifespan_context = original_lifespan

    def test_bytes_payload_decoded(self, fake_redis: Any, monkeypatch: Any) -> None:
        """Redis can deliver bytes — verify they are decoded and forwarded."""
        from fastapi.testclient import TestClient

        from app.routes import ws

        farm_id = str(uuid.uuid4())

        async def fake_farm_exists(_fid: Any) -> bool:
            return True

        async def fake_auth(_token: str) -> bool:
            return True

        monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
        monkeypatch.setattr(ws, "_authenticate_token", fake_auth)

        message = {
            "type": "message",
            "data": json.dumps({"evt": "bytes_test"}).encode("utf-8"),
        }
        fake_redis.payloads = [message]
        app.state.redis = fake_redis

        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def noop_lifespan(_: Any) -> AsyncIterator[None]:
            yield

        app.router.lifespan_context = noop_lifespan
        with (
            TestClient(app) as tc,
            tc.websocket_connect(f"/ws/{farm_id}/live") as ws_conn,
        ):
            ws_conn.send_json({"type": "auth", "token": "test-token"})
            payload = ws_conn.receive_json()
            assert payload["evt"] == "bytes_test"
        app.router.lifespan_context = original_lifespan


# ──────────────────────────────────────────────────────────────────────────
# 10. SYSTEM / CROSS-CUTTING — Health, CORS, OpenAPI, error formats
# ──────────────────────────────────────────────────────────────────────────


class TestSystemCrossCutting:
    """Health endpoints, CORS, OpenAPI doc completeness, error format consistency."""

    @pytest.mark.asyncio
    async def test_health_response_shape(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"status", "service", "version"}
        assert body["status"] == "ok"
        assert body["service"] == "agrisense"
        assert body["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_ready_response_shape(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app import main

        async def _all_ok(_app: object) -> dict[str, Any]:
            return {
                "database": {"ok": True, "message": "ok"},
                "redis": {"ok": True, "message": "ok"},
                "julia": {"ok": True, "message": "ok"},
            }

        monkeypatch.setattr(main, "_run_readiness_checks", _all_ok)
        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"status", "service", "version", "checks"}
        assert set(body["checks"].keys()) == {"database", "redis", "julia"}
        for check in body["checks"].values():
            assert "ok" in check
            assert "message" in check

    @pytest.mark.asyncio
    async def test_cors_preflight_headers(self, client: AsyncClient) -> None:
        resp = await client.options(
            "/api/v1/farms",
            headers={
                "origin": "https://example.com",
                "access-control-request-method": "POST",
            },
        )
        # CORS middleware should respond
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    @pytest.mark.asyncio
    async def test_request_id_unique_per_request(self, client: AsyncClient) -> None:
        resp1 = await client.get("/health")
        resp2 = await client.get("/health")
        id1 = resp1.headers.get("x-request-id")
        id2 = resp2.headers.get("x-request-id")
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_custom_request_id_echoed(self, client: AsyncClient) -> None:
        resp = await client.get("/health", headers={"x-request-id": "custom-123"})
        assert resp.headers.get("x-request-id") == "custom-123"

    @pytest.mark.asyncio
    async def test_openapi_all_endpoints_listed(self, client: AsyncClient) -> None:
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        expected_paths = [
            "/api/v1/farms",
            "/api/v1/farms/{farm_id}",
            "/api/v1/farms/{farm_id}/zones",
            "/api/v1/farms/{farm_id}/sensors",
            "/api/v1/farms/{farm_id}/graph",
            "/api/v1/ingest/soil",
            "/api/v1/ingest/weather",
            "/api/v1/ingest/irrigation",
            "/api/v1/ingest/npk",
            "/api/v1/ingest/vision",
            "/api/v1/ingest/bulk",
            "/api/v1/analytics/{farm_id}/status",
            "/api/v1/analytics/{farm_id}/zones/{zone_id}",
            "/api/v1/analytics/{farm_id}/vertices/{vertex_id}",
            "/api/v1/analytics/{farm_id}/irrigation/schedule",
            "/api/v1/analytics/{farm_id}/nutrients/report",
            "/api/v1/analytics/{farm_id}/yield/forecast",
            "/api/v1/analytics/{farm_id}/alerts",
            "/api/v1/ask/{farm_id}",
            "/api/v1/jobs/{farm_id}/recompute",
            "/api/v1/jobs/{job_id}/status",
        ]
        for path in expected_paths:
            assert path in paths, f"Missing path: {path}"

    @pytest.mark.asyncio
    async def test_openapi_schemas_present(self, client: AsyncClient) -> None:
        resp = await client.get("/openapi.json")
        body = resp.json()
        schemas = body["components"]["schemas"]
        expected_schemas = [
            "FarmCreate",
            "FarmRead",
            "FarmListRead",
            "FarmGraphRead",
            "ZoneCreate",
            "ZoneRead",
            "VertexCreate",
            "VertexRead",
            "IngestReceipt",
            "BulkIngestReceipt",
            "FarmStatusResponse",
            "ZoneDetailResponse",
            "IrrigationScheduleResponse",
            "NutrientReportResponse",
            "YieldForecastResponse",
            "AlertsResponse",
            "AskRequest",
            "AskResponse",
            "JobCreateResponse",
            "JobStatusResponse",
        ]
        for schema in expected_schemas:
            assert schema in schemas, f"Missing schema: {schema}"

    @pytest.mark.asyncio
    async def test_422_error_format_has_detail_array(self, client: AsyncClient) -> None:
        """FastAPI 422 responses should have a 'detail' list with validation errors."""
        resp = await client.post("/api/v1/farms", json={})
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0
        # Each element should have loc, msg, type
        error = body["detail"][0]
        assert "loc" in error
        assert "msg" in error
        assert "type" in error

    @pytest.mark.asyncio
    async def test_404_error_format_has_detail_string(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, fid: object) -> object:
            raise LookupError("not found")

        monkeypatch.setattr(FarmService, "get_farm", boom)
        resp = await client.get(f"/api/v1/farms/{uuid.uuid4()}")
        assert resp.status_code == 404
        body = resp.json()
        assert isinstance(body["detail"], str)

    @pytest.mark.asyncio
    async def test_500_error_format_has_detail_string(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(self: FarmService, fid: object) -> object:
            raise RuntimeError("critical")

        monkeypatch.setattr(FarmService, "get_farm", boom)
        resp = await client.get(f"/api/v1/farms/{uuid.uuid4()}")
        assert resp.status_code == 500
        body = resp.json()
        assert isinstance(body["detail"], str)

    @pytest.mark.asyncio
    async def test_nonexistent_route_404(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/does-not-exist")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_method_not_allowed_405(self, client: AsyncClient) -> None:
        resp = await client.delete("/api/v1/farms")
        assert resp.status_code == 405

    @pytest.mark.asyncio
    async def test_health_bypass_rate_limit(
        self, client: AsyncClient, fake_redis: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Health endpoints should bypass rate limiting entirely."""
        app.state.redis = fake_redis

        @dataclass
        class _SettingsStub:
            rate_limit_user_per_minute: int = 1
            rate_limit_api_key_per_minute: int = 1
            api_key_header_name: str = "x-api-key"

        monkeypatch.setattr("app.middleware.rate_limit.get_settings", lambda: _SettingsStub())

        for _ in range(5):
            resp = await client.get("/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_openapi_bypass_rate_limit(
        self, client: AsyncClient, fake_redis: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAPI endpoints should bypass rate limiting."""
        app.state.redis = fake_redis

        @dataclass
        class _SettingsStub:
            rate_limit_user_per_minute: int = 1
            rate_limit_api_key_per_minute: int = 1
            api_key_header_name: str = "x-api-key"

        monkeypatch.setattr("app.middleware.rate_limit.get_settings", lambda: _SettingsStub())

        for _ in range(5):
            resp = await client.get("/openapi.json")
            assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────
# 11. STRESS — Concurrent requests & large payloads
# ──────────────────────────────────────────────────────────────────────────


class TestStress:
    """Concurrent request handling and large payload tests."""

    @pytest.mark.asyncio
    async def test_concurrent_farm_list_requests(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fire 50 concurrent list-farms requests — all should succeed."""
        import asyncio

        async def fake_list(self: FarmService) -> list[object]:
            return []

        monkeypatch.setattr(FarmService, "list_farms", fake_list)

        tasks = [client.get("/api/v1/farms") for _ in range(50)]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_analytics_status_requests(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fire 50 concurrent analytics-status requests."""
        import asyncio

        farm_id = uuid.uuid4()

        async def fake_status(self: AnalyticsService, _fid: object) -> FarmStatusResponse:
            return FarmStatusResponse(farm_id=farm_id, generated_at=_NOW, zones=[])

        monkeypatch.setattr(AnalyticsService, "get_farm_status", fake_status)

        tasks = [client.get(f"/api/v1/analytics/{farm_id}/status") for _ in range(50)]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_bulk_ingest_large_payload(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bulk ingest with 100 soil readings — verify it processes."""
        farm_id = uuid.uuid4()

        async def fake_bulk(self: IngestService, _fid: object, **kw: object) -> BulkIngestReceipt:
            return BulkIngestReceipt(
                farm_id=farm_id, status="ok", inserted_count=100, failed_count=0
            )

        monkeypatch.setattr(IngestService, "ingest_bulk", fake_bulk)

        soil_readings = [
            {
                "sensor_id": str(uuid.uuid4()),
                "timestamp": _NOW.isoformat(),
                "moisture": 0.3 + (i * 0.001),
                "temperature": 22.0 + (i * 0.01),
            }
            for i in range(100)
        ]

        resp = await client.post(
            "/api/v1/ingest/bulk",
            json={"farm_id": str(farm_id), "soil": soil_readings},
        )
        assert resp.status_code == 200
        assert resp.json()["inserted_count"] == 100

    @pytest.mark.asyncio
    async def test_concurrent_ingest_different_endpoints(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fire concurrent requests to different ingest endpoints."""
        import asyncio

        farm_id = uuid.uuid4()

        async def fake_soil(self: IngestService, _fid: object, _r: object) -> IngestReceipt:
            return IngestReceipt(farm_id=farm_id, layer="soil", status="ok", inserted_count=1)

        async def fake_weather(self: IngestService, _fid: object, _r: object) -> IngestReceipt:
            return IngestReceipt(farm_id=farm_id, layer="weather", status="ok", inserted_count=1)

        monkeypatch.setattr(IngestService, "ingest_soil", fake_soil)
        monkeypatch.setattr(IngestService, "ingest_weather", fake_weather)

        soil_payload = {
            "farm_id": str(farm_id),
            "readings": [
                {
                    "sensor_id": str(uuid.uuid4()),
                    "timestamp": _NOW.isoformat(),
                    "moisture": 0.3,
                    "temperature": 24.0,
                }
            ],
        }
        weather_payload = {
            "farm_id": str(farm_id),
            "readings": [
                {
                    "station_id": str(uuid.uuid4()),
                    "timestamp": _NOW.isoformat(),
                    "temperature": 20.0,
                    "humidity": 55.0,
                    "precipitation_mm": 0.0,
                }
            ],
        }

        tasks = []
        for _ in range(25):
            tasks.append(client.post("/api/v1/ingest/soil", json=soil_payload))
            tasks.append(client.post("/api/v1/ingest/weather", json=weather_payload))

        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_mixed_endpoint_stress(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mix of GET and POST across different route groups."""
        import asyncio

        farm_id = uuid.uuid4()

        async def fake_list(self: FarmService) -> list[object]:
            return []

        async def fake_status(self: AnalyticsService, _fid: object) -> FarmStatusResponse:
            return FarmStatusResponse(farm_id=farm_id, generated_at=_NOW, zones=[])

        monkeypatch.setattr(FarmService, "list_farms", fake_list)
        monkeypatch.setattr(AnalyticsService, "get_farm_status", fake_status)

        tasks = []
        for _ in range(20):
            tasks.append(client.get("/api/v1/farms"))
            tasks.append(client.get(f"/api/v1/analytics/{farm_id}/status"))
            tasks.append(client.get("/health"))

        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 200


# ──────────────────────────────────────────────────────────────────────────
# 12. INGEST / ANALYTICS — Receipt and response field contracts
# ──────────────────────────────────────────────────────────────────────────


class TestResponseContracts:
    """Assert exact fields in IngestReceipt, BulkIngestReceipt, analytics responses."""

    @pytest.mark.asyncio
    async def test_ingest_receipt_fields(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()
        now = datetime.now(UTC)

        async def fake(self: IngestService, _fid: object, _r: object) -> IngestReceipt:
            return IngestReceipt(
                farm_id=farm_id,
                layer="soil",
                status="ok",
                inserted_count=2,
                failed_count=0,
                event_ids=[1, 2],
                timestamp_start=now,
                timestamp_end=now,
                warnings=[],
            )

        monkeypatch.setattr(IngestService, "ingest_soil", fake)
        resp = await client.post(
            "/api/v1/ingest/soil",
            json={
                "farm_id": str(farm_id),
                "readings": [
                    {
                        "sensor_id": str(uuid.uuid4()),
                        "timestamp": now.isoformat(),
                        "moisture": 0.3,
                        "temperature": 22.0,
                    }
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        expected_keys = {
            "farm_id",
            "layer",
            "status",
            "inserted_count",
            "failed_count",
            "event_ids",
            "timestamp_start",
            "timestamp_end",
            "warnings",
        }
        assert set(body.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_alerts_response_fields(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(self: AnalyticsService, _fid: object) -> AlertsResponse:
            return AlertsResponse(
                farm_id=farm_id,
                generated_at=_NOW,
                zones=[
                    ZoneAlerts(
                        zone_id=uuid.uuid4(),
                        alerts=[],
                    )
                ],
            )

        monkeypatch.setattr(AnalyticsService, "get_active_alerts", fake)
        resp = await client.get(f"/api/v1/analytics/{farm_id}/alerts")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"farm_id", "generated_at", "zones"}
        zone = body["zones"][0]
        assert set(zone.keys()) == {"zone_id", "alerts"}

    @pytest.mark.asyncio
    async def test_ask_response_fields(
        self, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        farm_id = uuid.uuid4()

        async def fake(
            self: LLMService,
            *,
            farm_id: object,
            question: str,
            language: object,
            user_id: object,
            conversation_id: str | None,
        ) -> AskResponse:
            return AskResponse(
                farm_id=str(farm_id),
                question=question,
                language=AskLanguage.en,
                intent="status",
                answer="All good",
                confidence=0.9,
                recommendations=[],
                sources=[],
                conversation_id=conversation_id or "conversation:test",
                tools_called=[],
            )

        monkeypatch.setattr(LLMService, "ask", fake)
        resp = await client.post(
            f"/api/v1/ask/{farm_id}",
            json={"question": "How is the farm?", "language": "en"},
        )
        assert resp.status_code == 200
        body = resp.json()
        expected_keys = {
            "farm_id",
            "question",
            "language",
            "intent",
            "answer",
            "confidence",
            "recommendations",
            "sources",
            "conversation_id",
            "tools_called",
            "telemetry",
        }
        assert set(body.keys()) == expected_keys
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0
