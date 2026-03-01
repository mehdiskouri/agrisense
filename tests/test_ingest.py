from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.schemas.ingest import (
    BulkIngestReceipt,
    IngestReceipt,
    IngestWarning,
    SoilReadingIn,
    WeatherReadingIn,
)
from app.services import julia_bridge
from app.services.ingest_service import IngestService


@pytest.mark.asyncio
async def test_ingest_soil_endpoint_success(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    farm_id = uuid4()

    async def fake_ingest(
        self: IngestService, _farm_id: object, _readings: object
    ) -> IngestReceipt:
        return IngestReceipt(
            farm_id=farm_id,
            layer="soil",
            status="ok",
            inserted_count=1,
            failed_count=0,
            event_ids=[1],
            warnings=[],
        )

    monkeypatch.setattr(IngestService, "ingest_soil", fake_ingest)

    response = await client.post(
        "/api/v1/ingest/soil",
        json={
            "farm_id": str(farm_id),
            "readings": [
                {
                    "sensor_id": str(uuid4()),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "moisture": 0.3,
                    "temperature": 24.2,
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["layer"] == "soil"
    assert body["inserted_count"] == 1


@pytest.mark.asyncio
async def test_ingest_weather_error_mapping(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ingest(
        self: IngestService, _farm_id: object, _readings: object
    ) -> IngestReceipt:
        raise ValueError("invalid weather station")

    monkeypatch.setattr(IngestService, "ingest_weather", fake_ingest)

    response = await client.post(
        "/api/v1/ingest/weather",
        json={
            "farm_id": str(uuid4()),
            "readings": [
                {
                    "station_id": str(uuid4()),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "temperature": 20.0,
                    "humidity": 60.0,
                    "precipitation_mm": 0.0,
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "invalid weather station" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_bulk_partial_success(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    farm_id = uuid4()

    async def fake_bulk(
        self: IngestService, _farm_id: object, **_kwargs: object
    ) -> BulkIngestReceipt:
        warning = IngestWarning(index=0, message="npk failed")
        return BulkIngestReceipt(
            farm_id=farm_id,
            status="partial",
            inserted_count=3,
            failed_count=1,
            warnings=[warning],
            layers={
                "soil": IngestReceipt(
                    farm_id=farm_id,
                    layer="soil",
                    status="ok",
                    inserted_count=2,
                    failed_count=0,
                    event_ids=[1, 2],
                    warnings=[],
                ),
                "npk": IngestReceipt(
                    farm_id=farm_id,
                    layer="npk",
                    status="failed",
                    inserted_count=0,
                    failed_count=1,
                    event_ids=[],
                    warnings=[warning],
                ),
            },
        )

    monkeypatch.setattr(IngestService, "ingest_bulk", fake_bulk)

    response = await client.post(
        "/api/v1/ingest/bulk",
        json={
            "farm_id": str(farm_id),
            "soil": [],
            "weather": [],
            "irrigation": [],
            "npk": [],
            "vision": [],
            "lighting": [],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["layers"]["npk"]["status"] == "failed"


def test_layer_normalization_aliases() -> None:
    assert IngestService.normalize_layer("solar") == "lighting"
    assert IngestService.normalize_layer("lighting") == "lighting"
    with pytest.raises(ValueError):
        IngestService.normalize_layer("unknown")


@pytest.mark.asyncio
async def test_ingest_openapi_receipt_contract_stability(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    body = response.json()

    ingest_receipt = body["components"]["schemas"]["IngestReceipt"]["properties"]
    bulk_receipt = body["components"]["schemas"]["BulkIngestReceipt"]["properties"]

    assert "timestamp_start" in ingest_receipt
    assert "timestamp_end" in ingest_receipt
    assert "timestamp_start" in bulk_receipt
    assert "timestamp_end" in bulk_receipt


class _NestedTx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeDb:
    def __init__(self) -> None:
        self._rows: list[object] = []

    def add_all(self, rows: list[object]) -> None:
        self._rows = rows

    async def flush(self) -> None:
        for index, row in enumerate(self._rows, start=1):
            row.id = index

    def begin_nested(self) -> _NestedTx:
        return _NestedTx()


@pytest.mark.asyncio
async def test_ingest_soil_graph_and_event_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    fake_redis = SimpleNamespace(publish=AsyncMock())
    service = IngestService(db, fake_redis)

    async def _noop_validate(_farm_id: object, _layer: object) -> None:
        return None

    zone_id = uuid4()

    async def _fake_vertex(
        _vertex_id: object, _farm_id: object, _allowed_types: object
    ) -> SimpleNamespace:
        return SimpleNamespace(config={"sensor_type": "soil"}, zone_id=zone_id)

    async def _fake_ensure_graph(_farm_id: object) -> None:
        return None

    batch_calls: list[tuple[str, list[dict[str, object]]]] = []

    def _fake_batch_update(
        farm_id: str, updates: list[dict[str, object]]
    ) -> dict[str, object]:
        batch_calls.append((farm_id, updates))
        return {"ok": True, "farm_id": farm_id, "version": 1, "n_updated": len(updates)}

    monkeypatch.setattr(service, "_validate_active_layer", _noop_validate)
    monkeypatch.setattr(service, "_require_vertex", _fake_vertex)
    monkeypatch.setattr(service, "_ensure_graph", _fake_ensure_graph)
    monkeypatch.setattr(julia_bridge, "batch_update_features", _fake_batch_update)

    farm_id = uuid4()
    readings = [
        SoilReadingIn(
            sensor_id=uuid4(),
            timestamp=datetime.now(UTC),
            moisture=0.31,
            temperature=23.4,
            conductivity=1.2,
            ph=6.6,
        ),
        SoilReadingIn(
            sensor_id=uuid4(),
            timestamp=datetime.now(UTC),
            moisture=0.35,
            temperature=23.9,
            conductivity=1.3,
            ph=6.5,
        ),
    ]

    receipt = await service.ingest_soil(farm_id, readings)

    assert len(batch_calls) == 1
    assert len(batch_calls[0][1]) == 2  # 2 readings batched into 1 call
    assert fake_redis.publish.await_count == 2
    channel, payload = fake_redis.publish.await_args_list[0].args
    payload_data = json.loads(payload)
    assert channel == f"farm:{farm_id}:live"
    assert payload_data["zone_id"] == str(zone_id)
    assert payload_data["vertex_id"] == str(readings[0].sensor_id)
    assert isinstance(payload_data["payload"], dict)
    assert isinstance(payload_data["warnings"], list)
    assert receipt.timestamp_start is not None
    assert receipt.timestamp_end is not None


@pytest.mark.asyncio
async def test_bulk_failed_layer_does_not_corrupt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeDb()
    service = IngestService(db, None)
    farm_id = uuid4()
    service._graph_built = True

    async def _ok_soil(_farm_id: object, _records: object) -> IngestReceipt:
        now = datetime.now(UTC)
        return IngestReceipt(
            farm_id=farm_id,
            layer="soil",
            status="ok",
            inserted_count=1,
            failed_count=0,
            event_ids=[1],
            timestamp_start=now,
            timestamp_end=now,
            warnings=[],
        )

    async def _fail_weather(_farm_id: object, _records: object) -> IngestReceipt:
        raise RuntimeError("weather exploded")

    monkeypatch.setattr(service, "ingest_soil", _ok_soil)
    monkeypatch.setattr(service, "ingest_weather", _fail_weather)

    weather_reading = WeatherReadingIn(
        station_id=uuid4(),
        timestamp=datetime.now(UTC),
        temperature=21.0,
        humidity=63.0,
        precipitation_mm=0.0,
    )

    receipt = await service.ingest_bulk(
        farm_id,
        soil=[
            SoilReadingIn(
                sensor_id=uuid4(),
                timestamp=datetime.now(UTC),
                moisture=0.3,
                temperature=22.0,
            )
        ],
        weather=[weather_reading],
        irrigation=[],
        npk=[],
        vision=[],
        lighting=[],
    )

    assert receipt.status == "partial"
    assert receipt.layers["weather"].status == "failed"
    # _graph_built remains True — no rollback needed, Julia cache is authoritative
    assert service._graph_built is True
