from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.schemas.ingest import BulkIngestReceipt, IngestReceipt, IngestWarning
from app.services.ingest_service import IngestService


@pytest.mark.asyncio
async def test_ingest_soil_endpoint_success(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    farm_id = uuid4()

    async def fake_ingest(self: IngestService, _farm_id: object, _readings: object) -> IngestReceipt:
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
    async def fake_ingest(self: IngestService, _farm_id: object, _readings: object) -> IngestReceipt:
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

    async def fake_bulk(self: IngestService, _farm_id: object, **_kwargs: object) -> BulkIngestReceipt:
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
        json={"farm_id": str(farm_id), "soil": [], "weather": [], "irrigation": [], "npk": [], "vision": [], "lighting": []},
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
