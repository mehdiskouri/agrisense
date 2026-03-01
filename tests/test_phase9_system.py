from __future__ import annotations

import pytest
from httpx import AsyncClient

from app import main


@pytest.mark.asyncio
async def test_health_ready_ok(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _ok(_app):
        return {
            "database": {"ok": True, "message": "ok"},
            "redis": {"ok": True, "message": "ok"},
            "julia": {"ok": True, "message": "ok"},
        }

    monkeypatch.setattr(main, "_run_readiness_checks", _ok)

    response = await client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["ok"] is True


@pytest.mark.asyncio
async def test_health_ready_degraded(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _bad(_app):
        return {
            "database": {"ok": False, "message": "db down"},
            "redis": {"ok": True, "message": "ok"},
            "julia": {"ok": True, "message": "ok"},
        }

    monkeypatch.setattr(main, "_run_readiness_checks", _bad)

    response = await client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["database"]["ok"] is False


@pytest.mark.asyncio
async def test_request_id_is_propagated(client: AsyncClient) -> None:
    request_id = "phase9-request-id"
    response = await client.get("/health", headers={"x-request-id": request_id})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == request_id


@pytest.mark.asyncio
async def test_request_id_generated_when_missing(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    generated = response.headers.get("x-request-id")
    assert generated is not None
    assert len(generated) >= 8
