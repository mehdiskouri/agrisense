from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
from app.routes import ws


@asynccontextmanager
async def _noop_lifespan(_: Any):
    yield


def test_websocket_live_feed_forwards_events(fake_redis: Any, monkeypatch: Any) -> None:
    farm_id = str(uuid4())

    async def fake_farm_exists(_farm_id: Any) -> bool:
        return True

    async def fake_auth(_token: str) -> bool:
        return True

    monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
    monkeypatch.setattr(ws, "_authenticate_token", fake_auth)

    message = {
        "type": "message",
        "data": json.dumps({"event_type": "ingest", "layer": "soil", "record_id": 1}),
    }
    fake_redis.payloads = [message]
    app.state.redis = fake_redis

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/{farm_id}/live?token=test-token") as websocket:
            payload = websocket.receive_json()
            assert payload["event_type"] == "ingest"
            assert payload["layer"] == "soil"
    assert fake_redis.last_pubsub is not None
    assert fake_redis.last_pubsub.unsubscribed_channel is not None
    assert fake_redis.last_pubsub.closed is True
    app.router.lifespan_context = original_lifespan


def test_websocket_without_redis_returns_error(monkeypatch: Any) -> None:
    farm_id = str(uuid4())

    async def fake_farm_exists(_farm_id: Any) -> bool:
        return True

    async def fake_auth(_token: str) -> bool:
        return True

    monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
    monkeypatch.setattr(ws, "_authenticate_token", fake_auth)

    app.state.redis = None

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/{farm_id}/live?token=test-token") as websocket:
            payload = websocket.receive_json()
            assert payload["error"] == "redis_unavailable"
    app.router.lifespan_context = original_lifespan


def test_websocket_invalid_farm_id_returns_error(fake_redis: Any) -> None:
    app.state.redis = fake_redis

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect("/ws/not-a-uuid/live?token=test-token") as websocket:
            payload = websocket.receive_json()
            assert payload["error"] == "invalid_farm_id"
    app.router.lifespan_context = original_lifespan


def test_websocket_unknown_farm_returns_error(fake_redis: Any, monkeypatch: Any) -> None:
    async def fake_farm_exists(_farm_id: Any) -> bool:
        return False

    async def fake_auth(_token: str) -> bool:
        return True

    monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
    monkeypatch.setattr(ws, "_authenticate_token", fake_auth)

    app.state.redis = fake_redis
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/{uuid4()}/live?token=test-token") as websocket:
            payload = websocket.receive_json()
            assert payload["error"] == "farm_not_found"
    app.router.lifespan_context = original_lifespan


def test_websocket_missing_token_rejected(fake_redis: Any, monkeypatch: Any) -> None:
    async def fake_farm_exists(_farm_id: Any) -> bool:
        return True

    monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
    app.state.redis = fake_redis

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/{uuid4()}/live") as websocket:
            payload = websocket.receive_json()
            assert payload["error"] == "auth_required"
    app.router.lifespan_context = original_lifespan


def test_websocket_invalid_token_rejected(fake_redis: Any, monkeypatch: Any) -> None:
    async def fake_farm_exists(_farm_id: Any) -> bool:
        return True

    async def fake_auth(_token: str) -> bool:
        return False

    monkeypatch.setattr(ws, "_farm_exists", fake_farm_exists)
    monkeypatch.setattr(ws, "_authenticate_token", fake_auth)
    app.state.redis = fake_redis

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/{uuid4()}/live?token=bad-token") as websocket:
            payload = websocket.receive_json()
            assert payload["error"] == "auth_invalid"
    app.router.lifespan_context = original_lifespan
