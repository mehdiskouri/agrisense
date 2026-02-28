from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi.testclient import TestClient

from app.main import app


@asynccontextmanager
async def _noop_lifespan(_: Any):
    yield


def test_websocket_live_feed_forwards_events(fake_redis: Any) -> None:
    message = {
        "type": "message",
        "data": json.dumps({"event_type": "ingest", "layer": "soil", "record_id": 1}),
    }
    fake_redis.payloads = [message]
    app.state.redis = fake_redis

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect("/ws/farm-1/live") as websocket:
            payload = websocket.receive_json()
            assert payload["event_type"] == "ingest"
            assert payload["layer"] == "soil"
    app.router.lifespan_context = original_lifespan


def test_websocket_without_redis_returns_error() -> None:
    app.state.redis = None

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as client:
        with client.websocket_connect("/ws/farm-1/live") as websocket:
            payload = websocket.receive_json()
            assert payload["error"] == "redis_unavailable"
    app.router.lifespan_context = original_lifespan
