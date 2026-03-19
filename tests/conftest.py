"""Shared pytest fixtures — async test client, test DB, Julia bridge mocks."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import AuthPrincipal, get_auth_principal, get_current_user
from app.auth.jwt import create_access_token
from app.database import get_db
from app.main import app
from app.models.enums import UserRoleEnum
from app.services import julia_bridge


class FakeAsyncSession:
    def __init__(self) -> None:
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.close = AsyncMock()
        self.execute = AsyncMock()


class FakePubSub:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.index = 0
        self.subscribed_channel: str | None = None
        self.unsubscribed_channel: str | None = None
        self.closed = False

    async def subscribe(self, _channel: str) -> None:
        self.subscribed_channel = _channel
        return None

    async def get_message(
        self, ignore_subscribe_messages: bool, timeout: float
    ) -> dict[str, Any] | None:
        if self.index >= len(self.payloads):
            return None
        message = self.payloads[self.index]
        self.index += 1
        return message

    async def unsubscribe(self, _channel: str) -> None:
        self.unsubscribed_channel = _channel
        return None

    async def close(self) -> None:
        self.closed = True
        return None


class FakeRedis:
    def __init__(self, payloads: list[dict[str, Any]] | None = None) -> None:
        self.payloads = payloads or []
        self.publish = AsyncMock()
        self.setex = AsyncMock()
        self.get = AsyncMock(return_value=None)
        self._counter: dict[str, int] = {}
        self.incr = AsyncMock(side_effect=self._incr)
        self.expire = AsyncMock(return_value=True)
        self.last_pubsub: FakePubSub | None = None

    async def _incr(self, key: str) -> int:
        value = self._counter.get(key, 0) + 1
        self._counter[key] = value
        return value

    def pubsub(self) -> FakePubSub:
        self.last_pubsub = FakePubSub(self.payloads)
        return self.last_pubsub

    def reset_counters(self) -> None:
        self._counter.clear()


@pytest.fixture
def fake_db_session() -> FakeAsyncSession:
    """A lightweight async-session stub for dependency overrides in API tests."""
    return FakeAsyncSession()


@pytest.fixture
def bridge_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Deterministic Julia-bridge stubs for API/service tests that need them."""

    graph_state = {
        "farm_id": "stub-farm",
        "n_vertices": 1,
        "vertex_index": {"v1": 1},
        "layers": {},
    }

    monkeypatch.setattr(julia_bridge, "build_graph", lambda _config: graph_state)
    monkeypatch.setattr(julia_bridge, "query_farm_status", lambda _farm_id, _zone: {"status": "ok"})
    monkeypatch.setattr(
        julia_bridge,
        "yield_forecast_ensemble",
        lambda _farm_id, include_members=False: [
            {
                "crop_bed_id": "bed-1",
                "yield_estimate_kg_m2": 4.0,
                "yield_lower": 3.5,
                "yield_upper": 4.6,
                "confidence": 0.9,
                "model_layer": "ensemble",
                "stress_factors": {},
                "ensemble_weights": {
                    "fao_single": 1.0 / 3.0,
                    "exp_smoothing": 1.0 / 3.0,
                    "quantile_regression": 1.0 / 3.0,
                },
                "ensemble_members": []
                if not include_members
                else [
                    {
                        "model_name": "fao_single",
                        "yield_estimate": 4.0,
                        "lower": 3.5,
                        "upper": 4.6,
                        "weight": 1.0 / 3.0,
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(
        julia_bridge,
        "backtest_yield",
        lambda _farm_id, n_folds=5, min_history=24: {
            "farm_id": _farm_id,
            "n_folds": n_folds,
            "status": "ok",
            "per_fold_metrics": [],
            "aggregate_metrics": {},
            "weights": {
                "fao_single": 1.0 / 3.0,
                "exp_smoothing": 1.0 / 3.0,
                "quantile_regression": 1.0 / 3.0,
            },
            "hyperparameters": {
                "exp_alpha": 0.3,
                "exp_beta": 0.1,
                "quantile_lambda": 0.001,
            },
        },
    )
    monkeypatch.setattr(
        julia_bridge,
        "nutrient_report",
        lambda _farm_id: [
            {
                "zone_id": "zone-1",
                "nitrogen_deficit": 0.12,
                "phosphorus_deficit": 0.08,
                "potassium_deficit": 0.06,
                "urgency": "medium",
            }
        ],
    )
    monkeypatch.setattr(
        julia_bridge,
        "detect_anomalies",
        lambda _farm_id, thresholds=None: [
            {
                "zone_id": "zone-1",
                "layer": "vision",
                "severity": "warning",
                "anomaly_type": "wilting",
            }
        ],
    )
    return graph_state


@pytest.fixture
def fake_redis() -> FakeRedis:
    """Reusable fake Redis client with async publish and pubsub behavior."""
    return FakeRedis()


@pytest.fixture
async def client(fake_db_session: FakeAsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """HTTPX async client with lifespan disabled and DB dependency mocked."""

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield fake_db_session

    async def override_current_user() -> Any:
        return type(
            "UserStub",
            (),
            {
                "id": uuid.uuid4(),
                "role": UserRoleEnum.admin,
                "is_active": True,
                "email": "admin@test.local",
            },
        )()

    async def override_auth_principal() -> AuthPrincipal:
        return AuthPrincipal(
            auth_type="api_key",
            subject_id=uuid.uuid4(),
            role=UserRoleEnum.field_operator,
            scopes={"ingest", "jobs"},
            api_key_id=uuid.uuid4(),
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_auth_principal] = override_auth_principal
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(_: Any) -> AsyncGenerator[None, None]:
        yield

    app.router.lifespan_context = noop_lifespan

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client

    app.router.lifespan_context = original_lifespan
    app.dependency_overrides.clear()


@pytest.fixture
async def auth_client(fake_db_session: FakeAsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """HTTPX async client with DB override only (real auth dependencies active)."""

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield fake_db_session

    app.dependency_overrides[get_db] = override_get_db
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(_: Any) -> AsyncGenerator[None, None]:
        yield

    app.router.lifespan_context = noop_lifespan

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client

    app.router.lifespan_context = original_lifespan
    app.dependency_overrides.clear()


@pytest.fixture
def auth_user_id() -> uuid.UUID:
    return uuid.UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def access_token(auth_user_id: uuid.UUID) -> str:
    return create_access_token(str(auth_user_id), expires_minutes=30)


@pytest.fixture
def ws_token(access_token: str) -> str:
    return access_token


@pytest.fixture
def api_key_plaintext() -> str:
    return "phase8-test-key"


@pytest.fixture
def api_key_sha256(api_key_plaintext: str) -> str:
    return hashlib.sha256(api_key_plaintext.encode("utf-8")).hexdigest()


@pytest.fixture
def now_utc() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def fake_chat_model() -> type:
    """Simple fake chat model class for LangChain tests via monkeypatching."""

    class FakeChatModel:
        def __init__(self, **_kwargs: Any) -> None:
            self.kwargs = _kwargs

    return FakeChatModel
