from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from httpx import AsyncClient

from app.auth.dependencies import (
    AuthPrincipal,
    _api_key_digest,
    _resolve_api_key,
    require_machine_scope,
)
from app.auth.jwt import AuthError, create_access_token, decode_token
from app.auth.models import APIKey
from app.main import app
from app.models.enums import UserRoleEnum
from app.services.farm_service import FarmService


class _ScalarRow:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object:
        return self._value


class _ScalarsResult:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def all(self) -> list[object]:
        return self._values


class _ListRow:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def scalars(self) -> _ScalarsResult:
        return _ScalarsResult(self._values)


@pytest.mark.asyncio
async def test_missing_jwt_rejected_on_protected_endpoint(auth_client: AsyncClient) -> None:
    response = await auth_client.get(f"/api/v1/analytics/{uuid4()}/status")
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "auth_required"


def test_jwt_create_decode_roundtrip(auth_user_id: UUID) -> None:
    token = create_access_token(str(auth_user_id), expires_minutes=5)
    payload = decode_token(token, expected_type="access")
    assert payload["sub"] == str(auth_user_id)
    assert payload["typ"] == "access"


def test_decode_invalid_token_raises_auth_error() -> None:
    with pytest.raises(AuthError):
        decode_token("invalid.token.payload", expected_type="access")


@pytest.mark.asyncio
async def test_rbac_forbidden_for_readonly_create_farm(client: AsyncClient) -> None:
    async def _readonly_user() -> object:
        return SimpleNamespace(
            id=uuid4(),
            role=UserRoleEnum.readonly,
            is_active=True,
            email="readonly@test.local",
        )

    from app.auth.dependencies import get_current_user

    app.dependency_overrides[get_current_user] = _readonly_user

    response = await client.post(
        "/api/v1/farms",
        json={"name": "demo", "farm_type": "greenhouse", "timezone": "UTC"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "forbidden"


@pytest.mark.asyncio
async def test_rate_limit_per_farm(
    fake_redis: object, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    app.state.redis = fake_redis

    @dataclass
    class _SettingsStub:
        rate_limit_user_per_minute: int = 1
        rate_limit_api_key_per_minute: int = 1
        api_key_header_name: str = "x-api-key"

    async def fake_graph(self: FarmService, _farm_id: object) -> dict[str, object]:
        return {
            "farm_id": str(farm_id),
            "n_vertices": 0,
            "vertex_index": {},
            "layers": {},
        }

    monkeypatch.setattr("app.middleware.rate_limit.get_settings", lambda: _SettingsStub())
    monkeypatch.setattr(FarmService, "get_graph", fake_graph)

    first = await client.get(f"/api/v1/farms/{farm_id}/graph")
    second = await client.get(f"/api/v1/farms/{farm_id}/graph")

    assert first.status_code in {200, 404}
    assert second.status_code == 429
    assert second.json()["detail"]["error"] == "rate_limited"


@pytest.mark.asyncio
async def test_machine_scope_enforcement_for_api_key_principal() -> None:
    dependency = require_machine_scope("jobs")
    principal = AuthPrincipal(
        auth_type="api_key",
        subject_id=uuid4(),
        role=UserRoleEnum.field_operator,
        scopes={"ingest"},
        api_key_id=uuid4(),
    )

    with pytest.raises(HTTPException):
        await dependency(principal)

    allowed = AuthPrincipal(
        auth_type="api_key",
        subject_id=uuid4(),
        role=UserRoleEnum.field_operator,
        scopes={"jobs"},
        api_key_id=uuid4(),
    )
    result = await dependency(allowed)
    assert result.scopes == {"jobs"}


@pytest.mark.asyncio
async def test_machine_scope_rejects_jwt_principal() -> None:
    dependency = require_machine_scope("ingest")
    principal = AuthPrincipal(
        auth_type="jwt",
        subject_id=uuid4(),
        role=UserRoleEnum.admin,
        scopes=set(),
        api_key_id=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await dependency(principal)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["error"] == "api_key_required"


@pytest.mark.asyncio
async def test_machine_scope_accepts_write_prefixed_legacy_scope() -> None:
    dependency = require_machine_scope("ingest")
    principal = AuthPrincipal(
        auth_type="api_key",
        subject_id=uuid4(),
        role=UserRoleEnum.field_operator,
        scopes={"write:ingest"},
        api_key_id=uuid4(),
    )

    result = await dependency(principal)
    assert result.api_key_id is not None


@pytest.mark.asyncio
async def test_resolve_api_key_prefers_digest_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    plaintext = "test-machine-key"
    digest = _api_key_digest(plaintext)
    api_key = APIKey(
        user_id=uuid4(),
        key_hash=digest,
        name="digest",
        scopes={"ingest": True},
        expires_at=datetime.now(UTC) + timedelta(days=1),
        is_active=True,
    )

    fake_db = SimpleNamespace(execute=AsyncMock(side_effect=[_ScalarRow(api_key)]))
    verify = AsyncMock(return_value=False)
    monkeypatch.setattr("app.auth.dependencies.pwd_context.verify", verify)

    resolved = await _resolve_api_key(fake_db, plaintext)  # type: ignore[arg-type]
    assert resolved is api_key
    verify.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_api_key_legacy_bcrypt_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    plaintext = "legacy-bcrypt-key"
    api_key = APIKey(
        user_id=uuid4(),
        key_hash="$2b$12$legacyplaceholderhash",
        name="legacy",
        scopes={"ingest": True},
        expires_at=datetime.now(UTC) + timedelta(days=1),
        is_active=True,
    )

    fake_db = SimpleNamespace(
        execute=AsyncMock(side_effect=[_ScalarRow(None), _ListRow([api_key])])
    )
    monkeypatch.setattr(
        "app.auth.dependencies.pwd_context.verify",
        lambda plain, _: plain == plaintext,
    )

    resolved = await _resolve_api_key(fake_db, plaintext)  # type: ignore[arg-type]
    assert resolved is api_key


@pytest.mark.asyncio
async def test_rate_limit_applies_to_ingest_body_farm_id(
    fake_redis: object,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    farm_id = uuid4()
    app.state.redis = fake_redis

    @dataclass
    class _SettingsStub:
        rate_limit_user_per_minute: int = 1
        rate_limit_api_key_per_minute: int = 1
        api_key_header_name: str = "x-api-key"

    monkeypatch.setattr("app.middleware.rate_limit.get_settings", lambda: _SettingsStub())

    async def _fake_ingest(self: object, _farm_id: object, _readings: object) -> object:
        return {
            "farm_id": str(farm_id),
            "layer": "soil",
            "status": "ok",
            "inserted_count": 1,
            "failed_count": 0,
            "event_ids": [1],
            "warnings": [],
            "timestamp_start": None,
            "timestamp_end": None,
        }

    from app.services.ingest_service import IngestService

    monkeypatch.setattr(IngestService, "ingest_soil", _fake_ingest)

    payload = {
        "farm_id": str(farm_id),
        "readings": [
            {
                "sensor_id": str(uuid4()),
                "timestamp": "2026-01-01T00:00:00Z",
                "moisture": 0.3,
                "temperature": 23.0,
            }
        ],
    }

    first = await client.post("/api/v1/ingest/soil", json=payload)
    second = await client.post("/api/v1/ingest/soil", json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"]["error"] == "rate_limited"
