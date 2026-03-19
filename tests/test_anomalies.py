from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import VertexTypeEnum
from app.schemas.anomalies import (
    AnomalyHistoryResponse,
    ThresholdListResponse,
    ThresholdRead,
    WebhookListResponse,
    WebhookRead,
    WebhookTestResponse,
)
from app.services import julia_bridge
from app.services.analytics_service import AnalyticsService
from app.services.anomaly_service import AnomalyService
from app.services.farm_service import FarmService
from app.services.webhook_service import WebhookService, run_dispatch_queue_worker


class _ScalarResult:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def all(self) -> list[Any]:
        return self._values


class _Result:
    def __init__(
        self, *, scalars_values: list[Any] | None = None, rows: list[Any] | None = None
    ) -> None:
        self._scalars_values = scalars_values or []
        self._rows = rows or []

    def scalars(self) -> _ScalarResult:
        return _ScalarResult(self._scalars_values)

    def all(self) -> list[Any]:
        return self._rows


@pytest.mark.asyncio
async def test_anomaly_history_endpoint_success(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    event_id = uuid4()

    async def fake_history(
        self: AnalyticsService,
        _farm_id: Any,
        _query: Any,
    ) -> AnomalyHistoryResponse:
        return AnomalyHistoryResponse(
            farm_id=farm_id,
            total_count=1,
            items=[
                {
                    "id": event_id,
                    "farm_id": farm_id,
                    "vertex_id": None,
                    "zone_id": None,
                    "layer": "vision",
                    "anomaly_type": "wilting",
                    "severity": "warning",
                    "feature": None,
                    "current_value": None,
                    "rolling_mean": None,
                    "rolling_std": None,
                    "sigma_deviation": None,
                    "anomaly_rules": ["rule2"],
                    "cross_layer_confirmed": False,
                    "payload": {"a": 1},
                    "detected_at": datetime.now(UTC),
                    "resolved_at": None,
                    "webhook_notified": False,
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                }
            ],
            filters_applied={},
        )

    monkeypatch.setattr(AnalyticsService, "get_anomaly_history", fake_history)

    response = await client.get(f"/api/v1/analytics/{farm_id}/anomalies/history")
    assert response.status_code == 200
    body = response.json()
    assert body["farm_id"] == str(farm_id)
    assert body["total_count"] == 1
    assert body["items"][0]["id"] == str(event_id)


@pytest.mark.asyncio
async def test_anomaly_history_filters_by_severity(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_history(
        self: AnalyticsService, _farm_id: Any, query: Any
    ) -> AnomalyHistoryResponse:
        assert str(query.severity) == "warning"
        return AnomalyHistoryResponse(farm_id=farm_id, total_count=0, items=[], filters_applied={})

    monkeypatch.setattr(AnalyticsService, "get_anomaly_history", fake_history)
    response = await client.get(f"/api/v1/analytics/{farm_id}/anomalies/history?severity=warning")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_anomaly_history_filters_by_date_range(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_history(
        self: AnalyticsService, _farm_id: Any, query: Any
    ) -> AnomalyHistoryResponse:
        assert query.since is not None
        assert query.until is not None
        return AnomalyHistoryResponse(farm_id=farm_id, total_count=0, items=[], filters_applied={})

    monkeypatch.setattr(AnalyticsService, "get_anomaly_history", fake_history)
    response = await client.get(
        f"/api/v1/analytics/{farm_id}/anomalies/history"
        "?since=2026-03-17T00:00:00Z&until=2026-03-18T00:00:00Z"
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_anomaly_history_pagination(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_history(
        self: AnalyticsService, _farm_id: Any, query: Any
    ) -> AnomalyHistoryResponse:
        assert query.limit == 25
        assert query.offset == 10
        return AnomalyHistoryResponse(farm_id=farm_id, total_count=0, items=[], filters_applied={})

    monkeypatch.setattr(AnalyticsService, "get_anomaly_history", fake_history)
    response = await client.get(f"/api/v1/analytics/{farm_id}/anomalies/history?limit=25&offset=10")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_anomaly_history_missing_farm_returns_404(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_history(
        self: AnalyticsService, _farm_id: Any, _query: Any
    ) -> AnomalyHistoryResponse:
        raise LookupError("farm not found")

    monkeypatch.setattr(AnalyticsService, "get_anomaly_history", fake_history)
    response = await client.get(f"/api/v1/analytics/{farm_id}/anomalies/history")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_threshold_201(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    farm_id = uuid4()
    threshold_id = uuid4()

    async def fake_create(self: Any, _farm_id: Any, _payload: Any) -> ThresholdRead:
        return ThresholdRead(
            id=threshold_id,
            farm_id=farm_id,
            vertex_type="sensor",
            layer="soil",
            sigma1=1.0,
            sigma2=2.0,
            sigma3=3.0,
            min_history=8,
            min_nan_run_outage=4,
            vision_anomaly_score_threshold=0.7,
            suppress_rule3_only=True,
            enabled=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    monkeypatch.setattr(AnomalyService, "create_threshold", fake_create)

    response = await client.post(
        f"/api/v1/anomalies/{farm_id}/thresholds",
        json={"vertex_type": "sensor", "layer": "soil"},
    )
    assert response.status_code == 201
    assert response.json()["id"] == str(threshold_id)


@pytest.mark.asyncio
async def test_create_threshold_duplicate_returns_409(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_create(self: Any, _farm_id: Any, _payload: Any) -> ThresholdRead:
        raise ValueError("threshold already exists for farm/vertex_type/layer")

    monkeypatch.setattr(AnomalyService, "create_threshold", fake_create)

    response = await client.post(
        f"/api/v1/anomalies/{farm_id}/thresholds",
        json={"vertex_type": "sensor", "layer": "soil"},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_list_thresholds(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    farm_id = uuid4()

    async def fake_list(self: Any, _farm_id: Any) -> ThresholdListResponse:
        return ThresholdListResponse(farm_id=farm_id, items=[])

    monkeypatch.setattr(AnomalyService, "get_thresholds", fake_list)
    response = await client.get(f"/api/v1/anomalies/{farm_id}/thresholds")
    assert response.status_code == 200
    assert response.json()["farm_id"] == str(farm_id)


@pytest.mark.asyncio
async def test_update_threshold_partial(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    threshold_id = uuid4()

    async def fake_update(
        self: Any, _farm_id: Any, _threshold_id: Any, payload: Any
    ) -> ThresholdRead:
        assert payload.sigma2 == 2.5
        return ThresholdRead(
            id=threshold_id,
            farm_id=farm_id,
            vertex_type="sensor",
            layer="soil",
            sigma1=1.0,
            sigma2=2.5,
            sigma3=3.0,
            min_history=8,
            min_nan_run_outage=4,
            vision_anomaly_score_threshold=0.7,
            suppress_rule3_only=True,
            enabled=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    monkeypatch.setattr(AnomalyService, "update_threshold", fake_update)
    response = await client.put(
        f"/api/v1/anomalies/{farm_id}/thresholds/{threshold_id}",
        json={"sigma2": 2.5},
    )
    assert response.status_code == 200
    assert response.json()["sigma2"] == 2.5


@pytest.mark.asyncio
async def test_delete_threshold_204(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    farm_id = uuid4()
    threshold_id = uuid4()

    async def fake_delete(self: Any, _farm_id: Any, _threshold_id: Any) -> None:
        return None

    monkeypatch.setattr(AnomalyService, "delete_threshold", fake_delete)
    response = await client.delete(f"/api/v1/anomalies/{farm_id}/thresholds/{threshold_id}")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_create_webhook_201_and_masked_secret(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    webhook_id = uuid4()

    async def fake_create(self: Any, _farm_id: Any, _payload: Any) -> WebhookRead:
        return WebhookRead(
            id=webhook_id,
            farm_id=farm_id,
            url="https://example.com/webhook",
            secret="***************7890",  # noqa: S106 - masked test fixture value
            event_types=["anomaly.*"],
            is_active=True,
            retry_max=3,
            last_triggered_at=None,
            last_status_code=None,
            failure_count=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    monkeypatch.setattr(WebhookService, "create_subscription", fake_create)
    response = await client.post(
        f"/api/v1/anomalies/{farm_id}/webhooks",
        json={"url": "https://example.com/webhook", "secret": "supersecretvalue7890"},
    )
    assert response.status_code == 201
    assert response.json()["secret"].endswith("7890")


@pytest.mark.asyncio
async def test_list_webhooks_masked_secret(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    webhook_id = uuid4()

    async def fake_list(self: Any, _farm_id: Any) -> WebhookListResponse:
        return WebhookListResponse(
            farm_id=farm_id,
            items=[
                WebhookRead(
                    id=webhook_id,
                    farm_id=farm_id,
                    url="https://example.com/webhook",
                    secret="***************abcd",  # noqa: S106 - masked test fixture value
                    event_types=["anomaly.critical"],
                    is_active=True,
                    retry_max=3,
                    last_triggered_at=None,
                    last_status_code=None,
                    failure_count=0,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            ],
        )

    monkeypatch.setattr(WebhookService, "list_subscriptions", fake_list)
    response = await client.get(f"/api/v1/anomalies/{farm_id}/webhooks")
    assert response.status_code == 200
    assert response.json()["items"][0]["secret"].endswith("abcd")


@pytest.mark.asyncio
async def test_webhook_test_ping_endpoint(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    webhook_id = uuid4()

    async def fake_test(self: Any, _farm_id: Any, _webhook_id: Any) -> WebhookTestResponse:
        return WebhookTestResponse(
            webhook_id=webhook_id,
            delivered=True,
            status_code=200,
            detail="delivered",
        )

    monkeypatch.setattr(WebhookService, "test_subscription", fake_test)
    response = await client.post(f"/api/v1/anomalies/{farm_id}/webhooks/{webhook_id}/test")
    assert response.status_code == 200
    assert response.json()["delivered"] is True


@pytest.mark.asyncio
async def test_delete_webhook_204(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    farm_id = uuid4()
    webhook_id = uuid4()

    async def fake_delete(self: Any, _farm_id: Any, _webhook_id: Any) -> None:
        return None

    monkeypatch.setattr(WebhookService, "delete_subscription", fake_delete)
    response = await client.delete(f"/api/v1/anomalies/{farm_id}/webhooks/{webhook_id}")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_severity_classification_mapping() -> None:
    service = AnomalyService(cast(AsyncSession, object()), None)

    critical = service._classify_severity(
        {"severity": "alarm", "cross_layer_confirmed": True}, None, None, None
    )
    warning = service._classify_severity({"severity": "warning"}, None, None, None)
    info = service._classify_severity({"anomaly_type": "nutrient_deficiency"}, None, None, None)

    assert critical.value == "critical"
    assert warning.value == "warning"
    assert info.value == "info"


@pytest.mark.asyncio
async def test_threshold_filtering_suppresses_rule3_only() -> None:
    service = AnomalyService(cast(AsyncSession, object()), None)
    threshold = SimpleNamespace(
        enabled=True,
        min_history=8,
        vision_anomaly_score_threshold=0.7,
        suppress_rule3_only=True,
        sigma1=1.0,
        sigma2=2.0,
        sigma3=3.0,
        min_nan_run_outage=4,
    )

    item = {
        "vertex_id": str(uuid4()),
        "layer": "soil",
        "anomaly_rules": ["rule3"],
        "sigma_deviation": 3.4,
        "history_count": 16,
    }

    vertex_id = UUID(item["vertex_id"])
    result = service._normalize_and_filter(
        item,
        {vertex_id: (VertexTypeEnum.sensor, None)},
        {(VertexTypeEnum.sensor, None): threshold},
    )
    assert result is None


@pytest.mark.asyncio
async def test_detect_and_persist_inserts_events(monkeypatch: pytest.MonkeyPatch) -> None:
    farm_id = uuid4()
    vertex_id = uuid4()
    zone_id = uuid4()

    class FakeDb:
        def __init__(self) -> None:
            self.execute = AsyncMock(
                side_effect=[
                    _Result(scalars_values=[]),
                    _Result(rows=[(vertex_id, VertexTypeEnum.sensor, zone_id)]),
                    _Result(scalars_values=[]),
                ]
            )
            self.flush = AsyncMock()
            self.add_all = lambda _events: None

    fake_db = FakeDb()
    service = AnomalyService(cast(AsyncSession, fake_db), None)

    async def fake_get_farm(self: FarmService, _farm_id: Any) -> Any:
        return SimpleNamespace(id=farm_id)

    monkeypatch.setattr(FarmService, "get_farm", fake_get_farm)
    monkeypatch.setattr(
        julia_bridge,
        "detect_anomalies",
        lambda _farm_id: [
            {
                "vertex_id": str(vertex_id),
                "layer": "soil",
                "severity": "warning",
                "anomaly_type": "wilting",
                "history_count": 12,
            }
        ],
    )

    def fake_create_task(coro: Any) -> Any:
        coro.close()
        return SimpleNamespace(add_done_callback=lambda _cb: None)

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    events = await service.detect_and_persist(farm_id)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_hmac_signature_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        content: bytes,
        headers: dict[str, str],
    ) -> Any:
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = headers
        return SimpleNamespace(status_code=200, text="ok")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    service = WebhookService(cast(AsyncSession, object()), None)
    subscription = SimpleNamespace(
        url="https://example.com/webhook",
        secret="my-signing-secret",  # noqa: S106 - deterministic test key
        retry_max=3,
    )
    payload = {"event": "anomaly", "items": [{"severity": "critical"}]}

    ok, status_code, _ = await service._post_with_retry(
        subscription, payload, event_header="anomaly"
    )
    assert ok is True
    assert status_code == 200

    expected_sig = hmac.new(
        b"my-signing-secret",
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert captured["headers"]["X-AgriSense-Signature"] == f"sha256={expected_sig}"


@pytest.mark.asyncio
async def test_webhook_retry_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        content: bytes,
        headers: dict[str, str],
    ) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(status_code=500, text="server error")
        return SimpleNamespace(status_code=200, text="ok")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(WebhookService, "_backoff_sleep", AsyncMock())

    service = WebhookService(cast(AsyncSession, object()), None)
    subscription = SimpleNamespace(
        url="https://example.com/webhook",
        secret="retry-secret",  # noqa: S106 - deterministic test key
        retry_max=3,
    )

    ok, status_code, _ = await service._post_with_retry(
        subscription, {"event": "anomaly"}, event_header="anomaly"
    )
    assert ok is True
    assert status_code == 200
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_event_type_filter_matching() -> None:
    service = WebhookService(cast(AsyncSession, object()), None)
    event = SimpleNamespace(severity=SimpleNamespace(value="critical"))

    assert service._event_matches(["anomaly.*"], event) is True
    assert service._event_matches(["anomaly.critical"], event) is True
    assert service._event_matches(["anomaly.warning"], event) is False


@pytest.mark.asyncio
async def test_webhook_dispatch_worker_processes_queued_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    farm_id = uuid4()
    event_id = uuid4()

    class FakeRedis:
        def __init__(self) -> None:
            self.calls = 0

        async def blpop(self, _key: str, timeout: int = 0) -> Any:
            self.calls += 1
            if self.calls == 1:
                payload = json.dumps(
                    {
                        "farm_id": str(farm_id),
                        "event_ids": [str(event_id)],
                    }
                )
                return ("anomaly:webhook:dispatch", payload)
            await asyncio.sleep(0.01)
            return None

    class _FakeSession:
        def __init__(self) -> None:
            self.execute = AsyncMock(return_value=_Result(scalars_values=[SimpleNamespace()]))
            self.commit = AsyncMock()

    class _FakeSessionCtx:
        def __init__(self, session: _FakeSession) -> None:
            self.session = session

        async def __aenter__(self) -> _FakeSession:
            return self.session

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            return False

    fake_session = _FakeSession()
    fake_redis = FakeRedis()
    dispatched: list[tuple[Any, Any]] = []

    def fake_session_factory() -> _FakeSessionCtx:
        return _FakeSessionCtx(fake_session)

    async def fake_dispatch(self: WebhookService, _farm_id: Any, events: Any) -> None:
        dispatched.append((_farm_id, events))

    monkeypatch.setattr("app.services.webhook_service.async_session_factory", fake_session_factory)
    monkeypatch.setattr(WebhookService, "dispatch_anomaly_events", fake_dispatch)

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        run_dispatch_queue_worker(cast(Any, fake_redis), stop_event, block_seconds=1)
    )

    await asyncio.sleep(0.05)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert dispatched
    assert dispatched[0][0] == farm_id
    assert len(dispatched[0][1]) == 1
    fake_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_anomaly_openapi_contract(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/analytics/{farm_id}/anomalies/history" in paths
    assert "/api/v1/anomalies/{farm_id}/thresholds" in paths
    assert "/api/v1/anomalies/{farm_id}/thresholds/{threshold_id}" in paths
    assert "/api/v1/anomalies/{farm_id}/webhooks" in paths
    assert "/api/v1/anomalies/{farm_id}/webhooks/{webhook_id}" in paths
    assert "/api/v1/anomalies/{farm_id}/webhooks/{webhook_id}/test" in paths
