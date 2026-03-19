"""Webhook subscription CRUD and anomaly event dispatching."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime
from typing import Any, cast

import httpx
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.anomalies import AnomalyEvent, WebhookSubscription
from app.schemas.anomalies import (
    WebhookCreate,
    WebhookListResponse,
    WebhookRead,
    WebhookTestResponse,
    WebhookUpdate,
)
from app.services.farm_service import FarmService

DISPATCH_QUEUE_KEY = "anomaly:webhook:dispatch"
DISPATCH_QUEUE_BLOCK_SECONDS = 2


class WebhookService:
    """Manages webhook subscriptions and signed event delivery."""

    def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
        self.db = db
        self.redis_client = redis_client

    async def create_subscription(
        self,
        farm_id: uuid.UUID,
        payload: WebhookCreate,
    ) -> WebhookRead:
        await FarmService(self.db).get_farm(farm_id)
        subscription = WebhookSubscription(
            farm_id=farm_id,
            url=str(payload.url),
            secret=payload.secret,
            event_types=payload.event_types,
            is_active=payload.is_active,
            retry_max=payload.retry_max,
        )
        self.db.add(subscription)
        await self.db.flush()
        await self.db.refresh(subscription)
        return self._to_webhook_read(subscription)

    async def list_subscriptions(self, farm_id: uuid.UUID) -> WebhookListResponse:
        await FarmService(self.db).get_farm(farm_id)
        rows = await self.db.execute(
            select(WebhookSubscription)
            .where(WebhookSubscription.farm_id == farm_id)
            .order_by(WebhookSubscription.created_at.asc())
        )
        items = [self._to_webhook_read(item) for item in rows.scalars().all()]
        return WebhookListResponse(farm_id=farm_id, items=items)

    async def get_subscription(self, farm_id: uuid.UUID, webhook_id: uuid.UUID) -> WebhookRead:
        subscription = await self._require_subscription(farm_id, webhook_id)
        return self._to_webhook_read(subscription)

    async def update_subscription(
        self,
        farm_id: uuid.UUID,
        webhook_id: uuid.UUID,
        payload: WebhookUpdate,
    ) -> WebhookRead:
        subscription = await self._require_subscription(farm_id, webhook_id)
        updates = payload.model_dump(exclude_none=True)
        if "url" in updates:
            updates["url"] = str(updates["url"])

        for key, value in updates.items():
            setattr(subscription, key, value)

        await self.db.flush()
        await self.db.refresh(subscription)
        return self._to_webhook_read(subscription)

    async def delete_subscription(self, farm_id: uuid.UUID, webhook_id: uuid.UUID) -> None:
        subscription = await self._require_subscription(farm_id, webhook_id)
        await self.db.delete(subscription)
        await self.db.flush()

    async def test_subscription(
        self,
        farm_id: uuid.UUID,
        webhook_id: uuid.UUID,
    ) -> WebhookTestResponse:
        subscription = await self._require_subscription(farm_id, webhook_id)
        payload: dict[str, Any] = {
            "event": "anomaly.test",
            "farm_id": str(farm_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "subscription_id": str(subscription.id),
        }

        ok, status_code, _ = await self._post_with_retry(
            subscription,
            payload,
            event_header="anomaly.test",
        )
        subscription.last_triggered_at = datetime.now(UTC)
        subscription.last_status_code = status_code
        subscription.failure_count = 0 if ok else subscription.failure_count + 1
        await self.db.flush()

        return WebhookTestResponse(
            webhook_id=subscription.id,
            delivered=ok,
            status_code=status_code,
            detail="delivered" if ok else "delivery_failed",
        )

    async def dispatch_anomaly_events(self, farm_id: uuid.UUID, events: list[AnomalyEvent]) -> None:
        subscriptions = await self._active_subscriptions(farm_id)
        if not subscriptions or not events:
            return

        for subscription in subscriptions:
            matched_events = [
                event for event in events if self._event_matches(subscription.event_types, event)
            ]
            if not matched_events:
                continue

            payload_events = [self._event_payload(event) for event in matched_events]
            payload: dict[str, Any] = {
                "event": "anomaly",
                "farm_id": str(farm_id),
                "sent_at": datetime.now(UTC).isoformat(),
                "subscription_id": str(subscription.id),
                "items": payload_events,
            }
            ok, status_code, _ = await self._post_with_retry(
                subscription,
                payload,
                event_header="anomaly",
            )

            subscription.last_triggered_at = datetime.now(UTC)
            subscription.last_status_code = status_code
            subscription.failure_count = 0 if ok else subscription.failure_count + 1
            if ok:
                for event in matched_events:
                    event.webhook_notified = True

        await self.db.flush()

    async def enqueue_dispatch_job(self, farm_id: uuid.UUID, event_ids: list[uuid.UUID]) -> bool:
        if self.redis_client is None or not event_ids:
            return False
        redis = cast(Any, self.redis_client)
        payload = {
            "farm_id": str(farm_id),
            "event_ids": [str(event_id) for event_id in event_ids],
            "enqueued_at": datetime.now(UTC).isoformat(),
        }
        await redis.rpush(DISPATCH_QUEUE_KEY, json.dumps(payload))
        return True

    async def process_dispatch_queue_once(self) -> bool:
        if self.redis_client is None:
            return False
        redis = cast(Any, self.redis_client)

        raw = await redis.lpop(DISPATCH_QUEUE_KEY)
        if raw is None:
            return False

        decoded = self._decode_queue_payload(raw)
        if decoded is None:
            return True

        farm_id, event_ids = decoded
        rows = await self.db.execute(
            select(AnomalyEvent).where(
                AnomalyEvent.farm_id == farm_id,
                AnomalyEvent.id.in_(event_ids),
            )
        )
        events = list(rows.scalars().all())

        if not events:
            # Events might not be committed yet in the producer transaction; retry later.
            retry_payload = {
                "farm_id": str(farm_id),
                "event_ids": [str(event_id) for event_id in event_ids],
            }
            await redis.rpush(
                DISPATCH_QUEUE_KEY,
                json.dumps(retry_payload),
            )
            return False

        await self.dispatch_anomaly_events(farm_id, events)
        return True

    async def _active_subscriptions(self, farm_id: uuid.UUID) -> list[WebhookSubscription]:
        rows = await self.db.execute(
            select(WebhookSubscription)
            .where(
                WebhookSubscription.farm_id == farm_id,
                WebhookSubscription.is_active.is_(True),
            )
            .order_by(WebhookSubscription.created_at.asc())
        )
        return list(rows.scalars().all())

    async def _require_subscription(
        self,
        farm_id: uuid.UUID,
        webhook_id: uuid.UUID,
    ) -> WebhookSubscription:
        await FarmService(self.db).get_farm(farm_id)
        rows = await self.db.execute(
            select(WebhookSubscription).where(WebhookSubscription.id == webhook_id)
        )
        subscription = rows.scalar_one_or_none()
        if subscription is None or subscription.farm_id != farm_id:
            raise LookupError(f"Webhook {webhook_id} not found")
        return subscription

    @staticmethod
    def _event_payload(event: AnomalyEvent) -> dict[str, Any]:
        return {
            "id": str(event.id),
            "farm_id": str(event.farm_id),
            "vertex_id": str(event.vertex_id) if event.vertex_id is not None else None,
            "zone_id": str(event.zone_id) if event.zone_id is not None else None,
            "layer": event.layer,
            "anomaly_type": event.anomaly_type,
            "severity": event.severity.value,
            "feature": event.feature,
            "current_value": event.current_value,
            "rolling_mean": event.rolling_mean,
            "rolling_std": event.rolling_std,
            "sigma_deviation": event.sigma_deviation,
            "anomaly_rules": event.anomaly_rules,
            "cross_layer_confirmed": event.cross_layer_confirmed,
            "detected_at": event.detected_at.isoformat(),
            "payload": event.payload,
        }

    @staticmethod
    def _event_matches(event_types: list[str], event: AnomalyEvent) -> bool:
        if not event_types:
            return True

        event_name = f"anomaly.{event.severity.value}"
        for candidate in event_types:
            token = str(candidate).strip().lower()
            if token in {"anomaly", "anomaly.*", "*"}:
                return True
            if token == event_name:
                return True
        return False

    @staticmethod
    def _masked_secret(secret: str) -> str:
        if len(secret) <= 4:
            return "*" * len(secret)
        return f"{'*' * (len(secret) - 4)}{secret[-4:]}"

    @staticmethod
    def _decode_queue_payload(raw: str | bytes) -> tuple[uuid.UUID, list[uuid.UUID]] | None:
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            payload = json.loads(text)
            farm_id = uuid.UUID(str(payload["farm_id"]))
            ids_raw = payload.get("event_ids")
            if not isinstance(ids_raw, list):
                return None
            event_ids = [uuid.UUID(str(item)) for item in ids_raw]
            if not event_ids:
                return None
            return farm_id, event_ids
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _to_webhook_read(self, subscription: WebhookSubscription) -> WebhookRead:
        return WebhookRead(
            id=subscription.id,
            farm_id=subscription.farm_id,
            url=subscription.url,
            secret=self._masked_secret(subscription.secret),
            event_types=list(subscription.event_types),
            is_active=subscription.is_active,
            retry_max=subscription.retry_max,
            last_triggered_at=subscription.last_triggered_at,
            last_status_code=subscription.last_status_code,
            failure_count=subscription.failure_count,
            created_at=subscription.created_at,
            updated_at=subscription.updated_at,
        )

    async def _post_with_retry(
        self,
        subscription: WebhookSubscription,
        payload: dict[str, Any],
        *,
        event_header: str,
    ) -> tuple[bool, int | None, str | None]:
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hmac.new(
            subscription.secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-AgriSense-Signature": f"sha256={digest}",
            "X-AgriSense-Event": event_header,
        }

        attempts = max(1, int(subscription.retry_max))
        status_code: int | None = None
        last_error: str | None = None

        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(attempts):
                try:
                    response = await client.post(
                        subscription.url, content=payload_bytes, headers=headers
                    )
                    status_code = response.status_code
                    if 200 <= response.status_code < 300:
                        return True, status_code, None
                    if response.status_code < 500:
                        return False, status_code, response.text
                    last_error = response.text
                except httpx.TimeoutException as exc:
                    last_error = str(exc)
                except httpx.RequestError as exc:
                    last_error = str(exc)

                if attempt < attempts - 1:
                    await self._backoff_sleep(attempt)

        return False, status_code, last_error

    @staticmethod
    async def _backoff_sleep(attempt: int) -> None:
        # 1s, 2s, 4s exponential backoff for retry attempts.
        delay_seconds = float(2**attempt)
        await asyncio.sleep(delay_seconds)


async def run_dispatch_queue_worker(
    redis_client: Redis,
    stop_event: asyncio.Event,
    *,
    block_seconds: int = DISPATCH_QUEUE_BLOCK_SECONDS,
) -> None:
    """Continuously consume queued webhook dispatch jobs from Redis.

    This worker is designed to run in app lifespan as a background task.
    """
    redis = cast(Any, redis_client)

    while not stop_event.is_set():
        raw_item: Any | None = None
        try:
            raw = await redis.blpop(DISPATCH_QUEUE_KEY, timeout=max(1, int(block_seconds)))
            if raw is None:
                continue

            # Redis returns (queue_name, payload)
            raw_item = raw[1] if isinstance(raw, (tuple, list)) and len(raw) == 2 else raw

            async with async_session_factory() as session:
                service = WebhookService(session, redis_client)
                decoded = service._decode_queue_payload(raw_item)
                if decoded is None:
                    continue

                farm_id, event_ids = decoded
                rows = await session.execute(
                    select(AnomalyEvent).where(
                        AnomalyEvent.farm_id == farm_id,
                        AnomalyEvent.id.in_(event_ids),
                    )
                )
                events = list(rows.scalars().all())

                # If producer transaction has not committed yet, re-queue for later.
                if not events:
                    await service.enqueue_dispatch_job(farm_id, event_ids)
                    continue

                await service.dispatch_anomaly_events(farm_id, events)
                await session.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Keep worker alive; failed payloads will be retried by producer paths.
            await asyncio.sleep(1.0)
