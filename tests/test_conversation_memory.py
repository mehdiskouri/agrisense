from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services import conversation_memory


class _FakeRedisClient:
    def __init__(self) -> None:
        self.expire = AsyncMock(return_value=True)
        self.delete = AsyncMock(return_value=1)
        self.aclose = AsyncMock(return_value=None)


@pytest.mark.asyncio
async def test_refresh_ttl_uses_history_key_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(conversation_memory, "from_url", lambda *args, **kwargs: fake_client)

    await conversation_memory.refresh_ttl("redis://localhost:6379/0", "conv-123", 3600)

    fake_client.expire.assert_awaited_once_with("message_store:conv-123", 3600)
    fake_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_clear_conversation_uses_history_key_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeRedisClient()
    monkeypatch.setattr(conversation_memory, "from_url", lambda *args, **kwargs: fake_client)

    farm_id = uuid4()
    user_id = uuid4()
    conversation_id = await conversation_memory.clear_conversation(
        "redis://localhost:6379/0", farm_id, user_id
    )

    expected = f"conversation:{farm_id}:{user_id}"
    assert conversation_id == expected
    fake_client.delete.assert_awaited_once_with(f"message_store:{expected}")
    fake_client.aclose.assert_awaited_once()
