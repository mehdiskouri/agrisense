from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app import database


class _FakeSession:
    def __init__(self, *, has_writes: bool) -> None:
        self.new = {object()} if has_writes else set()
        self.dirty = set()
        self.deleted = set()
        self._in_transaction = not has_writes
        self.commit = AsyncMock(return_value=None)
        self.rollback = AsyncMock(return_value=None)
        self.close = AsyncMock(return_value=None)

    def in_transaction(self) -> bool:
        return self._in_transaction


@pytest.mark.asyncio
async def test_get_db_commits_when_session_has_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession(has_writes=True)

    @asynccontextmanager
    async def _fake_factory() -> AsyncIterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(database, "async_session_factory", _fake_factory)

    dependency = database.get_db()
    yielded = await dependency.__anext__()
    assert yielded is fake_session

    with pytest.raises(StopAsyncIteration):
        await dependency.__anext__()

    fake_session.commit.assert_awaited_once()
    fake_session.rollback.assert_not_awaited()
    fake_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_db_rolls_back_read_only_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession(has_writes=False)

    @asynccontextmanager
    async def _fake_factory() -> AsyncIterator[_FakeSession]:
        yield fake_session

    monkeypatch.setattr(database, "async_session_factory", _fake_factory)

    dependency = database.get_db()
    yielded = await dependency.__anext__()
    assert yielded is fake_session

    with pytest.raises(StopAsyncIteration):
        await dependency.__anext__()

    fake_session.commit.assert_not_awaited()
    fake_session.rollback.assert_awaited_once()
    fake_session.close.assert_awaited_once()
