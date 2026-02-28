"""Shared pytest fixtures — async test client, test DB, Julia bridge mocks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from app.services import julia_bridge


class FakeAsyncSession:
	def __init__(self) -> None:
		self.commit = AsyncMock()
		self.rollback = AsyncMock()
		self.close = AsyncMock()
		self.execute = AsyncMock()


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
	monkeypatch.setattr(julia_bridge, "query_farm_status", lambda _graph, _zone: {"status": "ok"})
	return graph_state


@pytest.fixture
async def client(fake_db_session: FakeAsyncSession) -> AsyncGenerator[AsyncClient, None]:
	"""HTTPX async client with lifespan disabled and DB dependency mocked."""

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
