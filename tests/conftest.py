"""Shared pytest fixtures — async test client, test DB, Julia bridge mocks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
	"""HTTPX async client with lifespan disabled and DB dependency mocked."""

	fake_session: Any = object()

	async def override_get_db() -> AsyncGenerator[Any, None]:
		yield fake_session

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
