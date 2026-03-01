"""Redis-backed rate limiting middleware."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.dependencies import extract_identity_hint, extract_request_farm_id
from app.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
	"""Per-farm quota limiter backed by Redis atomic counters."""

	async def dispatch(self, request: Request, call_next):  # type: ignore[override]
		if self._is_bypass_path(request.url.path):
			return await call_next(request)

		farm_id = extract_request_farm_id(request)
		if farm_id is None:
			return await call_next(request)

		redis_client = getattr(request.app.state, "redis", None)
		if redis_client is None:
			return await call_next(request)

		settings = get_settings()
		identity = extract_identity_hint(request)
		quota = (
			settings.rate_limit_api_key_per_minute
			if identity == "api_key"
			else settings.rate_limit_user_per_minute
		)

		minute_bucket = datetime.now(UTC).strftime("%Y%m%d%H%M")
		key = f"ratelimit:farm:{farm_id}:{identity}:{minute_bucket}"
		current = await redis_client.incr(key)
		if current == 1:
			await redis_client.expire(key, 65)

		if current > quota:
			return JSONResponse(
				status_code=429,
				content={
					"detail": {
						"error": "rate_limited",
						"message": "Farm quota exceeded",
						"farm_id": str(farm_id),
						"quota": quota,
					}
				},
			)

		return await call_next(request)

	@staticmethod
	def _is_bypass_path(path: str) -> bool:
		return path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi") or path.startswith("/health")
