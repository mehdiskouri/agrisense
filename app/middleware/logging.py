"""Structured JSON logging with request ID propagation."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import LogFormat, get_settings

_configured = False


def configure_structured_logging() -> None:
	"""Configure stdlib + structlog once for API process."""
	global _configured
	if _configured:
		return

	settings = get_settings()
	log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

	timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
	shared_processors: list[Any] = [
		structlog.contextvars.merge_contextvars,
		structlog.processors.add_log_level,
		timestamper,
	]

	if settings.log_format == LogFormat.json:
		renderer: Any = structlog.processors.JSONRenderer()
		logging.basicConfig(level=log_level, format="%(message)s")
	else:
		renderer = structlog.dev.ConsoleRenderer()
		logging.basicConfig(level=log_level)

	structlog.configure(
		processors=[
			*shared_processors,
			structlog.processors.format_exc_info,
			renderer,
		],
		wrapper_class=structlog.make_filtering_bound_logger(log_level),
		logger_factory=structlog.PrintLoggerFactory(),
		cache_logger_on_first_use=True,
	)
	_configured = True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
	"""Attach request IDs and emit structured per-request timing logs."""

	async def dispatch(self, request: Request, call_next):  # type: ignore[override]
		request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
		request.state.request_id = request_id

		structlog.contextvars.clear_contextvars()
		structlog.contextvars.bind_contextvars(request_id=request_id)

		logger = structlog.get_logger("agrisense.request")
		start = time.perf_counter()

		try:
			response = await call_next(request)
		except Exception as exc:
			duration_ms = (time.perf_counter() - start) * 1000.0
			logger.exception(
				"http_request_failed",
				method=request.method,
				path=request.url.path,
				duration_ms=round(duration_ms, 2),
				error=str(exc),
			)
			raise

		duration_ms = (time.perf_counter() - start) * 1000.0
		response.headers["x-request-id"] = request_id
		logger.info(
			"http_request",
			method=request.method,
			path=request.url.path,
			status_code=response.status_code,
			duration_ms=round(duration_ms, 2),
		)
		return response
