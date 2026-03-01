"""JWT token creation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from jose import JWTError, jwt

from app.config import get_settings

TokenType = Literal["access", "refresh"]


@dataclass(slots=True)
class AuthError(Exception):
	"""Structured authentication error for consistent mapping at the edge."""

	code: str
	detail: str
	status_code: int = 401


def _settings() -> Any:
	return get_settings()


def _build_token(subject: str, token_type: TokenType, expires_delta: timedelta) -> str:
	now = datetime.now(UTC)
	claims: dict[str, Any] = {
		"sub": subject,
		"typ": token_type,
		"iat": int(now.timestamp()),
		"exp": int((now + expires_delta).timestamp()),
	}
	settings = _settings()
	return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str, expires_minutes: int | None = None) -> str:
	settings = _settings()
	ttl = expires_minutes or settings.jwt_access_token_expire_minutes
	return _build_token(subject, "access", timedelta(minutes=ttl))


def create_refresh_token(subject: str, expires_minutes: int | None = None) -> str:
	settings = _settings()
	ttl = expires_minutes or settings.jwt_refresh_token_expire_minutes
	return _build_token(subject, "refresh", timedelta(minutes=ttl))


def decode_token(token: str, expected_type: TokenType | None = None) -> dict[str, Any]:
	settings = _settings()
	try:
		payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
	except JWTError as exc:
		raise AuthError(code="token_invalid", detail="Invalid authentication token") from exc

	subject = payload.get("sub")
	if not isinstance(subject, str) or not subject:
		raise AuthError(code="token_invalid", detail="Token subject is missing")

	token_type = payload.get("typ")
	if expected_type is not None and token_type != expected_type:
		raise AuthError(code="token_type_invalid", detail=f"Expected {expected_type} token")

	exp_raw = payload.get("exp")
	if not isinstance(exp_raw, int):
		raise AuthError(code="token_invalid", detail="Token expiration is missing")
	if datetime.now(UTC).timestamp() >= exp_raw:
		raise AuthError(code="token_expired", detail="Authentication token has expired")

	return payload
