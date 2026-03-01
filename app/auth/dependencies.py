"""Authentication dependencies — get_current_user, require_role."""

from __future__ import annotations

import hashlib
import hmac
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import AuthError, decode_token
from app.auth.models import APIKey, User
from app.config import get_settings
from app.database import get_db
from app.models.enums import UserRoleEnum

bearer_scheme = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass(slots=True)
class AuthPrincipal:
	auth_type: str
	subject_id: uuid.UUID
	role: UserRoleEnum
	scopes: set[str]
	api_key_id: uuid.UUID | None = None


def _raise_auth(exc: AuthError) -> HTTPException:
	return HTTPException(
		status_code=exc.status_code,
		detail={"error": exc.code, "message": exc.detail},
	)


def _api_key_digest(plaintext: str) -> str:
	return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _coerce_scopes(raw: dict[str, Any] | None) -> set[str]:
	if raw is None:
		return set()
	scopes: set[str] = set()
	for key, value in raw.items():
		if isinstance(value, bool):
			if value:
				scopes.add(str(key))
			continue
		if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "allow"}:
			scopes.add(str(key))
	return scopes


def extract_request_farm_id(request: Request) -> uuid.UUID | None:
	token = request.path_params.get("farm_id")
	if token is not None:
		try:
			return uuid.UUID(str(token))
		except ValueError:
			return None

	match = re.search(
		r"/api/v1/(?:farms|ingest|analytics|jobs|ask)/([0-9a-fA-F\-]{36})(?:/|$)",
		request.url.path,
	)
	if match is None:
		return None
	try:
		return uuid.UUID(match.group(1))
	except ValueError:
		return None


def extract_identity_hint(request: Request) -> str:
	settings = get_settings()
	api_key_header = request.headers.get(settings.api_key_header_name)
	auth_header = request.headers.get("authorization", "")
	if api_key_header:
		return "api_key"
	if auth_header.lower().startswith("bearer "):
		return "jwt"
	return "anonymous"


async def _resolve_user_from_token(
	db: AsyncSession,
	credentials: HTTPAuthorizationCredentials | None,
) -> User:
	if credentials is None or credentials.scheme.lower() != "bearer":
		raise _raise_auth(AuthError(code="auth_required", detail="Bearer token is required"))

	payload = decode_token(credentials.credentials, expected_type="access")
	try:
		user_id = uuid.UUID(str(payload["sub"]))
	except (ValueError, KeyError) as exc:
		raise _raise_auth(AuthError(code="token_invalid", detail="Token subject is invalid")) from exc

	row = await db.execute(select(User).where(User.id == user_id))
	user = row.scalar_one_or_none()
	if user is None or not user.is_active:
		raise _raise_auth(AuthError(code="user_invalid", detail="User is not active"))
	return user


async def get_current_user(
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> User:
	credentials = await bearer_scheme(request)
	return await _resolve_user_from_token(db, credentials)


def require_role(*allowed: UserRoleEnum) -> Callable[[User], User]:
	allowed_set = set(allowed)

	async def dependency(current_user: User = Depends(get_current_user)) -> User:
		if current_user.role not in allowed_set:
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail={"error": "forbidden", "message": "Insufficient role"},
			)
		return current_user

	return dependency


async def _resolve_api_key(db: AsyncSession, plaintext_key: str) -> APIKey:
	digest = _api_key_digest(plaintext_key)
	rows = await db.execute(select(APIKey).where(APIKey.is_active.is_(True)))
	for api_key in rows.scalars().all():
		stored_hash = api_key.key_hash
		hash_match = hmac.compare_digest(digest, stored_hash)
		bcrypt_match = False
		try:
			bcrypt_match = pwd_context.verify(plaintext_key, stored_hash)
		except ValueError:
			bcrypt_match = False
		if not (hash_match or bcrypt_match):
			continue
		if api_key.expires_at is not None and api_key.expires_at <= datetime.now(UTC):
			raise _raise_auth(AuthError(code="api_key_expired", detail="API key expired"))
		return api_key
	raise _raise_auth(AuthError(code="api_key_invalid", detail="Invalid API key"))


async def get_api_key_principal(
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> AuthPrincipal:
	settings = get_settings()
	plaintext = request.headers.get(settings.api_key_header_name)
	if plaintext is None or not plaintext.strip():
		raise _raise_auth(AuthError(code="api_key_required", detail="API key header is required"))

	api_key = await _resolve_api_key(db, plaintext.strip())
	row = await db.execute(select(User).where(User.id == api_key.user_id))
	user = row.scalar_one_or_none()
	if user is None or not user.is_active:
		raise _raise_auth(AuthError(code="user_invalid", detail="API key owner is inactive"))

	return AuthPrincipal(
		auth_type="api_key",
		subject_id=user.id,
		role=user.role,
		scopes=_coerce_scopes(api_key.scopes),
		api_key_id=api_key.id,
	)


async def get_auth_principal(
	request: Request,
	db: AsyncSession = Depends(get_db),
) -> AuthPrincipal:
	settings = get_settings()
	api_key = request.headers.get(settings.api_key_header_name)
	credentials = await bearer_scheme(request)

	if api_key and api_key.strip():
		return await get_api_key_principal(request, db)

	user = await _resolve_user_from_token(db, credentials)
	return AuthPrincipal(
		auth_type="jwt",
		subject_id=user.id,
		role=user.role,
		scopes=set(),
	)


def require_machine_scope(scope: str) -> Callable[[AuthPrincipal], AuthPrincipal]:
	allowed_scopes = {"ingest", "jobs"}

	async def dependency(principal: AuthPrincipal = Depends(get_auth_principal)) -> AuthPrincipal:
		if principal.auth_type != "api_key":
			return principal
		if scope not in allowed_scopes:
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail={"error": "scope_invalid", "message": "Scope is not permitted"},
			)
		if scope not in principal.scopes:
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail={"error": "scope_missing", "message": f"Missing scope: {scope}"},
			)
		return principal

	return dependency
