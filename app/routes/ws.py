"""WebSocket live feed route."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from passlib.context import CryptContext
from sqlalchemy import select, true

from app.auth.jwt import AuthError, decode_token
from app.auth.models import APIKey, User
from app.database import async_session_factory
from app.models.farm import Farm

router = APIRouter(tags=["websocket"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def _farm_exists(farm_id: uuid.UUID) -> bool:
	async with async_session_factory() as session:
		row = await session.execute(select(Farm.id).where(Farm.id == farm_id))
		return row.scalar_one_or_none() is not None


async def _resolve_api_key(token: str) -> tuple[uuid.UUID, str] | None:
	digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
	async with async_session_factory() as session:
		rows = await session.execute(select(APIKey).where(APIKey.is_active == true()))
		for api_key in rows.scalars().all():
			stored_hash = api_key.key_hash
			digest_match = hmac.compare_digest(digest, stored_hash)
			bcrypt_match = False
			try:
				bcrypt_match = pwd_context.verify(token, stored_hash)
			except ValueError:
				bcrypt_match = False
			if not (digest_match or bcrypt_match):
				continue
			if api_key.expires_at is not None and api_key.expires_at <= datetime.now(UTC):
				return None
			scopes = api_key.scopes or {}
			scope_ok = bool(scopes.get("ingest") or scopes.get("jobs"))
			if not scope_ok:
				return None

			row = await session.execute(select(User).where(User.id == api_key.user_id))
			user = row.scalar_one_or_none()
			if user is None or not user.is_active:
				return None
			return user.id, user.role.value
	return None


async def _authenticate_token(token: str) -> bool:
	try:
		payload = decode_token(token, expected_type="access")
		user_id = uuid.UUID(str(payload["sub"]))
		async with async_session_factory() as session:
			row = await session.execute(select(User).where(User.id == user_id))
			user = row.scalar_one_or_none()
			return user is not None and user.is_active
	except (AuthError, ValueError, KeyError):
		pass

	api_key_user = await _resolve_api_key(token)
	return api_key_user is not None


@router.websocket("/ws/{farm_id}/live")
async def ws_live_feed(websocket: WebSocket, farm_id: str) -> None:
	await websocket.accept()
	try:
		farm_uuid = uuid.UUID(farm_id)
	except ValueError:
		await websocket.send_json({"error": "invalid_farm_id"})
		await websocket.close(code=1008)
		return

	token = websocket.query_params.get("token")
	if token is None or not token.strip():
		await websocket.send_json({"error": "auth_required"})
		await websocket.close(code=1008)
		return
	if not await _authenticate_token(token.strip()):
		await websocket.send_json({"error": "auth_invalid"})
		await websocket.close(code=1008)
		return

	if not await _farm_exists(farm_uuid):
		await websocket.send_json({"error": "farm_not_found"})
		await websocket.close(code=1008)
		return

	redis_client = getattr(websocket.app.state, "redis", None)
	if redis_client is None:
		await websocket.send_json({"error": "redis_unavailable"})
		await websocket.close(code=1011)
		return

	channel = f"farm:{farm_uuid}:live"
	pubsub = redis_client.pubsub()
	await pubsub.subscribe(channel)

	try:
		while True:
			message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
			if message is not None and message.get("type") == "message":
				payload = message.get("data")
				if isinstance(payload, bytes):
					payload = payload.decode("utf-8")
				if isinstance(payload, str):
					try:
						await websocket.send_json(json.loads(payload))
					except json.JSONDecodeError:
						await websocket.send_text(payload)
			await asyncio.sleep(0.05)
	except WebSocketDisconnect:
		return
	finally:
		await pubsub.unsubscribe(channel)
		await pubsub.close()
