"""WebSocket live feed route."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.auth.dependencies import _coerce_scopes, _resolve_api_key, scope_grants
from app.auth.jwt import AuthError, decode_token
from app.auth.models import User
from app.database import async_session_factory
from app.models.farm import Farm

router = APIRouter(tags=["websocket"])
AUTH_MESSAGE_TIMEOUT_SECONDS = 5.0


async def _farm_exists(farm_id: uuid.UUID) -> bool:
    async with async_session_factory() as session:
        row = await session.execute(select(Farm.id).where(Farm.id == farm_id))
        return row.scalar_one_or_none() is not None


async def _authenticate_api_key(token: str) -> bool:
    async with async_session_factory() as session:
        try:
            api_key = await _resolve_api_key(session, token)
        except HTTPException:
            return False

        scopes = _coerce_scopes(api_key.scopes)
        if not (scope_grants(scopes, "ingest") or scope_grants(scopes, "jobs")):
            return False

        row = await session.execute(select(User).where(User.id == api_key.user_id))
        user = row.scalar_one_or_none()
        return user is not None and user.is_active


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

    return await _authenticate_api_key(token)


async def _receive_auth_token(websocket: WebSocket) -> str | None:
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_MESSAGE_TIMEOUT_SECONDS)
    except (TimeoutError, WebSocketDisconnect):
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "auth":
        return None

    token = payload.get("token")
    if not isinstance(token, str) or not token.strip():
        return None
    return token.strip()


@router.websocket("/ws/{farm_id}/live")
async def ws_live_feed(websocket: WebSocket, farm_id: str) -> None:
    await websocket.accept()
    try:
        farm_uuid = uuid.UUID(farm_id)
    except ValueError:
        await websocket.send_json({"error": "invalid_farm_id"})
        await websocket.close(code=1008)
        return

    token = await _receive_auth_token(websocket)
    if token is None:
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
