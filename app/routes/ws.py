"""WebSocket live feed route."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.database import async_session_factory
from app.models.farm import Farm

router = APIRouter(tags=["websocket"])


async def _farm_exists(farm_id: uuid.UUID) -> bool:
	async with async_session_factory() as session:
		row = await session.execute(select(Farm.id).where(Farm.id == farm_id))
		return row.scalar_one_or_none() is not None


@router.websocket("/ws/{farm_id}/live")
async def ws_live_feed(websocket: WebSocket, farm_id: str) -> None:
	await websocket.accept()
	try:
		farm_uuid = uuid.UUID(farm_id)
	except ValueError:
		await websocket.send_json({"error": "invalid_farm_id"})
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
