"""Conversation memory helpers for LangChain ask workflows."""

from __future__ import annotations

import uuid

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from redis.asyncio import from_url

from app.config import Settings


def resolve_conversation_id(
    farm_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: str | None,
) -> str:
    """Build a stable conversation key scoped to farm+user."""
    if conversation_id is not None and conversation_id.strip():
        return conversation_id.strip()
    return f"conversation:{farm_id}:{user_id}"


def build_memory(
    redis_url: str,
    farm_id: uuid.UUID,
    user_id: uuid.UUID,
    settings: Settings,
    conversation_id: str | None = None,
    redis_available: bool = True,
) -> tuple[BaseChatMessageHistory, str]:
    """Create chat history backend and scoped conversation id."""
    resolved_id = resolve_conversation_id(farm_id, user_id, conversation_id)
    chat_history: BaseChatMessageHistory
    if redis_available:
        chat_history = RedisChatMessageHistory(
            session_id=resolved_id,
            url=redis_url,
            ttl=settings.langchain_conversation_ttl_seconds,
        )
    else:
        chat_history = InMemoryChatMessageHistory()
    return chat_history, resolved_id


async def get_window_messages(
    chat_history: BaseChatMessageHistory,
    max_messages: int,
) -> list[BaseMessage]:
    """Read bounded conversation history window for agent input."""
    messages = await chat_history.aget_messages()
    if max_messages <= 0:
        return []
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


async def refresh_ttl(redis_url: str, conversation_id: str, ttl_seconds: int) -> None:
    """Refresh conversation TTL to keep active sessions alive."""
    redis_client = from_url(redis_url, decode_responses=True)
    try:
        await redis_client.expire(conversation_id, ttl_seconds)
    finally:
        await redis_client.aclose()


async def clear_conversation(redis_url: str, farm_id: uuid.UUID, user_id: uuid.UUID) -> str:
    """Delete a scoped conversation history key and return its key."""
    conversation_id = resolve_conversation_id(farm_id, user_id, None)
    redis_client = from_url(redis_url, decode_responses=True)
    try:
        await redis_client.delete(conversation_id)
    finally:
        await redis_client.aclose()
    return conversation_id
