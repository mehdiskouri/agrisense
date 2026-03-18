from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.schemas.ask import AskLanguage, AskRecommendation, AskResponse, AskSource
from app.services.llm_service import LLMService


def _configure_langchain_path(
    monkeypatch: pytest.MonkeyPatch,
    *,
    agent_result: dict[str, Any],
    conversation_id: str = "conversation:test",
    capture: dict[str, Any] | None = None,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    class _FakeAgent:
        async def ainvoke(
            self,
            _payload: dict[str, Any],
            config: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            if capture is not None and config is not None:
                capture.update(config)
            return agent_result

    class _FakeHistory:
        def __init__(self) -> None:
            self.messages: list[Any] = []

        async def aget_messages(self) -> list[Any]:
            return self.messages

        async def aadd_messages(self, messages: list[Any]) -> None:
            self.messages.extend(messages)

    class _FakeChatModel:
        def __init__(self, **_kwargs: Any) -> None:
            self.kwargs = _kwargs

    async def _noop_refresh(_redis_url: str, _conversation_id: str, _ttl: int) -> None:
        return None

    async def _empty_window(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr("app.services.llm_service.ChatAnthropic", _FakeChatModel)
    monkeypatch.setattr("app.services.llm_service.create_agent", lambda **_kwargs: _FakeAgent())
    monkeypatch.setattr(
        "app.services.llm_service.build_memory",
        lambda **_kwargs: (_FakeHistory(), conversation_id),
    )
    monkeypatch.setattr("app.services.llm_service.get_window_messages", _empty_window)
    monkeypatch.setattr("app.services.llm_service.build_tools", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("app.services.llm_service.refresh_ttl", _noop_refresh)


@pytest.mark.asyncio
async def test_tool_routing_irrigation(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_langchain_path(
        monkeypatch,
        agent_result={
            "messages": [
                AIMessage(
                    content='{"answer": "Irrigate tomorrow", "confidence": 0.8}',
                    tool_calls=[
                        {
                            "name": "get_irrigation_schedule",
                            "args": {},
                            "id": "t1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
    )
    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]
    response = await service.ask(
        farm_id=uuid4(),
        question="When should I irrigate?",
        language=AskLanguage.en,
        user_id=uuid4(),
    )
    assert response.intent == "irrigation"
    assert response.tools_called == ["get_irrigation_schedule"]


@pytest.mark.asyncio
async def test_tool_routing_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_langchain_path(
        monkeypatch,
        agent_result={
            "messages": [
                AIMessage(
                    content='{"answer": "Yield looks stable", "confidence": 0.77}',
                    tool_calls=[
                        {
                            "name": "get_yield_forecast",
                            "args": {},
                            "id": "t2",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
    )
    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]
    response = await service.ask(
        farm_id=uuid4(),
        question="Give me forecast",
        language=AskLanguage.en,
        user_id=uuid4(),
    )
    assert response.intent == "yield"
    assert "get_yield_forecast" in response.tools_called


@pytest.mark.asyncio
async def test_tool_routing_alerts(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_langchain_path(
        monkeypatch,
        agent_result={
            "messages": [
                AIMessage(
                    content='{"answer": "Two active alerts", "confidence": 0.74}',
                    tool_calls=[
                        {
                            "name": "get_active_alerts",
                            "args": {},
                            "id": "t3",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
    )
    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]
    response = await service.ask(
        farm_id=uuid4(),
        question="Any alerts?",
        language=AskLanguage.en,
        user_id=uuid4(),
    )
    assert response.intent == "alerts"
    assert response.tools_called == ["get_active_alerts"]


@pytest.mark.asyncio
async def test_multi_tool_query_collects_multiple_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_langchain_path(
        monkeypatch,
        agent_result={
            "messages": [
                AIMessage(
                    content='{"answer": "Status and alerts ready", "confidence": 0.79}',
                    tool_calls=[
                        {
                            "name": "get_farm_status",
                            "args": {},
                            "id": "t4",
                            "type": "tool_call",
                        },
                        {
                            "name": "get_active_alerts",
                            "args": {},
                            "id": "t5",
                            "type": "tool_call",
                        },
                    ],
                )
            ]
        },
    )
    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]
    response = await service.ask(
        farm_id=uuid4(),
        question="status and alerts",
        language=AskLanguage.en,
        user_id=uuid4(),
    )
    assert len(response.tools_called) >= 2
    assert "get_farm_status" in response.tools_called
    assert "get_active_alerts" in response.tools_called


@pytest.mark.asyncio
async def test_conversation_continuity(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_ask(
        self: LLMService,
        *,
        farm_id: object,
        question: str,
        language: AskLanguage,
        user_id: object,
        conversation_id: str | None,
    ) -> AskResponse:
        return AskResponse(
            farm_id=str(farm_id),
            question=question,
            language=language,
            intent="status",
            answer=f"Echo: {question}",
            confidence=0.8,
            recommendations=[
                AskRecommendation(
                    action="Review farm status",
                    rationale="Conversation continuity test",
                )
            ],
            sources=[AskSource(layer="status", reference="test", payload={})],
            conversation_id=conversation_id or "conversation:auto",
            tools_called=["get_farm_status"],
        )

    monkeypatch.setattr(LLMService, "ask", fake_ask)

    payload = {
        "question": "first question",
        "language": "en",
        "conversation_id": "conversation:test-1",
    }
    first = await client.post(f"/api/v1/ask/{farm_id}", json=payload)
    assert first.status_code == 200

    payload["question"] = "second question"
    second = await client.post(f"/api/v1/ask/{farm_id}", json=payload)
    assert second.status_code == 200
    assert first.json()["conversation_id"] == second.json()["conversation_id"]


@pytest.mark.asyncio
async def test_conversation_clear_endpoint(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()
    history_by_conversation: dict[str, list[str]] = {}

    async def fake_ask(
        self: LLMService,
        *,
        farm_id: object,
        question: str,
        language: AskLanguage,
        user_id: object,
        conversation_id: str | None,
    ) -> AskResponse:
        conversation_key = conversation_id or f"conversation:{farm_id}:{user_id}"
        existing = history_by_conversation.get(conversation_key, [])
        prior_question = existing[-1] if existing else None
        history_by_conversation.setdefault(conversation_key, []).append(question)
        answer = (
            f"No memory before: {question}"
            if prior_question is None
            else f"Memory before was: {prior_question}"
        )
        return AskResponse(
            farm_id=str(farm_id),
            question=question,
            language=language,
            intent="status",
            answer=answer,
            confidence=0.85,
            recommendations=[],
            sources=[],
            conversation_id=conversation_key,
            tools_called=["get_farm_status"],
        )

    async def fake_clear(self: LLMService, farm_id: object, user_id: object) -> str:
        conversation_key = f"conversation:{farm_id}:{user_id}"
        history_by_conversation.pop(conversation_key, None)
        return conversation_key

    monkeypatch.setattr(LLMService, "ask", fake_ask)
    monkeypatch.setattr(LLMService, "clear_conversation", fake_clear)

    first = await client.post(
        f"/api/v1/ask/{farm_id}",
        json={"question": "first", "language": "en"},
    )
    assert first.status_code == 200
    assert "No memory before" in first.json()["answer"]

    clear_response = await client.delete(f"/api/v1/ask/{farm_id}/conversation")
    assert clear_response.status_code == 204

    second = await client.post(
        f"/api/v1/ask/{farm_id}",
        json={"question": "second", "language": "en"},
    )
    assert second.status_code == 200
    assert "No memory before" in second.json()["answer"]


@pytest.mark.asyncio
async def test_fallback_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]

    response = await service.ask(
        farm_id=uuid4(),
        question="status",
        language=AskLanguage.en,
        user_id=uuid4(),
    )
    assert response.tools_called == []
    assert response.conversation_id.startswith("conversation:")
    assert 0.0 <= response.confidence <= 1.0


@pytest.mark.asyncio
async def test_ask_response_schema_fields(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    farm_id = uuid4()

    async def fake_ask(
        self: LLMService,
        *,
        farm_id: object,
        question: str,
        language: AskLanguage,
        user_id: object,
        conversation_id: str | None,
    ) -> AskResponse:
        return AskResponse(
            farm_id=str(farm_id),
            question=question,
            language=language,
            intent="irrigation",
            answer="Irrigate in 6 hours",
            confidence=0.87,
            recommendations=[],
            sources=[],
            conversation_id=conversation_id or "conversation:test",
            tools_called=["get_irrigation_schedule"],
        )

    monkeypatch.setattr(LLMService, "ask", fake_ask)

    response = await client.post(
        f"/api/v1/ask/{farm_id}",
        json={"question": "When to irrigate?", "language": "en"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "conversation_id" in body
    assert "tools_called" in body
    assert "telemetry" in body


@pytest.mark.asyncio
async def test_ask_openapi_contract(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200

    body = response.json()
    assert "/api/v1/ask/{farm_id}" in body["paths"]
    assert "/api/v1/ask/{farm_id}/conversation" in body["paths"]
    assert "/api/v1/ask/{farm_id}/stream" in body["paths"]

    schema = body["components"]["schemas"]["AskResponse"]
    assert "conversation_id" in schema["properties"]
    assert "tools_called" in schema["properties"]
    assert "telemetry" in schema["properties"]


@pytest.mark.asyncio
async def test_agent_max_iterations_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _configure_langchain_path(
        monkeypatch,
        agent_result={
            "messages": [AIMessage(content='{"answer": "ok", "confidence": 0.7}')],
        },
        capture=captured,
    )

    settings = get_settings()
    monkeypatch.setattr(settings, "langchain_max_iterations", 3)

    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]
    await service.ask(
        farm_id=uuid4(),
        question="quick status",
        language=AskLanguage.en,
        user_id=uuid4(),
    )

    assert captured.get("recursion_limit") == 3


@pytest.mark.asyncio
async def test_ask_response_has_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_langchain_path(
        monkeypatch,
        agent_result={
            "messages": [
                AIMessage(
                    content='{"answer": "ok", "confidence": 0.7}',
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 25,
                        "total_tokens": 125,
                    },
                )
            ],
        },
    )

    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]
    response = await service.ask(
        farm_id=uuid4(),
        question="quick status",
        language=AskLanguage.en,
        user_id=uuid4(),
    )

    assert response.telemetry is not None
    assert response.telemetry.input_tokens == 100
    assert response.telemetry.output_tokens == 25
    assert response.telemetry.total_tokens == 125
    assert response.telemetry.estimated_cost_usd > 0.0


@pytest.mark.asyncio
async def test_stream_endpoint_emits_sse_events(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    farm_id = uuid4()

    async def fake_stream(
        self: LLMService,
        *,
        farm_id: object,
        question: str,
        language: AskLanguage,
        user_id: object,
        conversation_id: str | None,
    ) -> Any:
        del farm_id, question, language, user_id
        cid = conversation_id or "conversation:test-stream"
        yield {
            "type": "start",
            "conversation_id": cid,
            "data": {"model": "test-model"},
        }
        yield {
            "type": "token",
            "conversation_id": cid,
            "data": {"text": "hello"},
        }
        yield {
            "type": "final",
            "conversation_id": cid,
            "data": {"answer": "hello"},
        }

    monkeypatch.setattr(LLMService, "ask_stream", fake_stream)

    response = await client.post(
        f"/api/v1/ask/{farm_id}/stream",
        json={"question": "hello", "language": "en", "conversation_id": "conversation:sse"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: start" in body
    assert "event: token" in body
    assert "event: final" in body
