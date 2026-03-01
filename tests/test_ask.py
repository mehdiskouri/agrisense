from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.schemas.ask import AskLanguage, AskRecommendation, AskResponse, AskSource
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_ask_endpoint_success(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    farm_id = uuid4()

    async def fake_ask(self: LLMService, farm_id: object, question: str, language: AskLanguage) -> AskResponse:
        return AskResponse(
            farm_id=str(farm_id),
            question=question,
            language=language,
            intent="irrigation",
            answer="Irrigate zone A tomorrow morning.",
            confidence=0.88,
            recommendations=[
                AskRecommendation(
                    action="Open valve V1 for 20 minutes",
                    rationale="Soil moisture is below target",
                )
            ],
            sources=[
                AskSource(
                    layer="irrigation",
                    reference="farm-context",
                    payload={"window_hours": 24},
                )
            ],
        )

    monkeypatch.setattr(LLMService, "ask", fake_ask)

    response = await client.post(
        f"/api/v1/ask/{farm_id}",
        json={"question": "When should I irrigate?", "language": "en"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "irrigation"
    assert isinstance(body["confidence"], float)
    assert body["confidence"] >= 0.0


@pytest.mark.asyncio
async def test_ask_openapi_contract(client: AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200

    body = response.json()
    assert "/api/v1/ask/{farm_id}" in body["paths"]

    schema = body["components"]["schemas"]["AskResponse"]
    required = set(schema["required"])
    assert "confidence" in required
    assert "sources" in schema["properties"]
    assert "recommendations" in schema["properties"]


@pytest.mark.asyncio
async def test_llm_service_fallback_confidence_required(client: AsyncClient) -> None:
    service = LLMService(db=object(), redis_client=None)  # type: ignore[arg-type]
    response = service.parse_response(
        farm_id=uuid4(),
        question="Status?",
        language=AskLanguage.en,
        intent="status",
        context={"zones": []},
        llm_payload={"answer": "No data", "sources": []},
    )
    assert 0.0 <= response.confidence <= 1.0
