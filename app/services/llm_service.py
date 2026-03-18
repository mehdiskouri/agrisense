"""LangChain-based LLM integration for natural-language farm queries."""

from __future__ import annotations

import json
import uuid
from typing import Any, cast

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.schemas.ask import AskLanguage, AskRecommendation, AskResponse, AskSource
from app.services.agent_tools import build_tools
from app.services.analytics_service import AnalyticsService
from app.services.conversation_memory import (
    build_memory,
    clear_conversation,
    get_window_messages,
    refresh_ttl,
)


class LLMService:
    def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
        self.db = db
        self.redis_client = redis_client
        self.settings = get_settings()

    async def ask(
        self,
        *,
        farm_id: uuid.UUID,
        question: str,
        language: AskLanguage,
        user_id: uuid.UUID,
        conversation_id: str | None = None,
    ) -> AskResponse:
        if not self.settings.anthropic_api_key:
            return self._fallback_response(
                farm_id=farm_id,
                question=question,
                language=language,
                conversation_id=conversation_id or f"conversation:{farm_id}:{user_id}",
            )

        llm = ChatAnthropic(
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            timeout=self.settings.anthropic_timeout_seconds,
        )

        analytics = AnalyticsService(self.db, self.redis_client)
        tools = build_tools(farm_id, analytics)
        chat_history, resolved_conversation_id = build_memory(
            redis_url=self.settings.redis_url,
            farm_id=farm_id,
            user_id=user_id,
            settings=self.settings,
            conversation_id=conversation_id,
            redis_available=self.redis_client is not None,
        )

        history_messages = await get_window_messages(
            chat_history,
            self.settings.langchain_max_context_messages,
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=self._build_system_prompt(language),
            debug=self.settings.langchain_verbose,
        )

        messages: list[BaseMessage] = [
            *history_messages,
            HumanMessage(content=question),
        ]
        runtime_config: RunnableConfig = {
            "recursion_limit": self.settings.langchain_max_iterations,
        }
        agent_runtime = cast(Any, agent)
        raw_result = await agent_runtime.ainvoke(
            {"messages": messages},
            config=runtime_config,
        )
        result = cast(dict[str, Any], raw_result)

        final_ai_message = self._extract_final_ai_message(result)
        await self._persist_interaction(chat_history, question, final_ai_message)

        if self.redis_client is not None:
            await refresh_ttl(
                self.settings.redis_url,
                resolved_conversation_id,
                self.settings.langchain_conversation_ttl_seconds,
            )

        return self._parse_agent_output(
            farm_id=farm_id,
            question=question,
            language=language,
            conversation_id=resolved_conversation_id,
            agent_result=result,
        )

    def _build_system_prompt(self, language: AskLanguage) -> str:
        language_directive = {
            AskLanguage.en: "Respond in English.",
            AskLanguage.fr: "Reponds en francais.",
            AskLanguage.ar: "اجب باللغة العربية.",
        }[language]
        return (
            "You are an agricultural advisor for a precision farming platform. "
            "Answer only from tool outputs and never invent telemetry. "
            f"{language_directive} "
            "If required data is missing, explicitly say what is missing. "
            "When possible, return concise recommendations and a confidence score from 0 to 1. "
            "Prefer structured JSON with keys: answer, confidence, recommendations, sources."
        )

    def _parse_agent_output(
        self,
        *,
        farm_id: uuid.UUID,
        question: str,
        language: AskLanguage,
        conversation_id: str,
        agent_result: dict[str, Any],
    ) -> AskResponse:
        tools_called = self._extract_tools_called(agent_result)
        intent = self._intent_from_tools(tools_called)

        output_raw = self._extract_ai_content(agent_result)
        parsed_payload: dict[str, Any] = {}
        if isinstance(output_raw, str):
            output_text = output_raw.strip()
            try:
                candidate = json.loads(output_text)
                if isinstance(candidate, dict):
                    parsed_payload = candidate
            except json.JSONDecodeError:
                parsed_payload = {"answer": output_text}
        elif isinstance(output_raw, dict):
            parsed_payload = output_raw

        answer = str(parsed_payload.get("answer") or output_raw or "No grounded answer available.")
        confidence_raw = parsed_payload.get("confidence", 0.3)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.3
        confidence = max(0.0, min(1.0, confidence))

        recommendations_raw = parsed_payload.get("recommendations")
        recommendations: list[AskRecommendation] = []
        if isinstance(recommendations_raw, list):
            for item in recommendations_raw:
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action") or "Review latest sensor trends")
                rationale = str(item.get("rationale") or "Based on current farm telemetry")
                recommendations.append(AskRecommendation(action=action, rationale=rationale))

        sources_raw = parsed_payload.get("sources")
        sources: list[AskSource] = []
        if isinstance(sources_raw, list):
            for item in sources_raw:
                if not isinstance(item, dict):
                    continue
                payload_raw = item.get("payload")
                payload: dict[str, Any]
                if isinstance(payload_raw, dict):
                    payload = {str(key): value for key, value in payload_raw.items()}
                else:
                    payload = {}
                sources.append(
                    AskSource(
                        layer=str(item.get("layer") or intent),
                        reference=str(item.get("reference") or f"farm:{farm_id}:{intent}"),
                        payload=payload,
                    )
                )
        if not sources:
            sources.append(
                AskSource(
                    layer=intent,
                    reference=f"farm:{farm_id}:{intent}",
                    payload={"tools_called": tools_called},
                )
            )

        return AskResponse(
            farm_id=str(farm_id),
            question=question,
            language=language,
            intent=intent,
            answer=answer,
            confidence=confidence,
            recommendations=recommendations,
            sources=sources,
            conversation_id=conversation_id,
            tools_called=tools_called,
        )

    @staticmethod
    def _extract_tools_called(agent_result: dict[str, Any]) -> list[str]:
        tools_called: list[str] = []
        messages_raw = agent_result.get("messages")
        if not isinstance(messages_raw, list):
            return tools_called

        for message in messages_raw:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    call_name = tool_call.get("name")
                    if isinstance(call_name, str) and call_name and call_name not in tools_called:
                        tools_called.append(call_name)
            if isinstance(message, ToolMessage):
                tool_message_name = getattr(message, "name", None)
                if (
                    isinstance(tool_message_name, str)
                    and tool_message_name
                    and tool_message_name not in tools_called
                ):
                    tools_called.append(tool_message_name)
        return tools_called

    @staticmethod
    def _extract_final_ai_message(agent_result: dict[str, Any]) -> AIMessage:
        messages_raw = agent_result.get("messages")
        if isinstance(messages_raw, list):
            for message in reversed(messages_raw):
                if isinstance(message, AIMessage):
                    return message
        return AIMessage(content="")

    @staticmethod
    def _extract_ai_content(agent_result: dict[str, Any]) -> str:
        message = LLMService._extract_final_ai_message(agent_result)
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for chunk in content:
                if isinstance(chunk, str):
                    chunks.append(chunk)
                    continue
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return ""

    @staticmethod
    async def _persist_interaction(
        chat_history: BaseChatMessageHistory,
        question: str,
        ai_message: AIMessage,
    ) -> None:
        ai_content = ai_message.content
        ai_text = ai_content if isinstance(ai_content, str) else ""
        if not ai_text:
            ai_text = "No grounded answer available."
        await chat_history.aadd_messages(
            [
                HumanMessage(content=question),
                AIMessage(content=ai_text),
            ]
        )

    @staticmethod
    def _intent_from_tools(tools_called: list[str]) -> str:
        mapping = {
            "get_irrigation_schedule": "irrigation",
            "get_nutrient_report": "nutrients",
            "get_yield_forecast": "yield",
            "get_active_alerts": "alerts",
            "run_yield_backtest": "yield",
            "get_zone_detail": "status",
            "get_farm_status": "status",
        }
        for tool_name in tools_called:
            if tool_name in mapping:
                return mapping[tool_name]
        return "status"

    async def clear_conversation(self, farm_id: uuid.UUID, user_id: uuid.UUID) -> str:
        if self.redis_client is None:
            return f"conversation:{farm_id}:{user_id}"
        return await clear_conversation(self.settings.redis_url, farm_id, user_id)

    @staticmethod
    def _fallback_response(
        *,
        farm_id: uuid.UUID,
        question: str,
        language: AskLanguage,
        conversation_id: str,
    ) -> AskResponse:
        intent = "status"
        language_hint = {
            AskLanguage.en: "Based on current farm telemetry",
            AskLanguage.fr: "Selon les données actuelles de la ferme",
            AskLanguage.ar: "بناءً على بيانات المزرعة الحالية",
        }[language]
        return AskResponse(
            farm_id=str(farm_id),
            question=question,
            language=language,
            intent=intent,
            answer=f"{language_hint}, intent={intent}. Please review recommended actions.",
            confidence=0.62,
            recommendations=[
                AskRecommendation(
                    action="Review the latest analytics snapshot",
                    rationale="Grounded in current measured values",
                )
            ],
            sources=[
                AskSource(
                    layer=intent,
                    reference="analytics_context",
                    payload={"fallback": True},
                )
            ],
            conversation_id=conversation_id,
            tools_called=[],
        )
