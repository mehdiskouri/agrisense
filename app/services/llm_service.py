"""LangChain-based LLM integration for natural-language farm queries."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import structlog
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.schemas.ask import AskLanguage, AskRecommendation, AskResponse, AskSource, AskTelemetry
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
        self.logger = structlog.get_logger("agrisense.ask")

    async def ask(
        self,
        *,
        farm_id: uuid.UUID,
        question: str,
        language: AskLanguage,
        user_id: uuid.UUID,
        conversation_id: str | None = None,
    ) -> AskResponse:
        start = time.perf_counter()
        if not self.settings.anthropic_api_key:
            response = self._fallback_response(
                farm_id=farm_id,
                question=question,
                language=language,
                conversation_id=conversation_id or f"conversation:{farm_id}:{user_id}",
            )
            self._log_ask_response(
                response=response,
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
            return response

        llm = ChatAnthropic(
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            timeout=self.settings.anthropic_timeout_seconds,
            max_tokens=self.settings.langchain_max_tokens,
        )

        analytics = AnalyticsService(self.db, self.redis_client)
        tools = build_tools(farm_id, analytics, self.settings)
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

        response = self._parse_agent_output(
            farm_id=farm_id,
            question=question,
            language=language,
            conversation_id=resolved_conversation_id,
            agent_result=result,
            model_name=self.settings.anthropic_model,
            duration_ms=(time.perf_counter() - start) * 1000.0,
            fallback_used=False,
        )
        self._log_ask_response(
            response=response,
            duration_ms=(time.perf_counter() - start) * 1000.0,
        )
        return response

    async def ask_stream(
        self,
        *,
        farm_id: uuid.UUID,
        question: str,
        language: AskLanguage,
        user_id: uuid.UUID,
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        start = time.perf_counter()
        resolved_conversation_id = conversation_id or f"conversation:{farm_id}:{user_id}"

        if not self.settings.anthropic_api_key:
            fallback = self._fallback_response(
                farm_id=farm_id,
                question=question,
                language=language,
                conversation_id=resolved_conversation_id,
            )
            yield {
                "type": "final",
                "conversation_id": fallback.conversation_id,
                "data": fallback.model_dump(mode="json"),
            }
            self._log_ask_response(
                response=fallback,
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )
            return

        llm = ChatAnthropic(
            model=self.settings.anthropic_model,
            api_key=self.settings.anthropic_api_key,
            timeout=self.settings.anthropic_timeout_seconds,
            max_tokens=self.settings.langchain_max_tokens,
        )
        analytics = AnalyticsService(self.db, self.redis_client)
        tools = build_tools(farm_id, analytics, self.settings)
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

        yield {
            "type": "start",
            "conversation_id": resolved_conversation_id,
            "data": {"model": self.settings.anthropic_model},
        }

        raw_result = await agent_runtime.ainvoke(
            {"messages": messages},
            config=runtime_config,
        )
        result = cast(dict[str, Any], raw_result)

        for tool_name in self._extract_tools_called(result):
            yield {
                "type": "tool",
                "conversation_id": resolved_conversation_id,
                "data": {"name": tool_name},
            }

        final_content = self._extract_ai_content(result)
        for chunk in self._chunk_text(final_content, chunk_size=120):
            if chunk:
                yield {
                    "type": "token",
                    "conversation_id": resolved_conversation_id,
                    "data": {"text": chunk},
                }

        final_ai_message = self._extract_final_ai_message(result)
        await self._persist_interaction(chat_history, question, final_ai_message)
        if self.redis_client is not None:
            await refresh_ttl(
                self.settings.redis_url,
                resolved_conversation_id,
                self.settings.langchain_conversation_ttl_seconds,
            )

        response = self._parse_agent_output(
            farm_id=farm_id,
            question=question,
            language=language,
            conversation_id=resolved_conversation_id,
            agent_result=result,
            model_name=self.settings.anthropic_model,
            duration_ms=(time.perf_counter() - start) * 1000.0,
            fallback_used=False,
        )

        yield {
            "type": "final",
            "conversation_id": resolved_conversation_id,
            "data": response.model_dump(mode="json"),
        }
        self._log_ask_response(
            response=response,
            duration_ms=(time.perf_counter() - start) * 1000.0,
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
        model_name: str,
        duration_ms: float,
        fallback_used: bool,
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
            telemetry=self._build_telemetry(
                ai_message=self._extract_final_ai_message(agent_result),
                model_name=model_name,
                duration_ms=duration_ms,
                fallback_used=fallback_used,
            ),
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

    def _build_telemetry(
        self,
        *,
        ai_message: AIMessage,
        model_name: str,
        duration_ms: float,
        fallback_used: bool,
    ) -> AskTelemetry:
        usage = self._extract_usage(ai_message)
        estimated_cost_usd = self._estimate_cost_usd(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
        return AskTelemetry(
            model=model_name,
            latency_ms=round(max(0.0, duration_ms), 2),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            estimated_cost_usd=round(estimated_cost_usd, 8),
            fallback_used=fallback_used,
        )

    @staticmethod
    def _extract_usage(ai_message: AIMessage) -> dict[str, int]:
        usage = getattr(ai_message, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
            return {
                "input_tokens": max(0, input_tokens),
                "output_tokens": max(0, output_tokens),
                "total_tokens": max(0, total_tokens),
            }

        response_metadata = getattr(ai_message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage")
            if isinstance(token_usage, dict):
                input_tokens = int(token_usage.get("input_tokens") or 0)
                output_tokens = int(token_usage.get("output_tokens") or 0)
                total_tokens = int(
                    token_usage.get("total_tokens") or (input_tokens + output_tokens)
                )
                return {
                    "input_tokens": max(0, input_tokens),
                    "output_tokens": max(0, output_tokens),
                    "total_tokens": max(0, total_tokens),
                }

        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _estimate_cost_usd(self, *, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.settings.anthropic_input_cost_usd_per_million
        output_cost = (
            output_tokens / 1_000_000
        ) * self.settings.anthropic_output_cost_usd_per_million
        return max(0.0, input_cost + output_cost)

    def _log_ask_response(self, *, response: AskResponse, duration_ms: float) -> None:
        telemetry = response.telemetry
        self.logger.info(
            "ask_completed",
            conversation_id=response.conversation_id,
            farm_id=response.farm_id,
            intent=response.intent,
            tools_called=response.tools_called,
            duration_ms=round(max(0.0, duration_ms), 2),
            model=telemetry.model if telemetry is not None else self.settings.anthropic_model,
            input_tokens=telemetry.input_tokens if telemetry is not None else 0,
            output_tokens=telemetry.output_tokens if telemetry is not None else 0,
            estimated_cost_usd=(telemetry.estimated_cost_usd if telemetry is not None else 0.0),
            fallback_used=telemetry.fallback_used if telemetry is not None else True,
        )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        if chunk_size <= 0:
            return [text]
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

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
            telemetry=AskTelemetry(
                model="fallback",
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                fallback_used=True,
            ),
        )
