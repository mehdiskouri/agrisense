"""LLM integration — intent classification, context assembly, prompt, response parsing."""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.schemas.ask import AskLanguage, AskRecommendation, AskResponse, AskSource
from app.services.analytics_service import AnalyticsService


class LLMService:
	def __init__(self, db: AsyncSession, redis_client: Redis | None = None):
		self.db = db
		self.redis_client = redis_client
		self.settings = get_settings()

	def classify_intent(self, question: str) -> str:
		normalized = question.lower()
		if any(token in normalized for token in ["irrig", "water", "valve"]):
			return "irrigation"
		if any(token in normalized for token in ["npk", "nutr", "fertil", "phosph", "nitrogen", "potassium"]):
			return "nutrients"
		if any(token in normalized for token in ["yield", "harvest", "production", "forecast"]):
			return "yield"
		if any(token in normalized for token in ["alert", "anomaly", "disease", "pest"]):
			return "alerts"
		return "status"

	async def ask(self, farm_id: uuid.UUID, question: str, language: AskLanguage) -> AskResponse:
		intent = self.classify_intent(question)
		context = await self.assemble_context(farm_id, intent)
		llm_payload = await self.call_llm(question=question, language=language, intent=intent, context=context)
		return self.parse_response(
			farm_id=farm_id,
			question=question,
			language=language,
			intent=intent,
			context=context,
			llm_payload=llm_payload,
		)

	async def assemble_context(self, farm_id: uuid.UUID, intent: str) -> dict[str, Any]:
		analytics = AnalyticsService(self.db, self.redis_client)
		if intent == "irrigation":
			schedule = await analytics.get_irrigation_schedule(farm_id, horizon_days=3)
			return {
				"intent": intent,
				"schedule": [item.model_dump() for item in schedule.items],
				"generated_at": schedule.generated_at.isoformat(),
			}
		if intent == "nutrients":
			report = await analytics.get_nutrient_report(farm_id)
			return {
				"intent": intent,
				"nutrients": [item.model_dump() for item in report.items],
				"generated_at": report.generated_at.isoformat(),
			}
		if intent == "yield":
			forecast = await analytics.get_yield_forecast(farm_id)
			return {
				"intent": intent,
				"yield": [item.model_dump() for item in forecast.items],
				"generated_at": forecast.generated_at.isoformat(),
			}
		if intent == "alerts":
			alerts = await analytics.get_active_alerts(farm_id)
			return {
				"intent": intent,
				"alerts": [zone.model_dump() for zone in alerts.zones],
				"generated_at": alerts.generated_at.isoformat(),
			}

		status = await analytics.get_farm_status(farm_id)
		return {
			"intent": intent,
			"zones": [zone.model_dump() for zone in status.zones],
			"generated_at": status.generated_at.isoformat(),
		}

	async def call_llm(
		self,
		*,
		question: str,
		language: AskLanguage,
		intent: str,
		context: dict[str, Any],
	) -> dict[str, Any]:
		if not self.settings.anthropic_api_key:
			return self._fallback_response(intent=intent, language=language, context=context)

		system_prompt = (
			"You are an agricultural advisor. "
			"Answer only from the provided context. "
			"If context is missing, say so clearly. "
			"Return strict JSON with keys: answer, confidence, recommendations, sources."
		)
		user_prompt = {
			"language": language.value,
			"intent": intent,
			"question": question,
			"context": context,
		}

		headers = {
			"x-api-key": self.settings.anthropic_api_key,
			"anthropic-version": "2023-06-01",
			"content-type": "application/json",
		}
		body = {
			"model": self.settings.anthropic_model,
			"max_tokens": 700,
			"system": system_prompt,
			"messages": [{"role": "user", "content": json.dumps(user_prompt)}],
		}

		async with httpx.AsyncClient(timeout=self.settings.anthropic_timeout_seconds) as client:
			response = await client.post(self.settings.anthropic_base_url, headers=headers, json=body)
			response.raise_for_status()
			payload = response.json()

		content = payload.get("content")
		if not isinstance(content, list) or not content:
			return self._fallback_response(intent=intent, language=language, context=context)
		text = str(content[0].get("text") or "").strip()
		try:
			parsed = json.loads(text)
		except json.JSONDecodeError:
			return self._fallback_response(intent=intent, language=language, context=context)
		if not isinstance(parsed, dict):
			return self._fallback_response(intent=intent, language=language, context=context)
		return parsed

	def parse_response(
		self,
		*,
		farm_id: uuid.UUID,
		question: str,
		language: AskLanguage,
		intent: str,
		context: dict[str, Any],
		llm_payload: dict[str, Any],
	) -> AskResponse:
		answer = str(llm_payload.get("answer") or "No grounded answer available for this request.")
		confidence_raw = llm_payload.get("confidence", 0.3)
		try:
			confidence = float(confidence_raw)
		except (TypeError, ValueError):
			confidence = 0.3
		confidence = max(0.0, min(1.0, confidence))

		recommendations_raw = llm_payload.get("recommendations")
		recommendations: list[AskRecommendation] = []
		if isinstance(recommendations_raw, list):
			for item in recommendations_raw:
				if not isinstance(item, dict):
					continue
				action = str(item.get("action") or "Review latest sensor trends")
				rationale = str(item.get("rationale") or "Based on current farm telemetry")
				recommendations.append(AskRecommendation(action=action, rationale=rationale))

		sources_raw = llm_payload.get("sources")
		sources: list[AskSource] = []
		if isinstance(sources_raw, list):
			for item in sources_raw:
				if not isinstance(item, dict):
					continue
				sources.append(
					AskSource(
						layer=str(item.get("layer") or intent),
						reference=str(item.get("reference") or f"farm:{farm_id}:{intent}"),
						payload=item.get("payload") if isinstance(item.get("payload"), dict) else {},
					)
				)
		if not sources:
			sources.append(
				AskSource(
					layer=intent,
					reference=f"farm:{farm_id}:{intent}",
					payload={"context_keys": list(context.keys())},
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
		)

	@staticmethod
	def _fallback_response(
		*,
		intent: str,
		language: AskLanguage,
		context: dict[str, Any],
	) -> dict[str, Any]:
		language_hint = {
			AskLanguage.en: "Based on current farm telemetry",
			AskLanguage.fr: "Selon les données actuelles de la ferme",
			AskLanguage.ar: "بناءً على بيانات المزرعة الحالية",
		}[language]
		return {
			"answer": f"{language_hint}, intent={intent}. Please review recommended actions.",
			"confidence": 0.62,
			"recommendations": [
				{
					"action": "Review the latest analytics snapshot",
					"rationale": "Grounded in current measured values",
				}
			],
			"sources": [
				{
					"layer": intent,
					"reference": "analytics_context",
					"payload": {"keys": list(context.keys())},
				}
			],
		}
