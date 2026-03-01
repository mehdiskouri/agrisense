"""Pydantic schemas for the /ask NL endpoint."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AskLanguage(StrEnum):
	en = "en"
	fr = "fr"
	ar = "ar"


class AskRequest(BaseModel):
	question: str = Field(min_length=3, max_length=2000)
	language: AskLanguage = AskLanguage.en


class AskRecommendation(BaseModel):
	action: str = Field(min_length=1, max_length=300)
	rationale: str = Field(min_length=1, max_length=1000)


class AskSource(BaseModel):
	layer: str = Field(min_length=1, max_length=100)
	reference: str = Field(min_length=1, max_length=200)
	payload: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
	farm_id: str
	question: str
	language: AskLanguage
	intent: str
	answer: str
	confidence: float = Field(ge=0.0, le=1.0)
	recommendations: list[AskRecommendation] = Field(default_factory=list)
	sources: list[AskSource] = Field(default_factory=list)
