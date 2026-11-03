"""Pydantic request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceTicket(BaseModel):
    ticket_id: int
    similarity: float
    snippet: str


class SeverityInfo(BaseModel):
    label: Literal["low", "medium", "high"]
    score: float
    top_terms: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    route: Literal["resolved", "escalated"]
    intent: str
    answer: str | None = None
    sources: list[SourceTicket] = Field(default_factory=list)
    confidence: float = 0.0
    severity: SeverityInfo | None = None
    escalation_id: int | None = None
    latency_ms: int = 0


class FeedbackRequest(BaseModel):
    interaction_id: int
    thumbs: Literal["up", "down"]


class ResolveRequest(BaseModel):
    escalation_id: int
    resolution: str
    resolved_by: str = "senior"
