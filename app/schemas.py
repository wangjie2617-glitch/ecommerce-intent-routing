"""API request and response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, examples=["快递一直没到，我想退款"])

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("text cannot be blank")
        return normalized


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.strip().split()) for value in values]
        if any(not value for value in cleaned):
            raise ValueError("texts cannot contain blank items")
        if any(len(value) > 500 for value in cleaned):
            raise ValueError("each text must contain at most 500 characters")
        return cleaned


class IntentScore(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class RouteDecision(BaseModel):
    department: str
    priority: str
    sla_minutes: int
    manual_review: bool
    reasons: list[str]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    text: str
    intents: list[IntentScore]
    route: RouteDecision
    model_type: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    items: list[PredictionResponse]
    count: int
    total_latency_ms: float

