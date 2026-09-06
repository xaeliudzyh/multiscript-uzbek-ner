from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EntityLabel = Literal[
    "ORG",
    "NAME",
    "GEO",
]


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
    )


class Entity(BaseModel):
    text: str
    label: EntityLabel
    start: int
    end: int
    confidence: float | None = None
    source: str = "model"


class PredictResponse(BaseModel):
    text: str
    entities: list[Entity]
    latency_ms: float
    postprocess_stats: dict[str, int]


# Compatibility with the original service API.
class InputItem(BaseModel):
    hash: str = Field(min_length=1)
    text: str


class Prediction(BaseModel):
    hash: str
    entities: list[Entity]


class LegacyPredictResponse(BaseModel):
    data: list[Prediction]
