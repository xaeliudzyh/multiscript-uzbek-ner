from typing import Literal

from pydantic import BaseModel, Field


EntityLabel = Literal["ORG", "NAME", "GEO"]


class InputItem(BaseModel):
    hash: str = Field(min_length=1)
    text: str


class Entity(BaseModel):
    label: EntityLabel
    start: int
    end: int


class Prediction(BaseModel):
    hash: str
    entities: list[Entity]


class PredictResponse(BaseModel):
    data: list[Prediction]
