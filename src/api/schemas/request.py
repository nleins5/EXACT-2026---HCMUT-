from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """Unified BTC request for both logic and physics questions."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    question: str = Field(..., min_length=1)
    premises_nl: list[str] = Field(default_factory=list, alias="premises-NL")

    @field_validator("premises_nl", mode="before")
    @classmethod
    def normalize_premises(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return []

