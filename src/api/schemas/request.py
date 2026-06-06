from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskType = Literal["logic", "physics"]


class PredictRequest(BaseModel):
    """Unified BTC request for both logic and physics questions."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    question: str = Field(..., min_length=1)
    premises_nl: list[str] = Field(default_factory=list, alias="premises-NL")
    task_type: TaskType | None = Field(
        default=None,
        description="Optional explicit query type from the evaluator: logic/type1 or physics/type2.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_task_type_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if data.get("task_type") is not None:
            data["task_type"] = _normalize_task_type(data.get("task_type"))
            return data

        for key in ("query_type", "problem_type", "type", "task-type", "query-type"):
            if data.get(key) is not None:
                data["task_type"] = _normalize_task_type(data.get(key))
                break
        return data

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


def _normalize_task_type(value: Any) -> TaskType | None:
    """Accept the common EXACT variants while storing one internal label."""
    if value is None:
        return None

    raw = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if raw in {"1", "type1", "type-1"} or "logic" in raw:
        return "logic"
    if raw in {"2", "type2", "type-2", "physical"} or "physics" in raw:
        return "physics"
    return None
