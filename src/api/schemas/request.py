from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

TaskType = Literal["logic", "physics"]
QuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
]
PremiseText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]


class PredictRequest(BaseModel):
    """Normalize the EXACT 2026 unified input schema.

    The evaluation server sends:
    {
      "query_id": "T1_0001",
      "type": "type1",
      "query": "...",
      "premises": ["...", "..."],
      "options": ["Yes", "No", "Uncertain"]
    }
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    query_id: str = Field(
        default="",
        description="Unique query identifier from evaluation server.",
    )
    question: QuestionText
    premises_nl: list[PremiseText] = Field(
        default_factory=list,
        alias="premises-NL",
        max_length=200,
    )
    options: list[str] = Field(
        default_factory=list,
        description="Choice set for multiple-choice questions. Empty for free-form.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Optional explicit query type from the evaluator: logic/type1 or physics/type2.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_evaluator_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        data = dict(data)

        # ── query_id ──
        if data.get("query_id") is None:
            data["query_id"] = ""

        # ── question: accept "query" (official) and "questions" (legacy) ──
        if data.get("question") is None:
            if data.get("query") is not None:
                data["question"] = data["query"]
            elif data.get("questions") is not None:
                data["question"] = data["questions"]

        # ── premises: accept "premises" (official), "premises-NL", "premise-NL" ──
        if data.get("premises-NL") is None:
            if data.get("premises") is not None:
                data["premises-NL"] = data["premises"]
            elif data.get("premise-NL") is not None:
                data["premises-NL"] = data["premise-NL"]

        # ── options: accept "options" (official), "choices" ──
        if data.get("options") is None:
            if data.get("choices") is not None:
                data["options"] = data["choices"]

        # ── task_type: normalize from various field names ──
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
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [
                str(item)
                for item in value
                if item is not None and str(item).strip()
            ]
        return []

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if item is not None and str(item).strip()]
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
