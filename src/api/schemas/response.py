from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReasoningBlock(BaseModel):
    """Structured reasoning evidence (optional)."""
    type: str = Field(default="fol", description="e.g. 'fol', 'cot', 'proof'")
    steps: list[str] = Field(default_factory=list)


class PredictResult(BaseModel):
    """Official EXACT 2026 response object per the Submission Guide §4."""

    model_config = ConfigDict(extra="ignore")

    query_id: str = Field(default="")
    answer: str = Field(default="Unknown")
    unit: str = Field(default="", description="ASCII unit for Type 2; empty for Type 1")
    explanation: str = Field(default="")
    premises_used: list[int] = Field(
        default_factory=list,
        description="0-based indices of premises actually used (Type 1); empty for Type 2",
    )
    reasoning: ReasoningBlock | None = Field(
        default=None,
        description="Optional structured reasoning evidence",
    )


# Keep the old name as an alias for backward compat within the codebase
PredictResponse = PredictResult


class HealthResponse(BaseModel):
    status: str
    ready: bool
    busy: bool
    supervisor_running: bool
    active_role: str | None = None
    startup_error: str | None = None
    uptime: float


def fallback_response(query_id: str = "") -> PredictResult:
    return PredictResult(
        query_id=query_id,
        answer="Unknown",
        unit="",
        explanation="The system could not determine a definitive answer for this query.",
        premises_used=[],
        reasoning=None,
    )
