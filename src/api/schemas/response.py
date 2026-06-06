from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictResponse(BaseModel):
    """Official EXACT response shape returned by POST /predict."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(default="Unknown")
    explanation: str = Field(default="")
    fol: str = Field(default="")
    cot: list[str] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    supervisor_running: bool
    active_role: str | None = None
    uptime: float


def fallback_response() -> PredictResponse:
    return PredictResponse(
        answer="Unknown",
        explanation="The system could not determine a definitive answer for this query.",
        fol="",
        cot=[],
        premises=[],
        confidence=0.0,
    )

