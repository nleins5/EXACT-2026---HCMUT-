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


def fallback_response(error: str | None = None) -> PredictResponse:
    explanation = "The system could not complete reasoning within the request budget."
    if error:
        explanation = f"{explanation} Internal detail: {error}"
    return PredictResponse(
        answer="Unknown",
        explanation=explanation,
        fol="",
        cot=[],
        premises=[],
        confidence=0.0,
    )

