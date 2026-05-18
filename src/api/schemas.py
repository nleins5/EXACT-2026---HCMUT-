"""
Pydantic schemas for the EXACT 2026 HTTP API.

Mirrors slide pages 32-33 of EXACT_Slides.pdf:

    Type 1 input:  {"premises-NL": [...], "question": "..."}
    Type 2 input:  {"question": "..."}

    Response:
        Required:  answer, explanation
        Optional:  fol, cot, premises, confidence  (boost P3 score)

The hyphen in ``premises-NL`` is illegal as a Python identifier, so we alias
it via Pydantic's ``Field(alias=...)`` and enable population by both alias
and field name.
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Incoming request from the EXACT 2026 evaluation system.

    The competition merges Type 1 (logic) and Type 2 (physics) into a single
    stream. Type 1 carries premises, Type 2 omits them.
    """

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        ...,
        description="The problem statement (NL).",
        min_length=1,
    )
    premises_nl: Optional[List[str]] = Field(
        default=None,
        alias="premises-NL",
        description="Natural-language premises for Type 1 logic problems.",
    )


class PredictResponse(BaseModel):
    """Response sent back to the evaluation system (slide 33).

    Only ``answer`` and ``explanation`` are mandatory; the rest contribute to
    the Reasoning Depth (P3) score.
    """

    answer: str = Field(..., description="Final answer (A/B/C, Yes/No/Unknown, or numeric).")
    explanation: str = Field(..., description="Natural-language reasoning.")

    fol: Optional[str] = Field(default=None, description="First-order logic formalisation.")
    cot: Optional[List[str]] = Field(default=None, description="Chain-of-thought steps.")
    premises: Optional[List[str]] = Field(default=None, description="Rules or laws used.")
    confidence: Optional[float] = Field(default=None, description="Self-rated confidence in [0,1].")


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""
    status: str
    model_loaded: bool
    elapsed_warmup_seconds: Optional[float] = None
