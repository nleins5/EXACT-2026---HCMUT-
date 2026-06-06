from typing import List, Optional
from pydantic import BaseModel, Field


class ExactResponse(BaseModel):
    """Official response schema for EXACT 2026 competition."""

    answer: str = Field(..., description="Final answer (A, B, C, Yes, No, or numeric value)")
    explanation: str = Field(..., description="Natural-language reasoning explanation")

    fol: Optional[str] = Field(None, description="First-Order Logic translation")
    cot: Optional[List[str]] = Field(None, description="Chain-of-Thought reasoning steps")
    premises: Optional[List[str]] = Field(None, description="List of premises used")
    confidence: Optional[float] = Field(None, description="Model confidence score (0.0 - 1.0)")
