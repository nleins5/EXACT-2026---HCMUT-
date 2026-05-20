"""Physics knowledge distillation pipeline.

This module provides tools to build a knowledge base (KB) for RAG by distilling
physics problems into structured records containing formulas, symbols, and SymPy code.

Usage:
    python -m scripts.distill.distill_physics --source all
    python -m scripts.distill.verify_kb
    python -m scripts.distill.fetch_physics_formulae --include-constants
"""
from scripts.distill.prompts import (
    EXTRACT_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    build_extract_user_prompt,
    build_generate_user_prompt,
)
from scripts.distill.schema import KBRecord
from scripts.distill.teacher_client import GeminiTeacherClient, build_teacher_client

__all__ = [
    "KBRecord",
    "GeminiTeacherClient",
    "build_teacher_client",
    "EXTRACT_SYSTEM_PROMPT",
    "GENERATE_SYSTEM_PROMPT",
    "build_extract_user_prompt",
    "build_generate_user_prompt",
]
