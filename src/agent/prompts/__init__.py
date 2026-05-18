"""Prompt templates — moi node co prompt rieng (English content).

Quy uoc:
- *_SYSTEM_PROMPT: system message (huong dan format).
- *_USER_TEMPLATE: chuoi co `{placeholder}` cho `.format()`.
- *_OUTPUT_PROMPT / *_OUTPUT_ERROR_PROMPT: prompt cho explanation node
  (success branch vs error branch — match instruct.jsonl dataset).
"""
from src.agent.prompts.logic_formalizer import (
    Z3_SYSTEM_PROMPT,
    Z3_USER_TEMPLATE,
)
from src.agent.prompts.logic_explanation import (
    LOGIC_OUTPUT_PROMPT,
    LOGIC_OUTPUT_ERROR_PROMPT,
)
from src.agent.prompts.physics_formalizer import (
    PHYSICS_SYSTEM_PROMPT,
    PHYSICS_USER_TEMPLATE,
)
from src.agent.prompts.physics_explanation import (
    PHYSICS_OUTPUT_PROMPT,
    PHYSICS_OUTPUT_ERROR_PROMPT,
)

__all__ = [
    "Z3_SYSTEM_PROMPT",
    "Z3_USER_TEMPLATE",
    "LOGIC_OUTPUT_PROMPT",
    "LOGIC_OUTPUT_ERROR_PROMPT",
    "PHYSICS_SYSTEM_PROMPT",
    "PHYSICS_USER_TEMPLATE",
    "PHYSICS_OUTPUT_PROMPT",
    "PHYSICS_OUTPUT_ERROR_PROMPT",
]
