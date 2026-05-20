"""KB record schema for physics knowledge distillation.

Compact format: 1 record / problem, embed-friendly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KBRecord:
    """A knowledge base record after distillation from teacher LLM."""

    id: str                          # uid: "btc_pb_42", "electro_023"
    source: str                      # "btc_physics" | "electro" | "physics_formulae"
    problem: str                     # Problem statement (English)
    topic: str                       # e.g., "electric_circuits", "electrostatics", "other"
    formulas: list[str] = field(default_factory=list)   # plain math formulas
    symbols: dict[str, str] = field(default_factory=dict)  # var -> description + unit
    sympy_code: str = ""             # runnable SymPy code ending with print('FINAL_ANSWER: ...')
    answer: str = ""                 # final answer with unit (e.g., "20 Ohm")
    derivation: str = ""             # brief: 1-3 sentences explaining the reasoning

    # Verification metadata (set by verify_kb.py)
    verified: bool | None = None
    exec_output: str = ""
    exec_error: str = ""

    # Cost tracking (set by distill_physics.py)
    teacher_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "KBRecord":
        data: dict[str, Any] = json.loads(line)
        return cls(**data)
