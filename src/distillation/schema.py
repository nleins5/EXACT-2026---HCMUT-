"""KB record schema cho physics knowledge distillation.

Compact format: 1 record / problem, embed-friendly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KBRecord:
    """Mot record knowledge base sau khi distill tu teacher LLM."""

    id: str                          # uid: "btc_pb_42", "electro_023"
    source: str                      # "btc_physics" | "electro"
    problem: str                     # Problem statement (tieng Anh hoac Viet)
    topic: str                       # vd "electric_circuits", "kinematics", "thermodynamics"
    formulas: list[str] = field(default_factory=list)   # latex hoac plain math
    symbols: dict[str, str] = field(default_factory=dict)  # var -> mota + don vi
    sympy_code: str = ""             # code SymPy chuan, cham `print('FINAL_ANSWER: ...')`
    answer: str = ""                 # final answer voi unit (vd "20 Ohm")
    derivation: str = ""             # ngan: 1-3 cau giai thich

    # Verification metadata (set boi verify_kb.py)
    verified: bool | None = None
    exec_output: str = ""
    exec_error: str = ""

    # Cost tracking (set boi distill_physics.py)
    teacher_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "KBRecord":
        data: dict[str, Any] = json.loads(line)
        return cls(**data)
