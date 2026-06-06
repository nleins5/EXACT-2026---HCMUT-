"""Deterministic final answers when the explanation model is unavailable."""
from __future__ import annotations

import re

from src.agent.state import AgentState
from src.utils.z3_output_parser import parse_z3_output


def extract_physics_answer(code_output: str) -> str | None:
    match = re.search(r"FINAL_ANSWER:\s*(.+)", code_output, re.IGNORECASE)
    return match.group(1).strip() if match else None


def logic_solver_fallback(state: AgentState, reason: str) -> dict:
    intermediate = state.get("intermediate_answer", {})
    code_output = intermediate.get("code_output", "")
    code_error = intermediate.get("code_error", False)
    prediction = "Unknown" if code_error else parse_z3_output(code_output)

    if code_error:
        explanation = "The symbolic logic solver did not produce a verified result."
        confidence = 0.0
    else:
        explanation = f"The Z3 entailment check returned {prediction}."
        confidence = 0.9 if prediction != "Unknown" else 0.5

    return {
        "final_answer": {
            "answer": prediction,
            "explanation": explanation,
            "fol": "",
            "cot": [code_output] if code_output else [],
            "premises": list(state.get("premises", []) or []),
            "confidence": confidence,
        },
        "error": reason,
    }


def physics_solver_fallback(state: AgentState, reason: str) -> dict:
    intermediate = state.get("intermediate_answer", {})
    code_output = intermediate.get("code_output", "")
    code_error = intermediate.get("code_error", False)
    verified_answer = extract_physics_answer(code_output)
    answer = verified_answer if verified_answer and not code_error else "Unknown"

    if answer == "Unknown":
        explanation = "The symbolic physics solver did not produce a verified final answer."
        confidence = 0.0
    else:
        explanation = f"The SymPy calculation returned {answer}."
        confidence = 0.85

    steps = [line.strip() for line in code_output.splitlines() if line.strip()]
    return {
        "final_answer": {
            "answer": answer,
            "explanation": explanation,
            "fol": "",
            "cot": steps[-6:],
            "premises": [],
            "confidence": confidence,
        },
        "error": reason,
    }
