"""Deterministic final answers when the explanation model is unavailable."""
from __future__ import annotations

import re

from src.agent.state import AgentState
from src.utils.z3_output_parser import parse_z3_output


def extract_physics_answer(code_output: str) -> str | None:
    match = re.search(r"FINAL_ANSWER:\s*(.+)", code_output, re.IGNORECASE)
    return match.group(1).strip() if match else None


def split_physics_answer_unit(answer: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\s+([A-Za-z][A-Za-z0-9/*^._-]*)\s*",
        answer,
    )
    if not match:
        return answer, ""
    value, unit = match.groups()
    if unit.lower() == "ohm":
        unit = "ohm"
    return value, unit


def normalize_logic_answer(answer: str) -> str:
    normalized = str(answer).strip()
    aliases = {
        "true": "Yes",
        "false": "No",
        "yes": "Yes",
        "no": "No",
        "unknown": "Unknown",
        "uncertain": "Uncertain",
    }
    return aliases.get(normalized.lower(), normalized.upper() if normalized.upper() in {"A", "B", "C", "D"} else normalized)


def logic_solver_fallback(state: AgentState, reason: str) -> dict:
    intermediate = state.get("intermediate_answer", {})
    code_output = intermediate.get("code_output", "")
    code_error = intermediate.get("code_error", False)
    prediction = "Unknown" if code_error else normalize_logic_answer(parse_z3_output(code_output))
    premises = list(state.get("premises", []) or [])
    generated_code = intermediate.get("generated_code", "")
    conclusion_match = re.search(r"^\s*conclusion\s*=\s*(.+)$", generated_code, re.MULTILINE)
    fol = conclusion_match.group(1).strip() if conclusion_match else ""

    if code_error:
        explanation = "The symbolic logic solver did not produce a verified result."
        confidence = 0.0
    elif prediction == "Yes":
        explanation = "The Z3 entailment check verified that the conclusion follows from the supplied premises."
        confidence = 0.9
    elif prediction == "No":
        explanation = "The Z3 entailment check verified that the conclusion is contradicted by the supplied premises."
        confidence = 0.9
    elif prediction in {"A", "B", "C", "D"}:
        explanation = f"The logic check selected option {prediction} from the supplied premises."
        confidence = 0.85
    else:
        explanation = "Neither the conclusion nor its negation could be derived from the supplied premises."
        confidence = 0.5

    return {
        "final_answer": {
            "answer": prediction,
            "explanation": explanation,
            "fol": fol,
            "cot": [
                f"Loaded {len(premises)} natural-language premises into the Z3 verification program.",
                code_output,
            ] if code_output else [],
            "premises": premises,
            "premises_used": list(range(len(premises))),
            "unit": "",
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
        explanation = f"The generated SymPy program executed successfully and returned {answer}."
        confidence = 0.85

    steps = [line.strip() for line in code_output.splitlines() if line.strip()]
    unit = ""
    if answer != "Unknown":
        answer, unit = split_physics_answer_unit(answer)

    return {
        "final_answer": {
            "answer": answer,
            "explanation": explanation,
            "fol": "",
            "cot": steps[-6:],
            "premises": [],
            "premises_used": [],
            "unit": unit,
            "confidence": confidence,
        },
        "error": reason,
    }
