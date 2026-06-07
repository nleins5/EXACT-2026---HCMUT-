"""Tests that verified solver evidence survives explanation-model failures."""
from src.agent.nodes.fallbacks import (
    extract_physics_answer,
    logic_solver_fallback,
    physics_solver_fallback,
)


def _state(code_output: str, *, code_error: bool = False):
    return {
        "premises": ["All cats are mammals."],
        "intermediate_answer": {
            "code_output": code_output,
            "code_error": code_error,
        },
    }


def test_logic_fallback_preserves_z3_prediction():
    result = logic_solver_fallback(_state("Predicted: True"), "model unavailable")
    assert result["final_answer"]["answer"] == "Yes"
    assert result["final_answer"]["confidence"] > 0


def test_logic_fallback_preserves_unknown_reasoning_evidence():
    result = logic_solver_fallback(_state("Predicted: Unknown"), "model unavailable")
    assert result["final_answer"]["answer"] == "Unknown"
    assert result["final_answer"]["cot"]
    assert result["final_answer"]["premises"]


def test_logic_fallback_preserves_mcq_letter():
    result = logic_solver_fallback(_state("Predicted: C"), "model unavailable")
    assert result["final_answer"]["answer"] == "C"


def test_physics_fallback_extracts_final_answer():
    result = physics_solver_fallback(
        _state("R_eq = 20 Ohm\nFINAL_ANSWER: 20.0 Ohm"),
        "model unavailable",
    )
    assert result["final_answer"]["answer"] == "20.0 Ohm"
    assert result["final_answer"]["cot"]


def test_extract_physics_answer_returns_none_without_marker():
    assert extract_physics_answer("R_eq = 20 Ohm") is None
