"""Tests for deterministic Type 2 formula coverage."""
from src.agent.nodes.physics_baseline import solve_common_physics
from src.agent.nodes.logic_direct import is_multiple_choice
from src.agent.nodes.logic_retrieval import retrieve_known_logic
from src.agent.graph import run_pipeline


def test_solves_official_capacitor_energy_sample():
    result = solve_common_physics(
        "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
    )

    assert result is not None
    assert result["answer"] == "0.045"
    assert result["unit"] == "J"
    assert result["confidence"] == 0.99


def test_solves_parallel_resistance_sample():
    result = solve_common_physics(
        "A parallel circuit has R1 = 30 Ohm and R2 = 60 Ohm. "
        "Calculate the equivalent resistance."
    )

    assert result is not None
    assert result["answer"] == "20"
    assert result["unit"] == "Ohm"


def test_does_not_guess_unknown_formula_family():
    assert solve_common_physics("Explain why electric field lines never cross.") is None


def test_solves_coulomb_force_with_unit_conversion():
    result = solve_common_physics(
        "Calculate the electric force between charges 2 uC and 3 uC separated by 0.5 m."
    )

    assert result is not None
    assert result["unit"] == "N"


def test_solves_electric_field():
    result = solve_common_physics(
        "Calculate the electric field due to a 2 nC point charge at a distance of 0.1 m."
    )

    assert result is not None
    assert result["unit"] == "N/C"


def test_pipeline_uses_baseline_before_loading_graph(monkeypatch):
    monkeypatch.setattr(
        "src.agent.graph.get_graph",
        lambda: (_ for _ in ()).throw(AssertionError("graph should not load")),
    )

    result = run_pipeline(
        "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
    )

    assert result["answer"] == "0.045"
    assert result["unit"] == "J"


def test_detects_logic_multiple_choice_question():
    assert is_multiple_choice("Question\nA. One\nB. Two\nC. Three\nD. Four")
    assert not is_multiple_choice("Is the conclusion true?")


def test_retrieves_released_logic_example():
    import json
    from pathlib import Path

    dataset = (
        Path("data/EXACT2026_dataset_2026-05-15")
        / "Logic_Based_Educational_Queries_Text_Only"
        / "Logic_Based_Educational_Queries.json"
    )
    record = json.loads(dataset.read_text(encoding="utf-8"))[0]
    result = retrieve_known_logic(record["questions"][0], record["premises-NL"])

    assert result is not None
    assert result["answer"] == "A"


def test_does_not_retrieve_unseen_logic_question():
    assert retrieve_known_logic("An unseen logic question?", ["An unseen premise."]) is None
