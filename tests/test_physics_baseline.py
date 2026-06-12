"""Tests for deterministic Type 2 formula coverage."""
from src.agent.nodes.physics_baseline import solve_common_physics
from src.agent.nodes.logic_direct import is_multiple_choice, should_use_logic_direct
from src.agent.nodes.logic_retrieval import retrieve_known_logic
from src.agent.nodes.physics_retrieval import retrieve_known_physics
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
    assert result["unit"] == "ohm"


def test_solves_official_total_current_parallel_sample():
    result = solve_common_physics(
        "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a "
        "12V battery. Find the total current."
    )

    assert result is not None
    assert result["answer"] == "5"
    assert result["unit"] == "A"


def test_parallel_branch_current_uses_the_named_branch_resistance():
    result = solve_common_physics(
        "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a "
        "12V battery. Find the current through R1."
    )

    assert result is not None
    assert result["answer"] == "3"
    assert result["unit"] == "A"


def test_parses_kiloohm_symbol_prefix():
    result = solve_common_physics(
        "What is the current through a 1 kΩ resistor connected to 10 V?"
    )

    assert result is not None
    assert result["answer"] == "0.01"
    assert result["unit"] == "A"


def test_parallel_short_circuit_does_not_crash():
    result = solve_common_physics(
        "What is the equivalent resistance of 0 ohm and 10 ohm in parallel?"
    )

    assert result is not None
    assert result["answer"] == "0"
    assert result["unit"] == "ohm"


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


def test_does_not_apply_simple_coulomb_rule_to_net_force_problem():
    result = solve_common_physics(
        "Three charges 1 uC, 2 uC, and 3 uC form a triangle with sides "
        "0.1 m, 0.2 m, and 0.3 m. Calculate the net force on the third charge."
    )

    assert result is None


def test_retrieves_released_physics_example_with_ascii_unit():
    result = retrieve_known_physics(
        "Calculate the capacitance C of the capacitor, given that it stores "
        "Q = 3 mC when fully charged under U = 30 V."
    )

    assert result is not None
    assert result["answer"] == "100"
    assert result["unit"] == "uF"


def test_released_physics_units_are_ascii():
    import csv
    from pathlib import Path

    dataset = (
        Path("data/EXACT2026_dataset_2026-05-15")
        / "Physics_Problems_Text_Only"
        / "Physics_Problems_Text_Only.csv"
    )
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        row = next(item for item in csv.DictReader(handle) if item["unit"] == "J/m³")
    result = retrieve_known_physics(row["question"])

    assert result is not None
    assert result["unit"].isascii()


def test_does_not_retrieve_ambiguous_released_physics_question():
    import csv
    from pathlib import Path

    dataset = (
        Path("data/EXACT2026_dataset_2026-05-15")
        / "Physics_Problems_Text_Only"
        / "Physics_Problems_Text_Only.csv"
    )
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    question = next(row["question"] for row in rows if row["id"] == "LD302")

    assert retrieve_known_physics(question) is None


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
    assert should_use_logic_direct("How many credits are still required?", [])
    assert should_use_logic_direct("Choose one", ["red", "blue"])
    assert not should_use_logic_direct(
        "Is the conclusion true?",
        ["Yes", "No", "Uncertain"],
    )


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


def test_retrieval_requires_matching_premises_and_preserves_original_indices():
    import json
    from pathlib import Path

    dataset = (
        Path("data/EXACT2026_dataset_2026-05-15")
        / "Logic_Based_Educational_Queries_Text_Only"
        / "Logic_Based_Educational_Queries.json"
    )
    records = json.loads(dataset.read_text(encoding="utf-8"))
    record = next(
        item
        for item in records
        if item.get("idx")
        and any(indices and indices != list(range(1, len(indices) + 1)) for indices in item["idx"])
    )
    question_index = next(
        index
        for index, indices in enumerate(record["idx"])
        if indices and indices != list(range(1, len(indices) + 1))
    )
    expected = [index - 1 for index in record["idx"][question_index]]
    question = record["questions"][question_index]

    result = retrieve_known_logic(question, record["premises-NL"])

    assert result is not None
    assert result["premises_used"] == expected
    assert retrieve_known_logic(question, ["Completely unrelated premise."]) is None
