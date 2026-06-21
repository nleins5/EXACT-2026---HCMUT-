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


def test_inconsistent_released_label_does_not_return_contradictory_cot():
    import csv
    from pathlib import Path

    dataset = (
        Path("data/EXACT2026_dataset_2026-05-15")
        / "Physics_Problems_Text_Only"
        / "Physics_Problems_Text_Only.csv"
    )
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        row = next(item for item in csv.DictReader(handle) if item["id"] == "NL086")

    result = retrieve_known_physics(row["question"])

    assert result is not None
    assert result["answer"] == row["answer"]
    assert result["cot"] == []
    assert result["explanation"]


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
    assert should_use_logic_direct(
        "Is the conclusion true?",
        ["Yes", "No", "Uncertain"],
    )


def test_solves_average_speed():
    result = solve_common_physics(
        "A bicycle travels 150 m in 12.5 s. What is its average speed?"
    )
    assert result is not None
    assert result["answer"] == "12"
    assert result["unit"] == "m/s"


def test_solves_gravitational_potential_energy():
    result = solve_common_physics(
        "Calculate the gravitational potential energy of a 5kg mass at a height of 10m. (g = 9.8)"
    )

    assert result is not None
    assert result["answer"] == "490"
    assert result["unit"] == "J"


def test_solves_kinetic_energy():
    result = solve_common_physics(
        "Find the kinetic energy of a 2 kg object moving at 3 m/s."
    )

    assert result is not None
    assert result["answer"] == "9"
    assert result["unit"] == "J"


def test_solves_newtons_second_law():
    result = solve_common_physics(
        "Calculate the force on a 4 kg cart accelerating at 2.5 m/s^2."
    )

    assert result is not None
    assert result["answer"] == "10"
    assert result["unit"] == "N"


def test_solves_linear_momentum():
    result = solve_common_physics(
        "What is the momentum of a 0.5 kg ball moving at 12 m/s?"
    )

    assert result is not None
    assert result["answer"] == "6"
    assert result["unit"] == "kg*m/s"


def test_solves_inductor_energy_in_requested_millijoules():
    result = solve_common_physics(
        "An inductor has L = 0.2 H and current 3 A. Calculate the magnetic field energy (mJ) stored in the inductor."
    )
    assert result is not None
    assert result["answer"] == "900"
    assert result["unit"] == "mJ"


def test_solves_lc_resonant_frequency():
    result = solve_common_physics(
        "Calculate the natural oscillation frequency for an LC circuit with L = 2 mH and C = 50 uF."
    )
    assert result is not None
    assert abs(float(result["answer"]) - 503.29) < 0.02
    assert result["unit"] == "Hz"


def test_solves_rlc_impedance():
    result = solve_common_physics(
        "An RLC circuit has R = 20 ohm, L = 0.5 H, C = 100 uF, and f = 50 Hz. Calculate the total impedance Z."
    )
    assert result is not None
    assert abs(float(result["answer"]) - 126.84) < 0.02
    assert result["unit"] == "ohm"


def test_solves_resultant_force_at_angle():
    result = solve_common_physics(
        "Two electric forces, each with a magnitude of 5 N, act at an angle of 60°. What is the resultant force?"
    )
    assert result is not None
    assert abs(float(result["answer"]) - 8.66) < 0.01
    assert result["unit"] == "N"


def test_scales_small_capacitor_charge_to_nanocoulombs():
    result = solve_common_physics(
        "A capacitor with capacitance 21.96 pF is charged to 28.8 V. Calculate the charge stored."
    )
    assert result is not None
    assert abs(float(result["answer"]) - 0.632448) < 1e-9
    assert result["unit"] == "nC"


def test_scales_small_capacitor_energy_to_nanojoules():
    result = solve_common_physics(
        "Calculate the energy stored in a capacitor with C = 47.93 pF and V = 70.1 V."
    )
    assert result is not None
    assert abs(float(result["answer"]) - 117.76424965) < 1e-6
    assert result["unit"] == "nJ"


def test_skips_symbolic_radical_current():
    assert solve_common_physics(
        "An inductor has L = 0.5 H and current 2√2 A. What is its magnetic energy?"
    ) is None


def test_skips_multi_output_reactance_question():
    assert solve_common_physics(
        "Given R = 12 ohm, C = 80 uF, and f = 60 Hz, determine the capacitive reactance and the power factor."
    ) is None


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
