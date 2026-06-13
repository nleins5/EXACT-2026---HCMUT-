from src.agent.nodes.logic_direct import (
    _match_option,
    _minimal_rule_proof,
    _parse_direct_response,
)


def test_parse_direct_response_reads_answer_and_minimal_premises():
    answer, premises_used = _parse_direct_response(
        '{"answer":"Yes","premises_used":[0,2]}'
    )

    assert answer == "Yes"
    assert premises_used == [0, 2]


def test_parse_direct_response_keeps_empty_premise_set():
    answer, premises_used = _parse_direct_response(
        '```json\n{"answer":"Uncertain","premises_used":[]}\n```'
    )

    assert answer == "Uncertain"
    assert premises_used == []


def test_parse_direct_response_falls_back_to_plain_answer():
    answer, premises_used = _parse_direct_response("A")

    assert answer == "A"
    assert premises_used is None


def test_match_option_maps_content_to_letter_option():
    question = """Which option is supported?
A. Asha may join Study Alpha
B. Asha cannot join Study Alpha
C. Asha is a robot
D. None"""

    assert _match_option(
        "Asha may join Study Alpha",
        ["A", "B", "C", "D"],
        question,
    ) == "A"


def test_minimal_rule_proof_excludes_downstream_and_unrelated_premises():
    premises = [
        "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
        "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
        "Every researcher who may join Study Alpha is listed as an active contributor.",
        "Asha completed ethics training.",
        "Asha has lab access.",
        "Asha has supervisor approval.",
        "Study Alpha has 12 enrolled participants.",
        "No premise states whether Asha has budget approval.",
    ]

    assert _minimal_rule_proof("Asha may join Study Alpha", premises) == [0, 1, 3, 4, 5]
