from src.agent.nodes.logic_direct import _parse_direct_response


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
