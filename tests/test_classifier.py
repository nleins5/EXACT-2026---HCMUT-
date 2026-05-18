"""Test classifier rule-based — khong can LLM."""
from src.agent.nodes.classifier import classify_node, route_after_classify


def test_classify_logic_when_premises_present():
    state = {
        "question": "Co the qua mon nay khong?",
        "premises": ["Quy che 1", "Quy che 2"],
    }
    out = classify_node(state)
    assert out == {"task_type": "logic"}


def test_classify_physics_when_no_premises():
    state = {"question": "Tinh dien tro tuong duong?", "premises": []}
    out = classify_node(state)
    assert out == {"task_type": "physics"}


def test_classify_physics_when_premises_missing():
    state = {"question": "Tinh van toc?"}
    out = classify_node(state)
    assert out == {"task_type": "physics"}


def test_route_logic_to_formalizer():
    assert route_after_classify({"task_type": "logic"}) == "logic_formalizer"


def test_route_physics_to_rag():
    assert route_after_classify({"task_type": "physics"}) == "physics_rag"


def test_route_default_logic():
    assert route_after_classify({}) == "logic_formalizer"
