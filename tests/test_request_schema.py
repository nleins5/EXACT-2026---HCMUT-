"""Tests for evaluator-compatible request normalization."""
import pytest
from pydantic import ValidationError

from src.api.schemas.request import PredictRequest


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("task_type", "type-1", "logic"),
        ("query_type", "physics", "physics"),
        ("type", "2", "physics"),
        ("problem_type", "logic-query", "logic"),
    ],
)
def test_task_type_aliases(field, value, expected):
    request = PredictRequest.model_validate({"question": "q", field: value})
    assert request.task_type == expected


def test_premises_string_is_normalized():
    request = PredictRequest.model_validate(
        {"question": "q", "premises-NL": "All cats are mammals."}
    )
    assert request.premises_nl == ["All cats are mammals."]


def test_task_1_official_payload_is_normalized():
    request = PredictRequest.model_validate(
        {
            "question": "Are cats mammals?",
            "premises-NL": "All cats are mammals.",
        }
    )

    assert request.question == "Are cats mammals?"
    assert request.premises_nl == ["All cats are mammals."]


def test_legacy_task_1_aliases_are_normalized():
    request = PredictRequest.model_validate(
        {"questions": "Are cats mammals?", "premise-NL": "All cats are mammals."}
    )

    assert request.question == "Are cats mammals?"
    assert request.premises_nl == ["All cats are mammals."]


def test_task_2_official_payload_is_normalized():
    request = PredictRequest.model_validate({"question": "What is force?"})

    assert request.question == "What is force?"
    assert request.premises_nl == []


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"question": "   "})


def test_oversized_question_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"question": "x" * 20_001})
