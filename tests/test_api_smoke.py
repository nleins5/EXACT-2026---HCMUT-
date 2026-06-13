"""Fast API contract smoke test without loading local models."""
import time

import httpx
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import settings
import src.api.routes.models as models_route
import src.api.routes.predict as predict_route


def test_health_and_predict_contract(monkeypatch):
    monkeypatch.setattr(settings.api, "warmup_role", None)

    def fake_pipeline(question, premises, **kwargs):
        return {
            "answer": "True",
            "explanation": "Verified by the test solver.",
            "fol": "",
            "cot": ["Predicted: True"],
            "premises": premises,
            "premises_used": [0],
            "unit": "",
            "confidence": 0.9,
            "code_error": False,
            "error_message": "",
        }

    monkeypatch.setattr(predict_route, "run_pipeline", fake_pipeline)

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post(
            "/predict",
            json={
                "query_id": "T1_TEST",
                "type": "type1",
                "query": "Is A true?",
                "premises": ["A"],
                "options": [],
            },
        )

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["ready"] is False
    assert prediction.status_code == 200
    result = prediction.json()
    assert isinstance(result, list)
    assert result[0]["answer"] == "True"
    assert result[0]["query_id"] == "T1_TEST"


def test_predict_timeout_cancels_background_pipeline(monkeypatch):
    monkeypatch.setattr(settings.api, "warmup_role", None)
    monkeypatch.setattr(predict_route, "_request_timeout", lambda: 1.1)
    monkeypatch.setattr(settings.api, "cancellation_grace_seconds", 0.1)

    def cancellable_pipeline(question, premises, *, cancel_event, **kwargs):
        while not cancel_event.wait(0.01):
            time.sleep(0.01)
        raise RuntimeError("cancelled by test")

    monkeypatch.setattr(predict_route, "run_pipeline", cancellable_pipeline)

    with TestClient(app) as client:
        started_at = time.monotonic()
        prediction = client.post("/predict", json={"question": "slow"})
        elapsed = time.monotonic() - started_at

    assert elapsed < 1.5
    assert prediction.status_code == 200
    result = prediction.json()
    assert isinstance(result, list)
    assert result[0]["answer"] == "Unknown"
    assert predict_route._predict_gate.locked() is False


def test_unknown_logic_answer_keeps_reasoning_evidence(monkeypatch):
    monkeypatch.setattr(settings.api, "warmup_role", None)

    def fake_pipeline(question, premises, **kwargs):
        return {
            "answer": "Unknown",
            "explanation": "Neither conclusion can be derived.",
            "fol": "P(x)",
            "cot": ["Checked the conclusion.", "Checked its negation."],
            "premises": premises,
            "premises_used": [0],
            "unit": "",
            "confidence": 0.5,
            "code_error": False,
            "error_message": "",
        }

    monkeypatch.setattr(predict_route, "run_pipeline", fake_pipeline)

    with TestClient(app) as client:
        prediction = client.post(
            "/predict",
            json={
                "query_id": "T1_UNK",
                "type": "type1",
                "query": "Is A true?",
                "premises": ["A may be true."],
                "options": [],
            },
        )

    assert prediction.status_code == 200
    result = prediction.json()
    assert isinstance(result, list)
    assert result[0]["answer"] == "Unknown"
    assert result[0]["reasoning"] is not None
    assert result[0]["reasoning"]["steps"]


def test_unknown_logic_answer_maps_to_uncertain_option(monkeypatch):
    monkeypatch.setattr(settings.api, "warmup_role", None)
    monkeypatch.setattr(
        predict_route,
        "run_pipeline",
        lambda *args, **kwargs: {
            "task_type": "logic",
            "answer": "Unknown",
            "explanation": "Neither conclusion can be derived.",
            "premises_used": [0],
        },
    )

    with TestClient(app) as client:
        prediction = client.post(
            "/predict",
            json={
                "query_id": "T1_UNCERTAIN",
                "type": "type1",
                "query": "Can the conclusion be determined?",
                "premises": ["A may be true."],
                "options": ["Yes", "No", "Uncertain"],
            },
        )

    assert prediction.json()[0]["answer"] == "Uncertain"


def test_explicit_choice_marker_does_not_match_letters_inside_words():
    assert predict_route._constrain_answer_to_options(
        "Answer: C",
        ["A", "B", "C", "D"],
    ) == "C"
    assert predict_route._constrain_answer_to_options(
        "The answer is D",
        ["A", "B", "C", "D"],
    ) == "D"


def test_choice_fallback_still_returns_a_valid_option():
    payload = predict_route.PredictRequest.model_validate(
        {
            "query_id": "T1_FALLBACK",
            "type": "type1",
            "query": "Can this be determined?",
            "premises": ["A may be true."],
            "options": ["Yes", "No", "Uncertain"],
        }
    )

    result = predict_route._fallback_for_request(payload)

    assert result.answer == "Uncertain"
    assert result.query_id == "T1_FALLBACK"


def test_empty_logic_premises_used_is_preserved():
    result = predict_route._sanitize_response(
        {
            "task_type": "logic",
            "answer": "Yes",
            "explanation": "Verified.",
            "premises_used": [],
        },
        query_id="T1_PREMISES",
        options=[],
        num_premises=3,
        task_type_hint="logic",
    )

    assert result.premises_used == []


def test_models_endpoint_reports_unreachable_runtime(monkeypatch):
    monkeypatch.setattr(settings.api, "warmup_role", None)
    monkeypatch.setattr(models_route, "_cached_response", None)

    class FailingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            raise httpx.ConnectError("offline", request=httpx.Request("GET", url))

    monkeypatch.setattr(models_route.httpx, "AsyncClient", FailingClient)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 503
    assert len(response.json()["data"]) == 2


def test_models_endpoint_proxies_the_active_runtime_model(monkeypatch):
    monkeypatch.setattr(settings.api, "warmup_role", None)
    monkeypatch.setattr(models_route, "_cached_response", None)

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"object": "list", "data": [{"id": "active-model"}]}

    class WorkingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(models_route.httpx, "AsyncClient", WorkingClient)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    payload = response.json()
    assert response.status_code == 200
    assert payload == {"object": "list", "data": [{"id": "active-model"}]}
