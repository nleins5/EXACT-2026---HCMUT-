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
                "question": "Is A true?",
                "premises-NL": ["A"],
                "task_type": "logic",
            },
        )

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["ready"] is False
    assert prediction.status_code == 200
    assert prediction.json()["answer"] == "True"


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
    assert prediction.json()["answer"] == "Unknown"
    assert predict_route._predict_gate.locked() is False


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


def test_models_endpoint_discloses_both_configured_models(monkeypatch):
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
    assert len(payload["data"]) == 2
    assert payload["exact_runtime"]["reachable"] is True
