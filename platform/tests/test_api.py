"""API contract tests — structured errors, valid responses, no stack traces.

Model-dependent tests moved to test_predict_api.py.
Tests here work without model.joblib (health, drift, queue, registry, openapi).
"""

import asyncio
import pytest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.routers import drift, queue, registry


def _build_test_app():
    """FastAPI app with routers that work without a loaded model."""
    app = FastAPI(title="Test Platform")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.state.settings = Settings()
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    app.state.model = None
    app.state.threshold = app.state.settings.threshold
    app.state.drift_accumulator = [
        {
            "age": 40, "job": "admin.", "marital": "married",
            "education": "university.degree", "default": "no",
            "housing": "yes", "loan": "no", "contact": "cellular",
            "month": "may", "day_of_week": "mon", "campaign": 1,
            "pdays": (999 if i < 50 else 0),
            "previous": 0, "poutcome": "nonexistent",
            "emp.var.rate": (1.1 if i < 50 else -2.0),
            "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4,
            "euribor3m": (4.8 if i < 50 else 2.0),
            "nr.employed": 5191, "proba": (0.2 if i < 50 else 0.8),
        }
        for i in range(100)
    ]
    app.state.last_severity = "stable"

    app.include_router(drift.router, prefix="/drift", tags=["drift"])
    app.include_router(queue.router, prefix="/queue", tags=["queue"])
    app.include_router(registry.router, prefix="/registry", tags=["registry"])
    return app


_TEST_APP = _build_test_app()


@pytest.fixture
def test_client() -> TestClient:
    """TestClient that works without model.joblib. Predict tests use predict_client in test_predict_api.py."""
    _TEST_APP.state.last_severity = "stable"
    with TestClient(_TEST_APP) as client:
        yield client


def test_health(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema(test_client):
    response = test_client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/drift/report" in schema["paths"]
    assert "/registry/promote" in schema["paths"]
    assert "/registry/status" in schema["paths"]
    assert "/registry/history" in schema["paths"]
    assert "/registry/rollback" in schema["paths"]
    assert "/queue/status" in schema["paths"]


def test_drift_report_endpoint(test_client):
    """GET /drift/report returns webhook_sent true when the agent accepts the payload."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    original_client = test_client.app.state.http_client
    test_client.app.state.http_client = mock_client

    try:
        response = test_client.get("/drift/report")
    finally:
        test_client.app.state.http_client = original_client
        asyncio.run(mock_client.aclose())

    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "webhook_sent" in data
    assert data["report"]["severity"] == "critical"
    assert data["webhook_sent"] is True


def test_drift_report_endpoint_webhook_failure_sets_error(test_client):
    """GET /drift/report returns webhook_sent false and a useful error on agent failure."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "schema mismatch"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    original_client = test_client.app.state.http_client
    test_client.app.state.http_client = mock_client

    try:
        response = test_client.get("/drift/report")
    finally:
        test_client.app.state.http_client = original_client
        asyncio.run(mock_client.aclose())

    assert response.status_code == 200
    data = response.json()
    assert data["webhook_sent"] is False
    assert "422" in data["webhook_error"]


def test_queue_status_redis_failure(test_client):
    """GET /queue/status returns safe response when Redis is unavailable."""
    response = test_client.get("/queue/status")
    assert response.status_code == 200
    data = response.json()
    assert data["redis_connected"] is False
    assert data["queue_length"] is None
    assert data["dlq_length"] is None
    assert "Redis unavailable" in data["worker_note"]


def test_registry_status_ok(test_client, monkeypatch):
    """GET /registry/status returns model name and version info without live MLflow."""

    class FakeModelVersion:
        version = "1"
        last_updated_timestamp = None

    class FakeMlflowClient:
        def get_model_version_by_alias(self, name, alias):
            if alias == "candidate":
                return FakeModelVersion()
            raise RuntimeError("Production alias not set")

    monkeypatch.setattr(registry.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(registry, "MlflowClient", FakeMlflowClient)

    response = test_client.get("/registry/status")
    assert response.status_code == 200
    data = response.json()
    assert "registered_model_name" in data
    assert data["registered_model_name"] == "bank_marketing_pipeline"
    assert data["candidate_version"] == "1"
    assert data["production_version"] is None
    assert "status" in data


def test_promote_requires_approved_by(test_client):
    """POST /registry/promote rejects requests with empty approved_by."""
    payload = {
        "model_uri": "models:/bank_marketing_pipeline@candidate",
        "approved_by": "",
        "approval_id": "apr-1",
        "investigation_id": "test-inv-1",
        "timestamp": "2026-05-06T12:00:00Z",
    }
    response = test_client.post("/registry/promote", json=payload)
    assert response.status_code == 422
    assert "approved_by" in response.json()["detail"]


def test_rollback_requires_approval_id(test_client):
    """POST /registry/rollback rejects rollback without HIL approval_id."""
    payload = {
        "target_version": "1",
        "approved_by": "jad",
    }
    response = test_client.post("/registry/rollback", json=payload)
    assert response.status_code == 422
    assert "approval_id" in str(response.json()["detail"])
