"""
Tests for the /api/health endpoint.

Health is a LIVENESS + READINESS check: it always returns HTTP 200 while
the process is up, and uses the `status` field to distinguish a service
that can classify ("ok") from one that is running but unconfigured
("degraded"). It must never expose the API key.

Run with: pytest
"""

import pytest

from backend.app import create_app
from backend.config.settings import config


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_returns_200_and_service_name(client):
    """The process being up must always be a 200, whatever the config."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["service"] == "crop-disease-classifier-backend"
    assert data["status"] in ("ok", "degraded")


def test_health_reports_ok_when_provider_is_configured(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-a-real-looking-key")

    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert data["llm_configured"] is True
    assert data["llm_is_offline_stub"] is False


def test_health_reports_degraded_when_key_is_missing(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")

    data = client.get("/api/health").get_json()
    assert data["status"] == "degraded"
    assert data["llm_configured"] is False


def test_health_treats_placeholder_key_as_unconfigured(client, monkeypatch):
    """The .env.example placeholder must not count as a real key."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "your_api_key_here")

    assert client.get("/api/health").get_json()["llm_configured"] is False


def test_health_flags_the_offline_stub_provider(client, monkeypatch):
    """Echo needs no key, but must be reported as a stub, not as ready."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "echo", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")

    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert data["llm_is_offline_stub"] is True


def test_health_never_returns_the_api_key(client, monkeypatch):
    secret = "sk-do-not-leak-me-abcdef123456"
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setenv("LLM_API_KEY", secret)

    body = client.get("/api/health").data.decode("utf-8")
    assert secret not in body


def test_health_rejects_post(client):
    assert client.post("/api/health").status_code == 405
