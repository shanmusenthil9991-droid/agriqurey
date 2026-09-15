"""
Integration tests for the wiring between layers.

Covers the two ends of the system that unit tests miss:
  * GET /api/evaluation      — the dashboard's data source.
  * Frontend page serving    — the same-origin guarantee the browser JS
                               depends on for "/api/classify" to resolve.
  * Secret hygiene           — no response ever contains the API key.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app  # noqa: E402
from backend.routes import evaluation as evaluation_route  # noqa: E402


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# GET /api/evaluation
# ---------------------------------------------------------------------------

def test_evaluation_reports_unavailable_when_no_results(client, monkeypatch, tmp_path):
    """A missing results file is an expected state, not a server error."""
    monkeypatch.setattr(
        evaluation_route, "RESULTS_JSON_PATH", str(tmp_path / "missing.json")
    )
    response = client.get("/api/evaluation")
    assert response.status_code == 200

    body = response.get_json()
    assert body["available"] is False
    assert "message" in body
    # Crucially: no fabricated placeholder metrics.
    assert "metrics" not in body


def test_evaluation_returns_real_on_disk_values(client, monkeypatch, tmp_path):
    results = {
        "dataset": {"path": "data/test.csv", "total_rows": 45, "scored_rows": 45},
        "categories_evaluated": [
            "Symptoms", "Prevention", "Management", "General Information",
        ],
        "metrics": {
            "accuracy": 0.8222,
            "precision_macro": 0.8311,
            "recall_macro": 0.8194,
            "f1_macro": 0.8201,
        },
        "classification_report": {
            "Symptoms": {"support": 11},
            "Prevention": {"support": 12},
            "Management": {"support": 11},
            "General Information": {"support": 11},
        },
        "diagnostics": {"unable_to_classify_predictions": 0, "system_failures": 0},
        "generated_at_utc": "2026-01-01T00:00:00Z",
    }
    confusion = {
        "labels": ["Symptoms", "Prevention", "Management", "General Information"],
        "matrix": [[9, 0, 1, 1], [0, 11, 1, 0], [1, 1, 9, 0], [1, 0, 1, 9]],
    }

    results_path = tmp_path / "results.json"
    confusion_path = tmp_path / "confusion_matrix.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    confusion_path.write_text(json.dumps(confusion), encoding="utf-8")

    monkeypatch.setattr(evaluation_route, "RESULTS_JSON_PATH", str(results_path))
    monkeypatch.setattr(
        evaluation_route, "CONFUSION_MATRIX_JSON_PATH", str(confusion_path)
    )
    monkeypatch.setattr(
        evaluation_route, "METHOD_COMPARISON_JSON_PATH", str(tmp_path / "nope.json")
    )

    body = client.get("/api/evaluation").get_json()

    assert body["available"] is True
    # Values are passed through untouched — the route computes nothing.
    assert body["metrics"]["accuracy"] == 0.8222
    assert body["metrics"]["f1_macro"] == 0.8201
    assert body["total_test_questions"] == 45
    assert body["confusion_matrix"]["matrix"] == confusion["matrix"]
    # Support counts come from the saved classification report.
    assert body["category_distribution"]["Prevention"] == 12
    # Missing comparison degrades gracefully without failing the request.
    assert body["method_comparison"]["available"] is False


def test_evaluation_rejects_post(client):
    assert client.post("/api/evaluation").status_code == 405


# ---------------------------------------------------------------------------
# Frontend serving (same-origin guarantee)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path", ["/", "/index.html", "/ask.html", "/result.html", "/history.html", "/admin.html"]
)
def test_frontend_pages_are_served(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"CropCare" in response.data


@pytest.mark.parametrize(
    "path", ["/static/css/style.css", "/static/js/classifier.js", "/static/js/result.js", "/static/js/history.js", "/static/js/dashboard.js"]
)
def test_static_assets_are_served(client, path):
    assert client.get(path).status_code == 200


def test_ask_page_reaches_the_classify_endpoint_same_origin(client):
    """
    The ask page must be served from the same origin as the API it calls,
    otherwise the relative "/api/classify" fetch cannot resolve.
    """
    assert client.get("/ask.html").status_code == 200
    posted = client.post("/api/classify", json={"question": "What are rice blast symptoms?"})
    assert posted.status_code != 404


def test_landing_page_links_to_the_real_ask_page(client):
    """The home page CTA must reach ask.html, not a dead placeholder anchor."""
    html = client.get("/").data.decode("utf-8")
    assert 'href="ask.html"' in html
    assert "question page is on its way" not in html


def test_unlisted_pages_are_not_served(client):
    assert client.get("/secrets.html").status_code == 404


@pytest.mark.parametrize("path", ["/..%2f.env", "/%2e%2e/.env", "/../.env"])
def test_path_traversal_is_blocked(client, path):
    assert client.get(path).status_code in (400, 404)


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------

def test_health_never_exposes_the_api_key(client, monkeypatch):
    secret = "sk-super-secret-key-value-123456"
    monkeypatch.setenv("LLM_API_KEY", secret)

    body = client.get("/api/health").data.decode("utf-8")
    assert secret not in body
    # Readiness is reported as a boolean instead.
    assert "llm_configured" in body


def test_classify_error_never_exposes_the_api_key(client, monkeypatch):
    from backend.config.settings import config

    secret = "sk-super-secret-key-value-123456"
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setattr(config, "LLM_API_KEY", secret, raising=False)
    monkeypatch.setattr(config, "LLM_MODEL_NAME", "", raising=False)

    response = client.post("/api/classify", json={"question": "Why are my leaves yellow?"})
    assert secret not in response.data.decode("utf-8")


def test_no_stack_trace_leaks_on_bad_request(client):
    response = client.post("/api/classify", data="not json", content_type="text/plain")
    text = response.data.decode("utf-8")
    assert response.status_code == 400
    assert "Traceback" not in text
    assert "File \"" not in text


# ---------------------------------------------------------------------------
# Request size limit (DoS hardening)
# ---------------------------------------------------------------------------

def test_oversized_request_body_is_rejected_as_json(client):
    """
    Without a body-size cap, request.get_json() fully buffers and parses
    an arbitrarily large payload before any application-level validation
    runs. This must be capped, and the resulting error must still be
    JSON — Werkzeug's default 413 page is HTML.
    """
    huge_body = b"A" * (200 * 1024)
    response = client.post(
        "/api/classify", data=huge_body, content_type="application/json"
    )
    assert response.status_code == 413
    assert response.content_type.startswith("application/json")
    body = response.get_json()
    assert body["error_code"] == "payload_too_large"


def test_normal_sized_request_is_not_affected_by_the_size_cap(client):
    response = client.post(
        "/api/classify", json={"question": "What are rice blast symptoms?"}
    )
    # Not 413 — the cap must not interfere with legitimate traffic.
    assert response.status_code != 413
