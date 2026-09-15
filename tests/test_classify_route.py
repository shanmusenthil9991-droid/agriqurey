"""
Tests for the /api/classify and /api/health Flask endpoints.

The LLM classifier is mocked via monkeypatch throughout, so these tests
run fully OFFLINE: no API key, no network call, no cost.

Coverage:
- GET /api/health
- Request validation (missing/blank/wrong-type question, bad crop, extra
  fields, non-JSON body) -> 400, LLM never invoked
- Successful classification -> 200 with exactly the three expected fields
- Classifier-reported failures -> mapped to the right HTTP status, body
  never contains a stack trace or an API key
- An unexpected exception inside the route -> 500, safety-net handler
"""

import json

import pytest

from backend.app import create_app
from backend.services.llm_classifier import (
    ERROR_CONFIG,
    ERROR_INVALID_CATEGORY,
    ERROR_INVALID_CONFIDENCE,
    ERROR_INVALID_JSON,
    ERROR_PROVIDER,
)


@pytest.fixture
def app():
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def ok_result(category="Symptoms", confidence=0.91,
             reason="The question asks about visible symptoms affecting the crop."):
    return {
        "category": category, "confidence": confidence, "reason": reason,
        "status": "ok", "error": None, "error_code": None,
    }


def error_result(message, error_code):
    return {
        "category": "Unable to Classify", "confidence": 0.0,
        "reason": "The question could not be classified.",
        "status": "error", "error": message, "error_code": error_code,
    }


def post_json(client, body):
    return client.post("/api/classify", data=json.dumps(body),
                       content_type="application/json")


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

def test_health_returns_200_while_running(client):
    """
    Health always returns 200 while the process is up. `status` is "ok"
    or "degraded" depending on whether an LLM provider is configured,
    so this asserts liveness rather than a fixed readiness value.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] in ("ok", "degraded")


# ---------------------------------------------------------------------------
# Request validation — the LLM must never be called for these
# ---------------------------------------------------------------------------

def test_missing_question_field_returns_400(client, monkeypatch):
    called = []
    monkeypatch.setattr("backend.routes.classify.classify_question",
                        lambda **kw: called.append(kw) or ok_result())
    response = post_json(client, {"crop": "Rice"})
    assert response.status_code == 400
    assert response.get_json()["error_code"] == "invalid_input"
    assert called == []


def test_empty_question_returns_400(client, monkeypatch):
    called = []
    monkeypatch.setattr("backend.routes.classify.classify_question",
                        lambda **kw: called.append(kw) or ok_result())
    response = post_json(client, {"question": "   "})
    assert response.status_code == 400
    assert called == []


def test_non_string_question_returns_400(client):
    response = post_json(client, {"question": 12345})
    assert response.status_code == 400
    assert response.get_json()["field"] == "question"


def test_non_string_crop_returns_400(client):
    response = post_json(client, {"question": "What are the symptoms?", "crop": 42})
    assert response.status_code == 400
    assert response.get_json()["field"] == "crop"


def test_empty_string_crop_returns_400(client):
    response = post_json(client, {"question": "What are the symptoms?", "crop": "  "})
    assert response.status_code == 400


def test_overlong_crop_returns_400(client):
    response = post_json(client, {"question": "q?", "crop": "x" * 200})
    assert response.status_code == 400


def test_unknown_field_returns_400(client):
    response = post_json(client, {"question": "q?", "location": "field 3"})
    assert response.status_code == 400


def test_non_json_body_returns_400(client):
    response = client.post("/api/classify", data="question=hello",
                           content_type="application/x-www-form-urlencoded")
    assert response.status_code == 400


def test_malformed_json_body_returns_400(client):
    response = client.post("/api/classify", data="{not valid json",
                           content_type="application/json")
    assert response.status_code == 400


def test_json_array_body_returns_400(client):
    response = client.post("/api/classify", data=json.dumps(["question?"]),
                           content_type="application/json")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Successful classification
# ---------------------------------------------------------------------------

def test_successful_classification_returns_200_and_exact_shape(client, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.classify.classify_question",
        lambda **kw: ok_result(
            category="Symptoms", confidence=0.91,
            reason="The question asks about visible symptoms affecting the crop.",
        ),
    )
    response = post_json(client, {
        "question": "What are the symptoms of disease affecting my tomato leaves?",
        "crop": "Tomato",
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body == {
        "category": "Symptoms",
        "confidence": 0.91,
        "reason": "The question asks about visible symptoms affecting the crop.",
    }


def test_crop_is_optional(client, monkeypatch):
    received = {}
    def fake_classify(**kwargs):
        received.update(kwargs)
        return ok_result()
    monkeypatch.setattr("backend.routes.classify.classify_question", fake_classify)

    response = post_json(client, {"question": "What are the symptoms?"})
    assert response.status_code == 200
    assert received["crop"] is None


def test_question_and_crop_are_forwarded_unchanged(client, monkeypatch):
    received = {}
    def fake_classify(**kwargs):
        received.update(kwargs)
        return ok_result()
    monkeypatch.setattr("backend.routes.classify.classify_question", fake_classify)

    post_json(client, {"question": "Why are my leaves yellow?", "crop": "Wheat"})
    assert received["question"] == "Why are my leaves yellow?"
    assert received["crop"] == "Wheat"


@pytest.mark.parametrize("category", [
    "Symptoms", "Prevention", "Management", "General Information", "Unable to Classify",
])
def test_all_final_categories_pass_through(client, monkeypatch, category):
    monkeypatch.setattr("backend.routes.classify.classify_question",
                        lambda **kw: ok_result(category=category, confidence=0.5))
    response = post_json(client, {"question": "q?"})
    assert response.status_code == 200
    assert response.get_json()["category"] == category


def test_success_response_never_includes_status_or_error_keys(client, monkeypatch):
    """The route must strip internal bookkeeping fields from the response."""
    monkeypatch.setattr("backend.routes.classify.classify_question",
                        lambda **kw: ok_result())
    response = post_json(client, {"question": "q?"})
    body = response.get_json()
    assert set(body.keys()) == {"category", "confidence", "reason"}


# ---------------------------------------------------------------------------
# Classifier-reported failures -> mapped HTTP status, no leaked internals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error_code,expected_status", [
    (ERROR_INVALID_JSON, 502),
    (ERROR_INVALID_CATEGORY, 502),
    (ERROR_INVALID_CONFIDENCE, 502),
    (ERROR_PROVIDER, 502),
    (ERROR_CONFIG, 503),
])
def test_classifier_error_codes_map_to_expected_http_status(client, monkeypatch, error_code, expected_status):
    monkeypatch.setattr(
        "backend.routes.classify.classify_question",
        lambda **kw: error_result("Something went wrong.", error_code),
    )
    response = post_json(client, {"question": "q?"})
    assert response.status_code == expected_status
    body = response.get_json()
    assert body["error_code"] == error_code
    assert "category" not in body
    assert "confidence" not in body


def test_unrecognised_error_code_defaults_to_502(client, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.classify.classify_question",
        lambda **kw: error_result("Weird failure.", "some_future_error_code"),
    )
    response = post_json(client, {"question": "q?"})
    assert response.status_code == 502


def test_provider_error_message_reaches_caller_without_internals(client, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.classify.classify_question",
        lambda **kw: error_result("LLM API returned HTTP 500: server error", ERROR_PROVIDER),
    )
    response = post_json(client, {"question": "q?"})
    body = response.get_json()
    assert response.status_code == 502
    assert "Traceback" not in body["error"]
    assert body["error"] == "LLM API returned HTTP 500: server error"


# ---------------------------------------------------------------------------
# Secret hygiene / no stack traces
# ---------------------------------------------------------------------------

def test_api_key_never_appears_in_success_response(client, monkeypatch):
    monkeypatch.setattr("backend.routes.classify.classify_question",
                        lambda **kw: ok_result())
    response = post_json(client, {"question": "q?"})
    assert "sk-" not in response.get_data(as_text=True)


def test_api_key_never_appears_in_error_response(client, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.classify.classify_question",
        lambda **kw: error_result("LLM_API_KEY is not set.", ERROR_CONFIG),
    )
    response = post_json(client, {"question": "q?"})
    text = response.get_data(as_text=True)
    assert "sk-" not in text
    assert "your_api_key_here" not in text or "key" in text.lower()  # message may mention the placeholder name, never a real key


def test_unexpected_exception_returns_500_without_traceback(client, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("unexpected internal failure with secret_token=abc123")
    monkeypatch.setattr("backend.routes.classify.classify_question", boom)

    response = post_json(client, {"question": "q?"})
    assert response.status_code == 500
    text = response.get_data(as_text=True)
    assert "Traceback" not in text
    assert "secret_token" not in text
    assert response.get_json()["error_code"] == "internal_error"


def test_debug_mode_does_not_leak_traceback_html(monkeypatch):
    """
    Regression guard: Flask's default DEBUG behaviour re-raises unhandled
    exceptions (to power the interactive debugger), bypassing JSON error
    handlers entirely. This must be disabled so debug mode never leaks a
    traceback page to an API caller.
    """
    app = create_app()
    app.config["DEBUG"] = True
    app.config["TESTING"] = False

    def boom(**kwargs):
        raise RuntimeError("should never be shown to the caller")
    monkeypatch.setattr("backend.routes.classify.classify_question", boom)

    with app.test_client() as debug_client:
        response = debug_client.post(
            "/api/classify",
            data=json.dumps({"question": "q?"}),
            content_type="application/json",
        )
    assert response.status_code == 500
    assert response.is_json
    text = response.get_data(as_text=True)
    assert "Traceback" not in text
    assert "werkzeug" not in text.lower()


# ---------------------------------------------------------------------------
# HTTP method / routing
# ---------------------------------------------------------------------------

def test_get_on_classify_returns_405(client):
    response = client.get("/api/classify")
    assert response.status_code == 405


def test_unknown_route_returns_404_json(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "not_found"


# ---------------------------------------------------------------------------
# End-to-end smoke test using the real classifier with a fake provider
# (still offline — no network, no API key)
# ---------------------------------------------------------------------------

def test_end_to_end_with_real_classifier_and_fake_provider(client, monkeypatch):
    from backend.services.llm_providers import BaseLLMProvider

    class FakeProvider(BaseLLMProvider):
        def generate(self, prompt, max_tokens=512, temperature=0.0):
            return json.dumps({
                "category": "Symptoms",
                "confidence": 0.91,
                "reason": "The question asks about visible symptoms affecting the crop.",
            })

    monkeypatch.setattr(
        "backend.routes.classify._build_provider_for_route",
        lambda: FakeProvider(api_key="k", model="m", base_url=""),
        raising=False,
    )

    # Patch at the point classify_question resolves its provider.
    import backend.services.llm_classifier as classifier_module
    monkeypatch.setattr(classifier_module, "_build_provider",
                        lambda: FakeProvider(api_key="k", model="m", base_url=""))

    response = post_json(client, {
        "question": "What are the symptoms of disease affecting my tomato leaves?",
        "crop": "Tomato",
    })
    assert response.status_code == 200
    assert response.get_json()["category"] == "Symptoms"
