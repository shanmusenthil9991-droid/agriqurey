"""
Classification endpoint.

POST /api/classify
    Request:  {"question": "...", "crop": "Tomato"}   (crop optional)
    Response: {"category": ..., "confidence": ..., "reason": ...}

This route is a thin HTTP adapter. All real work — prompt building,
calling the LLM, parsing, validation — happens in
backend.services.llm_classifier.classify_question(), which never raises.
That is what lets this route avoid a blanket try/except: every expected
failure already arrives as a structured dict with a `status` field.

Security/safety requirements enforced here:
  - Request JSON is validated before anything touches the LLM layer.
  - No stack trace, exception object, or internal path is ever put in a
    response body — every response is built from fixed strings or from
    the classifier's own sanitized `error` message.
  - No API key can appear in a response: the classifier already keeps
    keys out of its result dict, and this route never echoes request
    headers or environment values back to the caller.
"""

from flask import Blueprint, jsonify, request

from backend.services.llm_classifier import (
    ERROR_CONFIG,
    ERROR_INVALID_CATEGORY,
    ERROR_INVALID_CONFIDENCE,
    ERROR_INVALID_INPUT,
    ERROR_INVALID_JSON,
    ERROR_PROVIDER,
    classify_question,
)

classify_bp = Blueprint("classify", __name__)

MAX_CROP_LENGTH = 100

# Maps the classifier's internal error_code to the HTTP status that best
# fits it. Anything not listed here (e.g. a future error_code) falls back
# to 502, since the safest default for "the LLM/provider misbehaved" is a
# bad-gateway response rather than pretending the request succeeded.
_ERROR_CODE_TO_STATUS = {
    ERROR_INVALID_INPUT: 400,        # caller's fault — bad request body
    ERROR_INVALID_JSON: 502,         # LLM returned unparseable output
    ERROR_INVALID_CATEGORY: 502,     # LLM returned a schema violation
    ERROR_INVALID_CONFIDENCE: 502,   # LLM returned a schema violation
    ERROR_CONFIG: 503,               # server misconfigured (e.g. no API key)
    ERROR_PROVIDER: 502,             # upstream LLM call failed
}
_DEFAULT_ERROR_STATUS = 502


def _bad_request(message: str, field: str = None):
    """Build a 400 response for a request-shape problem caught before the classifier runs."""
    body = {"error": message, "error_code": "invalid_input"}
    if field:
        body["field"] = field
    return jsonify(body), 400


@classify_bp.route("/api/classify", methods=["POST"])
def classify():
    """
    Classify the information need behind a farmer's question.

    Returns 200 with {category, confidence, reason} on success.
    Returns 400 for a malformed request (missing/blank/invalid fields).
    Returns 502/503 when the classifier itself reports a failure — the
    request was well-formed but the LLM call or its response failed.
    """
    # --- request body must be JSON ----------------------------------------
    if not request.is_json:
        return _bad_request("Request body must be JSON with a 'Content-Type: application/json' header.")

    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return _bad_request("Request body must be a JSON object.")

    # --- validate 'question' -----------------------------------------------
    question = payload.get("question")
    if "question" not in payload:
        return _bad_request("Missing required field: 'question'.", field="question")
    if not isinstance(question, str):
        return _bad_request("'question' must be a string.", field="question")
    if not question.strip():
        return _bad_request("'question' must not be empty.", field="question")

    # --- validate 'crop' (optional) -----------------------------------------
    crop = payload.get("crop")
    if crop is not None:
        if not isinstance(crop, str):
            return _bad_request("'crop' must be a string.", field="crop")
        if not crop.strip():
            return _bad_request("'crop' must not be an empty string. Omit it if unknown.", field="crop")
        if len(crop) > MAX_CROP_LENGTH:
            return _bad_request(
                f"'crop' exceeds the maximum length of {MAX_CROP_LENGTH} characters.",
                field="crop",
            )

    # --- reject unexpected extra fields defensively (not silently ignored) -
    allowed_fields = {"question", "crop"}
    unknown_fields = set(payload.keys()) - allowed_fields
    if unknown_fields:
        return _bad_request(
            f"Unexpected field(s): {sorted(unknown_fields)}. Allowed fields: {sorted(allowed_fields)}."
        )

    # --- run the classifier (never raises) ----------------------------------
    result = classify_question(question=question, crop=crop)

    if result["status"] == "ok":
        return jsonify({
            "category": result["category"],
            "confidence": result["confidence"],
            "reason": result["reason"],
        }), 200

    # Controlled failure: the classifier already sanitized the message —
    # it never contains a stack trace, an API key, or an internal path.
    status_code = _ERROR_CODE_TO_STATUS.get(result["error_code"], _DEFAULT_ERROR_STATUS)
    return jsonify({
        "error": result["error"],
        "error_code": result["error_code"],
    }), status_code


@classify_bp.errorhandler(Exception)
def _handle_unexpected_error(exc):
    """
    Last-resort safety net for this blueprint.

    classify_question() is designed to never raise, so this should not
    normally trigger. It exists so that even a bug in this route, or an
    exception Flask raises before the view runs (e.g. body-parsing
    issues), still returns a clean JSON error instead of a stack trace.
    """
    return jsonify({
        "error": "An unexpected server error occurred while classifying the question.",
        "error_code": "internal_error",
    }), 500
