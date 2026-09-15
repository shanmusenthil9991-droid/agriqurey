"""
LLM classification service.

Pipeline:
    question + crop
        -> build_classification_prompt()      (prompt_builder.py)
        -> provider.generate()                (llm_providers.py)
        -> extract + parse JSON               (this module)
        -> validate category and confidence   (this module)
        -> clean Python dict                  (returned to caller)

Design notes:

* Credentials are read from environment variables via backend.config.settings.
  No key is ever hardcoded, logged, or returned to the caller.

* Provider-specific details live entirely in llm_providers.py. This module
  only ever asks a provider for text, so swapping vendor/model requires no
  change here.

* Nothing in this module raises on a bad LLM response. Every failure path
  returns a structured error dict, so a Flask route can render it directly
  without try/except and the application cannot be crashed by malformed
  model output.

IMPORTANT — on the `confidence` field:
    Confidence is the model's SELF-REPORTED estimate. It is a soft signal
    for UI hinting and triage only. It is NOT a measured accuracy, NOT a
    calibrated probability, and must never be reported as model accuracy.
    Real accuracy comes only from evaluating predictions against the
    held-out test split.
"""

import json
import re
import time

from backend.config.settings import config
from backend.services.prompt_builder import (
    ALLOWED_CATEGORIES,
    FALLBACK_CATEGORY,
    build_classification_prompt,
    select_few_shot_examples,
)
from backend.services.llm_providers import LLMProviderError, get_provider

# The five labels the service may return. The four real categories plus
# the fallback the prompt permits for questions unrelated to crop disease.
FINAL_CATEGORIES = list(ALLOWED_CATEGORIES) + [FALLBACK_CATEGORY]

# Case-insensitive lookup so "symptoms" or "SYMPTOMS" normalise cleanly.
_CATEGORY_LOOKUP = {c.lower(): c for c in FINAL_CATEGORIES}

# Common near-miss labels seen from LLMs, mapped onto the canonical set.
# Kept deliberately small and conservative: only unambiguous synonyms.
_CATEGORY_ALIASES = {
    "symptom": "Symptoms",
    "signs": "Symptoms",
    "prevent": "Prevention",
    "preventive": "Prevention",
    "preventative": "Prevention",
    "manage": "Management",
    "control": "Management",
    "treatment": "Management",
    "general": "General Information",
    "general info": "General Information",
    "general_information": "General Information",
    "generalinformation": "General Information",
    "information": "General Information",
    "unable": FALLBACK_CATEGORY,
    "unable_to_classify": FALLBACK_CATEGORY,
    "unabletoclassify": FALLBACK_CATEGORY,
    "unknown": FALLBACK_CATEGORY,
    "other": FALLBACK_CATEGORY,
    "none": FALLBACK_CATEGORY,
}

# Machine-readable error codes for controlled failures.
ERROR_CONFIG = "configuration_error"
ERROR_PROVIDER = "provider_error"
ERROR_INVALID_JSON = "invalid_json"
ERROR_INVALID_CATEGORY = "invalid_category"
ERROR_INVALID_CONFIDENCE = "invalid_confidence"
ERROR_INVALID_INPUT = "invalid_input"

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0   # deterministic: classification is not creative work
MAX_QUESTION_LENGTH = 2000


class ClassificationError(Exception):
    """
    Internal signal used between the parse/validate helpers.

    Never escapes classify_question() — it is always converted into a
    structured error dict before returning.
    """

    def __init__(self, message: str, error_code: str):
        super().__init__(message)
        self.error_code = error_code


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------

def _success_result(category: str, confidence: float, reason: str, **meta) -> dict:
    """Build a successful classification result."""
    result = {
        "category": category,
        "confidence": confidence,
        "reason": reason,
        "status": "ok",
        "error": None,
        "error_code": None,
    }
    result.update(meta)
    return result


def _error_result(message: str, error_code: str, **meta) -> dict:
    """
    Build a controlled error result.

    The three core keys are always present so callers can treat every
    result uniformly. `category` is set to the fallback label, but
    `status` distinguishes a SYSTEM FAILURE from a genuine "this question
    isn't about crop disease" judgement by the model. Collapsing those
    two cases would silently turn outages into confident-looking
    classifications and corrupt any evaluation run.
    """
    result = {
        "category": FALLBACK_CATEGORY,
        "confidence": 0.0,
        "reason": "The question could not be classified.",
        "status": "error",
        "error": message,
        "error_code": error_code,
    }
    result.update(meta)
    return result


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_json_object(raw_text: str) -> dict:
    """
    Pull the JSON object out of the model's raw text output.

    Handles the usual real-world deviations: markdown code fences, a
    leading "Here is the classification:" preamble, and trailing commentary.
    The prompt asks for bare JSON, but models drift and the service must
    not fall over when they do.

    Raises:
        ClassificationError: If no parseable JSON object is found.
    """
    if raw_text is None or not str(raw_text).strip():
        raise ClassificationError("LLM returned an empty response.", ERROR_INVALID_JSON)

    text = str(raw_text).strip()

    # Strip ```json ... ``` / ``` ... ``` fences.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    # Fast path: the whole string is the object.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: scan for the first balanced {...} block, respecting strings
    # and escapes so braces inside the "reason" text don't break matching.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    raise ClassificationError(
        "LLM response did not contain a valid JSON object.", ERROR_INVALID_JSON
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_category(value) -> str:
    """
    Normalise and validate the category returned by the model.

    Accepts the canonical labels case-insensitively plus a short list of
    unambiguous synonyms. Anything else is rejected rather than guessed —
    silently coercing an unrecognised label would fabricate a result.

    Raises:
        ClassificationError: If the value is missing or unrecognised.
    """
    if value is None or not isinstance(value, str) or not value.strip():
        raise ClassificationError(
            "LLM response is missing the 'category' field.", ERROR_INVALID_CATEGORY
        )

    cleaned = value.strip()
    if cleaned.lower() in _CATEGORY_LOOKUP:
        return _CATEGORY_LOOKUP[cleaned.lower()]

    normalised = re.sub(r"[\s_-]+", " ", cleaned.lower()).strip()
    if normalised in _CATEGORY_LOOKUP:
        return _CATEGORY_LOOKUP[normalised]
    if normalised in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[normalised]
    if normalised.replace(" ", "") in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[normalised.replace(" ", "")]

    raise ClassificationError(
        f"LLM returned an invalid category '{cleaned}'. "
        f"Allowed categories: {FINAL_CATEGORIES}",
        ERROR_INVALID_CATEGORY,
    )


def validate_confidence(value) -> float:
    """
    Validate the self-reported confidence and coerce it to a float in [0, 1].

    Accepts ints/floats and numeric strings. Values slightly outside the
    range (e.g. 1.2, or a model reporting 91 for "91%") are clamped rather
    than rejected, since the number is only ever a soft signal — but a
    non-numeric value is a genuine schema violation and is rejected.

    Raises:
        ClassificationError: If the value is missing or non-numeric.
    """
    if value is None:
        raise ClassificationError(
            "LLM response is missing the 'confidence' field.", ERROR_INVALID_CONFIDENCE
        )

    if isinstance(value, bool):  # bool is a subclass of int — reject explicitly
        raise ClassificationError(
            "LLM returned a boolean confidence value.", ERROR_INVALID_CONFIDENCE
        )

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ClassificationError(
            f"LLM returned a non-numeric confidence value: {value!r}",
            ERROR_INVALID_CONFIDENCE,
        )

    if confidence != confidence or confidence in (float("inf"), float("-inf")):
        raise ClassificationError(
            "LLM returned a non-finite confidence value.", ERROR_INVALID_CONFIDENCE
        )

    # A model may report a percentage (e.g. 91) instead of a fraction.
    # Only values >= 2 are treated that way: a "confidence" of 1.5 is far
    # more likely to be a fraction overshooting 1.0 than a genuine 1.5%,
    # and dividing it by 100 would turn a high-confidence signal into a
    # near-zero one. Clamping preserves the intended meaning.
    if confidence >= 2.0:
        confidence = confidence / 100.0

    return max(0.0, min(1.0, confidence))


def validate_reason(value) -> str:
    """Validate the reason string, falling back to a neutral placeholder."""
    if value is None or not isinstance(value, str) or not value.strip():
        return "No reason was provided by the model."
    reason = " ".join(value.strip().split())
    return reason[:300]


def validate_response(payload: dict) -> dict:
    """
    Validate a parsed LLM payload and return the clean result dict.

    Raises:
        ClassificationError: If any field fails validation.
    """
    if not isinstance(payload, dict):
        raise ClassificationError(
            "LLM response was not a JSON object.", ERROR_INVALID_JSON
        )

    category = validate_category(payload.get("category"))
    confidence = validate_confidence(payload.get("confidence"))
    reason = validate_reason(payload.get("reason"))

    # The prompt defines the fallback label as carrying zero confidence.
    # Enforce it here so the contract holds regardless of model drift.
    if category == FALLBACK_CATEGORY:
        confidence = 0.0

    return _success_result(category, confidence, reason)


# ---------------------------------------------------------------------------
# Configuration / provider wiring
# ---------------------------------------------------------------------------

def _build_provider():
    """
    Construct the configured provider from environment settings.

    Raises:
        ClassificationError: If required configuration is missing.
    """
    provider_name = getattr(config, "LLM_PROVIDER", "") or ""
    model = getattr(config, "LLM_MODEL_NAME", "") or ""
    api_key = getattr(config, "LLM_API_KEY", "") or ""
    base_url = getattr(config, "LLM_API_BASE_URL", "") or ""
    timeout = int(getattr(config, "LLM_REQUEST_TIMEOUT", 30) or 30)

    needs_key = provider_name.strip().lower() != "echo"

    if needs_key and (not api_key.strip() or api_key.strip() == "your_api_key_here"):
        raise ClassificationError(
            "LLM_API_KEY is not set. Copy .env.example to .env and set a real "
            "API key (never hardcode it in source).",
            ERROR_CONFIG,
        )
    if not model.strip() and needs_key:
        raise ClassificationError(
            "LLM_MODEL_NAME is not set. Configure it in your .env file.",
            ERROR_CONFIG,
        )

    try:
        return get_provider(
            provider_name=provider_name,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )
    except LLMProviderError as exc:
        raise ClassificationError(str(exc), ERROR_CONFIG) from exc


_cached_examples = None


def _get_examples():
    """
    Select few-shot examples once and reuse them.

    Re-reading and re-sampling development.csv on every request would add
    disk I/O per call and — more importantly — the seeded selection is
    identical each time anyway, so caching also keeps prompts stable
    across requests within a process.
    """
    global _cached_examples
    if _cached_examples is None:
        try:
            _cached_examples = select_few_shot_examples()
        except Exception:
            # Missing/unreadable development.csv degrades to a zero-shot
            # prompt rather than failing the request outright.
            _cached_examples = []
    return _cached_examples


def reset_example_cache():
    """Clear the cached few-shot examples (used by tests)."""
    global _cached_examples
    _cached_examples = None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_question(
    question: str,
    crop: str = None,
    provider=None,
    examples: list = None,
    max_retries: int = None,
) -> dict:
    """
    Classify the information need behind a farmer's question.

    This is the single entry point the Flask route should call. It never
    raises: every failure is converted into a structured error dict.

    Args:
        question: The farmer's question text.
        crop: Optional crop name. Defaults to "Unknown" when not supplied.
        provider: Optional pre-built provider (dependency injection for
            tests and for callers that want to reuse one adapter).
        examples: Optional few-shot examples. Defaults to the cached,
            balanced selection from the development split.
        max_retries: Extra attempts for transient provider failures
            (timeouts, 429, 5xx). Non-transient errors are never retried.
            Defaults to the LLM_MAX_RETRIES environment setting.

    Returns:
        dict with keys:
            category    — one of FINAL_CATEGORIES
            confidence  — float in [0.0, 1.0] (SELF-REPORTED estimate only)
            reason      — short explanation string
            status      — "ok" or "error"
            error       — error message, or None on success
            error_code  — machine-readable code, or None on success

        `status` is what distinguishes a real "Unable to Classify"
        judgement (status "ok") from a system failure (status "error").
        Callers must not treat the two as equivalent.
    """
    # --- input validation -------------------------------------------------
    if question is None or not isinstance(question, str) or not question.strip():
        return _error_result(
            "Question must be a non-empty string.", ERROR_INVALID_INPUT
        )
    if len(question) > MAX_QUESTION_LENGTH:
        return _error_result(
            f"Question exceeds the maximum length of {MAX_QUESTION_LENGTH} characters.",
            ERROR_INVALID_INPUT,
        )

    question = question.strip()
    crop = (crop or "").strip() or "Unknown"

    # --- build prompt -----------------------------------------------------
    try:
        prompt = build_classification_prompt(
            question=question,
            crop=crop,
            examples=examples if examples is not None else _get_examples(),
        )
    except Exception as exc:
        return _error_result(f"Failed to build the classification prompt: {exc}",
                             ERROR_CONFIG)

    # --- resolve provider -------------------------------------------------
    if provider is None:
        try:
            provider = _build_provider()
        except ClassificationError as exc:
            return _error_result(str(exc), exc.error_code)

    # --- call the LLM (with retry on transient failures only) -------------
    raw_text = None
    last_error = None
    if max_retries is None:
        # LLM_MAX_RETRIES was loaded into config but never actually read,
        # so tuning it had no effect. Honour it here as the default.
        try:
            max_retries = int(getattr(config, "LLM_MAX_RETRIES", 1) or 0)
        except (TypeError, ValueError):
            max_retries = 1
    attempts = max(1, max_retries + 1)

    for attempt in range(attempts):
        try:
            raw_text = provider.generate(
                prompt,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
            )
            break
        except LLMProviderError as exc:
            last_error = exc
            if not exc.retryable or attempt == attempts - 1:
                return _error_result(str(exc), ERROR_PROVIDER)
            time.sleep(0.5 * (2 ** attempt))  # simple exponential backoff
        except Exception as exc:  # pragma: no cover - defensive
            return _error_result(f"Unexpected error calling the LLM: {exc}",
                                 ERROR_PROVIDER)

    if raw_text is None:
        return _error_result(
            f"LLM call failed: {last_error}", ERROR_PROVIDER
        )

    # --- parse and validate ----------------------------------------------
    try:
        payload = extract_json_object(raw_text)
        return validate_response(payload)
    except ClassificationError as exc:
        return _error_result(str(exc), exc.error_code)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _error_result(
            f"Unexpected error while parsing the LLM response: {exc}",
            ERROR_INVALID_JSON,
        )


def classify_questions(items: list) -> list:
    """
    Classify a batch of questions, reusing one provider instance.

    Args:
        items: list of dicts with "question" and optional "crop".

    Returns:
        A list of result dicts, one per input, in the same order.
    """
    try:
        provider = _build_provider()
    except ClassificationError as exc:
        return [_error_result(str(exc), exc.error_code) for _ in items]

    examples = _get_examples()
    return [
        classify_question(
            question=item.get("question"),
            crop=item.get("crop"),
            provider=provider,
            examples=examples,
        )
        for item in items
    ]
