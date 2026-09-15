"""
Tests for backend/services/llm_classifier.py and llm_providers.py

Every test runs fully OFFLINE. Fake providers are injected, so no test
requires an API key, makes a network call, or costs money.

Coverage:
- JSON extraction from messy real-world model output
- Category / confidence / reason validation
- Controlled error handling (never raises, never crashes Flask)
- Retry behaviour for transient vs permanent failures
- Provider registry and isolation
- Guarantee that API keys are never leaked into results
"""

import json

import pytest

from backend.services.llm_classifier import (
    ERROR_INVALID_CATEGORY,
    ERROR_INVALID_CONFIDENCE,
    ERROR_INVALID_INPUT,
    ERROR_INVALID_JSON,
    ERROR_PROVIDER,
    FINAL_CATEGORIES,
    ClassificationError,
    classify_question,
    classify_questions,
    extract_json_object,
    validate_category,
    validate_confidence,
    validate_reason,
    validate_response,
)
from backend.services.llm_providers import (
    AnthropicProvider,
    BaseLLMProvider,
    EchoProvider,
    LLMProviderError,
    OpenAICompatibleProvider,
    get_provider,
)
from backend.services.prompt_builder import ALLOWED_CATEGORIES, FALLBACK_CATEGORY


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------

class FakeProvider(BaseLLMProvider):
    """Returns a canned string. Records the prompt it was given."""

    def __init__(self, response_text):
        super().__init__(api_key="fake-key", model="fake-model", base_url="")
        self.response_text = response_text
        self.prompts = []

    def generate(self, prompt, max_tokens=512, temperature=0.0):
        self.prompts.append(prompt)
        return self.response_text


class FailingProvider(BaseLLMProvider):
    """Always raises. Counts calls so retry behaviour can be asserted."""

    def __init__(self, retryable=False, succeed_on_attempt=None):
        super().__init__(api_key="fake-key", model="fake-model", base_url="")
        self.retryable = retryable
        self.succeed_on_attempt = succeed_on_attempt
        self.calls = 0

    def generate(self, prompt, max_tokens=512, temperature=0.0):
        self.calls += 1
        if self.succeed_on_attempt and self.calls >= self.succeed_on_attempt:
            return json.dumps({
                "category": "Symptoms", "confidence": 0.9, "reason": "ok"
            })
        raise LLMProviderError("boom", retryable=self.retryable)


def valid_response(category="Symptoms", confidence=0.91, reason="Visible signs."):
    return json.dumps({
        "category": category, "confidence": confidence, "reason": reason
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_classify_question_returns_expected_shape():
    provider = FakeProvider(valid_response())
    result = classify_question("What are the symptoms of rice blast?", "Rice",
                               provider=provider, examples=[])

    assert result["category"] == "Symptoms"
    assert result["confidence"] == 0.91
    assert result["reason"] == "Visible signs."
    assert result["status"] == "ok"
    assert result["error"] is None


def test_core_keys_always_present_on_success_and_failure():
    ok = classify_question("q?", "Rice", provider=FakeProvider(valid_response()), examples=[])
    bad = classify_question("q?", "Rice", provider=FakeProvider("not json"), examples=[])
    for result in (ok, bad):
        for key in ("category", "confidence", "reason", "status", "error", "error_code"):
            assert key in result


def test_question_and_crop_reach_the_prompt():
    provider = FakeProvider(valid_response())
    classify_question("Why are my leaves yellow?", "Wheat", provider=provider, examples=[])
    prompt = provider.prompts[0]
    assert "Why are my leaves yellow?" in prompt
    assert "Wheat" in prompt


def test_missing_crop_defaults_to_unknown():
    provider = FakeProvider(valid_response())
    classify_question("Why are my leaves yellow?", provider=provider, examples=[])
    assert "Crop: Unknown" in provider.prompts[0]


@pytest.mark.parametrize("category", list(ALLOWED_CATEGORIES) + [FALLBACK_CATEGORY])
def test_all_final_categories_accepted(category):
    provider = FakeProvider(valid_response(category=category, confidence=0.5))
    result = classify_question("q?", "Rice", provider=provider, examples=[])
    assert result["category"] == category
    assert result["status"] == "ok"


def test_fallback_category_is_forced_to_zero_confidence():
    """The prompt defines the fallback as zero-confidence; enforce it."""
    provider = FakeProvider(valid_response(category=FALLBACK_CATEGORY, confidence=0.87))
    result = classify_question("What is the weather?", "Rice", provider=provider, examples=[])
    assert result["category"] == FALLBACK_CATEGORY
    assert result["confidence"] == 0.0


def test_genuine_unable_to_classify_is_not_an_error():
    """
    A real 'not about crop disease' judgement must have status 'ok'.
    Conflating it with a system failure would corrupt evaluation.
    """
    provider = FakeProvider(valid_response(category=FALLBACK_CATEGORY, confidence=0.0))
    result = classify_question("Who won the match?", "Rice", provider=provider, examples=[])
    assert result["status"] == "ok"
    assert result["error"] is None


def test_system_failure_is_distinguishable_from_unable_to_classify():
    result = classify_question("q?", "Rice", provider=FakeProvider("garbage"), examples=[])
    assert result["category"] == FALLBACK_CATEGORY
    assert result["status"] == "error"
    assert result["error_code"] == ERROR_INVALID_JSON


# ---------------------------------------------------------------------------
# JSON extraction from messy output
# ---------------------------------------------------------------------------

def test_extract_plain_json():
    assert extract_json_object('{"category": "Symptoms"}')["category"] == "Symptoms"


def test_extract_json_from_markdown_fence():
    raw = '```json\n{"category": "Prevention", "confidence": 0.8, "reason": "r"}\n```'
    assert extract_json_object(raw)["category"] == "Prevention"


def test_extract_json_from_bare_fence():
    raw = '```\n{"category": "Management", "confidence": 0.8, "reason": "r"}\n```'
    assert extract_json_object(raw)["category"] == "Management"


def test_extract_json_with_preamble_and_trailing_text():
    raw = 'Sure! Here is the result:\n{"category": "Symptoms", "confidence": 0.9, "reason": "r"}\nHope that helps.'
    assert extract_json_object(raw)["category"] == "Symptoms"


def test_extract_json_with_braces_inside_reason_string():
    """Braces inside a string must not break balanced-brace scanning."""
    raw = 'Result: {"category": "Symptoms", "confidence": 0.9, "reason": "uses {braces} inside"}'
    assert extract_json_object(raw)["reason"] == "uses {braces} inside"


def test_extract_json_with_escaped_quotes_in_reason():
    raw = '{"category": "Symptoms", "confidence": 0.9, "reason": "the \\"leaf\\" is spotted"}'
    assert 'leaf' in extract_json_object(raw)["reason"]


@pytest.mark.parametrize("raw", ["", "   ", None, "no json here", "{broken", "[1,2,3]"])
def test_extract_json_raises_on_unparseable_output(raw):
    with pytest.raises(ClassificationError):
        extract_json_object(raw)


# ---------------------------------------------------------------------------
# Category validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Symptoms", "Symptoms"),
    ("symptoms", "Symptoms"),
    ("SYMPTOMS", "Symptoms"),
    ("  Prevention  ", "Prevention"),
    ("general information", "General Information"),
    ("General_Information", "General Information"),
    ("unable to classify", FALLBACK_CATEGORY),
])
def test_category_normalisation(raw, expected):
    assert validate_category(raw) == expected


@pytest.mark.parametrize("raw", ["Diagnosis", "Blight", "", None, 123, [], "maybe symptoms"])
def test_invalid_categories_rejected(raw):
    with pytest.raises(ClassificationError):
        validate_category(raw)


def test_invalid_category_returns_controlled_error():
    provider = FakeProvider(valid_response(category="Diagnosis"))
    result = classify_question("q?", "Rice", provider=provider, examples=[])
    assert result["status"] == "error"
    assert result["error_code"] == ERROR_INVALID_CATEGORY
    assert result["category"] in FINAL_CATEGORIES


def test_model_must_not_be_allowed_to_invent_a_disease_label():
    """A diagnosis-shaped answer must be rejected, not passed through."""
    provider = FakeProvider(valid_response(category="Rice Blast"))
    result = classify_question("What's wrong with my rice?", "Rice",
                               provider=provider, examples=[])
    assert result["status"] == "error"
    assert result["category"] == FALLBACK_CATEGORY


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (0.91, 0.91), (0, 0.0), (1, 1.0), ("0.75", 0.75),
    (91, 0.91),        # percentage form
    (1.5, 1.0),        # overshoot of a fraction -> clamped, not inverted
    (2, 0.02),         # plausible percentage
    (-0.5, 0.0),       # clamped
    (250, 1.0),        # clamped
])
def test_confidence_coercion(raw, expected):
    assert validate_confidence(raw) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("raw", [None, "high", "", [], {}, True, False, float("nan"), float("inf")])
def test_invalid_confidence_rejected(raw):
    with pytest.raises(ClassificationError):
        validate_confidence(raw)


def test_confidence_always_within_unit_range():
    for raw in (-99, 0, 0.5, 1, 100, 1000):
        assert 0.0 <= validate_confidence(raw) <= 1.0


def test_invalid_confidence_returns_controlled_error():
    provider = FakeProvider(json.dumps({
        "category": "Symptoms", "confidence": "very high", "reason": "r"
    }))
    result = classify_question("q?", "Rice", provider=provider, examples=[])
    assert result["status"] == "error"
    assert result["error_code"] == ERROR_INVALID_CONFIDENCE


# ---------------------------------------------------------------------------
# Reason validation
# ---------------------------------------------------------------------------

def test_missing_reason_gets_placeholder():
    assert validate_reason(None)
    assert validate_reason("")


def test_reason_is_whitespace_normalised_and_truncated():
    assert validate_reason("  a   b  ") == "a b"
    assert len(validate_reason("x" * 900)) <= 300


def test_missing_reason_does_not_fail_classification():
    provider = FakeProvider(json.dumps({"category": "Symptoms", "confidence": 0.8}))
    result = classify_question("q?", "Rice", provider=provider, examples=[])
    assert result["status"] == "ok"
    assert result["reason"]


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------

def test_validate_response_rejects_non_dict():
    with pytest.raises(ClassificationError):
        validate_response(["not", "a", "dict"])


def test_validate_response_ignores_extra_fields():
    result = validate_response({
        "category": "Symptoms", "confidence": 0.9, "reason": "r",
        "disease": "Rice Blast", "treatment": "fungicide",
    })
    assert result["category"] == "Symptoms"
    assert "disease" not in result
    assert "treatment" not in result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, "", "   ", 123, [], {}])
def test_invalid_question_input_returns_controlled_error(bad):
    result = classify_question(bad, "Rice", provider=FakeProvider(valid_response()), examples=[])
    assert result["status"] == "error"
    assert result["error_code"] == ERROR_INVALID_INPUT


def test_overlong_question_rejected():
    result = classify_question("x" * 5000, "Rice",
                               provider=FakeProvider(valid_response()), examples=[])
    assert result["status"] == "error"
    assert result["error_code"] == ERROR_INVALID_INPUT


def test_classify_question_never_raises():
    """The Flask route must be safe without try/except."""
    for provider in (FakeProvider("garbage"), FakeProvider(""), FailingProvider(),
                     FakeProvider(valid_response(category="Nope"))):
        result = classify_question("q?", "Rice", provider=provider, examples=[])
        assert isinstance(result, dict)
        assert result["category"] in FINAL_CATEGORIES


# ---------------------------------------------------------------------------
# Provider errors and retries
# ---------------------------------------------------------------------------

def test_provider_error_returns_controlled_error():
    result = classify_question("q?", "Rice", provider=FailingProvider(), examples=[])
    assert result["status"] == "error"
    assert result["error_code"] == ERROR_PROVIDER


def test_non_retryable_error_is_not_retried():
    provider = FailingProvider(retryable=False)
    classify_question("q?", "Rice", provider=provider, examples=[], max_retries=3)
    assert provider.calls == 1


def test_retryable_error_is_retried_then_succeeds():
    provider = FailingProvider(retryable=True, succeed_on_attempt=2)
    result = classify_question("q?", "Rice", provider=provider, examples=[], max_retries=2)
    assert provider.calls == 2
    assert result["status"] == "ok"


def test_retries_give_up_and_return_error():
    provider = FailingProvider(retryable=True)
    result = classify_question("q?", "Rice", provider=provider, examples=[], max_retries=1)
    assert provider.calls == 2
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------

def test_api_key_never_appears_in_result():
    provider = FakeProvider(valid_response())
    provider.api_key = "sk-super-secret-value"
    result = classify_question("q?", "Rice", provider=provider, examples=[])
    assert "sk-super-secret-value" not in json.dumps(result)


def test_api_key_never_appears_in_error_result():
    provider = FailingProvider()
    provider.api_key = "sk-super-secret-value"
    result = classify_question("q?", "Rice", provider=provider, examples=[])
    assert "sk-super-secret-value" not in json.dumps(result)


def test_no_api_key_is_hardcoded_in_source():
    """Guardrail: credentials must come from the environment only."""
    import backend.services.llm_classifier as classifier_module
    import backend.services.llm_providers as providers_module

    for module in (classifier_module, providers_module):
        with open(module.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "sk-ant-" not in source
        assert "sk-proj-" not in source


# ---------------------------------------------------------------------------
# Provider registry / isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("anthropic", AnthropicProvider),
    ("openai", OpenAICompatibleProvider),
    ("echo", EchoProvider),
    ("ANTHROPIC", AnthropicProvider),
])
def test_provider_registry_resolves_names(name, expected):
    provider = get_provider(name, api_key="k", model="m")
    assert isinstance(provider, expected)


def test_unknown_provider_raises():
    with pytest.raises(LLMProviderError):
        get_provider("not-a-provider", api_key="k", model="m")


def test_echo_provider_makes_no_network_call():
    result = classify_question("q?", "Rice",
                               provider=get_provider("echo", api_key="", model=""),
                               examples=[])
    assert result["status"] == "ok"
    assert result["category"] == FALLBACK_CATEGORY


def test_base_provider_generate_is_abstract():
    with pytest.raises(NotImplementedError):
        BaseLLMProvider("k", "m", "").generate("prompt")


def test_classifier_module_contains_no_vendor_http_details():
    """
    Architectural guard: vendor specifics belong in llm_providers.py.
    If these strings appear in the classifier, the isolation has leaked.
    """
    import backend.services.llm_classifier as classifier_module
    with open(classifier_module.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    for token in ("x-api-key", "anthropic-version", "api.anthropic.com",
                  "Bearer ", "chat/completions", "requests.post"):
        assert token not in source, f"Vendor detail '{token}' leaked into the classifier"


# ---------------------------------------------------------------------------
# Batch interface
# ---------------------------------------------------------------------------

def test_classify_questions_returns_one_result_per_input(monkeypatch):
    monkeypatch.setattr(
        "backend.services.llm_classifier._build_provider",
        lambda: FakeProvider(valid_response()),
    )
    items = [
        {"question": "What are the symptoms?", "crop": "Rice"},
        {"question": "How do I prevent it?", "crop": "Wheat"},
        {"question": "", "crop": "Maize"},
    ]
    results = classify_questions(items)
    assert len(results) == len(items)
    assert results[2]["status"] == "error"
