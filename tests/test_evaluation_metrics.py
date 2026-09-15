"""
Tests for the evaluation calculations.

These verify that every reported number is genuinely COMPUTED from
predictions rather than fabricated, that system failures are excluded
from the scored metrics, and that the offline stub provider can never
produce evaluation numbers.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.evaluation.evaluate_classifier import (  # noqa: E402
    PRIMARY_CATEGORIES,
    EvaluationConfigError,
    _ensure_real_provider_configured,
    compute_metrics,
    split_records,
)
from backend.evaluation.error_analysis import (  # noqa: E402
    build_summary,
    count_directional_confusions,
    count_undirected_pair,
    identify_errors,
    split_rows,
)


def _record(true_category, predicted_category, status="ok"):
    return {
        "id": 1,
        "question": "q",
        "crop": "Rice",
        "true_category": true_category,
        "predicted_category": predicted_category,
        "confidence": 0.9,
        "reason": "r",
        "status": status,
        "error": None,
        "error_code": None,
    }


# ---------------------------------------------------------------------------
# split_records
# ---------------------------------------------------------------------------

def test_split_records_separates_system_failures():
    records = [
        _record("Symptoms", "Symptoms"),
        _record("Prevention", "Prevention"),
        _record("Management", "Unable to Classify", status="error"),
    ]
    scored, failed = split_records(records)
    assert len(scored) == 2
    assert len(failed) == 1


def test_system_failures_do_not_affect_accuracy():
    """A provider outage must not be scored as a classification mistake."""
    correct = [_record("Symptoms", "Symptoms") for _ in range(4)]
    scored_only, _ = split_records(correct)
    baseline = compute_metrics(scored_only)["accuracy"]

    with_failures = correct + [
        _record("Prevention", "Unable to Classify", status="error") for _ in range(5)
    ]
    scored, failed = split_records(with_failures)
    assert len(failed) == 5
    assert compute_metrics(scored)["accuracy"] == baseline == 1.0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_accuracy_matches_hand_computed_value():
    records = (
        [_record("Symptoms", "Symptoms")] * 3
        + [_record("Symptoms", "General Information")]
        + [_record("Prevention", "Prevention")] * 2
        + [_record("Prevention", "Management")]
        + [_record("Management", "Management")]
        + [_record("General Information", "General Information")]
    )
    metrics = compute_metrics(records)
    # 7 correct out of 9.
    assert metrics["accuracy"] == pytest.approx(7 / 9)
    assert metrics["n_scored"] == 9


def test_perfect_predictions_give_perfect_scores():
    records = [_record(c, c) for c in PRIMARY_CATEGORIES]
    metrics = compute_metrics(records)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision_macro"] == pytest.approx(1.0)
    assert metrics["recall_macro"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_all_wrong_predictions_give_zero_accuracy():
    records = [
        _record("Symptoms", "Prevention"),
        _record("Prevention", "Symptoms"),
    ]
    assert compute_metrics(records)["accuracy"] == pytest.approx(0.0)


def test_confusion_matrix_shape_and_placement():
    records = [
        _record("Symptoms", "Symptoms"),
        _record("Symptoms", "General Information"),
    ]
    matrix = compute_metrics(records)["confusion_matrix"]
    assert matrix.shape == (4, 4)

    symptoms = PRIMARY_CATEGORIES.index("Symptoms")
    general = PRIMARY_CATEGORIES.index("General Information")
    # matrix[true][predicted]
    assert matrix[symptoms][symptoms] == 1
    assert matrix[symptoms][general] == 1


def test_confusion_matrix_row_sums_equal_true_label_counts():
    records = (
        [_record("Symptoms", "Symptoms")] * 2
        + [_record("Symptoms", "Prevention")]
        + [_record("Management", "Management")]
    )
    matrix = compute_metrics(records)["confusion_matrix"]
    symptoms = PRIMARY_CATEGORIES.index("Symptoms")
    assert matrix[symptoms].sum() == 3


def test_fallback_prediction_counted_as_diagnostic_not_a_fifth_class():
    """
    'Unable to Classify' is never a ground-truth label. Predicting it on a
    real 4-class question must hurt recall and be surfaced as a
    diagnostic, without expanding the label space.
    """
    records = [
        _record("Symptoms", "Symptoms"),
        _record("Symptoms", "Unable to Classify"),
    ]
    metrics = compute_metrics(records)
    assert metrics["unable_to_classify_count"] == 1
    assert metrics["confusion_matrix"].shape == (4, 4)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_metrics_are_bounded_between_zero_and_one():
    records = [
        _record("Symptoms", "Symptoms"),
        _record("Prevention", "Management"),
        _record("Management", "Management"),
    ]
    metrics = compute_metrics(records)
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0.0 <= metrics[key] <= 1.0


# ---------------------------------------------------------------------------
# Guardrail: the offline stub can never produce evaluation numbers
# ---------------------------------------------------------------------------

def test_echo_provider_is_refused_for_evaluation(monkeypatch):
    from backend.config.settings import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "echo", raising=False)
    with pytest.raises(EvaluationConfigError):
        _ensure_real_provider_configured()


def test_missing_provider_is_refused_for_evaluation(monkeypatch):
    from backend.config.settings import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "", raising=False)
    with pytest.raises(EvaluationConfigError):
        _ensure_real_provider_configured()


def test_real_provider_with_credentials_is_accepted(monkeypatch):
    from backend.config.settings import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setattr(config, "LLM_MODEL_NAME", "some-model", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-a-real-looking-key")
    _ensure_real_provider_configured()  # must not raise


def test_missing_api_key_is_refused_before_any_api_call(monkeypatch):
    """
    Fail fast on missing credentials. Without this check the run fires one
    doomed request per test question and then aborts with a generic
    "every question failed" message that hides the real cause.
    """
    from backend.config.settings import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setattr(config, "LLM_MODEL_NAME", "some-model", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")

    with pytest.raises(EvaluationConfigError, match="LLM_API_KEY"):
        _ensure_real_provider_configured()


def test_missing_model_name_is_refused(monkeypatch):
    from backend.config.settings import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic", raising=False)
    monkeypatch.setattr(config, "LLM_MODEL_NAME", "", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "sk-a-real-looking-key")

    with pytest.raises(EvaluationConfigError, match="LLM_MODEL_NAME"):
        _ensure_real_provider_configured()


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def test_identify_errors_returns_only_mismatches():
    rows = [
        {"true_category": "Symptoms", "predicted_category": "Symptoms"},
        {"true_category": "Symptoms", "predicted_category": "Prevention"},
    ]
    errors = identify_errors(rows)
    assert len(errors) == 1
    assert errors[0]["predicted_category"] == "Prevention"


def test_directional_confusion_counts():
    rows = [
        {"true_category": "Symptoms", "predicted_category": "General Information"},
        {"true_category": "Symptoms", "predicted_category": "General Information"},
        {"true_category": "Prevention", "predicted_category": "Management"},
    ]
    counts = count_directional_confusions(rows)
    assert counts["Symptoms -> General Information"] == 2
    assert counts["Prevention -> Management"] == 1


def test_undirected_pair_counts_both_directions():
    rows = [
        {"true_category": "Symptoms", "predicted_category": "General Information"},
        {"true_category": "General Information", "predicted_category": "Symptoms"},
    ]
    assert count_undirected_pair(rows, "Symptoms", "General Information") == 2


def test_error_summary_counts_reconcile():
    rows = [
        {"true_category": "Symptoms", "predicted_category": "Symptoms", "status": "ok"},
        {"true_category": "Symptoms", "predicted_category": "Prevention", "status": "ok"},
        {"true_category": "Management", "predicted_category": "Management", "status": "error"},
    ]
    scored, failed = split_rows(rows)
    errors = identify_errors(scored)
    summary = build_summary(rows, scored, failed, errors)

    assert summary["total_test_questions"] == 3
    assert summary["scored_questions"] == 2
    assert summary["excluded_system_failures"] == 1
    assert summary["correct_predictions"] + summary["incorrect_predictions"] == 2
    assert summary["accuracy_on_scored"] == pytest.approx(0.5)
