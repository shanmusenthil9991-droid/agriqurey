"""
Evaluation reporting endpoint.

GET /api/evaluation
    Reads the artifacts already written to disk by the evaluation
    scripts (backend/evaluation/evaluate_classifier.py and, optionally,
    backend/evaluation/compare_methods.py) and returns them as JSON for
    the admin dashboard (frontend/admin.html + static/js/dashboard.js).

This route does NOT run the classifier, does NOT compute metrics, and
does NOT invent any number. It is a thin, read-only adapter over:

    evaluation/results.json            <- backend.evaluation.evaluate_classifier
    evaluation/confusion_matrix.json   <- backend.evaluation.evaluate_classifier
    evaluation/method_comparison.json  <- backend.evaluation.compare_methods (optional)

If evaluation/results.json does not exist yet (the evaluation scripts
have never been run), this endpoint returns HTTP 200 with
`{"available": false, "message": "No evaluation results available yet."}`
rather than an error — an empty evaluation report is an expected state
early in the project, not a server failure. The frontend is expected to
render that message verbatim rather than fabricating placeholder charts.

If evaluation/method_comparison.json is missing, the rest of the
payload (accuracy/precision/recall/F1, category distribution, confusion
matrix) is still returned; only the `method_comparison` section reports
itself unavailable, since the two artifacts are produced by separate,
independently-run scripts.
"""

import json
import os

from flask import Blueprint, jsonify

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
RESULTS_JSON_PATH = os.path.join(EVALUATION_DIR, "results.json")
CONFUSION_MATRIX_JSON_PATH = os.path.join(EVALUATION_DIR, "confusion_matrix.json")
METHOD_COMPARISON_JSON_PATH = os.path.join(EVALUATION_DIR, "method_comparison.json")

NO_RESULTS_MESSAGE = (
    "No evaluation results available yet. Run "
    "`python -m backend.evaluation.evaluate_classifier` to generate "
    "evaluation/results.json, then refresh this dashboard."
)

evaluation_bp = Blueprint("evaluation", __name__)


def _read_json(path: str):
    """Return the parsed JSON at path, or None if it doesn't exist / can't be read."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _category_distribution(results: dict) -> dict:
    """
    Extract the ground-truth category distribution (support counts) from
    the classification report already computed by evaluate_classifier.py.

    This is the actual number of TEST-SET questions per category — not
    predicted counts — taken from scikit-learn's classification_report,
    which evaluate_classifier.py saves under "classification_report".
    """
    categories = results.get("categories_evaluated", [])
    report = results.get("classification_report", {})
    distribution = {}
    for category in categories:
        category_report = report.get(category, {})
        support = category_report.get("support")
        distribution[category] = int(support) if support is not None else None
    return distribution


def _build_method_comparison_payload() -> dict:
    """Build the zero-shot vs. few-shot section, or report it unavailable."""
    comparison = _read_json(METHOD_COMPARISON_JSON_PATH)
    if comparison is None:
        return {
            "available": False,
            "message": (
                "No zero-shot vs. few-shot comparison available yet. Run "
                "`python -m backend.evaluation.compare_methods` to generate "
                "evaluation/method_comparison.json."
            ),
        }
    return {
        "available": True,
        "baseline_zero_shot": comparison.get("baseline_zero_shot"),
        "proposed_few_shot": comparison.get("proposed_few_shot"),
        "delta_few_shot_minus_zero_shot": comparison.get("delta_few_shot_minus_zero_shot"),
        "conclusion": comparison.get("conclusion"),
        "generated_at_utc": comparison.get("generated_at_utc"),
    }


@evaluation_bp.route("/api/evaluation", methods=["GET"])
def get_evaluation():
    """
    Return the most recently generated evaluation report.

    200 {"available": false, "message": ...}  — scripts have never been run.
    200 {"available": true, ...}               — real, on-disk evaluation data.
    """
    results = _read_json(RESULTS_JSON_PATH)
    if results is None:
        return jsonify({"available": False, "message": NO_RESULTS_MESSAGE}), 200

    confusion = _read_json(CONFUSION_MATRIX_JSON_PATH)

    payload = {
        "available": True,
        "source": "evaluation/results.json",
        "generated_at_utc": results.get("generated_at_utc"),
        "dataset": results.get("dataset", {}),
        "total_test_questions": results.get("dataset", {}).get("total_rows"),
        "categories": results.get("categories_evaluated", []),
        "metrics": {
            "accuracy": results.get("metrics", {}).get("accuracy"),
            "precision_macro": results.get("metrics", {}).get("precision_macro"),
            "recall_macro": results.get("metrics", {}).get("recall_macro"),
            "f1_macro": results.get("metrics", {}).get("f1_macro"),
        },
        "category_distribution": _category_distribution(results),
        "confusion_matrix": (
            {"labels": confusion.get("labels", []), "matrix": confusion.get("matrix", [])}
            if confusion is not None
            else {"labels": [], "matrix": [], "note": "confusion_matrix.json not found."}
        ),
        "diagnostics": results.get("diagnostics", {}),
        "method_comparison": _build_method_comparison_payload(),
    }
    return jsonify(payload), 200
