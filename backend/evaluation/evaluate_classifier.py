"""
Academic evaluation of the crop-disease farmer-question classifier.

What this script does
----------------------
1. Loads the held-out test split: data/test.csv.
2. Sends every test question through the SAME production classification
   pipeline the Flask API uses (backend.services.llm_classifier.classify_questions).
3. Compares each prediction against the ground-truth category.
4. Computes accuracy, macro precision, macro recall, macro F1, a
   confusion matrix, and a full classification report using scikit-learn.
5. Writes evaluation/results.json, evaluation/confusion_matrix.json, and
   evaluation/predictions.csv, and prints a human-readable report.

Test / few-shot separation
---------------------------
Few-shot examples embedded in the classification prompt are always drawn
from data/development.csv (see
backend.services.prompt_builder.select_few_shot_examples). This script
only ever reads data/test.csv. It never opens development.csv, so the
held-out test set can never leak into the prompt — the separation is
structural, not just a convention.

Scope of the four-class evaluation
------------------------------------
data/test.csv only contains the four primary information-need
categories — Symptoms, Prevention, Management, General Information.
backend.data_loader enforces this on load (it raises if any other
category value is present). "Unable to Classify" is a label the MODEL
may emit (a genuine "this isn't answerable" judgement, or the fallback
used when the LLM response fails validation) — it is never a ground-truth
label in this dataset, and it is deliberately NOT added as a 5th class to
the primary metrics.

Concretely:
  * accuracy / precision / recall / F1 / confusion matrix / classification
    report are computed with scikit-learn's `labels=` parameter restricted
    to the four primary categories. A prediction outside that set (e.g.
    "Unable to Classify") is still scored — it simply cannot match any
    true label, so it correctly counts against that example's recall
    without inflating the label space.
  * How often the model fell back to "Unable to Classify" on a genuine
    4-class question is still tracked and reported, under `diagnostics`
    in results.json and in the console output — visible, not hidden.
  * Outright system failures (network/provider/config errors — status
    "error" from classify_question, meaning no real judgement was ever
    made) are excluded from the scored metrics and reported separately.
    Scoring them as "wrong" would conflate infrastructure reliability
    with classification quality; silently dropping them without saying
    so would misrepresent the sample size. This script does neither.

No number in the saved output is hardcoded. Every value comes from
actually calling the classifier on data/test.csv during this run.

Usage
-----
    python -m backend.evaluation.evaluate_classifier
    # or
    python backend/evaluation/evaluate_classifier.py

Requires a real LLM provider configured in .env (LLM_PROVIDER=anthropic
or openai, with LLM_API_KEY / LLM_MODEL_NAME set). The offline "echo"
provider is refused — it returns a fixed canned response for every
question and would produce meaningless, non-representative numbers.
"""

import csv
import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config.settings import config  # noqa: E402
from backend.data_loader import load_dataset  # noqa: E402
from backend.services.llm_classifier import FALLBACK_CATEGORY, classify_questions  # noqa: E402
from backend.services.prompt_builder import ALLOWED_CATEGORIES  # noqa: E402

TEST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "test.csv")
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
RESULTS_JSON_PATH = os.path.join(EVALUATION_DIR, "results.json")
CONFUSION_MATRIX_JSON_PATH = os.path.join(EVALUATION_DIR, "confusion_matrix.json")
PREDICTIONS_CSV_PATH = os.path.join(EVALUATION_DIR, "predictions.csv")

# The four primary information-need categories. This is the ONLY label
# set used for accuracy / precision / recall / F1 / confusion matrix.
PRIMARY_CATEGORIES = list(ALLOWED_CATEGORIES)


class EvaluationConfigError(Exception):
    """Raised when the environment is not set up for a genuine evaluation run."""


# ---------------------------------------------------------------------------
# Setup / guardrails
# ---------------------------------------------------------------------------

def _ensure_real_provider_configured() -> None:
    """
    Refuse to run against the offline "echo" stub.

    EchoProvider returns the same canned "Unable to Classify" response
    for every question (see backend/services/llm_providers.py) — running
    the evaluation against it would produce numbers that look real but
    measure nothing. This is a hard stop, not a warning.
    """
    provider_name = (getattr(config, "LLM_PROVIDER", "") or "").strip().lower()
    if not provider_name:
        raise EvaluationConfigError(
            "LLM_PROVIDER is not configured. Set it in .env before running this evaluation."
        )
    if provider_name == "echo":
        raise EvaluationConfigError(
            "LLM_PROVIDER is set to 'echo', the offline stub provider. It returns a "
            "fixed canned response for every question and must never be used to "
            "produce evaluation numbers. Set LLM_PROVIDER=anthropic (or openai) and "
            "configure LLM_API_KEY / LLM_MODEL_NAME in your .env file, then re-run."
        )

    # Check credentials up front. Without this the run proceeds to fire one
    # doomed API call per test question, then aborts with a generic
    # "every question failed" message that hides the real cause.
    if not config.is_llm_configured():
        raise EvaluationConfigError(
            f"LLM_PROVIDER is '{provider_name}' but LLM_API_KEY is not set. "
            "Copy .env.example to .env and set a real API key, then re-run. "
            "(The key is read from the environment and is never logged.)"
        )
    if not (getattr(config, "LLM_MODEL_NAME", "") or "").strip():
        raise EvaluationConfigError(
            f"LLM_PROVIDER is '{provider_name}' but LLM_MODEL_NAME is not set. "
            "Configure it in your .env file, then re-run."
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_set() -> pd.DataFrame:
    """Load and validate data/test.csv. Raises if the file is missing or malformed."""
    print(f"[evaluate] Loading held-out test set from: {TEST_CSV_PATH}")
    df = load_dataset(TEST_CSV_PATH, drop_invalid_rows=False)
    print(f"[evaluate] Loaded {len(df)} test question(s).")

    counts = df["category"].value_counts().to_dict()
    print("[evaluate] Ground-truth category distribution:")
    for category in PRIMARY_CATEGORIES:
        print(f"    {category:<22} {counts.get(category, 0)}")

    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def run_classification(df: pd.DataFrame) -> list:
    """
    Send every test-set question through the production classifier.

    Returns one dict per row combining the ground-truth label with the
    raw classify_question() result (category, confidence, reason, status,
    error, error_code).
    """
    items = [{"question": row.question, "crop": row.crop} for row in df.itertuples(index=False)]

    print(f"[evaluate] Sending {len(items)} question(s) to the LLM classifier...")
    start_time = time.time()
    raw_results = classify_questions(items)
    elapsed = time.time() - start_time
    per_item = elapsed / max(len(items), 1)
    print(f"[evaluate] Finished in {elapsed:.1f}s ({per_item:.2f}s/question average).")

    records = []
    for row, result in zip(df.itertuples(index=False), raw_results):
        records.append({
            "id": row.id,
            "question": row.question,
            "crop": row.crop,
            "true_category": row.category,
            "predicted_category": result["category"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "status": result["status"],
            "error": result["error"],
            "error_code": result["error_code"],
        })
    return records


def split_records(records: list) -> tuple:
    """
    Separate genuine classification judgements from system failures.

    status == "error" means classify_question() never got a usable
    answer from the LLM (config, network, or a malformed response that
    failed validation) — it is not a prediction and must not be scored.
    """
    scored = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    return scored, failed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(scored_records: list) -> dict:
    """
    Compute accuracy, macro precision/recall/F1, a confusion matrix, and a
    full classification report over the four primary categories.

    All metrics are restricted to PRIMARY_CATEGORIES via scikit-learn's
    `labels=` argument. A prediction outside that set (e.g. the model
    returning "Unable to Classify" for a genuine 4-class question) is
    left in y_pred as-is: it simply cannot match the true label, so it
    correctly reduces that class's recall instead of being dropped or
    silently remapped.
    """
    y_true = [r["true_category"] for r in scored_records]
    y_pred = [r["predicted_category"] for r in scored_records]

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=PRIMARY_CATEGORIES, average="macro", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, labels=PRIMARY_CATEGORIES, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, labels=PRIMARY_CATEGORIES, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=PRIMARY_CATEGORIES)

    # Diagnostic only (not part of the primary 4-class metrics): how many
    # genuinely 4-class questions the model routed to the fallback label.
    unable_to_classify_count = sum(
        1 for r in scored_records if r["predicted_category"] == FALLBACK_CATEGORY
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "unable_to_classify_count": unable_to_classify_count,
        "n_scored": len(scored_records),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(metrics: dict, scored_records: list, failed_records: list, output_path: str) -> None:
    payload = {
        "dataset": {
            "path": os.path.relpath(TEST_CSV_PATH, PROJECT_ROOT).replace(os.sep, "/"),
            "total_rows": len(scored_records) + len(failed_records),
            "scored_rows": len(scored_records),
            "excluded_system_failures": len(failed_records),
        },
        "categories_evaluated": PRIMARY_CATEGORIES,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        },
        "classification_report": metrics["classification_report_dict"],
        "diagnostics": {
            "unable_to_classify_predictions": metrics["unable_to_classify_count"],
            "system_failures": len(failed_records),
            "system_failure_error_codes": sorted(
                {r["error_code"] for r in failed_records if r["error_code"]}
            ),
        },
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[evaluate] Wrote metrics to {output_path}")


def save_confusion_matrix(cm, output_path: str) -> None:
    payload = {
        "labels": PRIMARY_CATEGORIES,
        "matrix": cm.tolist(),
        "note": (
            "matrix[i][j] is the number of test examples whose true category is "
            "labels[i] and whose predicted category is labels[j]."
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[evaluate] Wrote confusion matrix to {output_path}")


def save_predictions_csv(records: list, output_path: str) -> None:
    fieldnames = [
        "id", "question", "crop", "true_category", "predicted_category",
        "correct", "confidence", "reason", "status", "error", "error_code",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            is_correct = r["status"] == "ok" and r["predicted_category"] == r["true_category"]
            writer.writerow({
                "id": r["id"],
                "question": r["question"],
                "crop": r["crop"],
                "true_category": r["true_category"],
                "predicted_category": r["predicted_category"],
                "correct": is_correct,
                "confidence": r["confidence"],
                "reason": r["reason"],
                "status": r["status"],
                "error": r["error"] or "",
                "error_code": r["error_code"] or "",
            })
    print(f"[evaluate] Wrote per-question predictions to {output_path}")


def print_console_report(metrics: dict, scored_records: list, failed_records: list) -> None:
    print("\n" + "=" * 64)
    print("CROP DISEASE QUESTION CLASSIFIER — EVALUATION RESULTS")
    print("=" * 64)
    print(f"Test questions scored                : {len(scored_records)}")
    if failed_records:
        print(f"Excluded (system failures, not scored): {len(failed_records)}")
    print("-" * 64)
    print(f"Accuracy             : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)    : {metrics['precision_macro']:.4f}")
    print(f"Recall (macro)       : {metrics['recall_macro']:.4f}")
    print(f"F1 Score (macro)     : {metrics['f1_macro']:.4f}")
    print("-" * 64)
    print("Classification Report:")
    print(metrics["classification_report_text"])
    print("-" * 64)
    print("Confusion Matrix (rows = true category, columns = predicted category):")
    col_header = " " * 24 + "".join(f"{c[:12]:>14}" for c in PRIMARY_CATEGORIES)
    print(col_header)
    for label, row in zip(PRIMARY_CATEGORIES, metrics["confusion_matrix"]):
        print(f"{label:<24}" + "".join(f"{v:>14}" for v in row))
    print("-" * 64)
    if metrics["unable_to_classify_count"]:
        print(
            f"'{FALLBACK_CATEGORY}' predicted for a genuine 4-class question: "
            f"{metrics['unable_to_classify_count']} time(s)"
        )
    if failed_records:
        codes = sorted({r["error_code"] for r in failed_records if r["error_code"]})
        print(f"System failure error code(s) encountered: {codes}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    try:
        _ensure_real_provider_configured()
    except EvaluationConfigError as exc:
        print(f"[evaluate] ABORTED: {exc}")
        sys.exit(1)

    df = load_test_set()
    records = run_classification(df)
    scored_records, failed_records = split_records(records)

    if not scored_records:
        # Report the real reasons. The previous message pointed the reader
        # at "error_code values above", but nothing had printed them.
        codes = sorted({r["error_code"] for r in failed_records if r["error_code"]})
        messages = sorted({r["error"] for r in failed_records if r["error"]})
        print(
            f"[evaluate] ABORTED: all {len(failed_records)} test question(s) resulted "
            "in a system failure, so there are no genuine predictions to score."
        )
        if codes:
            print(f"[evaluate] Error code(s): {codes}")
        for message in messages[:3]:
            print(f"[evaluate]   - {message}")
        sys.exit(1)

    metrics = compute_metrics(scored_records)
    print_console_report(metrics, scored_records, failed_records)

    save_results(metrics, scored_records, failed_records, RESULTS_JSON_PATH)
    save_confusion_matrix(metrics["confusion_matrix"], CONFUSION_MATRIX_JSON_PATH)
    save_predictions_csv(records, PREDICTIONS_CSV_PATH)


if __name__ == "__main__":
    main()
