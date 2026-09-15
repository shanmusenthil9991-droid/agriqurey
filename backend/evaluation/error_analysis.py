"""
Error analysis for the crop-disease farmer-question classifier.

What this script does
----------------------
1. Loads evaluation/predictions.csv, the per-question output already
   produced by backend.evaluation.evaluate_classifier (one row per test
   question, with the ground-truth category, the model's predicted
   category, and a run status).
2. Isolates every test question the classifier got wrong.
3. Writes one row per error to evaluation/errors.csv with exactly:
   question, crop, actual_category, predicted_category (plus the row id,
   kept for traceability back to data/test.csv).
4. Counts how often each (actual -> predicted) category pair occurs
   among the errors, and reports the pairs in descending frequency.
5. Reports, by name, the count for each of three specific confusions
   the project wants tracked regardless of whether they turn out to be
   frequent: Symptoms vs General Information, Prevention vs Management,
   and Management vs General Information. "vs" here is a single
   undirected pair — a Symptoms question predicted as General
   Information and a General Information question predicted as
   Symptoms both count toward the same "Symptoms vs General
   Information" total. The directional confusion table above still
   shows each direction separately for readers who want that detail.
6. Writes a summary (total test questions, correct, incorrect, most
   confused category pairs, and the three tracked pairs) to
   evaluation/error_analysis_summary.json and prints it to the console.

What this script deliberately does NOT do
-------------------------------------------
* It does not modify data/test.csv or any ground-truth label. It only
  ever reads evaluation/predictions.csv.
* It does not generate any explanation for WHY a question was
  misclassified. It reports what happened (actual vs. predicted
  category, and confusion counts) and leaves interpretation to the
  report's discussion section.
* It does not score system failures (status != "ok", e.g. a network or
  provider error where the model never produced a real judgement) as
  classification errors. This mirrors evaluate_classifier.py, which
  excludes the same rows from its accuracy/precision/recall/F1
  numbers. Any such rows are counted and reported separately so the
  sample size is never silently changed.

Usage
-----
    python -m backend.evaluation.error_analysis
    # or
    python backend/evaluation/error_analysis.py

Requires evaluation/predictions.csv to already exist. Generate it first
with:
    python -m backend.evaluation.evaluate_classifier
"""

import csv
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
PREDICTIONS_CSV_PATH = os.path.join(EVALUATION_DIR, "predictions.csv")
ERRORS_CSV_PATH = os.path.join(EVALUATION_DIR, "errors.csv")
SUMMARY_JSON_PATH = os.path.join(EVALUATION_DIR, "error_analysis_summary.json")

# Confusions the project specifically wants surfaced, independent of
# whether they end up being the most frequent ones. Each is an
# unordered pair: "A vs B" counts (A predicted as B) + (B predicted as A).
TRACKED_PAIRS = [
    ("Symptoms", "General Information"),
    ("Prevention", "Management"),
    ("Management", "General Information"),
]


class ErrorAnalysisConfigError(Exception):
    """Raised when evaluation/predictions.csv is missing or unreadable."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(path: str = PREDICTIONS_CSV_PATH) -> list:
    """
    Load evaluation/predictions.csv as a list of dict rows.

    Raises ErrorAnalysisConfigError if the file does not exist, so this
    script fails loudly instead of silently analyzing nothing.
    """
    if not os.path.exists(path):
        raise ErrorAnalysisConfigError(
            f"{path} does not exist. Run "
            "`python -m backend.evaluation.evaluate_classifier` first to "
            "generate it, then re-run this script."
        )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ErrorAnalysisConfigError(f"{path} exists but contains no rows.")
    return rows


# ---------------------------------------------------------------------------
# Splitting scored predictions from system failures
# ---------------------------------------------------------------------------

def split_rows(rows: list) -> tuple:
    """
    Separate genuine classification judgements (status == "ok") from
    system failures (status != "ok", e.g. a network/provider error).

    System failures are excluded from error analysis for the same
    reason evaluate_classifier.py excludes them from accuracy/F1: no
    real classification judgement was made, so there is no
    "misclassification" to analyze — only an infrastructure failure.
    """
    scored = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") != "ok"]
    return scored, failed


def identify_errors(scored_rows: list) -> list:
    """Return the subset of scored rows where predicted != actual category."""
    return [
        r for r in scored_rows
        if r.get("predicted_category") != r.get("true_category")
    ]


# ---------------------------------------------------------------------------
# Confusion pair counting
# ---------------------------------------------------------------------------

def count_directional_confusions(error_rows: list) -> Counter:
    """
    Count (actual_category -> predicted_category) pairs among the errors.

    Returns a Counter keyed by "Actual -> Predicted" strings, most
    frequent first when iterated via .most_common().
    """
    counter = Counter()
    for r in error_rows:
        key = f"{r['true_category']} -> {r['predicted_category']}"
        counter[key] += 1
    return counter


def count_undirected_pair(error_rows: list, category_a: str, category_b: str) -> int:
    """
    Count errors where the (actual, predicted) categories are
    {category_a, category_b} in either direction.
    """
    total = 0
    for r in error_rows:
        pair = {r["true_category"], r["predicted_category"]}
        if pair == {category_a, category_b}:
            total += 1
    return total


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_errors_csv(error_rows: list, output_path: str = ERRORS_CSV_PATH) -> None:
    fieldnames = ["id", "question", "crop", "actual_category", "predicted_category"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in error_rows:
            writer.writerow({
                "id": r.get("id", ""),
                "question": r.get("question", ""),
                "crop": r.get("crop", ""),
                "actual_category": r.get("true_category", ""),
                "predicted_category": r.get("predicted_category", ""),
            })
    print(f"[error_analysis] Wrote {len(error_rows)} error record(s) to {output_path}")


def build_summary(all_rows: list, scored_rows: list, failed_rows: list, error_rows: list) -> dict:
    directional = count_directional_confusions(error_rows)
    most_confused_directional = [
        {"confusion": key, "count": count} for key, count in directional.most_common()
    ]

    tracked = []
    for category_a, category_b in TRACKED_PAIRS:
        tracked.append({
            "pair": f"{category_a} vs {category_b}",
            "count": count_undirected_pair(error_rows, category_a, category_b),
        })

    correct_count = len(scored_rows) - len(error_rows)

    return {
        "total_test_questions": len(all_rows),
        "scored_questions": len(scored_rows),
        "excluded_system_failures": len(failed_rows),
        "correct_predictions": correct_count,
        "incorrect_predictions": len(error_rows),
        "accuracy_on_scored": (
            round(correct_count / len(scored_rows), 4) if scored_rows else None
        ),
        "most_confused_category_pairs": most_confused_directional,
        "tracked_confusion_pairs": tracked,
    }


def save_summary_json(summary: dict, output_path: str = SUMMARY_JSON_PATH) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[error_analysis] Wrote summary to {output_path}")


def print_console_report(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("CROP DISEASE QUESTION CLASSIFIER — ERROR ANALYSIS")
    print("=" * 64)
    print(f"Total test questions (in predictions.csv) : {summary['total_test_questions']}")
    if summary["excluded_system_failures"]:
        print(f"Excluded (system failures, not scored)    : {summary['excluded_system_failures']}")
    print(f"Scored questions                           : {summary['scored_questions']}")
    print(f"Correct predictions                        : {summary['correct_predictions']}")
    print(f"Incorrect predictions                      : {summary['incorrect_predictions']}")
    if summary["accuracy_on_scored"] is not None:
        print(f"Accuracy (on scored questions)             : {summary['accuracy_on_scored']:.4f}")
    print("-" * 64)
    print("Confusion pairs among errors (actual -> predicted), most frequent first:")
    if summary["most_confused_category_pairs"]:
        for entry in summary["most_confused_category_pairs"]:
            print(f"    {entry['confusion']:<45} {entry['count']}")
    else:
        print("    (no errors to report)")
    print("-" * 64)
    print("Tracked pairs of interest (undirected: either direction counted):")
    for entry in summary["tracked_confusion_pairs"]:
        print(f"    {entry['pair']:<40} {entry['count']}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    try:
        rows = load_predictions()
    except ErrorAnalysisConfigError as exc:
        print(f"[error_analysis] ABORTED: {exc}")
        sys.exit(1)

    scored_rows, failed_rows = split_rows(rows)
    error_rows = identify_errors(scored_rows)

    summary = build_summary(rows, scored_rows, failed_rows, error_rows)

    save_errors_csv(error_rows)
    save_summary_json(summary)
    print_console_report(summary)


if __name__ == "__main__":
    main()
