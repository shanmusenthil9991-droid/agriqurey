"""
Academic comparison: zero-shot vs. few-shot LLM classification.

Research question
------------------
Does few-shot prompting (representative examples from data/development.csv
embedded in the prompt) improve classification performance over direct
zero-shot classification (no examples at all), on the SAME held-out test
set (data/test.csv)?

Experiment A — BASELINE (zero-shot)
    Each test question is classified with an empty example list, so the
    prompt contains only the category definitions/rules and the question
    itself — no demonstrations of any kind.

Experiment B — PROPOSED (few-shot)
    The same test questions are classified again, this time with the
    project's standard few-shot example set
    (backend.services.prompt_builder.select_few_shot_examples()), drawn
    exclusively from data/development.csv.

Both experiments reuse the exact same prompt template, category rules,
validation logic, and LLM provider/model — the only variable that changes
between Experiment A and Experiment B is the presence of few-shot
examples. This isolates the effect of few-shot prompting from every other
factor.

No test-set leakage
--------------------
Few-shot examples are selected ONLY from data/development.csv via
select_few_shot_examples(); this script never reads test.csv rows into
an examples list, and the zero-shot condition uses an explicit empty
list. The separation is structural: there is no code path in this file
that could place a test-set question inside a prompt as a demonstration.

Honesty about results
-----------------------
This script does not assert that few-shot prompting is better. It
computes both experiments' metrics from actual classifier runs on
data/test.csv, reports the measured difference, and only ever describes
that difference — never a predetermined conclusion. If the numbers show
no improvement, that is what gets printed and saved.

Usage
-----
    python -m backend.evaluation.compare_methods
    # or
    python backend/evaluation/compare_methods.py

Requires a real LLM provider configured in .env (LLM_PROVIDER=anthropic
or openai, with LLM_API_KEY / LLM_MODEL_NAME set). The offline "echo"
stub is refused, for the same reason as evaluate_classifier.py: it
returns a fixed canned response regardless of prompt content, so it
cannot demonstrate any zero-shot/few-shot difference.
"""

import json
import os
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.evaluation.evaluate_classifier import (  # noqa: E402
    PRIMARY_CATEGORIES,
    EvaluationConfigError,
    _ensure_real_provider_configured,
    compute_metrics,
    load_test_set,
    split_records,
)
from backend.services.llm_classifier import (  # noqa: E402
    ClassificationError,
    _build_provider,
    classify_question,
)
from backend.services.prompt_builder import select_few_shot_examples  # noqa: E402

EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation")
COMPARISON_JSON_PATH = os.path.join(EVALUATION_DIR, "method_comparison.json")

ZERO_SHOT_EXAMPLES = []  # BASELINE: no demonstrations at all.


# ---------------------------------------------------------------------------
# Running one experiment condition
# ---------------------------------------------------------------------------

def run_experiment(df: pd.DataFrame, provider, examples: list, label: str) -> list:
    """
    Classify every row of df through classify_question() using a fixed
    `examples` list (either [] for zero-shot, or the few-shot set), reusing
    one provider instance across all calls.

    Returns one record per row: ground truth plus the raw classifier result.
    """
    print(f"[compare] Running {label} on {len(df)} test question(s)...")
    start_time = time.time()

    records = []
    for row in df.itertuples(index=False):
        result = classify_question(
            question=row.question,
            crop=row.crop,
            provider=provider,
            examples=examples,
        )
        records.append({
            "id": row.id,
            "question": row.question,
            "crop": row.crop,
            "true_category": row.category,
            "predicted_category": result["category"],
            "confidence": result["confidence"],
            "status": result["status"],
            "error_code": result["error_code"],
        })

    elapsed = time.time() - start_time
    print(f"[compare] {label} finished in {elapsed:.1f}s "
          f"({elapsed / max(len(df), 1):.2f}s/question average).")
    return records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _metrics_summary(metrics: dict, scored_records: list, failed_records: list) -> dict:
    return {
        "n_scored": len(scored_records),
        "n_excluded_system_failures": len(failed_records),
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "unable_to_classify_predictions": metrics["unable_to_classify_count"],
    }


def save_comparison(zero_shot_summary: dict, few_shot_summary: dict, deltas: dict,
                     conclusion: str, output_path: str) -> None:
    payload = {
        "experiment": {
            "test_set": "data/test.csv",
            "few_shot_source": "data/development.csv",
            "categories_evaluated": PRIMARY_CATEGORIES,
        },
        "baseline_zero_shot": zero_shot_summary,
        "proposed_few_shot": few_shot_summary,
        "delta_few_shot_minus_zero_shot": deltas,
        "conclusion": conclusion,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[compare] Wrote method comparison to {output_path}")


def print_comparison_table(zero_shot_summary: dict, few_shot_summary: dict, deltas: dict) -> None:
    metric_rows = [
        ("Accuracy", "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)", "recall_macro"),
        ("F1 Score (macro)", "f1_macro"),
    ]

    print("\n" + "=" * 72)
    print("ZERO-SHOT vs. FEW-SHOT CLASSIFICATION — METHOD COMPARISON")
    print("=" * 72)
    print(f"{'Metric':<20}{'Zero-shot':>15}{'Few-shot':>15}{'Difference':>18}")
    print("-" * 72)
    for label, key in metric_rows:
        zs = zero_shot_summary[key]
        fs = few_shot_summary[key]
        diff = deltas[key]
        sign = "+" if diff >= 0 else ""
        print(f"{label:<20}{zs:>15.4f}{fs:>15.4f}{sign + format(diff, '.4f'):>18}")
    print("-" * 72)
    print(f"{'Scored questions':<20}{zero_shot_summary['n_scored']:>15}{few_shot_summary['n_scored']:>15}")
    if zero_shot_summary["n_excluded_system_failures"] or few_shot_summary["n_excluded_system_failures"]:
        print(f"{'Excluded failures':<20}"
              f"{zero_shot_summary['n_excluded_system_failures']:>15}"
              f"{few_shot_summary['n_excluded_system_failures']:>15}")
    if zero_shot_summary["unable_to_classify_predictions"] or few_shot_summary["unable_to_classify_predictions"]:
        unable_label = "'Unable to Classify'"
        print(f"{unable_label:<20}"
              f"{zero_shot_summary['unable_to_classify_predictions']:>15}"
              f"{few_shot_summary['unable_to_classify_predictions']:>15}")
    print("=" * 72 + "\n")


def determine_conclusion(deltas: dict) -> str:
    """
    Describe the measured outcome plainly — never assert improvement the
    numbers don't show. Judged primarily on macro F1 (the standard single
    summary metric for multiclass performance with class imbalance);
    accuracy is reported alongside for context.
    """
    f1_delta = deltas["f1_macro"]
    acc_delta = deltas["accuracy"]

    if f1_delta > 0 and acc_delta > 0:
        return (
            f"Few-shot prompting outperformed zero-shot on this test set: "
            f"macro F1 improved by {f1_delta:+.4f} and accuracy by {acc_delta:+.4f}."
        )
    if f1_delta < 0 and acc_delta < 0:
        return (
            f"Few-shot prompting underperformed zero-shot on this test set: "
            f"macro F1 changed by {f1_delta:+.4f} and accuracy by {acc_delta:+.4f}."
        )
    if f1_delta == 0 and acc_delta == 0:
        return "Few-shot and zero-shot produced identical measured performance on this test set."
    return (
        f"Mixed result on this test set: macro F1 changed by {f1_delta:+.4f} while "
        f"accuracy changed by {acc_delta:+.4f}. The two methods do not agree on which "
        f"performed better by every metric, so no overall improvement can be claimed."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    try:
        _ensure_real_provider_configured()
    except EvaluationConfigError as exc:
        print(f"[compare] ABORTED: {exc}")
        sys.exit(1)

    df = load_test_set()

    # _build_provider() raises ClassificationError on bad/missing config.
    # Uncaught, that surfaced as a raw traceback; convert it to the same
    # clean, actionable message style the rest of the script uses.
    try:
        provider = _build_provider()
    except ClassificationError as exc:
        print(f"[compare] ABORTED: {exc}")
        sys.exit(1)

    # Few-shot examples come exclusively from development.csv — test.csv
    # is never touched here or anywhere else in this file.
    few_shot_examples = select_few_shot_examples()
    print(f"[compare] Selected {len(few_shot_examples)} few-shot example(s) from data/development.csv.")

    zero_shot_records = run_experiment(df, provider, ZERO_SHOT_EXAMPLES, "Experiment A (zero-shot)")
    few_shot_records = run_experiment(df, provider, few_shot_examples, "Experiment B (few-shot)")

    zero_shot_scored, zero_shot_failed = split_records(zero_shot_records)
    few_shot_scored, few_shot_failed = split_records(few_shot_records)

    if not zero_shot_scored or not few_shot_scored:
        print(
            "[compare] ABORTED: one or both experiments produced no genuine "
            "predictions (every call was a system failure). Nothing to compare."
        )
        sys.exit(1)

    zero_shot_metrics = compute_metrics(zero_shot_scored)
    few_shot_metrics = compute_metrics(few_shot_scored)

    zero_shot_summary = _metrics_summary(zero_shot_metrics, zero_shot_scored, zero_shot_failed)
    few_shot_summary = _metrics_summary(few_shot_metrics, few_shot_scored, few_shot_failed)

    deltas = {
        key: few_shot_summary[key] - zero_shot_summary[key]
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    }

    conclusion = determine_conclusion(deltas)

    print_comparison_table(zero_shot_summary, few_shot_summary, deltas)
    print(conclusion + "\n")

    save_comparison(zero_shot_summary, few_shot_summary, deltas, conclusion, COMPARISON_JSON_PATH)


if __name__ == "__main__":
    main()
