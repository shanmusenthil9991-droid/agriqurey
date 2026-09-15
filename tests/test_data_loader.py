"""
Tests for backend/data_loader.py

Covers:
- Successful loading of the real dataset
- Column validation
- Category validation
- Missing value detection
- Duplicate question detection
"""

import os
import pandas as pd
import pytest

from backend.data_loader import (
    load_dataset,
    get_dataset_summary,
    DatasetValidationError,
    REQUIRED_COLUMNS,
    ALLOWED_CATEGORIES,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "crop_questions.csv")


# ---------------------------------------------------------------------------
# Tests against the real, shipped dataset
# ---------------------------------------------------------------------------

def test_real_dataset_loads_successfully():
    df = load_dataset(REAL_DATASET_PATH)
    assert not df.empty
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_real_dataset_size_within_expected_range():
    df = load_dataset(REAL_DATASET_PATH)
    assert 200 <= len(df) <= 500


def test_real_dataset_categories_are_allowed():
    df = load_dataset(REAL_DATASET_PATH)
    assert set(df["category"].unique()).issubset(ALLOWED_CATEGORIES)


def test_real_dataset_has_no_missing_values():
    df = load_dataset(REAL_DATASET_PATH)
    assert df[REQUIRED_COLUMNS].isnull().sum().sum() == 0


def test_real_dataset_has_no_duplicate_questions():
    df = load_dataset(REAL_DATASET_PATH)
    normalized = df["question"].str.strip().str.lower()
    assert normalized.duplicated().sum() == 0


def test_real_dataset_categories_are_reasonably_balanced():
    df = load_dataset(REAL_DATASET_PATH)
    counts = df["category"].value_counts()
    assert len(counts) == 4
    # No single category should dominate more than ~40% of the dataset.
    assert counts.max() / len(df) < 0.40


def test_get_dataset_summary_structure():
    df = load_dataset(REAL_DATASET_PATH)
    summary = get_dataset_summary(df)
    assert summary["total_rows"] == len(df)
    assert set(summary["category_counts"].keys()).issubset(ALLOWED_CATEGORIES)


def test_missing_dataset_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/does_not_exist.csv")


# ---------------------------------------------------------------------------
# Tests against small synthetic CSVs (edge cases)
# ---------------------------------------------------------------------------

def test_missing_required_column_raises_validation_error(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({
        "id": [1, 2],
        "question": ["What is rice blast?", "How to prevent it?"],
        "category": ["General Information", "Prevention"],
        # "crop" column intentionally omitted
    }).to_csv(bad_csv, index=False)

    with pytest.raises(DatasetValidationError):
        load_dataset(str(bad_csv))


def test_invalid_category_value_raises_validation_error(tmp_path):
    bad_csv = tmp_path / "bad_category.csv"
    pd.DataFrame({
        "id": [1, 2],
        "question": ["What is rice blast?", "How to treat it?"],
        "crop": ["Rice", "Rice"],
        "category": ["General Information", "Diagnosis"],  # "Diagnosis" not allowed
    }).to_csv(bad_csv, index=False)

    with pytest.raises(DatasetValidationError):
        load_dataset(str(bad_csv))


def test_duplicate_questions_are_dropped_when_requested(tmp_path):
    dup_csv = tmp_path / "dup.csv"
    pd.DataFrame({
        "id": [1, 2, 3],
        "question": [
            "What is rice blast?",
            "what is rice blast?",  # duplicate (case-insensitive)
            "How to prevent it?",
        ],
        "crop": ["Rice", "Rice", "Rice"],
        "category": ["General Information", "General Information", "Prevention"],
    }).to_csv(dup_csv, index=False)

    df = load_dataset(str(dup_csv), drop_invalid_rows=True)
    assert len(df) == 2


def test_missing_values_are_dropped_when_requested(tmp_path):
    missing_csv = tmp_path / "missing.csv"
    pd.DataFrame({
        "id": [1, 2],
        "question": ["What is rice blast?", None],
        "crop": ["Rice", "Rice"],
        "category": ["General Information", "Prevention"],
    }).to_csv(missing_csv, index=False)

    df = load_dataset(str(missing_csv), drop_invalid_rows=True)
    assert len(df) == 1
    assert df.iloc[0]["question"] == "What is rice blast?"


def test_drop_invalid_rows_false_keeps_rows_but_still_reports(tmp_path, capsys):
    dup_csv = tmp_path / "dup_keep.csv"
    pd.DataFrame({
        "id": [1, 2],
        "question": ["What is rice blast?", "what is rice blast?"],
        "crop": ["Rice", "Rice"],
        "category": ["General Information", "General Information"],
    }).to_csv(dup_csv, index=False)

    df = load_dataset(str(dup_csv), drop_invalid_rows=False)
    assert len(df) == 2  # nothing dropped
    captured = capsys.readouterr()
    assert "duplicate" in captured.out.lower()
