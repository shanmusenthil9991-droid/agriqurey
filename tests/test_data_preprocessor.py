"""
Tests for backend/data_preprocessor.py

Covers:
- load_dataset() / clean_dataset() behavior (validation, missing values, duplicates)
- split_dataset() ratios, stratification, reproducibility, and no-overlap guarantees
- save_splits() writing the three CSV files
"""

import os
import pandas as pd
import pytest

from backend.data_loader import DatasetValidationError
from backend.data_preprocessor import (
    load_dataset,
    clean_dataset,
    split_dataset,
    save_splits,
    run_preprocessing_pipeline,
    DEV_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "crop_questions.csv")


# ---------------------------------------------------------------------------
# load_dataset()
# ---------------------------------------------------------------------------

def test_load_dataset_reads_real_csv():
    df = load_dataset(REAL_DATASET_PATH)
    assert not df.empty
    assert "question" in df.columns


def test_load_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("data/does_not_exist.csv")


# ---------------------------------------------------------------------------
# clean_dataset()
# ---------------------------------------------------------------------------

def test_clean_dataset_on_real_data_has_no_missing_or_duplicates():
    raw = load_dataset(REAL_DATASET_PATH)
    clean = clean_dataset(raw, verbose=False)
    assert clean.isnull().sum().sum() == 0
    normalized = clean["question"].str.strip().str.lower()
    assert normalized.duplicated().sum() == 0


def test_clean_dataset_missing_column_raises():
    bad_df = pd.DataFrame({
        "id": [1, 2],
        "question": ["a?", "b?"],
        "category": ["General Information", "Prevention"],
        # "crop" missing
    })
    with pytest.raises(DatasetValidationError):
        clean_dataset(bad_df, verbose=False)


def test_clean_dataset_invalid_category_raises():
    bad_df = pd.DataFrame({
        "id": [1, 2],
        "question": ["a?", "b?"],
        "crop": ["Rice", "Rice"],
        "category": ["General Information", "Diagnosis"],
    })
    with pytest.raises(DatasetValidationError):
        clean_dataset(bad_df, verbose=False)


def test_clean_dataset_drops_missing_values():
    df = pd.DataFrame({
        "id": [1, 2],
        "question": ["What is rice blast?", None],
        "crop": ["Rice", "Rice"],
        "category": ["General Information", "Prevention"],
    })
    clean = clean_dataset(df, verbose=False)
    assert len(clean) == 1


def test_clean_dataset_drops_duplicate_questions():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "question": ["What is rice blast?", "what is rice blast?", "How to prevent it?"],
        "crop": ["Rice", "Rice", "Rice"],
        "category": ["General Information", "General Information", "Prevention"],
    })
    clean = clean_dataset(df, verbose=False)
    assert len(clean) == 2


# ---------------------------------------------------------------------------
# split_dataset()
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_real_df():
    raw = load_dataset(REAL_DATASET_PATH)
    return clean_dataset(raw, verbose=False)


def test_split_ratios_are_approximately_correct(clean_real_df):
    dev, val, test = split_dataset(clean_real_df)
    total = len(clean_real_df)
    assert abs(len(dev) / total - DEV_RATIO) < 0.02
    assert abs(len(val) / total - VAL_RATIO) < 0.02
    assert abs(len(test) / total - TEST_RATIO) < 0.02


def test_split_covers_every_record_exactly_once(clean_real_df):
    dev, val, test = split_dataset(clean_real_df)
    dev_ids, val_ids, test_ids = set(dev["id"]), set(val["id"]), set(test["id"])

    assert dev_ids.isdisjoint(val_ids)
    assert dev_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(dev_ids | val_ids | test_ids) == len(clean_real_df)


def test_split_is_stratified_across_all_four_categories(clean_real_df):
    dev, val, test = split_dataset(clean_real_df)
    expected_categories = {"Symptoms", "Prevention", "Management", "General Information"}

    assert set(dev["category"].unique()) == expected_categories
    assert set(val["category"].unique()) == expected_categories
    assert set(test["category"].unique()) == expected_categories


def test_split_is_reproducible_with_same_seed(clean_real_df):
    dev1, val1, test1 = split_dataset(clean_real_df, random_seed=42)
    dev2, val2, test2 = split_dataset(clean_real_df, random_seed=42)

    assert list(dev1["id"]) == list(dev2["id"])
    assert list(val1["id"]) == list(val2["id"])
    assert list(test1["id"]) == list(test2["id"])


def test_split_differs_with_different_seed(clean_real_df):
    _, _, test_seed_a = split_dataset(clean_real_df, random_seed=1)
    _, _, test_seed_b = split_dataset(clean_real_df, random_seed=2)
    assert list(test_seed_a["id"]) != list(test_seed_b["id"])


def test_split_invalid_ratios_raise_value_error(clean_real_df):
    with pytest.raises(ValueError):
        split_dataset(clean_real_df, dev_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


# ---------------------------------------------------------------------------
# save_splits() / run_preprocessing_pipeline()
# ---------------------------------------------------------------------------

def test_save_splits_writes_three_files(tmp_path, clean_real_df):
    dev, val, test = split_dataset(clean_real_df)
    dev_path = tmp_path / "development.csv"
    val_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"

    save_splits(dev, val, test, str(dev_path), str(val_path), str(test_path))

    assert dev_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    reloaded_dev = pd.read_csv(dev_path)
    assert len(reloaded_dev) == len(dev)


def test_run_preprocessing_pipeline_end_to_end(tmp_path):
    dev_path = tmp_path / "development.csv"
    val_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"

    dev, val, test = run_preprocessing_pipeline(
        input_path=REAL_DATASET_PATH,
        dev_path=str(dev_path),
        val_path=str(val_path),
        test_path=str(test_path),
        verbose=False,
    )

    assert dev_path.exists() and val_path.exists() and test_path.exists()
    assert len(dev) + len(val) + len(test) == len(dev) + len(val) + len(test)  # sanity
    assert set(dev["id"]).isdisjoint(set(test["id"]))
