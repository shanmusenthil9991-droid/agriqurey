"""
Dataset preprocessing and splitting.

This module takes the raw labeled dataset (data/crop_questions.csv) and:
  1. Loads it.
  2. Cleans it (validates structure/categories, handles missing values,
     removes duplicate questions).
  3. Splits it into stratified Development/Validation/Test sets.

The split is used downstream as follows:
  - Development/Prompt Set (70%): used to design prompts, pick few-shot
    examples, and iterate on the LLM classifier.
  - Validation Set (15%): used to tune prompt/parameter choices while
    developing, without touching the test set.
  - Test Set (15%): held out for FINAL evaluation only.

IMPORTANT: The test set must NEVER be used as a source of few-shot
examples, prompt design, or any other form of development. It exists
purely to report an unbiased final classification quality score.

This module does not call any LLM — it is a pure data-preparation step.
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Running this file directly (`python backend/data_preprocessor.py`, as the
# README documents) puts backend/ on sys.path instead of the project root,
# so "import backend.config..." would fail with ModuleNotFoundError. Adding
# the project root makes the file work both as a script and as a module
# (`python -m backend.data_preprocessor`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.config.settings import config  # noqa: E402
from backend.data_loader import (  # noqa: E402
    REQUIRED_COLUMNS,
    ALLOWED_CATEGORIES,
    DatasetValidationError,
)

RANDOM_SEED = 42

# Split proportions — must sum to 1.0
DEV_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Honour the DATASET_PATH environment variable rather than hardcoding the
# filename, so the configured value is actually used somewhere instead of
# being dead configuration. Falls back to data/crop_questions.csv.
DEFAULT_INPUT_PATH = config.resolve_dataset_path()
DEFAULT_DEV_PATH = os.path.join(PROJECT_ROOT, "data", "development.csv")
DEFAULT_VAL_PATH = os.path.join(PROJECT_ROOT, "data", "validation.csv")
DEFAULT_TEST_PATH = os.path.join(PROJECT_ROOT, "data", "test.csv")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """
    Load the raw labeled dataset from CSV.

    This is intentionally a thin, unopinionated read — no cleaning or
    validation happens here. Use clean_dataset() afterwards.

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Validate and clean the raw dataset.

    Steps:
        - Validate required columns are present (hard failure).
        - Validate all category values are in the allowed set (hard failure).
        - Detect and report rows with missing values in required columns,
          then drop them.
        - Detect and report duplicate questions (case/whitespace-insensitive),
          then drop the later occurrences, keeping the first.

    Args:
        df: Raw dataframe, as returned by load_dataset().
        verbose: If True, prints a short report of what was cleaned.

    Returns:
        A clean DataFrame with a fresh, sequential index.

    Raises:
        DatasetValidationError: If required columns are missing or invalid
            category values are found.
    """
    # --- Structural validation ---
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DatasetValidationError(
            f"Dataset is missing required column(s): {missing_columns}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    invalid_categories = set(df["category"].unique()) - ALLOWED_CATEGORIES
    if invalid_categories:
        raise DatasetValidationError(
            f"Dataset contains invalid category value(s): {sorted(invalid_categories)}. "
            f"Allowed categories: {sorted(ALLOWED_CATEGORIES)}"
        )

    cleaned = df.copy()

    # --- Missing values ---
    missing_mask = cleaned[REQUIRED_COLUMNS].isnull().any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        if verbose:
            print(f"[data_preprocessor] Found {n_missing} row(s) with missing values — dropping them.")
        cleaned = cleaned[~missing_mask]

    # --- Duplicate questions (case-insensitive, whitespace-trimmed) ---
    normalized = cleaned["question"].astype(str).str.strip().str.lower()
    duplicate_mask = normalized.duplicated(keep="first")
    n_duplicates = int(duplicate_mask.sum())
    if n_duplicates > 0:
        if verbose:
            print(f"[data_preprocessor] Found {n_duplicates} duplicate question(s) — keeping first occurrence, dropping the rest.")
        cleaned = cleaned[~duplicate_mask]

    # --- Light text cleanup: strip stray whitespace ---
    cleaned["question"] = cleaned["question"].astype(str).str.strip()
    cleaned["crop"] = cleaned["crop"].astype(str).str.strip()
    cleaned["category"] = cleaned["category"].astype(str).str.strip()

    cleaned = cleaned.reset_index(drop=True)

    if verbose:
        print(f"[data_preprocessor] Clean dataset ready: {len(cleaned)} rows "
              f"(from {len(df)} raw rows).")

    return cleaned


# ---------------------------------------------------------------------------
# 3. Split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    dev_ratio: float = DEV_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
):
    """
    Perform a stratified split of the cleaned dataset into
    Development, Validation, and Test sets.

    Stratification is done on the 'category' column so that all four
    categories are proportionally represented in every split.

    Args:
        df: A cleaned DataFrame (output of clean_dataset()).
        dev_ratio: Fraction of data for the development/prompt set (default 0.70).
        val_ratio: Fraction of data for the validation set (default 0.15).
        test_ratio: Fraction of data for the test set (default 0.15).
        random_seed: Seed for reproducibility (default 42).

    Returns:
        (development_df, validation_df, test_df) — three DataFrames.

    Raises:
        ValueError: If the ratios do not sum to 1.0 (within floating tolerance).
    """
    total_ratio = dev_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"dev_ratio + val_ratio + test_ratio must sum to 1.0, got {total_ratio}"
        )

    labels = df["category"]

    # Step 1: carve off the test set first (stratified).
    dev_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed,
    )

    # Step 2: split the remainder into development and validation.
    # val_ratio is expressed as a fraction of the ORIGINAL dataset, so we
    # convert it to a fraction of the remaining (dev+val) data.
    remaining_ratio = dev_ratio + val_ratio
    val_fraction_of_remaining = val_ratio / remaining_ratio

    dev_df, val_df = train_test_split(
        dev_val_df,
        test_size=val_fraction_of_remaining,
        stratify=dev_val_df["category"],
        random_state=random_seed,
    )

    dev_df = dev_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return dev_df, val_df, test_df


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------

def save_splits(
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dev_path: str = DEFAULT_DEV_PATH,
    val_path: str = DEFAULT_VAL_PATH,
    test_path: str = DEFAULT_TEST_PATH,
) -> None:
    """Write the three split DataFrames to their respective CSV files."""
    dev_df.to_csv(dev_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    input_path: str = DEFAULT_INPUT_PATH,
    dev_path: str = DEFAULT_DEV_PATH,
    val_path: str = DEFAULT_VAL_PATH,
    test_path: str = DEFAULT_TEST_PATH,
    random_seed: int = RANDOM_SEED,
    verbose: bool = True,
):
    """
    Convenience function that runs the full pipeline:
    load -> clean -> split -> save, and returns the three split DataFrames.
    """
    raw_df = load_dataset(input_path)
    clean_df = clean_dataset(raw_df, verbose=verbose)
    dev_df, val_df, test_df = split_dataset(clean_df, random_seed=random_seed)
    save_splits(dev_df, val_df, test_df, dev_path, val_path, test_path)
    return dev_df, val_df, test_df


# ---------------------------------------------------------------------------
# Script entry point: prints a summary report.
# ---------------------------------------------------------------------------

def _print_report(clean_df, dev_df, val_df, test_df):
    print("=" * 60)
    print("DATASET PREPROCESSING & SPLIT REPORT")
    print("=" * 60)
    print(f"Total records (after cleaning): {len(clean_df)}")
    print(f"Development records: {len(dev_df)} ({len(dev_df) / len(clean_df):.1%})")
    print(f"Validation records:  {len(val_df)} ({len(val_df) / len(clean_df):.1%})")
    print(f"Test records:        {len(test_df)} ({len(test_df) / len(clean_df):.1%})")
    print("-" * 60)
    print("Records per category — Development set:")
    print(dev_df["category"].value_counts().to_string())
    print("-" * 60)
    print("Records per category — Validation set:")
    print(val_df["category"].value_counts().to_string())
    print("-" * 60)
    print("Records per category — Test set:")
    print(test_df["category"].value_counts().to_string())
    print("-" * 60)
    print("Records per category — Full cleaned dataset:")
    print(clean_df["category"].value_counts().to_string())
    print("=" * 60)
    print(f"Random seed used: {RANDOM_SEED} (reproducible split)")
    print("NOTE: The Test Set must NEVER be used for few-shot examples")
    print("      or any other form of prompt/model development.")
    print("=" * 60)


if __name__ == "__main__":
    raw = load_dataset()
    clean = clean_dataset(raw)
    development, validation, test = split_dataset(clean)
    save_splits(development, validation, test)
    _print_report(clean, development, validation, test)
