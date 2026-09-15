"""
Dataset loading and validation utilities.

Responsible ONLY for reading data/crop_questions.csv and returning a
clean, validated Pandas DataFrame. This module does not do any
classification — it is purely a data-access layer used by later
components (LLM classification, evaluation, etc.).
"""

import os
import pandas as pd

REQUIRED_COLUMNS = ["id", "question", "crop", "category"]

ALLOWED_CATEGORIES = {
    "Symptoms",
    "Prevention",
    "Management",
    "General Information",
}


class DatasetValidationError(Exception):
    """Raised when the dataset fails a structural or content validation check."""


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure all required columns are present in the dataframe."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DatasetValidationError(
            f"Dataset is missing required column(s): {missing_columns}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )


def _validate_categories(df: pd.DataFrame) -> None:
    """Ensure every row's category is one of the four allowed labels."""
    invalid_categories = set(df["category"].unique()) - ALLOWED_CATEGORIES
    if invalid_categories:
        raise DatasetValidationError(
            f"Dataset contains invalid category value(s): {sorted(invalid_categories)}. "
            f"Allowed categories: {sorted(ALLOWED_CATEGORIES)}"
        )


def _detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of rows with missing values in required columns.
    Does not raise — missing values are reported so the caller can decide
    how to handle them (e.g. drop, log, or fail the pipeline).
    """
    missing_mask = df[REQUIRED_COLUMNS].isnull().any(axis=1)
    return df[missing_mask]


def _detect_duplicate_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows whose 'question' text is a duplicate (case-insensitive,
    whitespace-trimmed) of an earlier row. The first occurrence is kept;
    later occurrences are flagged as duplicates.
    """
    normalized = df["question"].astype(str).str.strip().str.lower()
    duplicate_mask = normalized.duplicated(keep="first")
    return df[duplicate_mask]


def load_dataset(csv_path: str, drop_invalid_rows: bool = True) -> pd.DataFrame:
    """
    Load, validate, and clean the labeled farmer-question dataset.

    Args:
        csv_path: Path to the CSV file (e.g. "data/crop_questions.csv").
        drop_invalid_rows: If True (default), rows with missing values or
            duplicate questions are dropped from the returned dataframe.
            If False, they are left in place but still reported via the
            printed/logged warnings.

    Returns:
        A cleaned Pandas DataFrame with columns: id, question, crop, category.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        DatasetValidationError: If required columns are missing or the
            dataset contains category values outside the allowed set.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- Structural validation (hard failures) ---
    _validate_columns(df)
    _validate_categories(df)

    # --- Content quality checks (soft — reported, optionally cleaned) ---
    missing_rows = _detect_missing_values(df)
    duplicate_rows = _detect_duplicate_questions(df)

    if not missing_rows.empty:
        print(f"[data_loader] Warning: {len(missing_rows)} row(s) with missing values detected.")

    if not duplicate_rows.empty:
        print(f"[data_loader] Warning: {len(duplicate_rows)} duplicate question(s) detected.")

    if drop_invalid_rows:
        if not missing_rows.empty:
            df = df.drop(index=missing_rows.index)
        if not duplicate_rows.empty:
            # Recompute duplicates against the (possibly already-cleaned) df
            # to keep index alignment safe after the missing-value drop.
            duplicate_rows = _detect_duplicate_questions(df)
            df = df.drop(index=duplicate_rows.index)

    df = df.reset_index(drop=True)
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return basic summary statistics about a loaded dataset — useful for
    debugging, logging, and later for the evaluation module.
    """
    return {
        "total_rows": len(df),
        "category_counts": df["category"].value_counts().to_dict(),
        "crop_counts": df["crop"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    # Simple manual smoke check: `python backend/data_loader.py`
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "crop_questions.csv",
    )
    dataframe = load_dataset(default_path)
    print(dataframe.head())
    print(get_dataset_summary(dataframe))
