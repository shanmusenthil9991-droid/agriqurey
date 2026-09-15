"""
Tests for backend/services/prompt_builder.py

Covers:
- Prompt structure (system role, definitions, rules, examples, question, output format)
- Few-shot example formatting and validation
- Error handling (empty question, malformed examples, invalid categories)
- Few-shot example selection from the development set only
"""

import os
import pandas as pd
import pytest

from backend.services.prompt_builder import (
    build_classification_prompt,
    format_few_shot_examples,
    select_few_shot_examples,
    PromptBuildError,
    describe_selection,
    ALLOWED_CATEGORIES,
    FALLBACK_CATEGORY,
    DEFAULT_DEVELOPMENT_PATH,
    DEFAULT_EXAMPLES_PER_CATEGORY,
    MAX_EXAMPLES_PER_CATEGORY,
    DataLeakageError,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# build_classification_prompt() — structure
# ---------------------------------------------------------------------------

def test_prompt_contains_all_required_sections():
    prompt = build_classification_prompt(
        question="What are the symptoms of rice blast?",
        crop="Rice",
        examples=[],
    )
    # 1. System role
    assert "classifier" in prompt.lower()
    # 2. Category definitions
    for category in ["SYMPTOMS", "PREVENTION", "MANAGEMENT", "GENERAL INFORMATION"]:
        assert category in prompt
    # 3. Classification rules
    assert "CLASSIFICATION RULES" in prompt
    # 4. Few-shot examples section header (even if empty)
    assert "FEW-SHOT EXAMPLES" in prompt
    # 5. Farmer question
    assert "What are the symptoms of rice blast?" in prompt
    assert "Rice" in prompt
    # 6. Required JSON output format
    assert "REQUIRED OUTPUT FORMAT" in prompt
    assert '"category"' in prompt
    assert '"confidence"' in prompt
    assert '"reason"' in prompt


def test_prompt_never_asks_for_diagnosis():
    prompt = build_classification_prompt("What is bacterial leaf blight?", "Rice", [])
    assert "must never diagnose" in prompt.lower() or "not diagnose" in prompt.lower()


def test_prompt_supports_unable_to_classify_fallback():
    prompt = build_classification_prompt("What's the weather like today?", "Rice", [])
    assert FALLBACK_CATEGORY in prompt


def test_prompt_with_empty_crop_defaults_to_unknown():
    prompt = build_classification_prompt("What causes this disease?", "", [])
    assert "Crop: Unknown" in prompt


def test_prompt_raises_on_empty_question():
    with pytest.raises(ValueError):
        build_classification_prompt("", "Rice", [])


def test_prompt_raises_on_whitespace_only_question():
    with pytest.raises(ValueError):
        build_classification_prompt("   ", "Rice", [])


# ---------------------------------------------------------------------------
# format_few_shot_examples()
# ---------------------------------------------------------------------------

def test_format_few_shot_examples_empty_list():
    result = format_few_shot_examples([])
    assert "No examples provided" in result


def test_format_few_shot_examples_valid_examples():
    examples = [
        {"question": "What are the symptoms of rice blast?", "crop": "Rice", "category": "Symptoms"},
        {"question": "How can I prevent late blight in potato?", "crop": "Potato", "category": "Prevention"},
    ]
    result = format_few_shot_examples(examples)
    assert "Example 1:" in result
    assert "Example 2:" in result
    assert "Symptoms" in result
    assert "Prevention" in result
    # JSON braces should not be double-escaped
    assert "{{" not in result
    assert "}}" not in result


def test_format_few_shot_examples_missing_key_raises():
    with pytest.raises(PromptBuildError):
        format_few_shot_examples([{"question": "x?", "crop": "Rice"}])  # missing category


def test_format_few_shot_examples_invalid_category_raises():
    with pytest.raises(PromptBuildError):
        format_few_shot_examples([{"question": "x?", "crop": "Rice", "category": "Diagnosis"}])


def test_format_few_shot_examples_rejects_fallback_category():
    """Few-shot examples must never demonstrate the fallback category."""
    with pytest.raises(PromptBuildError):
        format_few_shot_examples([
            {"question": "x?", "crop": "Rice", "category": FALLBACK_CATEGORY}
        ])


# ---------------------------------------------------------------------------
# select_few_shot_examples() — must only ever use the development set
# ---------------------------------------------------------------------------

DEV_PATH = os.path.join(PROJECT_ROOT, "data", "development.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "test.csv")


def test_default_path_is_development_csv():
    assert DEFAULT_DEVELOPMENT_PATH.endswith("development.csv")
    assert os.path.basename(DEFAULT_DEVELOPMENT_PATH) != "test.csv"


def test_default_selection_is_two_per_category():
    examples = select_few_shot_examples(development_csv_path=DEV_PATH)
    counts = describe_selection(examples)
    assert set(counts.values()) == {DEFAULT_EXAMPLES_PER_CATEGORY}
    assert len(examples) == DEFAULT_EXAMPLES_PER_CATEGORY * len(ALLOWED_CATEGORIES)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_selection_is_balanced_across_categories(n):
    """Every category must contribute exactly the same number of examples."""
    examples = select_few_shot_examples(development_csv_path=DEV_PATH, examples_per_category=n)
    counts = describe_selection(examples)
    assert set(counts.keys()) == set(ALLOWED_CATEGORIES)
    assert set(counts.values()) == {n}, f"Unbalanced selection: {counts}"
    assert len(examples) == n * len(ALLOWED_CATEGORIES)


@pytest.mark.parametrize("n", [0, 1, 5, 10, -1])
def test_selection_rejects_counts_outside_supported_range(n):
    with pytest.raises(ValueError):
        select_few_shot_examples(development_csv_path=DEV_PATH, examples_per_category=n)


def test_selection_never_returns_whole_dataset():
    """Guard against dumping the full development set into the prompt."""
    dev_df = pd.read_csv(DEV_PATH)
    examples = select_few_shot_examples(
        development_csv_path=DEV_PATH,
        examples_per_category=MAX_EXAMPLES_PER_CATEGORY,
    )
    assert len(examples) == MAX_EXAMPLES_PER_CATEGORY * len(ALLOWED_CATEGORIES)
    assert len(examples) < len(dev_df) / 4


def test_selection_is_reproducible_with_same_seed():
    a = select_few_shot_examples(development_csv_path=DEV_PATH, random_seed=42)
    b = select_few_shot_examples(development_csv_path=DEV_PATH, random_seed=42)
    assert [e["question"] for e in a] == [e["question"] for e in b]


def test_different_seeds_select_different_examples():
    a = select_few_shot_examples(development_csv_path=DEV_PATH, random_seed=1)
    b = select_few_shot_examples(development_csv_path=DEV_PATH, random_seed=99)
    assert sorted(e["question"] for e in a) != sorted(e["question"] for e in b)


def test_selected_examples_have_required_keys_and_valid_categories():
    examples = select_few_shot_examples(development_csv_path=DEV_PATH)
    for example in examples:
        assert set(example.keys()) == {"question", "crop", "category"}
        assert example["category"] in ALLOWED_CATEGORIES
        assert example["category"] != FALLBACK_CATEGORY
        assert example["question"].strip()


def test_selected_examples_are_unique():
    examples = select_few_shot_examples(
        development_csv_path=DEV_PATH, examples_per_category=MAX_EXAMPLES_PER_CATEGORY
    )
    questions = [e["question"] for e in examples]
    assert len(questions) == len(set(questions))


def test_selection_is_not_grouped_by_category():
    """
    Examples must be shuffled. If they arrive in fixed category blocks the
    model can learn position instead of meaning.
    """
    examples = select_few_shot_examples(
        development_csv_path=DEV_PATH, examples_per_category=MAX_EXAMPLES_PER_CATEGORY
    )
    ordered = [e["category"] for e in examples]
    grouped = sorted(ordered, key=ALLOWED_CATEGORIES.index)
    assert ordered != grouped


def test_selection_refuses_the_test_split():
    """The held-out test set must never be a few-shot source."""
    with pytest.raises(DataLeakageError):
        select_few_shot_examples(development_csv_path=TEST_PATH)


def test_selection_refuses_the_validation_split():
    val_path = os.path.join(PROJECT_ROOT, "data", "validation.csv")
    with pytest.raises(DataLeakageError):
        select_few_shot_examples(development_csv_path=val_path)


def test_no_selected_example_appears_in_the_test_set():
    """
    Strongest leakage check: no question handed to the model as a
    demonstration may also exist in the held-out test split.
    """
    examples = select_few_shot_examples(
        development_csv_path=DEV_PATH, examples_per_category=MAX_EXAMPLES_PER_CATEGORY
    )
    test_questions = set(pd.read_csv(TEST_PATH)["question"].str.strip())
    for example in examples:
        assert example["question"].strip() not in test_questions


def test_selection_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        select_few_shot_examples(development_csv_path="data/does_not_exist.csv")


def test_selection_handles_missing_category_gracefully(tmp_path):
    """A category absent from the source must not break selection."""
    csv_path = tmp_path / "development.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4],
        "question": ["a?", "b?", "c?", "d?"],
        "crop": ["Rice"] * 4,
        "category": ["Symptoms", "Symptoms", "Prevention", "Prevention"],
    }).to_csv(csv_path, index=False)

    examples = select_few_shot_examples(development_csv_path=str(csv_path), examples_per_category=2)
    counts = describe_selection(examples)
    assert counts["Symptoms"] == 2
    assert counts["Prevention"] == 2
    assert counts["Management"] == 0


def test_balance_reduces_count_when_a_category_is_short(tmp_path):
    """
    With balanced=True, a thin category caps every category so no single
    label is over-represented.
    """
    csv_path = tmp_path / "development.csv"
    rows = []
    for category, n in [("Symptoms", 4), ("Prevention", 4), ("Management", 4), ("General Information", 1)]:
        for i in range(n):
            rows.append({"id": len(rows), "question": f"{category} q{i}?", "crop": "Rice", "category": category})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    examples = select_few_shot_examples(
        development_csv_path=str(csv_path), examples_per_category=4, balanced=True
    )
    assert set(describe_selection(examples).values()) == {1}


def test_unbalanced_mode_takes_what_is_available(tmp_path):
    csv_path = tmp_path / "development.csv"
    rows = []
    for category, n in [("Symptoms", 4), ("Prevention", 4), ("Management", 4), ("General Information", 1)]:
        for i in range(n):
            rows.append({"id": len(rows), "question": f"{category} q{i}?", "crop": "Rice", "category": category})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    counts = describe_selection(select_few_shot_examples(
        development_csv_path=str(csv_path), examples_per_category=4, balanced=False
    ))
    assert counts["Symptoms"] == 4
    assert counts["General Information"] == 1


def test_describe_selection_counts_all_categories():
    counts = describe_selection([{"question": "q", "crop": "Rice", "category": "Symptoms"}])
    assert counts["Symptoms"] == 1
    assert counts["Management"] == 0


# ---------------------------------------------------------------------------
# Prompt size — must stay small enough for a lightweight application
# ---------------------------------------------------------------------------

def test_prompt_stays_small_with_max_examples():
    examples = select_few_shot_examples(
        development_csv_path=DEV_PATH, examples_per_category=MAX_EXAMPLES_PER_CATEGORY
    )
    prompt = build_classification_prompt("Why are my leaves spotted?", "Tomato", examples)
    assert len(prompt) < 8000, "Prompt grew too large for a lightweight app"


def test_more_examples_produce_a_longer_prompt():
    small = build_classification_prompt(
        "q?", "Rice", select_few_shot_examples(development_csv_path=DEV_PATH, examples_per_category=2)
    )
    large = build_classification_prompt(
        "q?", "Rice", select_few_shot_examples(development_csv_path=DEV_PATH, examples_per_category=4)
    )
    assert len(large) > len(small)


def test_every_selected_example_appears_in_the_prompt():
    examples = select_few_shot_examples(development_csv_path=DEV_PATH)
    prompt = build_classification_prompt("Why are my leaves spotted?", "Tomato", examples)
    for example in examples:
        assert example["question"] in prompt
        assert example["category"] in prompt


# ---------------------------------------------------------------------------
# End-to-end: build a full prompt using real selected examples
# ---------------------------------------------------------------------------

def test_end_to_end_prompt_with_real_development_examples():
    dev_path = os.path.join(PROJECT_ROOT, "data", "development.csv")
    examples = select_few_shot_examples(development_csv_path=dev_path)
    prompt = build_classification_prompt(
        question="My tomato leaves are turning brown and curling",
        crop="Tomato",
        examples=examples,
    )
    assert "My tomato leaves are turning brown and curling" in prompt
    assert "Example 1:" in prompt
    assert "REQUIRED OUTPUT FORMAT" in prompt
