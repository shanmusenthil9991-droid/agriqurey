"""
Prompt construction for the crop-disease farmer-question classifier.

This module is responsible ONLY for building the text prompt that will
later be sent to an LLM. It does NOT call any LLM API — that is added in
a later step. Keeping prompt construction separate from the API call
makes the prompt easy to test, version, and reuse (e.g. from the
Flask route, from evaluation scripts, or from a notebook).
"""

import os
import random
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPT_TEMPLATE_PATH = os.path.join(
    PROJECT_ROOT, "backend", "prompts", "classification_prompt.txt"
)
DEFAULT_DEVELOPMENT_PATH = os.path.join(PROJECT_ROOT, "data", "development.csv")

ALLOWED_CATEGORIES = [
    "Symptoms",
    "Prevention",
    "Management",
    "General Information",
]

# The classifier may also emit this label for genuinely unrelated/garbled
# questions. It is intentionally excluded from ALLOWED_CATEGORIES because
# it is a fallback, not one of the four information-need categories being
# measured.
FALLBACK_CATEGORY = "Unable to Classify"

# Few-shot sizing. Two per category (8 total) is the default: enough to
# demonstrate each label and the exact JSON shape, small enough to keep the
# prompt light for a low-latency application.
MIN_EXAMPLES_PER_CATEGORY = 2
MAX_EXAMPLES_PER_CATEGORY = 4
DEFAULT_EXAMPLES_PER_CATEGORY = 2

# Filenames that must never be used as a few-shot source.
FORBIDDEN_SOURCE_FILENAMES = {"test.csv", "validation.csv"}

# Default "reason" text used for a few-shot example that does not carry its
# own. These are deliberately written in the same voice we want the model to
# use at inference time: a short sentence describing the INFORMATION NEED,
# never a diagnosis. Generic filler like "matches the X category" would teach
# the model to emit equally uninformative reasons.
DEFAULT_EXAMPLE_REASONS = {
    "Symptoms": "The question asks about visible signs of crop disease.",
    "Prevention": "The question asks how to prevent disease before it occurs.",
    "Management": "The question asks how to control an existing disease problem.",
    "General Information": "The question asks for background knowledge about a crop disease.",
}


class PromptBuildError(Exception):
    """Raised when a classification prompt cannot be built correctly."""


class DataLeakageError(Exception):
    """Raised when few-shot selection is pointed at a held-out split."""


def _load_prompt_template(template_path: str = PROMPT_TEMPLATE_PATH) -> str:
    """Read the raw prompt template from disk."""
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template not found at: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def format_few_shot_examples(examples: list) -> str:
    """
    Format a list of labeled example dicts into the few-shot block used
    inside the prompt template.

    Each example dict is expected to have the keys:
        - "question" (str)
        - "crop" (str)
        - "category" (str) — one of ALLOWED_CATEGORIES

    Args:
        examples: List of example dicts. May be empty (zero-shot prompt).

    Returns:
        A formatted string block, one example per entry, e.g.:

            Example 1:
            Crop: Rice
            Question: "What are the symptoms of rice blast?"
            Output: {"category": "Symptoms", "confidence": 0.95, "reason": "..."}

    Raises:
        PromptBuildError: If an example is missing required keys or uses
            a category outside the four allowed labels (few-shot examples
            must always be genuine, unambiguous demonstrations — never
            "Unable to Classify" examples, since that would encourage the
            model to overuse the fallback).
    """
    if not examples:
        return "(No examples provided — classify using the definitions and rules above only.)"

    lines = []
    for i, example in enumerate(examples, start=1):
        missing = [k for k in ("question", "crop", "category") if k not in example]
        if missing:
            raise PromptBuildError(
                f"Few-shot example #{i} is missing required key(s): {missing}"
            )
        if example["category"] not in ALLOWED_CATEGORIES:
            raise PromptBuildError(
                f"Few-shot example #{i} has invalid category '{example['category']}'. "
                f"Few-shot examples must use one of: {ALLOWED_CATEGORIES} "
                f"('{FALLBACK_CATEGORY}' is not allowed as a few-shot example)."
            )

        reason = example.get("reason") or DEFAULT_EXAMPLE_REASONS[example["category"]]
        confidence = example.get("confidence", 0.95)

        example_json = (
            '{"category": "%s", "confidence": %s, "reason": "%s"}'
            % (example["category"], confidence, reason)
        )
        lines.append(
            f'Example {i}:\n'
            f'Crop: {example["crop"]}\n'
            f'Question: "{example["question"]}"\n'
            f'Output: {example_json}'
        )

    return "\n\n".join(lines)


def build_classification_prompt(question: str, crop: str, examples: list = None) -> str:
    """
    Build the full classification prompt for a single farmer question.

    Args:
        question: The farmer's raw question text.
        crop: The crop the question is about (may be an empty string if
            unknown — the model can still classify from the question text).
        examples: Optional list of few-shot example dicts (see
            format_few_shot_examples()). Defaults to no examples.

    Returns:
        A complete prompt string ready to send to an LLM, containing:
            1. System role
            2. Category definitions
            3. Classification rules
            4. Few-shot examples
            5. The farmer's question (with crop)
            6. The required JSON output format

    Raises:
        ValueError: If question is empty/blank.
        PromptBuildError: If the few-shot examples are malformed.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    examples = examples or []
    template = _load_prompt_template()

    few_shot_block = format_few_shot_examples(examples)

    prompt = template.format(
        few_shot_examples=few_shot_block,
        crop=crop.strip() if crop else "Unknown",
        question=question.strip(),
    )
    return prompt


# ---------------------------------------------------------------------------
# Few-shot example selection (development set ONLY — never the test set)
# ---------------------------------------------------------------------------

def _guard_not_test_split(csv_path: str) -> None:
    """
    Hard guardrail: refuse to read any file that looks like the held-out
    test split. The test set exists purely to produce an unbiased final
    score; using it to pick few-shot examples would leak it into prompt
    design and silently inflate that score. Enforced at runtime rather
    than left to convention so a stray argument cannot bypass it.
    """
    filename = os.path.basename(os.path.abspath(csv_path)).lower()
    if filename in FORBIDDEN_SOURCE_FILENAMES:
        raise DataLeakageError(
            f"Refusing to select few-shot examples from '{filename}'. "
            f"Few-shot examples must come from data/development.csv only. "
            f"The test split is reserved for final evaluation."
        )


def select_few_shot_examples(
    development_csv_path: str = DEFAULT_DEVELOPMENT_PATH,
    examples_per_category: int = DEFAULT_EXAMPLES_PER_CATEGORY,
    random_seed: int = 42,
    balanced: bool = True,
) -> list:
    """
    Select a small, balanced set of few-shot examples from the
    DEVELOPMENT split only.

    Selection strategy (deliberately simple and explainable):

      1. Read data/development.csv and keep only the four allowed categories.
      2. For each category, draw `examples_per_category` rows at random
         using a fixed seed, so the same seed always yields the same
         examples (reproducible prompts and reproducible evaluation runs).
      3. If `balanced` is True, every category contributes the SAME number
         of examples. When one category has fewer rows available than
         requested, the per-category count is reduced for ALL categories
         so the prompt never over-represents one label.
      4. Shuffle the final list so the categories are not presented in a
         fixed block order — otherwise the model can pick up on position
         ("the last example is always Symptoms") instead of on meaning.

    Only a handful of rows are ever returned; the full dataset is never
    embedded in a prompt.

    Args:
        development_csv_path: Path to data/development.csv.
        examples_per_category: Examples to draw per category (2-4).
            Default 2 -> 8 total examples.
        random_seed: Seed for reproducible selection.
        balanced: If True (default), force an equal count per category.

    Returns:
        A list of example dicts: {"question", "crop", "category"}.

    Raises:
        FileNotFoundError: If the development CSV does not exist.
        DataLeakageError: If the path points at the held-out test split.
        ValueError: If examples_per_category is outside the supported range.
    """
    if not (MIN_EXAMPLES_PER_CATEGORY <= examples_per_category <= MAX_EXAMPLES_PER_CATEGORY):
        raise ValueError(
            f"examples_per_category must be between {MIN_EXAMPLES_PER_CATEGORY} "
            f"and {MAX_EXAMPLES_PER_CATEGORY} (got {examples_per_category}). "
            f"Too few examples gives the model no signal; too many bloat the "
            f"prompt without improving accuracy."
        )

    _guard_not_test_split(development_csv_path)

    if not os.path.exists(development_csv_path):
        raise FileNotFoundError(
            f"Development dataset not found at: {development_csv_path}. "
            f"Run backend/data_preprocessor.py first to generate the splits."
        )

    dev_df = pd.read_csv(development_csv_path)

    missing_columns = [c for c in ("question", "crop", "category") if c not in dev_df.columns]
    if missing_columns:
        raise PromptBuildError(
            f"Development dataset is missing required column(s): {missing_columns}"
        )

    # Keep only rows whose label is one of the four real categories.
    dev_df = dev_df[dev_df["category"].isin(ALLOWED_CATEGORIES)]

    available = {
        category: dev_df[dev_df["category"] == category]
        for category in ALLOWED_CATEGORIES
    }
    non_empty = {c: rows for c, rows in available.items() if not rows.empty}

    if not non_empty:
        raise PromptBuildError(
            "Development dataset contains no rows in any of the allowed categories."
        )

    if balanced:
        # Equal representation: every category gives the same count.
        per_category = min(examples_per_category, min(len(r) for r in non_empty.values()))
    else:
        per_category = examples_per_category

    selected = []
    for category, rows in non_empty.items():
        take = min(per_category, len(rows))
        if take <= 0:
            continue
        sampled = rows.sample(n=take, random_state=random_seed)
        for _, row in sampled.iterrows():
            selected.append({
                "question": str(row["question"]).strip(),
                "crop": str(row["crop"]).strip(),
                "category": str(row["category"]).strip(),
            })

    # Shuffle so category order does not become a positional cue.
    random.Random(random_seed).shuffle(selected)
    return selected


def describe_selection(examples: list) -> dict:
    """
    Return a {category: count} summary of a selected example set.

    Useful for logging/debugging prompt composition and for asserting
    balance in tests without re-deriving the counts by hand.
    """
    counts = {category: 0 for category in ALLOWED_CATEGORIES}
    for example in examples:
        category = example.get("category")
        if category in counts:
            counts[category] += 1
    return counts


if __name__ == "__main__":
    # Manual smoke check: build one example prompt and print it.
    demo_examples = select_few_shot_examples()
    print("Selected example counts:", describe_selection(demo_examples))
    demo_prompt = build_classification_prompt(
        question="What are the symptoms of rice blast?",
        crop="Rice",
        examples=demo_examples,
    )
    print(demo_prompt)
