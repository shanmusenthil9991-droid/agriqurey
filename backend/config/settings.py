"""
Centralized application configuration.

Loads values from environment variables (via python-dotenv) so that
sensitive data such as API keys never live in source code.
"""

import os
from dotenv import load_dotenv

# Project root = two levels up from backend/config/settings.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load variables from a .env file if present (no-op in production
# environments where real env vars are already set).
load_dotenv()


class Config:
    """Base configuration shared by all environments."""

    # --- Flask ---
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "True") == "True"
    HOST = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_RUN_PORT", 5000))

    # --- LLM provider (kept generic/configurable, not hardcoded) ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
    LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", 30))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 1))

    @staticmethod
    def is_llm_configured() -> bool:
        """
        True when a usable API key is present.

        Lets the health endpoint and startup checks report readiness
        without ever exposing the key itself.
        """
        key = os.getenv("LLM_API_KEY", "").strip()
        return bool(key) and key != "your_api_key_here"

    # --- Dataset ---
    # Default must match the file that actually ships in data/. The old
    # default ("data/farmer_questions.csv") pointed at a non-existent file,
    # so anything honouring this setting would fail with FileNotFoundError.
    # Relative paths are resolved against the project root, not the current
    # working directory, so the value works no matter where you run from.
    DATASET_PATH = os.getenv("DATASET_PATH", "data/crop_questions.csv")

    @staticmethod
    def resolve_dataset_path() -> str:
        """Return DATASET_PATH as an absolute path anchored at the project root."""
        raw = os.getenv("DATASET_PATH", "data/crop_questions.csv")
        if os.path.isabs(raw):
            return raw
        return os.path.join(PROJECT_ROOT, raw)


config = Config()
