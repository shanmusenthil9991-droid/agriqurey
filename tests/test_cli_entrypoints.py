"""
Regression tests for the documented command-line entrypoints.

Every command the README tells a user to run must either work or fail
with a clean, actionable message. None may crash with a raw traceback,
and none may fail with ModuleNotFoundError because of how Python sets
sys.path when a file is run directly rather than as a module.

These run each script in a real subprocess, because the bugs they guard
against only appear through the actual interpreter startup path — an
in-process import would not reproduce them.
"""

import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, env_overrides=None):
    """Run a command from the project root with a clean environment."""
    env = dict(os.environ)
    # Neutralise any developer .env so results do not depend on local setup.
    env.pop("LLM_API_KEY", None)
    env.pop("LLM_MODEL_NAME", None)
    env["LLM_PROVIDER"] = "anthropic"
    if env_overrides:
        env.update(env_overrides)

    return subprocess.run(
        [sys.executable] + args,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# Scripts that must SUCCEED with no credentials
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", [
    "backend/data_loader.py",
    "backend/data_preprocessor.py",
    "backend/services/prompt_builder.py",
])
def test_offline_scripts_run_as_direct_scripts(script):
    """
    Guards the ModuleNotFoundError class of bug: running a file directly
    puts its own directory on sys.path instead of the project root, so
    'import backend.x' fails unless the script bootstraps sys.path.
    """
    result = _run([script])
    assert result.returncode == 0, (
        f"`python {script}` failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("module", [
    "backend.data_preprocessor",
    "backend.services.prompt_builder",
])
def test_offline_scripts_run_as_modules(module):
    """The same files must also work via `python -m`."""
    result = _run(["-m", module])
    assert result.returncode == 0, (
        f"`python -m {module}` failed:\n{result.stdout}\n{result.stderr}"
    )


def test_preprocessor_split_is_reproducible():
    """Re-running the split must not change the row counts (fixed seed)."""
    import pandas as pd

    before = {
        name: len(pd.read_csv(os.path.join(PROJECT_ROOT, "data", f"{name}.csv")))
        for name in ("development", "validation", "test")
    }
    assert _run(["backend/data_preprocessor.py"]).returncode == 0
    after = {
        name: len(pd.read_csv(os.path.join(PROJECT_ROOT, "data", f"{name}.csv")))
        for name in ("development", "validation", "test")
    }
    assert before == after


# ---------------------------------------------------------------------------
# Scripts that must FAIL CLEANLY without credentials
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "backend.evaluation.evaluate_classifier",
    "backend.evaluation.compare_methods",
])
def test_evaluation_scripts_fail_cleanly_without_api_key(module):
    """
    Missing credentials is a configuration problem, not a crash. The user
    must get an actionable message, never a stack trace.
    """
    result = _run(["-m", module])
    combined = result.stdout + result.stderr

    assert result.returncode == 1
    assert "Traceback" not in combined, f"raw traceback leaked:\n{combined}"
    assert "ABORTED" in combined
    assert "LLM_API_KEY" in combined


@pytest.mark.parametrize("module", [
    "backend.evaluation.evaluate_classifier",
    "backend.evaluation.compare_methods",
])
def test_evaluation_scripts_refuse_the_offline_stub(module):
    """The echo stub must never be able to produce evaluation numbers."""
    result = _run(["-m", module], {"LLM_PROVIDER": "echo"})
    combined = result.stdout + result.stderr

    assert result.returncode == 1
    assert "Traceback" not in combined
    assert "echo" in combined


def test_evaluation_scripts_never_print_the_api_key():
    """A failure message must not echo the credential back to the console."""
    secret = "sk-must-never-be-printed-123456"
    result = _run(
        ["-m", "backend.evaluation.evaluate_classifier"],
        {"LLM_API_KEY": secret, "LLM_MODEL_NAME": ""},
    )
    assert secret not in (result.stdout + result.stderr)


def test_error_analysis_reports_missing_predictions_cleanly(tmp_path):
    """
    error_analysis depends on an artifact from a previous step. When it is
    absent the script must say so and point at the command that creates
    it, rather than raising.
    """
    predictions = os.path.join(PROJECT_ROOT, "evaluation", "predictions.csv")
    if os.path.exists(predictions):
        pytest.skip("predictions.csv exists; this covers the absent case only")

    result = _run(["-m", "backend.evaluation.error_analysis"])
    combined = result.stdout + result.stderr

    assert result.returncode == 1
    assert "Traceback" not in combined
    assert "evaluate_classifier" in combined


# ---------------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------------

def test_app_imports_and_exposes_every_route():
    """`python run.py` must import cleanly and wire up all endpoints."""
    result = _run([
        "-c",
        "import run; "
        "rules = sorted(str(r) for r in run.app.url_map.iter_rules()); "
        "print('\\n'.join(rules))",
    ])
    assert result.returncode == 0, result.stderr
    for route in ("/api/health", "/api/classify", "/api/evaluation", "/static/"):
        assert route in result.stdout


def test_startup_banner_never_prints_the_api_key():
    secret = "sk-banner-must-not-leak-9876"
    result = _run(
        ["-c", "import run; run._print_startup_banner()"],
        {"LLM_API_KEY": secret},
    )
    assert result.returncode == 0, result.stderr
    assert secret not in result.stdout
    assert "API key set" in result.stdout
