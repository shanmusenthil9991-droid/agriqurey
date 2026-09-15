"""
Health check endpoint.

Verifies that the Flask server is running and reports whether the
backend is actually READY to classify — i.e. whether an LLM provider is
configured and few-shot examples are available on disk.

SECURITY: this endpoint reports only booleans and non-sensitive names.
It never returns the API key, any part of it, or even its length. The
`llm_configured` flag comes from Config.is_llm_configured(), which
checks for a usable key and returns a plain bool.
"""

import os

from flask import Blueprint, jsonify

from backend.config.settings import config

health_bp = Blueprint("health", __name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVELOPMENT_CSV = os.path.join(PROJECT_ROOT, "data", "development.csv")


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """
    Liveness + readiness check for the API.

    Always returns 200 while the process is up. `status` is "ok" when the
    service can classify and "degraded" when it is running but not
    configured to reach an LLM — a distinction that makes a
    misconfigured deployment visible instead of silent.
    """
    provider_name = (getattr(config, "LLM_PROVIDER", "") or "").strip().lower()
    llm_configured = config.is_llm_configured()

    # The offline "echo" stub needs no key, but it cannot really classify,
    # so it is reported as a stub rather than as a ready provider.
    is_stub = provider_name == "echo"
    ready = is_stub or llm_configured

    return jsonify({
        "status": "ok" if ready else "degraded",
        "service": "crop-disease-classifier-backend",
        "message": (
            "Flask server is running."
            if ready
            else "Flask server is running, but no LLM API key is configured. "
                 "Copy .env.example to .env and set LLM_API_KEY."
        ),
        # Names and booleans only — never the key itself.
        "llm_provider": provider_name or None,
        "llm_configured": llm_configured,
        "llm_is_offline_stub": is_stub,
        "few_shot_examples_available": os.path.exists(DEVELOPMENT_CSV),
    }), 200
