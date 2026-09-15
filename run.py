"""
Entry point to start the Flask backend server.

Serves BOTH the API and the frontend pages on one origin, so the
browser's same-origin calls to /api/classify and /api/evaluation resolve
without any CORS configuration or hardcoded backend URL.

Usage:
    python run.py

Then open http://localhost:5000 in a browser.
"""

from backend.app import app
from backend.config.settings import config


def _print_startup_banner():
    """
    Print a short readiness report at startup.

    Deliberately prints the provider NAME and a boolean only — never the
    API key, any prefix of it, or its length.
    """
    provider = (config.LLM_PROVIDER or "").strip().lower() or "(unset)"
    configured = config.is_llm_configured()

    print("=" * 60)
    print("CropCare AI — crop disease question classifier")
    print("=" * 60)
    print(f"  URL           : http://localhost:{config.PORT}")
    print(f"  LLM provider  : {provider}")
    print(f"  API key set   : {'yes' if configured else 'no'}")

    if provider == "echo":
        print()
        print("  NOTE: the offline 'echo' stub is active. It returns a fixed")
        print("        canned response and does NOT classify. Set")
        print("        LLM_PROVIDER + LLM_API_KEY in .env for real results.")
    elif not configured:
        print()
        print("  WARNING: no LLM_API_KEY is set, so /api/classify will return")
        print("           a 503 configuration error. Copy .env.example to .env")
        print("           and set a real key.")
    print("=" * 60)


if __name__ == "__main__":
    _print_startup_banner()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
