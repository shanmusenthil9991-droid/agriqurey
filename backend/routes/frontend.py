"""
Frontend page serving.

The browser JavaScript calls the API with SAME-ORIGIN paths
(static/js/classifier.js -> "/api/classify", static/js/dashboard.js ->
"/api/evaluation"). That only resolves if the HTML is served by this
Flask app. Opening frontend/index.html directly from the filesystem
makes the browser resolve "/api/classify" against the file:// origin,
which can never reach the backend — so the whole
frontend -> API -> LLM -> result chain silently breaks.

This blueprint closes that gap: Flask serves the four HTML pages, and
create_app() points Flask's static folder at the project-level static/
directory so "../static/css/style.css" resolves to "/static/css/style.css".

Only an explicit allow-list of filenames is served. A generic
send_from_directory on user input would be a path-traversal risk
(e.g. "/../.env"); an allow-list makes that structurally impossible.
"""

import os

from flask import Blueprint, send_from_directory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Allow-list: only these pages may be served. Nothing else on disk is reachable.
ALLOWED_PAGES = {
    "index.html",
    "ask.html",
    "result.html",
    "history.html",
    "admin.html",
}

frontend_bp = Blueprint("frontend", __name__)


def _send_page(filename: str):
    return send_from_directory(FRONTEND_DIR, filename)


@frontend_bp.route("/", methods=["GET"])
def home():
    """Serve the landing page at the site root."""
    return _send_page("index.html")


@frontend_bp.route("/<page>.html", methods=["GET"])
def page(page: str):
    """
    Serve one of the allow-listed frontend pages.

    The in-page links are relative ("ask.html", "admin.html"), so they
    resolve to these routes directly and the whole app works from one
    origin with no absolute URLs to configure.
    """
    filename = f"{page}.html"
    if filename not in ALLOWED_PAGES:
        # Fall through to the app-level 404 JSON handler.
        from flask import abort
        abort(404)
    return _send_page(filename)
