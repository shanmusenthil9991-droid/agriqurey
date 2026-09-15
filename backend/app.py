"""
Flask application factory.

This module wires together configuration and route blueprints.
Later steps may add additional blueprints for:
  - dataset management
"""

import os

from flask import Flask, jsonify

from backend.config.settings import config
from backend.routes.health import health_bp
from backend.routes.classify import classify_bp
from backend.routes.evaluation import evaluation_bp
from backend.routes.frontend import frontend_bp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")


def create_app():
    """Application factory: builds and configures the Flask app."""
    # Flask defaults its static folder to a "static" dir next to this
    # module (backend/static), which does not exist. The real assets live
    # at the project root in static/, and the HTML references them as
    # "../static/css/style.css" -> "/static/css/style.css". Pointing
    # static_folder here is what makes those requests resolve.
    app = Flask(
        __name__,
        static_folder=STATIC_DIR,
        static_url_path="/static",
    )
    app.config["DEBUG"] = config.DEBUG

    # Cap request body size. Without this, request.get_json() reads and
    # parses the ENTIRE body into memory before any application-level
    # validation (e.g. the 2000-char question limit) ever runs — so a
    # multi-hundred-megabyte POST would be fully buffered and decoded on
    # every request, a cheap memory-exhaustion vector. 64 KB is generously
    # larger than any legitimate {question, crop} payload can be.
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    # IMPORTANT: Flask's default behaviour is to re-raise unhandled
    # exceptions when DEBUG is True, bypassing @errorhandler so the
    # interactive debugger/traceback page can render. That would leak
    # internal stack traces to API callers. Disabling PROPAGATE_EXCEPTIONS
    # forces every unhandled error through our JSON error handlers
    # instead, in both debug and production.
    app.config["PROPAGATE_EXCEPTIONS"] = False

    # Allow the frontend (served separately) to call the API.
    # flask-cors is an optional dependency at this stage; the app
    # still runs without it (e.g. in offline/dev sandboxes), but it
    # is listed in requirements.txt and should be installed for
    # real cross-origin frontend/backend usage.
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        app.logger.warning(
            "flask-cors not installed; CORS is disabled. "
            "Run 'pip install -r requirements.txt' to enable it."
        )

    # Register blueprints. Modular so future features (dataset,
    # evaluation) can be added as their own blueprints.
    app.register_blueprint(health_bp)
    app.register_blueprint(classify_bp)
    app.register_blueprint(evaluation_bp)

    # Serves the HTML pages so the frontend and API share one origin.
    # Without this the browser's same-origin "/api/classify" calls have
    # nothing to resolve against.
    app.register_blueprint(frontend_bp)

    @app.errorhandler(404)
    def _not_found(_exc):
        return jsonify({"error": "Not found.", "error_code": "not_found"}), 404

    @app.errorhandler(405)
    def _method_not_allowed(_exc):
        return jsonify({"error": "Method not allowed.", "error_code": "method_not_allowed"}), 405

    @app.errorhandler(413)
    def _payload_too_large(_exc):
        # Flask/Werkzeug's default 413 page is HTML, not JSON. Without this
        # handler a client that trips MAX_CONTENT_LENGTH gets a different
        # response shape than every other error this API returns.
        return jsonify({
            "error": "Request body is too large.",
            "error_code": "payload_too_large",
        }), 413

    @app.errorhandler(Exception)
    def _unhandled_error(_exc):
        # Global safety net: no stack trace, exception message, or
        # internal detail is ever included in the response body.
        app.logger.exception("Unhandled exception while processing a request.")
        return jsonify({
            "error": "An unexpected server error occurred.",
            "error_code": "internal_error",
        }), 500

    return app


# Module-level app instance for `flask run` / WSGI servers.
app = create_app()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

