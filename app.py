"""
Face Recognition Web Application — Flask entry point.
Production-ready for Render Free Web Service with Gunicorn.
"""
import os
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

from config import Config
from backend.routes import api

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# CORS — restrict in production via CORS_ORIGINS env var
cors_origins = Config.CORS_ORIGINS
if cors_origins == "*":
    CORS(app, resources={r"/api/*": {"origins": "*"}})
else:
    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origins}})

app.register_blueprint(api, url_prefix="/api")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.errorhandler(413)
def request_entity_too_large(e):
    return {"success": False, "error": "Request too large", "code": "payload_too_large"}, 413


@app.errorhandler(404)
def not_found(e):
    return {"success": False, "error": "Not found", "code": "not_found"}, 404


@app.errorhandler(500)
def internal_error(e):
    # Never leak stack traces
    return {"success": False, "error": "Internal server error", "code": "server_error"}, 500


if __name__ == "__main__":
    # Development only. Production uses Gunicorn via Procfile.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=Config.DEBUG)
