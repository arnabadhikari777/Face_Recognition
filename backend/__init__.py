"""Application factory."""
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from .database import MongoRepository, RepositoryError
from .routes import RateLimiter, bp
from .validation import ApiError

CSP = ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'wasm-unsafe-eval'; "
       "style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
       "connect-src 'self' https://cdn.jsdelivr.net; worker-src 'self' blob:; "
       "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


class Services:
    def __init__(self, cfg, repository=None):
        self.cfg, self.limiter, self._repo = cfg, RateLimiter(), repository

    def repo(self):
        """Create the DB connection lazily so the app boots even if the DB is down."""
        if self._repo is None:
            self._repo = MongoRepository(self.cfg.MONGODB_URI, self.cfg.MONGODB_DB_NAME,
                                         self.cfg.MONGODB_COLLECTION, self.cfg.RETENTION_DAYS)
        return self._repo


def create_app(cfg=None, repository=None):
    cfg = cfg or Config.from_env()
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_LENGTH
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 86400
    if cfg.TRUST_PROXY:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    app.extensions["facerec"] = Services(cfg, repository)
    app.register_blueprint(bp)

    def err(status, code, message, headers=None):
        resp = jsonify({"status": "error", "code": code, "message": message})
        resp.status_code = status
        for k, v in (headers or {}).items():
            resp.headers[k] = v
        return resp

    @app.errorhandler(ApiError)
    def _api(e):
        return err(e.status, e.code, e.message, e.headers)

    @app.errorhandler(RepositoryError)
    def _db(e):
        app.logger.error("database error: %s", e)
        return err(503, "database_unavailable", "The database is temporarily unavailable. Please try again later.")

    @app.errorhandler(HTTPException)
    def _http(e):
        if request.path.startswith("/api/"):
            code = "payload_too_large" if e.code == 413 else "http_error"
            return err(e.code, code, "Request is too large." if e.code == 413 else e.name)
        return e

    @app.errorhandler(Exception)
    def _unexpected(e):
        app.logger.exception("unhandled error")  # stack trace stays in server logs only
        return err(500, "internal_error", "Something went wrong. Please try again.")

    @app.before_request
    def _cors_preflight():
        if request.method == "OPTIONS" and request.path.startswith("/api/"):
            return app.make_response(("", 204))

    @app.after_request
    def _headers(resp):
        h = resp.headers
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "no-referrer"
        h["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
        h["Content-Security-Policy"] = CSP
        if request.is_secure:
            h["Strict-Transport-Security"] = "max-age=31536000"
        if request.path.startswith("/api/"):
            h["Cache-Control"] = "no-store"
            origin = request.headers.get("Origin")
            if origin and origin in cfg.ALLOWED_ORIGINS:
                h["Access-Control-Allow-Origin"] = origin
                h["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                h["Access-Control-Allow-Headers"] = "Content-Type"
                h["Vary"] = "Origin"
        return resp

    return app
