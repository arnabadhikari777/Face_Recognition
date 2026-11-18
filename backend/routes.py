import threading
import time

from flask import Blueprint, current_app, jsonify, render_template, request

from . import registration
from .database import RepositoryError
from .validation import ApiError, parse_recognize, parse_register

bp = Blueprint("main", __name__)


class RateLimiter:
    """Sliding-window limiter kept in process memory (one Gunicorn worker is
    used on Render, so the limit is effectively global; see README)."""

    def __init__(self):
        self._hits, self._lock, self._last_prune = {}, threading.Lock(), 0.0

    def check(self, key, limit, window, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            if now - self._last_prune > 60 or len(self._hits) > 10000:
                pruned = {k: [t for t in v if now - t < 3600] for k, v in self._hits.items()}
                self._hits = {k: v for k, v in pruned.items() if v}
                self._last_prune = now
            hits = [t for t in self._hits.get(key, []) if now - t < window]
            if len(hits) >= limit:
                self._hits[key] = hits
                return False, max(1, int(window - (now - hits[0])) + 1)
            hits.append(now)
            self._hits[key] = hits
            return True, 0


def svc():
    return current_app.extensions["facerec"]


def rate_limit(name, limit, window):
    ok, retry = svc().limiter.check((name, request.remote_addr or "unknown"), limit, window)
    if not ok:
        raise ApiError(429, "rate_limited", "Too many requests. Please wait a moment and try again.",
                       headers={"Retry-After": str(retry)})


def read_json():
    if not request.is_json:
        raise ApiError(415, "unsupported_media_type", "Content-Type must be application/json.")
    data = request.get_json(silent=True)
    if data is None:
        raise ApiError(400, "invalid_json", "Request body is not valid JSON.")
    return data


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/api/config")
def public_config():
    c = svc().cfg
    rate_limit("default", c.RATE_DEFAULT_PER_MINUTE, 60)
    s = svc()
    return jsonify(s.manifest.public(s.calibration.consistency_min(s.manifest.primary)))


@bp.get("/api/health")
def health():
    rate_limit("default", svc().cfg.RATE_DEFAULT_PER_MINUTE, 60)
    try:
        svc().repo().ping()
        s = svc()
        return jsonify({"status": "ok", "database": "ok", "encryption": "enabled" if s.cipher.enabled else "disabled",
                        "thresholds_calibrated": s.calibration.calibrated})
    except RepositoryError as exc:
        current_app.logger.warning("health: database unavailable: %s", exc)
        return jsonify({"status": "degraded", "database": "unavailable"}), 503


@bp.post("/api/register-face")
def register_face():
    cfg = svc().cfg
    rate_limit("register", cfg.RATE_REGISTER_PER_HOUR, 3600)
    s = svc()
    name, frames = parse_register(read_json(), s.manifest, cfg)
    status, body = registration.register_face(s.repo(), cfg, s.manifest, s.calibration, name, frames)
    return jsonify(body), status


@bp.post("/api/recognize-face")
def recognize_face():
    cfg = svc().cfg
    rate_limit("recognize", cfg.RATE_RECOGNIZE_PER_MINUTE, 60)
    s = svc()
    frames, final = parse_recognize(read_json(), s.manifest)
    status, body = registration.recognize_face(s.repo(), cfg, s.manifest, s.calibration, frames, final)
    return jsonify(body), status
