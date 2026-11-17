"""
Flask API routes.
"""
import time
from collections import defaultdict
from functools import wraps
from typing import Callable

from flask import Blueprint, request, jsonify, current_app

from config import Config
from backend import database, face_matching, registration

api = Blueprint("api", __name__)

# Simple in-memory rate limiter (per IP). Resets on process restart — acceptable for free-tier demo.
_rate_store: dict = defaultdict(list)


def rate_limit(f: Callable):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
        ip = ip.split(",")[0].strip()
        now = time.time()
        window = Config.RATE_LIMIT_WINDOW_SECONDS
        max_req = Config.RATE_LIMIT_REQUESTS
        # Prune old entries
        _rate_store[ip] = [t for t in _rate_store[ip] if now - t < window]
        if len(_rate_store[ip]) >= max_req:
            return jsonify({
                "success": False,
                "error": "Too many requests. Please wait a moment and try again.",
                "code": "rate_limited",
            }), 429
        _rate_store[ip].append(now)
        return f(*args, **kwargs)
    return decorated


@api.route("/health", methods=["GET"])
def health():
    db_ok = database.check_connection()
    status = "ok" if db_ok else "degraded"
    code = 200 if db_ok else 503
    return jsonify({
        "status": status,
        "database": "connected" if db_ok else "unreachable",
        "model": Config.MODEL_NAME,
        "model_version": Config.MODEL_VERSION,
        "embedding_dim": Config.EMBEDDING_DIM,
    }), code


@api.route("/register-face", methods=["POST"])
@rate_limit
def register_face():
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json", "code": "invalid_content"}), 400

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON body", "code": "invalid_json"}), 400

    name = data.get("name")
    embeddings = data.get("embeddings")
    consent = data.get("consent")
    model_name = data.get("model_name")
    model_version = data.get("model_version")

    result, status = registration.register_face(
        name=name,
        embeddings=embeddings,
        consent=consent,
        model_name=model_name,
        model_version=model_version,
    )
    return jsonify(result), status


@api.route("/recognize-face", methods=["POST"])
@rate_limit
def recognize_face():
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json", "code": "invalid_content"}), 400

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON body", "code": "invalid_json"}), 400

    consent = data.get("consent")
    ok, err = registration.validate_consent(consent)
    if not ok:
        return jsonify({"success": False, "error": err, "code": "consent_required"}), 400

    # Accept either single embedding or list of embeddings
    embedding = data.get("embedding")
    embeddings = data.get("embeddings")
    if embeddings is None and embedding is not None:
        embeddings = [embedding]
    if embeddings is None:
        return jsonify({"success": False, "error": "embedding or embeddings required", "code": "missing_embedding"}), 400

    ok, err, cleaned = registration.validate_embeddings_list(embeddings)
    if not ok:
        return jsonify({"success": False, "error": err, "code": "invalid_embedding"}), 400

    model_name = data.get("model_name") or Config.MODEL_NAME
    model_version = data.get("model_version") or Config.MODEL_VERSION
    if model_name != Config.MODEL_NAME or model_version != Config.MODEL_VERSION:
        return jsonify({
            "success": False,
            "error": f"Model mismatch. Server expects {Config.MODEL_NAME} v{Config.MODEL_VERSION}",
            "code": "model_mismatch",
        }), 400

    try:
        registered = database.find_all_faces(model_name=model_name, model_version=model_version)
    except Exception:
        return jsonify({
            "success": False,
            "error": "Database unavailable. Please try again later.",
            "code": "db_error",
        }), 503

    match_result = face_matching.find_best_match(
        cleaned,
        registered,
        threshold=Config.RECOGNITION_THRESHOLD,
        uncertain_threshold=Config.UNCERTAIN_THRESHOLD,
    )

    # Do not expose full embeddings or internal ids beyond what's needed
    response = {
        "success": True,
        "status": match_result["status"],
        "message": match_result.get("message", ""),
        "score": match_result.get("score"),
    }
    if match_result["status"] == "match":
        response["name"] = match_result.get("name")
    # For uncertain we optionally surface the closest name but still flag as uncertain
    if match_result["status"] == "uncertain" and match_result.get("name"):
        response["closest_name"] = match_result.get("name")

    return jsonify(response), 200
