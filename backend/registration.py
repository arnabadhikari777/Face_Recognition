"""
Registration business logic: validation, duplicate check, persistence.
"""
from typing import Any, Dict, List, Tuple

from config import Config
from backend import database, face_matching


def validate_name(name: Any) -> Tuple[bool, str]:
    if not isinstance(name, str):
        return False, "Name must be a string"
    name = name.strip()
    if len(name) < 1:
        return False, "Name cannot be empty"
    if len(name) > 100:
        return False, "Name too long (max 100 characters)"
    # Basic sanitization: allow letters, numbers, spaces, common punctuation
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'-_")
    if not all(c in allowed for c in name):
        return False, "Name contains invalid characters"
    return True, name


def validate_consent(consent: Any) -> Tuple[bool, str]:
    if consent is not True and consent != "true" and consent != 1:
        return False, "Informed consent is required"
    return True, ""


def validate_embeddings_list(embeddings: Any) -> Tuple[bool, str, List[List[float]]]:
    if not isinstance(embeddings, list):
        return False, "embeddings must be a list", []
    if len(embeddings) < 1 or len(embeddings) > 5:
        return False, "Provide between 1 and 5 face embeddings", []
    cleaned: List[List[float]] = []
    for i, emb in enumerate(embeddings):
        ok, msg = face_matching.validate_embedding(emb, Config.EMBEDDING_DIM)
        if not ok:
            return False, f"Sample {i + 1}: {msg}", []
        cleaned.append([float(v) for v in emb])
    return True, "", cleaned


def register_face(
    name: str,
    embeddings: List[List[float]],
    consent: bool,
    model_name: str = None,
    model_version: str = None,
) -> Dict[str, Any]:
    """
    Full registration flow.
    Returns a response dict with success/error status suitable for JSON.
    """
    ok, name_or_err = validate_name(name)
    if not ok:
        return {"success": False, "error": name_or_err, "code": "invalid_name"}, 400

    ok, err = validate_consent(consent)
    if not ok:
        return {"success": False, "error": err, "code": "consent_required"}, 400

    ok, err, cleaned = validate_embeddings_list(embeddings)
    if not ok:
        return {"success": False, "error": err, "code": "invalid_embedding"}, 400

    model_name = model_name or Config.MODEL_NAME
    model_version = model_version or Config.MODEL_VERSION

    if model_name != Config.MODEL_NAME or model_version != Config.MODEL_VERSION:
        return {
            "success": False,
            "error": f"Model mismatch. Server expects {Config.MODEL_NAME} v{Config.MODEL_VERSION}",
            "code": "model_mismatch",
        }, 400

    # Load existing faces for the same model
    try:
        registered = database.find_all_faces(model_name=model_name, model_version=model_version)
    except Exception as e:
        return {
            "success": False,
            "error": "Database unavailable. Please try again later.",
            "code": "db_error",
        }, 503

    # Duplicate detection
    dup = face_matching.check_duplicate(cleaned, registered, Config.DUPLICATE_THRESHOLD)
    if dup:
        return {
            "success": False,
            "duplicate": True,
            "name": dup["name"],
            "score": dup["score"],
            "message": dup["message"],
            "code": "duplicate_face",
        }, 409

    # Persist
    doc = {
        "name": name_or_err,
        "embeddings": cleaned,
        "model_name": model_name,
        "model_version": model_version,
        "created_at": database.utc_now_iso(),
    }
    try:
        face_id = database.insert_face(doc)
    except Exception:
        return {
            "success": False,
            "error": "Failed to save registration. Please try again.",
            "code": "db_error",
        }, 503

    return {
        "success": True,
        "message": "Face registered successfully.",
        "face_id": face_id,
        "name": name_or_err,
    }, 201
