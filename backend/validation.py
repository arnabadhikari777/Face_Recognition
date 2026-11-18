"""Strict request validation. Nothing here touches the database."""
import math
import re

import numpy as np

from .manifest import POSES


class ApiError(Exception):
    def __init__(self, status, code, message, headers=None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
NAME_MAX = 60
MAX_SAMPLES = 16


def validate_name(name):
    if not isinstance(name, str):
        raise ApiError(400, "invalid_name", "Name must be text.")
    name = " ".join(name.split())
    if not name:
        raise ApiError(400, "invalid_name", "Please enter a name.")
    if len(name) > NAME_MAX:
        raise ApiError(400, "invalid_name", f"Name must be at most {NAME_MAX} characters.")
    if _CONTROL.search(name):
        raise ApiError(400, "invalid_name", "Name contains invalid characters.")
    return name


def validate_consent(data):
    if data.get("consent") is not True:
        raise ApiError(400, "consent_required", "Informed consent must be confirmed.")


def validate_vector(vec, dim):
    if not isinstance(vec, list) or len(vec) != dim:
        raise ApiError(400, "invalid_embedding", f"Embedding must be a list of {dim} numbers.")
    for v in vec:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ApiError(400, "invalid_embedding", "Embedding must contain only numbers.")
        if not math.isfinite(v):
            raise ApiError(400, "invalid_embedding", "Embedding contains NaN or infinite values.")
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not (0.98 <= norm <= 1.02):
        raise ApiError(400, "invalid_embedding", "Embedding is not L2-normalised.")
    return arr / norm


def _num(q, key, lo, hi, required=False):
    v = q.get(key)
    if v is None:
        if required:
            raise ApiError(400, "invalid_quality", f"Quality field '{key}' is required.")
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not (lo <= v <= hi):
        raise ApiError(400, "invalid_quality", f"Quality field '{key}' is out of range.")
    return float(v)


def _frame(item, manifest, require_primary):
    if not isinstance(item, dict):
        raise ApiError(400, "invalid_request", "Each sample must be an object.")
    pose = item.get("pose")
    if pose not in POSES:
        raise ApiError(400, "invalid_pose", "Unknown pose.")
    q = item.get("quality")
    if not isinstance(q, dict):
        raise ApiError(400, "invalid_quality", "Quality information is required.")
    quality = {"score": _num(q, "score", 0, 1, True), "yaw": _num(q, "yaw", -5, 5), "pitch": _num(q, "pitch", -5, 5),
               "sharpness": _num(q, "sharpness", 0, 1e6), "brightness": _num(q, "brightness", 0, 255),
               "detection": _num(q, "detection", 0, 1), "agreement": _num(q, "agreement", 0, 1)}
    embs = item.get("embeddings")
    if not isinstance(embs, dict) or not embs:
        raise ApiError(400, "invalid_embedding", "Embeddings are required.")
    out = {}
    for mid, e in embs.items():
        if mid not in manifest.recognizers:
            raise ApiError(409, "model_mismatch", "Unknown recognition model. Please reload the page.")
        if not isinstance(e, dict) or e.get("version") != manifest.version(mid):
            raise ApiError(409, "model_mismatch", "The face model on this device does not match the server. Please reload the page.")
        out[mid] = validate_vector(e.get("vector"), manifest.dim(mid))
    if require_primary and manifest.primary not in out:
        raise ApiError(400, "invalid_embedding", "The primary model embedding is required.")
    return {"pose": pose, "quality": quality, "vecs": out}


def _frames(items, manifest, require_primary):
    if not isinstance(items, list) or not (1 <= len(items) <= MAX_SAMPLES):
        raise ApiError(400, "invalid_request", f"Between 1 and {MAX_SAMPLES} samples are required.")
    frames = [_frame(i, manifest, require_primary) for i in items]
    ids = {frozenset(f["vecs"]) for f in frames}
    if len(ids) != 1:
        raise ApiError(400, "invalid_embedding", "Every sample must contain the same set of models.")
    return frames


def parse_register(data, manifest, cfg):
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_request", "Request body must be a JSON object.")
    name = validate_name(data.get("name"))
    validate_consent(data)
    frames = _frames(data.get("samples"), manifest, True)
    if cfg.ENROLL_REQUIRE_ALL and set(frames[0]["vecs"]) != set(manifest.recognizers):
        raise ApiError(400, "invalid_embedding", "All recognition models are required for registration.")
    return name, frames


def parse_recognize(data, manifest):
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_request", "Request body must be a JSON object.")
    validate_consent(data)
    return _frames(data.get("frames"), manifest, False), data.get("final") is True
