"""Strict request validation. Nothing here touches the database."""
import math
import re

import numpy as np


class ApiError(Exception):
    def __init__(self, status, code, message, headers=None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
NAME_MAX = 60


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


def validate_model(data, cfg):
    if data.get("model_name") != cfg.MODEL_NAME or data.get("model_version") != cfg.MODEL_VERSION:
        raise ApiError(409, "model_mismatch",
                       "The face model on this device does not match the server. Please reload the page.")


def validate_embedding(emb, dim):
    """Return a unit-length float64 vector or raise ApiError."""
    if not isinstance(emb, list) or len(emb) != dim:
        raise ApiError(400, "invalid_embedding", f"Embedding must be a list of {dim} numbers.")
    for v in emb:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ApiError(400, "invalid_embedding", "Embedding must contain only numbers.")
        if not math.isfinite(v):
            raise ApiError(400, "invalid_embedding", "Embedding contains NaN or infinite values.")
    arr = np.asarray(emb, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or not (0.98 <= norm <= 1.02):
        raise ApiError(400, "invalid_embedding", "Embedding is not L2-normalised.")
    return arr / norm


def parse_register(data, cfg):
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_request", "Request body must be a JSON object.")
    name = validate_name(data.get("name"))
    validate_consent(data)
    validate_model(data, cfg)
    embs = data.get("embeddings")
    if not isinstance(embs, list) or len(embs) != cfg.SAMPLES_REQUIRED:
        raise ApiError(400, "invalid_embedding", f"Exactly {cfg.SAMPLES_REQUIRED} embeddings are required.")
    arr = np.stack([validate_embedding(e, cfg.EMBEDDING_DIM) for e in embs])
    sims = arr @ arr.T
    iu = np.triu_indices(len(arr), k=1)
    if float(sims[iu].min()) < cfg.SELF_CONSISTENCY_MIN:
        raise ApiError(422, "inconsistent_samples",
                       "The three samples do not look like the same person. Please try again.")
    return name, arr


def parse_recognize(data, cfg):
    if not isinstance(data, dict):
        raise ApiError(400, "invalid_request", "Request body must be a JSON object.")
    validate_consent(data)
    validate_model(data, cfg)
    return validate_embedding(data.get("embedding"), cfg.EMBEDDING_DIM)
