"""
Face embedding similarity and matching logic.

Uses cosine similarity on L2-normalized embeddings (standard for ArcFace / MobileFaceNet).
Thresholds are configurable via Config and should be re-calibrated on real enrollment data.
"""
from typing import Any, Dict, List, Optional, Tuple
import math

from config import Config


def _is_valid_number(x: float) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def validate_embedding(emb: Any, expected_dim: int = Config.EMBEDDING_DIM) -> Tuple[bool, str]:
    """Validate embedding structure and values. Returns (ok, error_message)."""
    if not isinstance(emb, (list, tuple)):
        return False, "Embedding must be a list of numbers"
    if len(emb) != expected_dim:
        return False, f"Embedding dimension must be {expected_dim}, got {len(emb)}"
    for i, v in enumerate(emb):
        if not _is_valid_number(v):
            return False, f"Invalid numeric value at index {i}"
    return True, ""


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity. Assumes embeddings are already L2-normalized; still computes safely."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


def average_embedding(embeddings: List[List[float]]) -> List[float]:
    """Average multiple embeddings then L2-normalize."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    avg = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            avg[i] += v
    n = float(len(embeddings))
    avg = [v / n for v in avg]
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in avg))
    if norm > 0:
        avg = [v / norm for v in avg]
    return avg


def max_similarity_to_record(
    query_embs: List[List[float]],
    stored_embs: List[List[float]],
) -> float:
    """
    Aggregation strategy for multi-sample comparison:
    For each query sample, take the max cosine similarity against any stored sample,
    then average those max scores. This is more robust than pure mean-of-all-pairs
    when pose/illumination varies.
    """
    if not query_embs or not stored_embs:
        return 0.0
    scores = []
    for q in query_embs:
        best = max(cosine_similarity(q, s) for s in stored_embs)
        scores.append(best)
    return sum(scores) / len(scores)


def find_best_match(
    query_embeddings: List[List[float]],
    registered: List[Dict[str, Any]],
    threshold: float = Config.RECOGNITION_THRESHOLD,
    uncertain_threshold: float = Config.UNCERTAIN_THRESHOLD,
) -> Dict[str, Any]:
    """
    Compare query embedding(s) against all registered faces.

    Returns a structured result:
      - status: "match" | "unknown" | "uncertain"
      - name, score, face_id when applicable
    """
    if not query_embeddings:
        return {"status": "unknown", "message": "No embedding provided"}

    best_score = -1.0
    best_record: Optional[Dict] = None

    for rec in registered:
        stored = rec.get("embeddings") or []
        if not stored:
            continue
        score = max_similarity_to_record(query_embeddings, stored)
        if score > best_score:
            best_score = score
            best_record = rec

    if best_record is None:
        return {
            "status": "unknown",
            "message": "No matching registered face found.",
            "score": 0.0,
        }

    if best_score >= threshold:
        return {
            "status": "match",
            "name": best_record.get("name"),
            "score": round(best_score, 4),
            "face_id": best_record.get("_id"),
            "message": "Match Found",
        }
    if best_score >= uncertain_threshold:
        return {
            "status": "uncertain",
            "name": best_record.get("name"),
            "score": round(best_score, 4),
            "face_id": best_record.get("_id"),
            "message": "Unable to identify reliably. Please try again.",
        }
    return {
        "status": "unknown",
        "score": round(best_score, 4),
        "message": "No matching registered face found.",
    }


def check_duplicate(
    new_embeddings: List[List[float]],
    registered: List[Dict[str, Any]],
    threshold: float = Config.DUPLICATE_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """
    If a sufficiently similar face already exists, return that record info.
    Otherwise return None.
    """
    result = find_best_match(
        new_embeddings,
        registered,
        threshold=threshold,
        uncertain_threshold=threshold,  # for duplicate we treat above threshold as match
    )
    if result.get("status") == "match":
        return {
            "name": result.get("name"),
            "score": result.get("score"),
            "face_id": result.get("face_id"),
            "message": f"This face appears already registered as '{result.get('name')}'.",
        }
    return None
