"""Business logic for registering and recognising faces."""
from . import face_matching as fm
from .validation import ApiError


def _round(x):
    return round(float(x), 4)


def register_face(repo, cfg, name, embeddings):
    """-> (http_status, body). Raises RepositoryError on database failure."""
    if repo.count() >= cfg.MAX_RECORDS:
        raise ApiError(503, "capacity_reached", "Registration is temporarily closed (storage limit reached).")
    records = repo.load_embeddings(cfg.MODEL_NAME, cfg.MODEL_VERSION)
    ranked = fm.rank_candidates(embeddings, records, cfg.MATCH_THRESHOLD)
    verdict, best = fm.classify_duplicate(ranked, cfg, len(embeddings))
    if verdict == "duplicate":
        return 409, {"status": "duplicate",
                     "message": "This face appears to be already registered.",
                     "existing_name": best.name, "similarity_score": _round(best.score)}
    if verdict == "uncertain":
        return 409, {"status": "uncertain",
                     "message": "Could not rule out that this face is already registered. "
                                "Please try again with better lighting.",
                     "similarity_score": _round(best.score)}
    repo.insert(name, embeddings, cfg.MODEL_NAME, cfg.MODEL_VERSION)
    return 201, {"status": "registered", "message": "Face registered successfully."}


def recognize_face(repo, cfg, embedding):
    """-> (http_status, body)."""
    records = repo.load_embeddings(cfg.MODEL_NAME, cfg.MODEL_VERSION)
    ranked = fm.rank_candidates(embedding, records, cfg.MATCH_THRESHOLD)
    verdict, best = fm.classify_recognition(ranked, cfg)
    if verdict == "match":
        return 200, {"status": "match", "message": "Match Found", "name": best.name,
                     "similarity_score": _round(best.score)}
    if verdict == "uncertain":
        return 200, {"status": "uncertain", "message": "Unable to identify reliably. Please try again."}
    return 200, {"status": "unknown", "message": "Unknown Face",
                 "detail": "No matching registered face found."}
