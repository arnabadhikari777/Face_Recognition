"""Pure matching logic (numpy only). Embeddings are expected to be L2-normalised,
so cosine similarity is a dot product.

Aggregation strategy (documented in README):
  * For each stored person, every query sample is compared against all of that
    person's stored samples and the BEST similarity is kept (handles pose variety).
  * The person's score is the MEAN of those per-query bests.
  * Registration (3 queries) also counts "supporting" samples: queries whose best
    similarity alone reaches MATCH_THRESHOLD.
The score is a cosine similarity, NOT a probability of correct identification.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    id: str
    name: str
    score: float
    supporting: int


def rank_candidates(queries, records, threshold):
    queries = np.atleast_2d(queries)
    out = []
    for rec in records:
        stored = np.atleast_2d(rec["embeddings"])
        per_query = (queries @ stored.T).max(axis=1)
        out.append(Candidate(rec["id"], rec["name"], float(per_query.mean()),
                             int((per_query >= threshold).sum())))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def classify_duplicate(ranked, cfg, n_queries):
    """-> ('duplicate' | 'uncertain' | 'new', best_candidate_or_None)"""
    if not ranked:
        return "new", None
    best = ranked[0]
    needed = min(cfg.MIN_SUPPORTING_SAMPLES, n_queries)
    if best.score >= cfg.MATCH_THRESHOLD and best.supporting >= needed:
        return "duplicate", best
    if best.score >= cfg.UNCERTAIN_LOW:
        return "uncertain", best
    return "new", best


def classify_recognition(ranked, cfg):
    """-> ('match' | 'uncertain' | 'unknown', best_candidate_or_None)"""
    if not ranked:
        return "unknown", None
    best = ranked[0]
    second = ranked[1].score if len(ranked) > 1 else None
    if best.score >= cfg.MATCH_THRESHOLD:
        if second is not None and second >= cfg.UNCERTAIN_LOW and best.score - second < cfg.MATCH_MARGIN:
            return "uncertain", best  # two registered people look equally similar
        return "match", best
    if best.score >= cfg.UNCERTAIN_LOW:
        return "uncertain", best
    return "unknown", best
