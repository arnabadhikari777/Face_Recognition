"""Template hygiene + construction (numpy only): duplicate rejection, outlier rejection, consistency validation,
quality-weighted master template and pose prototypes. Mirrors static/js/engine/templates.js (client does it first, server re-checks).
Embeddings of DIFFERENT models are never combined: every function here works on one model's vectors."""
import numpy as np

POSE_ORDER = ("front", "left", "right", "up", "down")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def select_samples(vecs, qualities, poses, *, duplicate_similarity, mad_k, consistency_min, max_templates):
    """Decide which samples become templates, using ONE model's vectors (the primary).
    -> (keep_indices_best_first, [(index, reason)]) with reason in duplicate|outlier|limit."""
    vecs = np.asarray(vecs, dtype=np.float64)
    order = sorted(range(len(vecs)), key=lambda i: -qualities[i])
    kept, dropped = [], []
    for i in order:
        if any(poses[k] == poses[i] and float(vecs[k] @ vecs[i]) >= duplicate_similarity for k in kept):
            dropped.append((i, "duplicate"))
        else:
            kept.append(i)
    pool = kept
    if len(pool) >= 4:
        sub = vecs[pool]
        sims = sub @ sub.T
        means = (sims.sum(axis=1) - 1.0) / (len(pool) - 1)
        med = float(np.median(means))
        mad = float(np.median(np.abs(means - med))) * 1.4826
        floor = max(consistency_min, med - mad_k * max(mad, 0.02))
        new_pool = []
        for idx, m in zip(pool, means):
            (new_pool if m >= floor else dropped).append(idx if m >= floor else (idx, "outlier"))
        pool = new_pool
    pool = sorted(pool, key=lambda i: -qualities[i])
    for i in pool[max_templates:]:
        dropped.append((i, "limit"))
    return pool[:max_templates], dropped


def min_pairwise(vecs):
    vecs = np.asarray(vecs, dtype=np.float64)
    if len(vecs) < 2:
        return 1.0
    sims = vecs @ vecs.T
    return float(sims[np.triu_indices(len(vecs), k=1)].min())


def weighted_master(vecs, qualities):
    """Quality-weighted mean, L2-normalised. Quality is clipped so one very bad frame cannot dominate or vanish."""
    w = np.clip(np.asarray(qualities, dtype=np.float64), 0.05, 1.0)
    return unit((np.asarray(vecs, dtype=np.float64) * w[:, None]).sum(axis=0))


def build_model_block(vecs, qualities, poses):
    """-> {'master': vec, 'prototypes': {pose: vec}} for one model."""
    vecs = np.asarray(vecs, dtype=np.float64)
    protos = {}
    for pose in POSE_ORDER:
        idx = [i for i, p in enumerate(poses) if p == pose]
        if idx:
            protos[pose] = weighted_master(vecs[idx], [qualities[i] for i in idx])
    return {"master": weighted_master(vecs, qualities), "prototypes": protos}
