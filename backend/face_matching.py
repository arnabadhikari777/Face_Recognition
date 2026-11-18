"""Multi-model, multi-template, multi-frame matching (numpy only).

Per identity and per model m, for every query frame f:
    frame_score = a*cos(q, master) + b*mean(top-k cos(q, individual templates)) + c*cos(q, pose prototype or master)
The per-frame scores are aggregated robustly (lower outliers dropped by MAD, quality-weighted mean). The result is a COSINE
SIMILARITY, never a probability. Each model's score is converted to a z-like value with its OWN calibration
    z_m = (score_m - threshold_m) / scale_m
and models are fused at score level: fused_z = sum(w_m z_m) / sum(w_m). Embeddings of different models are never mixed.
"""
import math
from dataclasses import dataclass, field

import numpy as np

MATCH, UNCERTAIN, UNKNOWN, NEED_SECONDARY = "match", "uncertain", "unknown", "need_secondary"


@dataclass
class ModelEvidence:
    score: float
    z: float
    supporting: int
    frames: int
    accepts: bool


@dataclass
class IdentityResult:
    identity: object
    fused_z: float
    per_model: dict = field(default_factory=dict)


@dataclass
class Verdict:
    status: str
    reason: str = ""
    best: object = None
    margin: float = math.inf
    need_models: list = field(default_factory=list)


def frame_scores(vecs, block, poses, cal):
    fs, k = cal.frame, int(cal.frame["individual_top_k"])
    master = vecs @ block["master"]
    sims = vecs @ block["templates"].T
    top = np.sort(sims, axis=1)[:, -min(k, sims.shape[1]):].mean(axis=1)
    pose = np.array([float(v @ block["prototypes"][p]) if p in block["prototypes"] else m for v, p, m in zip(vecs, poses, master)])
    return fs["master"] * master + fs["individual"] * top + fs["pose"] * pose


def robust_aggregate(scores, weights, mad_k):
    scores, w = np.asarray(scores, float), np.clip(np.asarray(weights, float), 0.05, 1.0)
    if len(scores) >= 4:
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med))) * 1.4826
        keep = scores >= med - mad_k * max(mad, 0.02)
        if keep.sum() >= 2:
            scores, w = scores[keep], w[keep]
    return float((scores * w).sum() / w.sum())


def score_identity(frames, ident, cal):
    """frames: {model_id: {'vecs': (F,d), 'quality': (F,), 'poses': [F]}}"""
    per, zs, ws = {}, [], []
    for mid, f in frames.items():
        block = ident.models.get(mid)
        if block is None:
            continue
        n = len(f["vecs"])
        fs = frame_scores(f["vecs"], block, f["poses"], cal)
        s = robust_aggregate(fs, f["quality"], cal.dec["frame_outlier_mad_k"])
        thr = cal.thr(mid)
        z = (s - thr) / cal.scale(mid)
        sup = int((fs >= thr).sum())
        per[mid] = ModelEvidence(s, z, sup, n, z >= 0 and sup >= math.ceil(cal.dec["min_support_frac"] * n))
        zs.append(cal.weight(mid) * z)
        ws.append(cal.weight(mid))
    if not per:
        return None
    return IdentityResult(ident, sum(zs) / sum(ws), per)


def rank(frames, identities, cal):
    out = [r for r in (score_identity(frames, i, cal) for i in identities) if r is not None]
    out.sort(key=lambda r: r.fused_z, reverse=True)
    return out


def _flags(best, cal):
    zs = [e.z for e in best.per_model.values()]
    disagree = len(zs) > 1 and max(zs) >= 0 and min(zs) < -cal.dec["model_agreement_tol_z"]
    support_ok = all(e.supporting >= math.ceil(cal.dec["min_support_frac"] * e.frames) for e in best.per_model.values())
    return disagree, support_ok


def decide_recognition(ranked, supplied, cal, final):
    """1:N decision. supplied = set of model ids the client sent. -> Verdict(MATCH|UNCERTAIN|UNKNOWN|NEED_SECONDARY)."""
    d = cal.dec
    if not ranked:
        return Verdict(UNKNOWN)
    best = ranked[0]
    margin = best.fused_z - ranked[1].fused_z if len(ranked) > 1 else math.inf
    if best.fused_z < d["uncertain_z"]:
        return Verdict(UNKNOWN, best=best, margin=margin)
    disagree, support_ok = _flags(best, cal)
    if best.fused_z >= d["match_z"] and support_ok and not disagree and margin >= d["margin_z"]:
        reason = ""
        # confident enough for a single model? otherwise ask for the models the person is also enrolled in
        missing = [m for m in best.identity.models if m not in supplied]
        if missing and not final and best.fused_z < d["secondary_z"]:
            return Verdict(NEED_SECONDARY, "borderline", best, margin, missing)
        return Verdict(MATCH, reason, best, margin)
    reason = ("model_disagreement" if disagree else "ambiguous" if margin < d["margin_z"] and best.fused_z >= d["match_z"]
              else "low_support" if not support_ok and best.fused_z >= d["match_z"] else "borderline")
    missing = [m for m in best.identity.models if m not in supplied]
    if missing and not final:
        return Verdict(NEED_SECONDARY, reason, best, margin, missing)
    return Verdict(UNCERTAIN, reason, best, margin)


def decide_duplicate(ranked, cal):
    """Registration check -> ('duplicate'|'uncertain'|'new', best). No candidate margin here: ANY close identity blocks a new record."""
    d = cal.dec
    if not ranked:
        return "new", None
    best = ranked[0]
    disagree, support_ok = _flags(best, cal)
    if best.fused_z >= d["match_z"] and support_ok and not disagree:
        return "duplicate", best
    if best.fused_z >= d["uncertain_z"] or (disagree and max(e.z for e in best.per_model.values()) >= 0):
        return "uncertain", best
    return "new", best
