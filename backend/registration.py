"""Business logic: register (multi-template, duplicate-safe) and recognise (MATCH / UNCERTAIN / UNKNOWN / retry)."""
import numpy as np

from . import face_matching as fm
from . import templates as tp
from .database import LockBusy, name_key
from .validation import ApiError


def _stack(frames, model_ids=None):
    """-> {model_id: {'vecs','quality','poses'}} from validated frames."""
    ids = model_ids or list(frames[0]["vecs"])
    return {m: {"vecs": np.stack([f["vecs"][m] for f in frames]), "quality": np.array([f["quality"]["score"] for f in frames]),
                "poses": [f["pose"] for f in frames]} for m in ids}


def clean_for_enrolment(frames, manifest, cal):
    """Duplicate/outlier rejection on the primary model, then per-model consistency validation.
    -> kept frame list, or ApiError 422."""
    reg = manifest.registration
    prim = manifest.primary
    keep, _dropped = tp.select_samples(
        np.stack([f["vecs"][prim] for f in frames]), [f["quality"]["score"] for f in frames], [f["pose"] for f in frames],
        duplicate_similarity=reg["duplicate_similarity"], mad_k=reg["outlier_mad_k"],
        consistency_min=cal.consistency_min(prim), max_templates=reg["max_templates"])
    if len(keep) < reg["min_templates"]:
        raise ApiError(422, "insufficient_samples", f"Not enough distinct, consistent samples ({len(keep)} usable, {reg['min_templates']} needed). Please try again.")
    kept = [frames[i] for i in keep]
    low = manifest.quality["min_frame_quality"]
    if sum(1 for f in kept if f["quality"]["score"] >= low) < reg["min_templates"]:
        raise ApiError(422, "low_quality", "The samples were not clear enough. Please try again in better light.")
    for mid in kept[0]["vecs"]:
        if tp.min_pairwise([f["vecs"][mid] for f in kept]) < cal.consistency_min(mid):
            raise ApiError(422, "inconsistent_samples", "The samples do not look like the same person. Please try again.")
    return kept


def _blocks(kept, manifest):
    blocks, versions, dims, tpls = {}, {}, {}, {}
    for mid in kept[0]["vecs"]:
        vecs = [f["vecs"][mid] for f in kept]
        q = [f["quality"]["score"] for f in kept]
        blocks[mid] = tp.build_model_block(vecs, q, [f["pose"] for f in kept])
        versions[mid], dims[mid] = manifest.version(mid), manifest.dim(mid)
        tpls[mid] = [{"vec": f["vecs"][mid], "pose": f["pose"], "quality": f["quality"]} for f in kept]
    return blocks, versions, dims, tpls


def register_face(repo, cfg, manifest, cal, name, frames):
    """-> (http_status, body). Raises RepositoryError on database failure."""
    kept = clean_for_enrolment(frames, manifest, cal)
    try:
        with repo.registration_lock():
            if repo.count() >= cfg.MAX_RECORDS:
                raise ApiError(503, "capacity_reached", "Registration is temporarily closed (storage limit reached).")
            idents = repo.load_identities(manifest.active_versions())
            ranked = fm.rank(_stack(kept), idents, cal)
            verdict, best = fm.decide_duplicate(ranked, cal)
            if verdict == "duplicate":
                ident = best.identity
                new_models = [m for m in kept[0]["vecs"] if m not in ident.models]
                if new_models and name_key(ident.name) == name_key(name):      # same person, same name, extra model => upgrade
                    blocks, versions, dims, tpls = _blocks(kept, manifest)
                    sel = lambda d: {m: d[m] for m in new_models}  # noqa: E731
                    repo.add_models(ident.id, sel(blocks), sel(versions), sel(dims), sel(tpls))
                    return 200, {"status": "upgraded", "message": f"Added {len(new_models)} new recognition model(s) to the existing registration."}
                return 409, {"status": "duplicate", "message": "This face appears to be already registered.", "existing_name": ident.name}
            if verdict == "uncertain":
                return 409, {"status": "uncertain", "message": "Could not rule out that this face is already registered. Please try again with better lighting."}
            blocks, versions, dims, tpls = _blocks(kept, manifest)
            repo.insert_identity(name, blocks, versions, dims, tpls)
            return 201, {"status": "registered", "message": "Face registered successfully.", "templates": len(kept), "models": sorted(blocks)}
    except LockBusy:
        raise ApiError(503, "busy", "Another registration is in progress. Please try again in a few seconds.")


def recognize_face(repo, cfg, manifest, cal, frames, final):
    """-> (http_status, body)."""
    low, need = manifest.quality["min_frame_quality"], manifest.recognition["min_frames"]
    good = [f for f in frames if f["quality"]["score"] >= low]
    if len(good) < need:
        return 200, {"status": "retry", "message": "Image quality was too low. Face the camera in good light, hold still and try again."}
    supplied = set(good[0]["vecs"])
    ranked = fm.rank(_stack(good), repo.load_identities(manifest.active_versions()), cal)
    v = fm.decide_recognition(ranked, supplied, cal, final)
    ev = {}
    if v.best is not None:
        ev = {"models_used": len(v.best.per_model), "models_agreeing": sum(1 for e in v.best.per_model.values() if e.accepts),
              "frames": len(good), "supporting_frames": min(e.supporting for e in v.best.per_model.values())}
    if v.status == fm.MATCH:
        return 200, {"status": "match", "message": "Match Found", "name": v.best.identity.name, "evidence": ev}
    if v.status == fm.NEED_SECONDARY:
        return 200, {"status": "need_secondary", "reason": v.reason, "need_models": v.need_models}
    if v.status == fm.UNCERTAIN:
        return 200, {"status": "uncertain", "reason": v.reason, "message": "Unable to identify reliably. Please try again.", "evidence": ev}
    return 200, {"status": "unknown", "message": "Unknown Face", "detail": "No matching registered face found."}
