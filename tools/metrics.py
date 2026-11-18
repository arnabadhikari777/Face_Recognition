"""Biometric verification / identification metrics (numpy only)."""
import numpy as np


def far_frr_curve(genuine, impostor):
    """Sweep every observed score as threshold. -> thresholds, FAR, FRR (accept if score >= threshold)."""
    g, i = np.sort(np.asarray(genuine, float)), np.sort(np.asarray(impostor, float))
    th = np.unique(np.concatenate([g, i, [g.min() - 1e-9, i.max() + 1e-9]]))
    far = 1.0 - np.searchsorted(i, th, side="left") / len(i)
    frr = np.searchsorted(g, th, side="left") / len(g)
    return th, far, frr


def eer(genuine, impostor):
    th, far, frr = far_frr_curve(genuine, impostor)
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(th[k])


def tar_at_far(genuine, impostor, target_far):
    """Lowest-FAR operating point not exceeding target_far -> (TAR, threshold, achieved FAR). Impostor pairs must be >> 1/target_far to mean anything."""
    th, far, frr = far_frr_curve(genuine, impostor)
    ok = np.where(far <= target_far)[0]
    k = int(ok[0]) if len(ok) else len(th) - 1
    return float(1 - frr[k]), float(th[k]), float(far[k])


def roc_points(genuine, impostor, n=200):
    th, far, frr = far_frr_curve(genuine, impostor)
    idx = np.unique(np.linspace(0, len(th) - 1, n).astype(int))
    return [{"threshold": float(th[i]), "far": float(far[i]), "frr": float(frr[i])} for i in idx]


def rank1_accuracy(sim_matrix, probe_labels, gallery_labels):
    """1:N closed-set identification. sim_matrix: (probes, gallery)."""
    pred = np.asarray(gallery_labels)[np.argmax(sim_matrix, axis=1)]
    return float(np.mean(pred == np.asarray(probe_labels)))


def open_set_rates(sim_matrix, probe_labels, gallery_labels, threshold):
    """1:N open-set: DIR (correct id above threshold), false-accept rate for probes NOT in the gallery."""
    sim, gl = np.asarray(sim_matrix), np.asarray(gallery_labels)
    best, arg = sim.max(axis=1), sim.argmax(axis=1)
    known = np.array([p in set(gl) for p in probe_labels])
    correct = (gl[arg] == np.asarray(probe_labels)) & (best >= threshold)
    dir_ = float(correct[known].mean()) if known.any() else None
    fa = float((best[~known] >= threshold).mean()) if (~known).any() else None
    return {"detection_identification_rate": dir_, "false_accept_rate_unknown": fa}
