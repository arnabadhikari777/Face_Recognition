#!/usr/bin/env python3
"""Accuracy benchmark for every recognition model, the detector choice and the score-fusion ensemble.

DATASET LAYOUT (your own consented images):  <dataset>/<person_id>/<condition>/<image>.jpg|png     (condition is optional: lighting,
pose, glasses, phone_a ... - it is used to break the results down). Never commit this folder.
USAGE: python tools/benchmark.py --models static/models --dataset /path/to/dataset --out benchmark_out [--detector scrfd_500m]
OUTPUT: benchmark_out/results.json (+ scores for tools/calibrate.py), roc_<model>.csv, report.md.
HONESTY RULES: with fewer than 20 people or fewer than 1000 impostor pairs the report says the numbers are NOT meaningful.
The same pipeline as the browser is used (SCRFD -> similarity alignment -> model-specific preprocessing); see tests/reference."""
import argparse
import glob
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.reference.make_reference import ARC, estimate, scrfd_detect  # noqa: E402
from tools import metrics as M  # noqa: E402

MODELS = {"w600k_mbf": "w600k_mbf.onnx", "w600k_r50": "w600k_r50.onnx"}


def find_images(root):
    items = []
    for path in sorted(glob.glob(os.path.join(root, "*", "**", "*.*"), recursive=True)):
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            rel = os.path.relpath(path, root).split(os.sep)
            items.append({"path": path, "person": rel[0], "condition": rel[1] if len(rel) > 2 else "unspecified"})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="static/models")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="benchmark_out")
    ap.add_argument("--far", type=float, nargs="*", default=[1e-2, 1e-3, 1e-4])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    so = ort.SessionOptions(); so.log_severity_level = 3
    det = ort.InferenceSession(os.path.join(a.models, "det_500m.onnx"), so, providers=["CPUExecutionProvider"])
    recs = {m: ort.InferenceSession(os.path.join(a.models, f), so, providers=["CPUExecutionProvider"]) for m, f in MODELS.items()}

    items, embs, times, no_face = find_images(a.dataset), {m: [] for m in MODELS}, {m: [] for m in MODELS}, 0
    kept = []
    for it in items:
        img = cv2.imread(it["path"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue
        b, k, s = scrfd_detect(det, img)
        if len(s) == 0:
            no_face += 1
            continue
        i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
        crop = cv2.warpAffine(img, estimate(k[i], ARC), (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImages([crop], 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        for m, sess in recs.items():
            t0 = time.perf_counter()
            e = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
            times[m].append((time.perf_counter() - t0) * 1000)
            embs[m].append(e / np.linalg.norm(e))
        kept.append(it)
    persons = sorted({k["person"] for k in kept})
    labels = np.array([k["person"] for k in kept]); conds = np.array([k["condition"] for k in kept])
    results = {"images": len(items), "no_face_detected": no_face, "usable": len(kept), "persons": len(persons), "models": {}, "caveats": []}
    pairs = list(itertools.combinations(range(len(kept)), 2))
    gi = [(i, j) for i, j in pairs if labels[i] == labels[j]]
    ii = [(i, j) for i, j in pairs if labels[i] != labels[j]]
    if len(persons) < 20 or len(ii) < 1000 or len(gi) < 100:
        results["caveats"].append(f"INSUFFICIENT DATA: {len(persons)} persons, {len(gi)} genuine and {len(ii)} impostor pairs. FAR/EER values below are NOT meaningful. "
                                  "Use >= 20 people, >= 100 genuine and >= 10000 impostor pairs (and 10x more than 1/FAR impostor pairs for TAR@FAR).")
    scores = {}
    for m in MODELS:
        E = np.stack(embs[m]) if embs[m] else np.zeros((0, 512))
        S = E @ E.T
        g = np.array([S[i, j] for i, j in gi]); imp = np.array([S[i, j] for i, j in ii])
        scores[m] = (g, imp, S)
    # score-level fusion: mean of per-model z-scores using each model's impostor statistics (no thresholds needed)
    zs = {m: (scores[m][2] - scores[m][1].mean()) / (scores[m][1].std() + 1e-9) for m in MODELS if len(scores[m][1])}
    if len(zs) == len(MODELS):
        Sf = sum(zs.values()) / len(zs)
        scores["fusion_mean_z"] = (np.array([Sf[i, j] for i, j in gi]), np.array([Sf[i, j] for i, j in ii]), Sf)
    for m, (g, imp, S) in scores.items():
        if len(g) == 0 or len(imp) == 0:
            results["models"][m] = {"error": "no genuine or no impostor pairs"}; continue
        e, eth = M.eer(g, imp)
        r = {"genuine_pairs": len(g), "impostor_pairs": len(imp), "eer": e, "eer_threshold": eth, "tar_at_far": {}, "per_condition_eer": {}}
        for f in a.far:
            tar, th, achieved = M.tar_at_far(g, imp, f)
            r["tar_at_far"][str(f)] = {"tar": tar, "threshold": th, "achieved_far": achieved}
        for c in sorted(set(conds)):                                   # genuine pairs where at least one image has this condition
            gc = np.array([S[i, j] for i, j in gi if c in (conds[i], conds[j])])
            if len(gc) >= 5:
                r["per_condition_eer"][c] = {"genuine_pairs": len(gc), "eer_vs_all_impostors": M.eer(gc, imp)[0], "mean_genuine_cosine": float(gc.mean())}
        # 1:N identification: first image of each person is the gallery, the rest are probes
        first = {}
        for idx, p in enumerate(labels):
            first.setdefault(p, idx)
        gal = list(first.values()); probes = [i for i in range(len(kept)) if i not in gal]
        if probes:
            sim = S[np.ix_(probes, gal)]
            r["identification_rank1"] = M.rank1_accuracy(sim, labels[probes], labels[gal])
            r["open_set_at_eer_threshold"] = M.open_set_rates(sim, labels[probes], labels[gal], eth)
        if m in times:
            r["inference_ms_cpu_python_mean"] = float(np.mean(times[m])); r["note_timing"] = "desktop CPU via onnxruntime-python; phones are slower - measure in the browser"
        results["models"][m] = r
        with open(os.path.join(a.out, f"roc_{m}.csv"), "w") as fh:
            fh.write("threshold,far,frr\n" + "\n".join(f"{p['threshold']:.5f},{p['far']:.6f},{p['frr']:.6f}" for p in M.roc_points(g, imp)))
    results["_scores"] = {m: {"genuine": scores[m][0].tolist(), "impostor": scores[m][1].tolist()} for m in scores}
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"))
    lines = ["# Benchmark report", "", f"Images: {results['images']} (usable {results['usable']}, no face {no_face}); people: {len(persons)}", ""]
    lines += [f"**{c}**" for c in results["caveats"]] + [""]
    lines += ["| model | EER | TAR@FAR=1e-3 | rank-1 (1:N) | CPU ms |", "|---|---|---|---|---|"]
    for m, r in results["models"].items():
        if "eer" in r:
            t = r["tar_at_far"].get("0.001", {}).get("tar")
            lines.append(f"| {m} | {r['eer']:.4f} | {t if t is None else round(t, 4)} | {r.get('identification_rank1')} | {r.get('inference_ms_cpu_python_mean', '-') if isinstance(r.get('inference_ms_cpu_python_mean'), str) else round(r.get('inference_ms_cpu_python_mean', 0), 1)} |")
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
