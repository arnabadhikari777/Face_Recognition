#!/usr/bin/env python3
"""Turn benchmark scores into models/calibration.json (per-model threshold at a target FAR, scale = impostor std).
USAGE: python tools/calibrate.py --results benchmark_out/results.json --far 0.001 --out models/calibration.json
Refuses to mark the file calibrated=true when the data are too small to support the requested FAR."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import metrics as M  # noqa: E402


def build(results, far, base):
    cal = json.loads(json.dumps(base))
    ok = True
    for m, s in results["_scores"].items():
        if m not in cal["models"]:
            continue
        g, imp = np.array(s["genuine"]), np.array(s["impostor"])
        _tar, th, achieved = M.tar_at_far(g, imp, far)
        enough = len(imp) >= 10 / far and len(g) >= 100
        ok &= enough
        cal["models"][m].update({"threshold": round(float(th), 4), "scale": round(float(imp.std()), 4),
                                 "consistency_min": round(float(max(0.05, np.percentile(g, 1))), 4)})
        cal["models"][m]["fit"] = {"target_far": far, "achieved_far": achieved, "genuine_pairs": len(g), "impostor_pairs": len(imp), "enough_data": bool(enough)}
    cal["calibrated"] = bool(ok and results["persons"] >= 20)
    cal["note"] = ("Calibrated from benchmark results." if cal["calibrated"] else
                   "Computed from TOO LITTLE data (need >=20 people, >=100 genuine pairs and >=10/FAR impostor pairs): still provisional.")
    return cal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--far", type=float, default=1e-3)
    ap.add_argument("--base", default="models/calibration.json"); ap.add_argument("--out", default="models/calibration.json")
    a = ap.parse_args()
    cal = build(json.load(open(a.results)), a.far, json.load(open(a.base)))
    json.dump(cal, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}: calibrated={cal['calibrated']} - {cal['note']}")
