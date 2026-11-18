import numpy as np
from tools import metrics as M
from tools.calibrate import build
import json


def test_metrics_on_known_distributions():
    rng = np.random.default_rng(0)
    g, i = rng.normal(0.7, 0.08, 4000), rng.normal(0.0, 0.08, 40000)
    e, th = M.eer(g, i)
    assert e < 1e-3 and 0.2 < th < 0.5
    tar, t, far = M.tar_at_far(g, i, 1e-3)
    assert far <= 1e-3 and tar > 0.99
    overlap = M.eer(rng.normal(0.3, 0.1, 5000), rng.normal(0.2, 0.1, 5000))[0]
    assert 0.3 < overlap < 0.45
    sim = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.2]])
    assert M.rank1_accuracy(sim[:2], ["a", "b"], ["a", "b"]) == 1.0
    r = M.open_set_rates(sim, ["a", "b", "zz"], ["a", "b"], 0.5)
    assert r["detection_identification_rate"] == 1.0 and r["false_accept_rate_unknown"] == 0.0


def test_calibration_refuses_small_data():
    base = json.load(open("models/calibration.json"))
    res = {"persons": 3, "_scores": {"w600k_mbf": {"genuine": [0.7, 0.65, 0.8], "impostor": [0.0, 0.05, -0.02, 0.1]}}}
    cal = build(res, 1e-3, base)
    assert cal["calibrated"] is False and "TOO LITTLE" in cal["note"]
