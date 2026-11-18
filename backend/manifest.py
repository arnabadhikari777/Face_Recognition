"""Loads models/model_manifest.json and models/calibration.json (single source of truth for models and thresholds)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSES = ("front", "left", "right", "up", "down", "other", "any")


class Manifest:
    def __init__(self, data):
        self.data = data
        self.models = {m["id"]: m for m in data["models"]}
        self.recognizers = [m["id"] for m in data["models"] if m["category"] == "recognition"]
        self.primary = next(m["id"] for m in data["models"] if m.get("role") == "primary_recognizer")
        self.quality = data["quality"]
        self.registration = data["registration"]
        self.recognition = data["recognition"]
        for p in data["profiles"].values():
            for mid in p["detectors"] + p["recognizers"]:
                if mid not in self.models:
                    raise ValueError(f"profile references unknown model {mid}")

    @classmethod
    def load(cls, path=None):
        with open(path or os.path.join(ROOT, "models", "model_manifest.json"), encoding="utf-8") as fh:
            return cls(json.load(fh))

    def dim(self, model_id):
        return int(self.models[model_id]["embedding_dim"])

    def version(self, model_id):
        return self.models[model_id]["version"]

    def active_versions(self):
        return {mid: self.version(mid) for mid in self.recognizers}

    def public(self, consistency_min):
        """What the browser needs. No thresholds and no secrets."""
        d = self.data
        return {
            "default_profile": d["default_profile"], "profiles": d["profiles"], "detection": d["detection"],
            "quality": d["quality"], "registration": d["registration"], "recognition": d["recognition"],
            "consistency_min": consistency_min,
            "models": [{k: m[k] for k in ("id", "name", "category", "role", "version", "filename", "sha256", "size_bytes", "embedding_dim")}
                       for m in d["models"]],
        }


class Calibration:
    def __init__(self, data, manifest):
        self.data = data
        self.calibrated = bool(data.get("calibrated"))
        self.models = data["models"]
        self.frame = data["frame_score"]
        self.dec = data["decision"]
        for mid in manifest.recognizers:
            if mid not in self.models:
                raise ValueError(f"calibration.json has no entry for model {mid}")

    @classmethod
    def load(cls, manifest, path=None):
        with open(path or os.path.join(ROOT, "models", "calibration.json"), encoding="utf-8") as fh:
            return cls(json.load(fh), manifest)

    def thr(self, mid): return float(self.models[mid]["threshold"])
    def scale(self, mid): return float(self.models[mid]["scale"])
    def weight(self, mid): return float(self.models[mid].get("weight", 1.0))
    def consistency_min(self, mid): return float(self.models[mid]["consistency_min"])
