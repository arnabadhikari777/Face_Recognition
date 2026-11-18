"""SYNTHETIC vectors for server-logic unit tests only. They prove the code paths, NOT real-world recognition accuracy."""
import mongomock
import numpy as np

from backend.crypto import TemplateCipher
from backend.database import MongoRepository
from backend.manifest import Calibration, Manifest

MANIFEST = Manifest.load()
CAL = Calibration.load(MANIFEST)
DIM = 512
POSES = ["front", "front", "front", "left", "left", "right", "right", "up", "down"]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Person:
    """A person has one latent identity vector PER MODEL (different embedding spaces)."""
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.centers = {m: unit(rng.normal(size=DIM)) for m in MANIFEST.recognizers}

    def sample(self, model, rng, noise=0.45):
        return unit(self.centers[model] + noise * unit(rng.normal(size=DIM)))


def payload_samples(person, seed, models=None, poses=POSES, noise=0.45, quality=0.8, version_override=None):
    rng = np.random.default_rng(seed)
    models = models or list(MANIFEST.recognizers)
    out = []
    for pose in poses:
        out.append({"pose": pose, "quality": {"score": quality, "yaw": 0.0, "pitch": 0.0, "sharpness": 100.0, "brightness": 120.0, "detection": 0.9, "agreement": 0.95},
                    "embeddings": {m: {"version": version_override or MANIFEST.version(m), "vector": person.sample(m, rng, noise).tolist()} for m in models}})
    return out


def register_body(name, person, seed=1, **kw):
    return {"name": name, "consent": True, "samples": payload_samples(person, seed, **kw)}


def recognize_body(person, seed=100, n=5, final=False, **kw):
    return {"consent": True, "final": final, "frames": payload_samples(person, seed, poses=["any"] * n, **kw)}


def make_repo(key=""):
    client = mongomock.MongoClient(tz_aware=True)
    cipher = TemplateCipher(key)
    return MongoRepository(client=client, db_name="t", collection="identities", retention_days=0, cipher=cipher), client
