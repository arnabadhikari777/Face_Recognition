"""Central configuration. Every value can be overridden by an environment variable."""
import os
from dataclasses import dataclass


def load_dotenv_file(path=".env", env=None):
    """Tiny .env reader (no extra dependency). Never overrides variables that are already set."""
    env = os.environ if env is None else env
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return 0
    loaded = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in env:
            env[key] = value
            loaded += 1
    return loaded


def _int(env, name, default):
    return int(env.get(name, default))


def _float(env, name, default):
    return float(env.get(name, default))


@dataclass(frozen=True)
class Config:
    # --- Database -------------------------------------------------------
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "face_recognition_db"
    MONGODB_COLLECTION: str = "registered_faces"

    # --- Model contract (must match static/js/face-engine.js + the model files) ---
    MODEL_NAME: str = "opencv-zoo-sface"
    MODEL_VERSION: str = "2021dec"
    EMBEDDING_DIM: int = 128
    SAMPLES_REQUIRED: int = 3

    # --- Matching (cosine similarity of L2-normalised embeddings) -------
    # 0.363 is the cosine threshold published for SFace by OpenCV (opencv_zoo
    # demo / FaceRecognizerSF docs). It is a starting point, NOT a validated
    # operating point for your camera population. Calibrate before relying on it.
    MATCH_THRESHOLD: float = 0.363
    UNCERTAIN_LOW: float = 0.30        # [UNCERTAIN_LOW, MATCH_THRESHOLD) => "uncertain"
    MATCH_MARGIN: float = 0.05         # best must beat the runner-up by this much
    MIN_SUPPORTING_SAMPLES: int = 2    # duplicate check: >= this many of the 3 new samples must match
    SELF_CONSISTENCY_MIN: float = 0.15 # sanity check: the 3 registration samples must resemble each other

    # --- Data retention / capacity --------------------------------------
    RETENTION_DAYS: int = 90           # MongoDB TTL index deletes records after this many days
    MAX_RECORDS: int = 5000            # protects the free tier and Render RAM

    # --- HTTP -----------------------------------------------------------
    MAX_CONTENT_LENGTH: int = 65536
    ALLOWED_ORIGINS: tuple = ()        # empty => same-origin only (no CORS headers)
    TRUST_PROXY: bool = False          # set true on Render (one proxy hop)
    RATE_REGISTER_PER_HOUR: int = 10
    RATE_RECOGNIZE_PER_MINUTE: int = 30
    RATE_DEFAULT_PER_MINUTE: int = 60

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        origins = tuple(o.strip() for o in env.get("ALLOWED_ORIGINS", "").split(",") if o.strip())
        d = cls()
        return cls(
            MONGODB_URI=env.get("MONGODB_URI", ""),
            MONGODB_DB_NAME=env.get("MONGODB_DB_NAME", d.MONGODB_DB_NAME),
            MONGODB_COLLECTION=env.get("MONGODB_COLLECTION", d.MONGODB_COLLECTION),
            MATCH_THRESHOLD=_float(env, "MATCH_THRESHOLD", d.MATCH_THRESHOLD),
            UNCERTAIN_LOW=_float(env, "UNCERTAIN_LOW", d.UNCERTAIN_LOW),
            MATCH_MARGIN=_float(env, "MATCH_MARGIN", d.MATCH_MARGIN),
            MIN_SUPPORTING_SAMPLES=_int(env, "MIN_SUPPORTING_SAMPLES", d.MIN_SUPPORTING_SAMPLES),
            SELF_CONSISTENCY_MIN=_float(env, "SELF_CONSISTENCY_MIN", d.SELF_CONSISTENCY_MIN),
            RETENTION_DAYS=_int(env, "RETENTION_DAYS", d.RETENTION_DAYS),
            MAX_RECORDS=_int(env, "MAX_RECORDS", d.MAX_RECORDS),
            MAX_CONTENT_LENGTH=_int(env, "MAX_CONTENT_LENGTH", d.MAX_CONTENT_LENGTH),
            ALLOWED_ORIGINS=origins,
            TRUST_PROXY=env.get("TRUST_PROXY", "").lower() in ("1", "true", "yes"),
            RATE_REGISTER_PER_HOUR=_int(env, "RATE_REGISTER_PER_HOUR", d.RATE_REGISTER_PER_HOUR),
            RATE_RECOGNIZE_PER_MINUTE=_int(env, "RATE_RECOGNIZE_PER_MINUTE", d.RATE_RECOGNIZE_PER_MINUTE),
            RATE_DEFAULT_PER_MINUTE=_int(env, "RATE_DEFAULT_PER_MINUTE", d.RATE_DEFAULT_PER_MINUTE),
        )
