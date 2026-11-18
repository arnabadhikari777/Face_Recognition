"""Central configuration. Every value can be overridden by an environment variable (see .env.example)."""
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


def _bool(env, name, default):
    v = env.get(name)
    return default if v is None else v.lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Config:
    # --- Database -------------------------------------------------------
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "face_recognition_db"
    MONGODB_COLLECTION: str = "identities"       # schema v2. The old 'registered_faces' collection is only read by scripts/migrate_db.py
    TEMPLATE_ENCRYPTION_KEY: str = ""            # Fernet key; empty => templates stored unencrypted (reported by /api/health)

    # --- Models / thresholds: models/model_manifest.json and models/calibration.json (override paths here) ---
    MANIFEST_FILE: str = ""
    CALIBRATION_FILE: str = ""
    ENROLL_REQUIRE_ALL: bool = False             # True: registration must carry every recognition model (Accuracy mode only)

    # --- Data retention / capacity --------------------------------------
    RETENTION_DAYS: int = 0                      # 0 = keep forever (no automatic deletion). >0 adds a MongoDB TTL index.
    MAX_RECORDS: int = 1500                      # ~80 KB per identity with 2 models => ~120 MB; protects Atlas M0 (512 MB)

    # --- HTTP -----------------------------------------------------------
    MAX_CONTENT_LENGTH: int = 524288
    ALLOWED_ORIGINS: tuple = ()
    TRUST_PROXY: bool = False
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
            TEMPLATE_ENCRYPTION_KEY=env.get("TEMPLATE_ENCRYPTION_KEY", ""),
            MANIFEST_FILE=env.get("MANIFEST_FILE", ""),
            CALIBRATION_FILE=env.get("CALIBRATION_FILE", ""),
            ENROLL_REQUIRE_ALL=_bool(env, "ENROLL_REQUIRE_ALL", d.ENROLL_REQUIRE_ALL),
            RETENTION_DAYS=_int(env, "RETENTION_DAYS", d.RETENTION_DAYS),
            MAX_RECORDS=_int(env, "MAX_RECORDS", d.MAX_RECORDS),
            MAX_CONTENT_LENGTH=_int(env, "MAX_CONTENT_LENGTH", d.MAX_CONTENT_LENGTH),
            ALLOWED_ORIGINS=origins,
            TRUST_PROXY=_bool(env, "TRUST_PROXY", d.TRUST_PROXY),
            RATE_REGISTER_PER_HOUR=_int(env, "RATE_REGISTER_PER_HOUR", d.RATE_REGISTER_PER_HOUR),
            RATE_RECOGNIZE_PER_MINUTE=_int(env, "RATE_RECOGNIZE_PER_MINUTE", d.RATE_RECOGNIZE_PER_MINUTE),
            RATE_DEFAULT_PER_MINUTE=_int(env, "RATE_DEFAULT_PER_MINUTE", d.RATE_DEFAULT_PER_MINUTE),
        )
