import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB = os.environ.get("MONGODB_DB", "face_recognition_db")
    MONGODB_COLLECTION = "registered_faces"

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-use-long-random-string")
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    # Face model settings (must match frontend model)
    MODEL_NAME = "mobilefacenet-arcface"
    MODEL_VERSION = "1.0"
    EMBEDDING_DIM = 512

    # Similarity thresholds (cosine similarity). Documented ranges for MobileFaceNet/ArcFace style models:
    # Typical same-person cosine similarity often falls ~0.40–0.80 depending on quality/pose/lighting.
    # Different people usually < 0.30–0.40. Thresholds below are conservative starting points;
    # owners should re-calibrate on their own enrollment data.
    DUPLICATE_THRESHOLD = float(os.environ.get("DUPLICATE_THRESHOLD", "0.55"))
    RECOGNITION_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.48"))
    UNCERTAIN_THRESHOLD = float(os.environ.get("UNCERTAIN_THRESHOLD", "0.40"))

    # Rate limiting (simple in-memory; for production consider Redis)
    RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "30"))
    RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

    # Request size
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB

    # CORS
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
