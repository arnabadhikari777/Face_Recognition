"""
MongoDB Atlas connection and collection helpers.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId

from config import Config

_client: Optional[MongoClient] = None
_db = None
_collection = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
    return _client


def get_db():
    global _db
    if _db is None:
        client = get_client()
        _db = client[Config.MONGODB_DB]
    return _db


def get_collection():
    global _collection
    if _collection is None:
        db = get_db()
        _collection = db[Config.MONGODB_COLLECTION]
        # Ensure useful indexes (idempotent)
        try:
            _collection.create_index([("name", ASCENDING)])
            _collection.create_index([("created_at", ASCENDING)])
            _collection.create_index([("model_name", ASCENDING), ("model_version", ASCENDING)])
        except Exception:
            # Index creation may fail on free-tier limits or permissions; non-fatal
            pass
    return _collection


def check_connection() -> bool:
    """Return True if MongoDB is reachable."""
    try:
        client = get_client()
        client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception):
        return False


def insert_face(document: Dict[str, Any]) -> str:
    """Insert a registered face document. Returns inserted_id as string."""
    col = get_collection()
    result = col.insert_one(document)
    return str(result.inserted_id)


def find_all_faces(model_name: Optional[str] = None, model_version: Optional[str] = None) -> List[Dict]:
    """Return all registered faces, optionally filtered by model."""
    col = get_collection()
    query: Dict[str, Any] = {}
    if model_name:
        query["model_name"] = model_name
    if model_version:
        query["model_version"] = model_version
    cursor = col.find(query)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results


def find_by_id(face_id: str) -> Optional[Dict]:
    col = get_collection()
    try:
        doc = col.find_one({"_id": ObjectId(face_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception:
        return None


def delete_by_id(face_id: str) -> bool:
    """Owner-operated deletion helper. Returns True if a document was deleted."""
    col = get_collection()
    try:
        result = col.delete_one({"_id": ObjectId(face_id)})
        return result.deleted_count > 0
    except Exception:
        return False


def count_faces() -> int:
    col = get_collection()
    return col.count_documents({})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
