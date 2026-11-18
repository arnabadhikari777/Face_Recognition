"""MongoDB access. pymongo is imported lazily so the rest of the app (and the
unit tests) work without it installed."""
from datetime import datetime, timezone

import numpy as np


class RepositoryError(Exception):
    """Any database failure. Message is safe to log, never shown to visitors."""


class MongoRepository:
    def __init__(self, uri, db_name, collection, retention_days):
        if not uri:
            raise RepositoryError("MONGODB_URI is not set")
        try:
            import pymongo
            from pymongo.errors import PyMongoError
        except ImportError as exc:  # pragma: no cover
            raise RepositoryError("pymongo is not installed") from exc
        self._err = PyMongoError
        try:
            self._client = pymongo.MongoClient(
                uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000,
                socketTimeoutMS=10000, tz_aware=True, appname="face-recognition")
            self._db = self._client[db_name]
            self._coll = self._db[collection]
            self._coll_name = collection
            self._client.admin.command("ping")
            self._ensure_indexes(retention_days)
        except PyMongoError as exc:
            raise RepositoryError(f"connection failed: {type(exc).__name__}") from exc

    def _ensure_indexes(self, retention_days):
        from pymongo.errors import OperationFailure
        ttl = int(retention_days) * 86400
        self._coll.create_index([("model_name", 1), ("model_version", 1)], name="model_idx")
        try:
            self._coll.create_index([("created_at", 1)], name="created_at_ttl", expireAfterSeconds=ttl)
        except OperationFailure:
            # TTL value changed since the index was created: update it in place.
            self._db.command("collMod", self._coll_name,
                             index={"name": "created_at_ttl", "expireAfterSeconds": ttl})

    def ping(self):
        try:
            self._client.admin.command("ping")
        except self._err as exc:
            raise RepositoryError(f"ping failed: {type(exc).__name__}") from exc

    def count(self):
        try:
            return self._coll.estimated_document_count()
        except self._err as exc:
            raise RepositoryError(type(exc).__name__) from exc

    def load_embeddings(self, model_name, model_version):
        try:
            cur = self._coll.find({"model_name": model_name, "model_version": model_version},
                                  {"name": 1, "embeddings": 1})
            return [{"id": str(d["_id"]), "name": d["name"],
                     "embeddings": np.asarray(d["embeddings"], dtype=np.float64)} for d in cur]
        except self._err as exc:
            raise RepositoryError(type(exc).__name__) from exc

    def insert(self, name, embeddings, model_name, model_version):
        doc = {"name": name, "embeddings": [[float(x) for x in e] for e in embeddings],
               "embedding_dim": int(len(embeddings[0])), "sample_count": len(embeddings),
               "model_name": model_name, "model_version": model_version,
               "created_at": datetime.now(timezone.utc)}
        try:
            return str(self._coll.insert_one(doc).inserted_id)
        except self._err as exc:
            raise RepositoryError(type(exc).__name__) from exc
