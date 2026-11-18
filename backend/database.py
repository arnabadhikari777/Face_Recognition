"""MongoDB access, schema v2 (hybrid biometric storage). pymongo is imported lazily so the unit tests run without it.

identities (one document per person):
  name, name_key, schema_version=2, created_at, updated_at, template_version
  models.<model_id> = {version, dim, master (encrypted), prototypes{pose: encrypted}, template_count}
  templates[] = {template_id, model_id, model_version, dim, embedding (encrypted), pose, quality{...}, captured_at, template_version}
meta   : {_id:'revision', value:int}  -> bumped on every write; lets every worker cache decrypted matrices safely
locks  : {_id:'registration', expires_at}  -> cross-process lock (TTL) for atomic duplicate protection
Templates of another model version are kept untouched but never loaded for matching (no silent comparison of incompatible vectors)."""
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from .crypto import CryptoError

SCHEMA_VERSION, TEMPLATE_VERSION = 2, 1


class RepositoryError(Exception):
    """Any database failure. Message is safe to log, never shown to visitors."""


class LockBusy(Exception):
    pass


@dataclass
class IdentityView:
    id: str
    name: str
    models: dict = field(default_factory=dict)   # model_id -> {'master','prototypes','templates'(n,d),'template_poses'}


def name_key(name):
    return " ".join(name.split()).casefold()


class MongoRepository:
    def __init__(self, uri="", db_name="face_recognition_db", collection="identities", retention_days=0, cipher=None, client=None):
        if client is None:
            if not uri:
                raise RepositoryError("MONGODB_URI is not set")
            try:
                import pymongo
            except ImportError as exc:  # pragma: no cover
                raise RepositoryError("pymongo is not installed") from exc
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000, socketTimeoutMS=10000,
                                         tz_aware=True, appname="face-recognition")
        try:
            from pymongo.errors import PyMongoError
        except ImportError:  # pragma: no cover
            PyMongoError = Exception  # noqa: N806
        self._err = PyMongoError
        self._cipher = cipher
        self._cache, self._cache_key = None, None
        self._mem_lock = threading.Lock()
        try:
            self._client = client
            self._db = client[db_name]
            self._coll = self._db[collection]
            self._meta, self._locks = self._db["meta"], self._db["locks"]
            self._client.admin.command("ping")
            self._ensure_indexes(retention_days)
        except self._err as exc:
            raise RepositoryError(f"connection failed: {type(exc).__name__}") from exc

    # ---- setup -----------------------------------------------------------
    def _ensure_indexes(self, retention_days):
        self._coll.create_index([("name_key", 1)], name="name_key_idx")
        self._locks.create_index([("expires_at", 1)], name="lock_ttl", expireAfterSeconds=0)
        existing = self._coll.index_information()
        if retention_days and int(retention_days) > 0:
            ttl = int(retention_days) * 86400
            if "retention_ttl" in existing:
                self._db.command("collMod", self._coll.name, index={"name": "retention_ttl", "expireAfterSeconds": ttl})
            else:
                self._coll.create_index([("created_at", 1)], name="retention_ttl", expireAfterSeconds=ttl)
        elif "retention_ttl" in existing:        # retention switched off: remove the automatic deletion
            self._coll.drop_index("retention_ttl")

    def ping(self):
        try:
            self._client.admin.command("ping")
        except self._err as exc:
            raise RepositoryError(f"ping failed: {type(exc).__name__}") from exc

    def count(self):
        try:
            return self._coll.count_documents({})
        except self._err as exc:
            raise RepositoryError(type(exc).__name__) from exc

    # ---- revision cache --------------------------------------------------
    def _revision(self):
        doc = self._meta.find_one({"_id": "revision"})
        return int(doc["value"]) if doc else 0

    def _bump(self):
        self._meta.update_one({"_id": "revision"}, {"$inc": {"value": 1}}, upsert=True)

    def load_identities(self, active_versions):
        """Decrypted, matrix-form identities for the CURRENT model versions only. Cached until any write changes the revision."""
        try:
            key = (self._revision(), tuple(sorted(active_versions.items())))
            if self._cache is not None and self._cache_key == key:
                return self._cache
            out = []
            for d in self._coll.find({"schema_version": SCHEMA_VERSION}):
                view = IdentityView(str(d["_id"]), d["name"])
                for mid, ver in active_versions.items():
                    meta = d.get("models", {}).get(mid)
                    if not meta or meta.get("version") != ver:
                        continue
                    dim = int(meta["dim"])
                    tpls = [t for t in d.get("templates", []) if t["model_id"] == mid and t["model_version"] == ver and int(t["dim"]) == dim]
                    if not tpls:
                        continue
                    view.models[mid] = {
                        "master": self._cipher.decrypt(meta["master"], dim),
                        "prototypes": {p: self._cipher.decrypt(v, dim) for p, v in meta.get("prototypes", {}).items()},
                        "templates": np.stack([self._cipher.decrypt(t["embedding"], dim) for t in tpls]),
                        "template_poses": [t["pose"] for t in tpls],
                    }
                if view.models:
                    out.append(view)
            self._cache, self._cache_key = out, key
            return out
        except self._err as exc:
            raise RepositoryError(type(exc).__name__) from exc
        except CryptoError as exc:
            raise RepositoryError(f"decryption failed: {exc}") from exc

    # ---- writes ----------------------------------------------------------
    def _model_docs(self, name, blocks, versions, dims, templates, now):
        models = {mid: {"version": versions[mid], "dim": dims[mid], "master": self._cipher.encrypt(b["master"]),
                        "prototypes": {p: self._cipher.encrypt(v) for p, v in b["prototypes"].items()},
                        "template_count": len(templates[mid]), "created_at": now} for mid, b in blocks.items()}
        tpl = [{"template_id": uuid.uuid4().hex, "model_id": mid, "model_version": versions[mid], "dim": dims[mid],
                "embedding": self._cipher.encrypt(t["vec"]), "pose": t["pose"], "quality": t["quality"], "captured_at": now,
                "template_version": TEMPLATE_VERSION} for mid, lst in templates.items() for t in lst]
        return models, tpl

    def insert_identity(self, name, blocks, versions, dims, templates):
        """blocks/templates are per model id. Returns the new id."""
        now = datetime.now(timezone.utc)
        models, tpl = self._model_docs(name, blocks, versions, dims, templates, now)
        doc = {"name": name, "name_key": name_key(name), "schema_version": SCHEMA_VERSION, "template_version": TEMPLATE_VERSION,
               "created_at": now, "updated_at": now, "models": models, "templates": tpl, "migration": {"status": "native"}}
        try:
            _id = str(self._coll.insert_one(doc).inserted_id)
            self._bump()
            return _id
        except self._err as exc:
            raise RepositoryError(type(exc).__name__) from exc

    def add_models(self, identity_id, blocks, versions, dims, templates):
        """Attach templates of ADDITIONAL models to an existing identity (never overwrites an existing model)."""
        from bson import ObjectId
        now = datetime.now(timezone.utc)
        models, tpl = self._model_docs("", blocks, versions, dims, templates, now)
        try:
            for mid in models:
                res = self._coll.update_one({"_id": ObjectId(identity_id), f"models.{mid}": {"$exists": False}},
                                            {"$set": {f"models.{mid}": models[mid], "updated_at": now}})
                if res.matched_count == 0:
                    raise RepositoryError(f"model {mid} already present or identity missing")
            self._coll.update_one({"_id": ObjectId(identity_id)}, {"$push": {"templates": {"$each": tpl}}})
            self._bump()
        except self._err as exc:
            raise RepositoryError(type(exc).__name__) from exc

    # ---- atomic registration --------------------------------------------
    @contextmanager
    def registration_lock(self, ttl_seconds=30, wait_seconds=10):
        """Serialises check-then-insert across threads AND processes (a lock document with a TTL guards against crashed holders)."""
        deadline = time.monotonic() + wait_seconds
        token = uuid.uuid4().hex
        got = False
        with self._mem_lock:
            while True:
                try:
                    self._locks.delete_many({"_id": "registration", "expires_at": {"$lt": datetime.now(timezone.utc)}})
                    self._locks.insert_one({"_id": "registration", "token": token,
                                            "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)})
                    got = True
                    break
                except Exception as exc:  # noqa: BLE001 - DuplicateKeyError = somebody else holds it
                    if type(exc).__name__ != "DuplicateKeyError":
                        raise RepositoryError(type(exc).__name__) from exc
                    if time.monotonic() > deadline:
                        raise LockBusy() from exc
                    time.sleep(0.1)
            try:
                yield
            finally:
                if got:
                    try:
                        self._locks.delete_one({"_id": "registration", "token": token})
                    except self._err:
                        pass
