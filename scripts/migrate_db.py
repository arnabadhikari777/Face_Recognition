#!/usr/bin/env python3
"""Migrate the OLD collection (schema v1: registered_faces, one SFace 128-d vector list per person) to the new hybrid schema.

What it does (nothing is deleted unless you pass --delete-legacy):
  * copies every legacy record into the new `identities` collection as schema v2 with migration.status = "legacy_unmatched"
    (name, created_at, the legacy vectors under legacy.<model_name> - kept for audit only)
  * removes the old 90-day TTL index from the legacy collection so existing records stop expiring (--keep-ttl to leave it)
  * records are SKIPPED if already migrated (idempotent)
Why they cannot be matched: embeddings of a different model live in a different vector space. SFace (128-d) vectors can not be
converted to w600k_mbf / w600k_r50 vectors, so these people have to be registered again once. The new code ignores
"legacy_unmatched" records for matching (never compares incompatible vectors).
Usage: MONGODB_URI=... python scripts/migrate_db.py [--dry-run] [--delete-legacy] [--keep-ttl]
"""
import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import SCHEMA_VERSION, TEMPLATE_VERSION, name_key  # noqa: E402


def migrate(client, db_name, legacy_name="registered_faces", new_name="identities", dry_run=False, delete_legacy=False, keep_ttl=False, log=print):
    db = client[db_name]
    legacy, new = db[legacy_name], db[new_name]
    stats = {"found": 0, "migrated": 0, "skipped": 0, "ttl_removed": False, "deleted_legacy": 0}
    if not keep_ttl and "created_at_ttl" in legacy.index_information():
        stats["ttl_removed"] = True
        if not dry_run:
            legacy.drop_index("created_at_ttl")
        log("legacy TTL index (automatic deletion) removed" if not dry_run else "[dry-run] would remove legacy TTL index")
    for d in legacy.find({}):
        stats["found"] += 1
        if new.find_one({"migration.legacy_id": str(d["_id"])}):
            stats["skipped"] += 1
            continue
        doc = {"name": d["name"], "name_key": name_key(d["name"]), "schema_version": SCHEMA_VERSION, "template_version": TEMPLATE_VERSION,
               "created_at": d.get("created_at", datetime.now(timezone.utc)), "updated_at": datetime.now(timezone.utc),
               "models": {}, "templates": [],
               "migration": {"status": "legacy_unmatched", "legacy_id": str(d["_id"]), "legacy_collection": legacy_name,
                             "reason": "legacy embeddings belong to another model; re-registration required"},
               "legacy": {"model_name": d.get("model_name"), "model_version": d.get("model_version"),
                          "embedding_dim": d.get("embedding_dim"), "embeddings": d.get("embeddings")}}
        if not dry_run:
            new.insert_one(doc)
        stats["migrated"] += 1
    if stats["migrated"] and not dry_run:
        db["meta"].update_one({"_id": "revision"}, {"$inc": {"value": 1}}, upsert=True)
    if delete_legacy and not dry_run:
        stats["deleted_legacy"] = legacy.delete_many({}).deleted_count
    log(f"found={stats['found']} migrated={stats['migrated']} skipped={stats['skipped']} deleted_legacy={stats['deleted_legacy']}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--delete-legacy", action="store_true", help="delete the legacy collection AFTER copying (irreversible)")
    ap.add_argument("--keep-ttl", action="store_true")
    ap.add_argument("--legacy-collection", default="registered_faces")
    a = ap.parse_args()
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        sys.exit("MONGODB_URI is not set")
    import pymongo
    client = pymongo.MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=8000)
    migrate(client, os.environ.get("MONGODB_DB_NAME", "face_recognition_db"), a.legacy_collection,
            os.environ.get("MONGODB_COLLECTION", "identities"), a.dry_run, a.delete_legacy, a.keep_ttl)


if __name__ == "__main__":
    main()
