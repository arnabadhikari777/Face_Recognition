import mongomock

from scripts.migrate_db import migrate
from tests.synthetic import Person, make_repo, recognize_body, register_body
from backend import create_app
from config import Config
from datetime import datetime, timezone


def test_migration_is_safe_idempotent_and_legacy_never_matches():
    repo, client = make_repo()
    db = client["t"]
    db["registered_faces"].create_index("created_at", name="created_at_ttl", expireAfterSeconds=7776000)
    db["registered_faces"].insert_many([{"name": "Old Alice", "embeddings": [[0.1] * 128] * 3, "embedding_dim": 128, "sample_count": 3,
                                          "model_name": "opencv-zoo-sface", "model_version": "2021dec", "created_at": datetime.now(timezone.utc)}])
    s = migrate(client, "t", log=lambda *_: None, dry_run=True)
    assert s["migrated"] == 1 and db["identities"].count_documents({}) == 0 and "created_at_ttl" in db["registered_faces"].index_information()
    s = migrate(client, "t", log=lambda *_: None)
    assert s["migrated"] == 1 and s["ttl_removed"] and "created_at_ttl" not in db["registered_faces"].index_information()
    assert db["registered_faces"].count_documents({}) == 1                      # legacy data kept
    assert migrate(client, "t", log=lambda *_: None)["skipped"] == 1             # idempotent
    doc = db["identities"].find_one({})
    assert doc["migration"]["status"] == "legacy_unmatched" and doc["models"] == {}
    c = create_app(Config(RATE_REGISTER_PER_HOUR=99, RATE_RECOGNIZE_PER_MINUTE=99), repo).test_client()
    assert c.post("/api/recognize-face", json=recognize_body(Person(1))).json["status"] == "unknown"   # never matched
    assert c.post("/api/register-face", json=register_body("Old Alice", Person(1))).status_code == 201  # re-registration works
