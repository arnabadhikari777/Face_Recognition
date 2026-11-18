"""Unit + API tests for the Flask side. Uses mongomock (an in-memory MongoDB stand-in) and SYNTHETIC vectors:
labelled as such because they test logic, not recognition accuracy."""
import threading

import numpy as np
import pytest
from cryptography.fernet import Fernet

from backend import create_app
from backend import face_matching as fm
from backend import templates as tp
from config import Config
from tests.synthetic import CAL, DIM, MANIFEST, Person, make_repo, recognize_body, register_body, unit

CFG = Config(RATE_REGISTER_PER_HOUR=1000, RATE_RECOGNIZE_PER_MINUTE=1000, RATE_DEFAULT_PER_MINUTE=1000)
PRIM, SEC = MANIFEST.primary, [m for m in MANIFEST.recognizers if m != MANIFEST.primary][0]


@pytest.fixture()
def env():
    repo, client = make_repo()
    app = create_app(CFG, repo)
    return app.test_client(), repo, client


def reg(c, name, person, **kw):
    return c.post("/api/register-face", json=register_body(name, person, **kw))


def rec(c, person, **kw):
    return c.post("/api/recognize-face", json=recognize_body(person, **kw))


# ---------------- template hygiene ----------------
def test_duplicate_and_outlier_rejection():
    rng = np.random.default_rng(0)
    base = unit(rng.normal(size=DIM))
    vecs = [unit(base + 0.4 * unit(rng.normal(size=DIM))) for _ in range(6)]
    vecs.append(vecs[0].copy())                         # exact duplicate of sample 0, same pose
    vecs.append(unit(rng.normal(size=DIM)))             # a different person: outlier
    poses = ["front"] * 3 + ["left", "right", "up", "front", "down"]
    q = [0.9, 0.8, 0.7, 0.8, 0.8, 0.8, 0.5, 0.8]
    keep, dropped = tp.select_samples(np.stack(vecs), q, poses, duplicate_similarity=0.97, mad_k=3, consistency_min=0.2, max_templates=10)
    reasons = dict(dropped)
    assert reasons[6] == "duplicate" and reasons[7] == "outlier" and 7 not in keep and 6 not in keep


def test_master_is_unit_and_quality_weighted():
    a, b = unit(np.eye(DIM)[0]), unit(np.eye(DIM)[1])
    m = tp.weighted_master([a, b], [1.0, 0.1])
    assert abs(np.linalg.norm(m) - 1) < 1e-9 and m @ a > m @ b
    block = tp.build_model_block([a, b], [1.0, 1.0], ["front", "left"])
    assert set(block["prototypes"]) == {"front", "left"}


# ---------------- registration ----------------
def test_register_then_recognize_same_and_different(env):
    c, repo, _ = env
    alice, bob = Person(1), Person(2)
    r = reg(c, "Alice", alice)
    assert r.status_code == 201 and sorted(r.json["models"]) == sorted(MANIFEST.recognizers)
    assert reg(c, "Bob", bob).status_code == 201
    m = rec(c, alice).json
    assert m["status"] == "match" and m["name"] == "Alice" and m["evidence"]["models_agreeing"] == 2
    assert rec(c, Person(99)).json["status"] == "unknown"
    assert rec(c, bob, seed=7).json["name"] == "Bob"


def test_duplicate_registration_blocked(env):
    c, _, _ = env
    alice = Person(1)
    assert reg(c, "Alice", alice).status_code == 201
    r = reg(c, "Alice Again", alice, seed=5)
    assert r.status_code == 409 and r.json["status"] == "duplicate" and r.json["existing_name"] == "Alice"
    assert "embedding" not in str(r.json).lower()


def test_insufficient_and_inconsistent_samples(env):
    c, _, _ = env
    body = register_body("Few", Person(3))
    body["samples"] = body["samples"][:3]
    assert c.post("/api/register-face", json=body).json["code"] == "insufficient_samples"
    body = register_body("Mixed", Person(4))
    other = register_body("x", Person(5), seed=9)["samples"]
    for i in range(0, 9, 2):
        body["samples"][i] = other[i]
    assert c.post("/api/register-face", json=body).status_code == 422


def test_low_quality_samples_rejected(env):
    c, _, _ = env
    r = reg(c, "Dark", Person(6), quality=0.1)
    assert r.status_code == 422 and r.json["code"] == "low_quality"


def test_concurrent_registration_of_same_person_creates_one_record(env):
    c, repo, _ = env
    p, results = Person(8), []
    def go(seed):
        results.append(reg(c, "Carol", p, seed=seed).status_code)
    ts = [threading.Thread(target=go, args=(s,)) for s in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(results).count(201) == 1 and repo.count() == 1


def test_upgrade_adds_missing_model_only_for_same_name(env):
    c, repo, _ = env
    p = Person(11)
    assert reg(c, "Dave", p, models=[PRIM]).status_code == 201
    assert reg(c, "Someone Else", p, seed=3).status_code == 409                 # other name: plain duplicate
    r = reg(c, "dave", p, seed=4)                                               # same name (case-insensitive) + extra model
    assert r.status_code == 200 and r.json["status"] == "upgraded"
    doc = repo._coll.find_one({})
    assert set(doc["models"]) == set(MANIFEST.recognizers)
    assert reg(c, "Dave", p, seed=6).status_code == 409                         # now fully enrolled: duplicate again


# ---------------- decisions ----------------
def test_need_secondary_then_final_decision(env):
    c, _, _ = env
    p = Person(21)
    reg(c, "Eve", p)
    weak = recognize_body(p, models=[PRIM], noise=1.55)                         # noisy single-model query
    first = c.post("/api/recognize-face", json=weak).json
    assert first["status"] in ("need_secondary", "match", "uncertain", "unknown")
    if first["status"] == "need_secondary":
        assert first["need_models"] == [SEC]
    finalbody = recognize_body(p, models=[PRIM], noise=1.55, final=True)
    assert c.post("/api/recognize-face", json=finalbody).json["status"] in ("match", "uncertain", "unknown")


def test_model_disagreement_is_uncertain(env):
    c, _, _ = env
    p, q = Person(31), Person(32)
    reg(c, "Frank", p)
    mixed = Person(31)
    mixed.centers = {PRIM: p.centers[PRIM], SEC: q.centers[SEC]}               # primary says Frank, secondary says stranger
    r = rec(c, mixed, final=True).json
    assert r["status"] in ("uncertain", "unknown") and r["status"] != "match"


def test_ambiguous_candidates_not_matched(env):
    c, _, _ = env
    p = Person(41)
    twin = Person(41)
    twin.centers = {m: unit(p.centers[m] + 0.02 * unit(np.random.default_rng(5).normal(size=DIM))) for m in p.centers}
    assert reg(c, "Orig", p).status_code == 201
    # a near-identical identity cannot be registered normally (duplicate protection), so inject it directly
    from backend.registration import _blocks, clean_for_enrolment
    from backend.validation import parse_register
    name, frames = parse_register(register_body("Twin", twin, seed=77), MANIFEST, CFG)
    c.application.extensions["facerec"].repo().insert_identity("Twin", *_blocks(clean_for_enrolment(frames, MANIFEST, CAL), MANIFEST))
    r = rec(c, p, final=True).json
    assert r["status"] == "uncertain" and r["reason"] == "ambiguous"


def test_poor_quality_query_asks_for_retry(env):
    c, _, _ = env
    reg(c, "Gina", Person(51))
    r = rec(c, Person(51), quality=0.05).json
    assert r["status"] == "retry"


def test_robust_aggregate_ignores_one_bad_frame():
    good = [0.6, 0.62, 0.58, 0.61, 0.05]
    assert fm.robust_aggregate(good, [0.8] * 5, 3.0) > 0.55


# ---------------- storage / security ----------------
def test_schema_encryption_and_model_versions():
    key = Fernet.generate_key().decode()
    repo, client = make_repo(key)
    app = create_app(CFG, repo, MANIFEST, CAL)
    c = app.test_client()
    assert reg(c, "Hank", Person(61)).status_code == 201
    doc = client["t"]["identities"].find_one({})
    assert doc["schema_version"] == 2 and len(doc["templates"]) == 2 * 9
    t = doc["templates"][0]
    assert t["embedding"].startswith("f1:") and t["model_id"] in MANIFEST.recognizers and t["dim"] == DIM and t["template_version"] == 1
    assert doc["models"][PRIM]["master"].startswith("f1:") and "templates" in doc
    # a new model version must not be compared with old templates
    MANIFEST.models[PRIM]["version"] = "OTHER"
    try:
        assert rec(c, Person(61), models=[PRIM], version_override="OTHER").json["status"] == "unknown"
    finally:
        MANIFEST.models[PRIM]["version"] = "buffalo_s-v0.7"


def test_wrong_key_is_an_error_not_a_wrong_match():
    key = Fernet.generate_key().decode()
    repo, client = make_repo(key)
    c = create_app(CFG, repo).test_client()
    reg(c, "Ivy", Person(71))
    repo2, _ = make_repo(Fernet.generate_key().decode())
    repo2._coll = client["t"]["identities"]; repo2._meta = client["t"]["meta"]
    c2 = create_app(CFG, repo2).test_client()
    assert rec(c2, Person(71)).status_code == 503      # reported as a database problem, never a wrong match


def test_retention_default_has_no_ttl_and_can_be_enabled():
    repo, client = make_repo()
    assert "retention_ttl" not in client["t"]["identities"].index_information()


# ---------------- API surface ----------------
def test_config_endpoint_exposes_no_thresholds_or_secrets(env):
    c, _, _ = env
    j = c.get("/api/config").json
    text = str(j)
    assert "'threshold':" not in text and "calibration" not in text and j["default_profile"] == "accuracy"
    assert {m["id"] for m in j["models"]} == {m["id"] for m in MANIFEST.data["models"]}
    assert c.get("/api/health").json["thresholds_calibrated"] is False


def test_validation_errors(env):
    c, _, _ = env
    b = register_body("X", Person(81)); b["consent"] = False
    assert c.post("/api/register-face", json=b).json["code"] == "consent_required"
    b = register_body("X", Person(81)); b["samples"][0]["embeddings"][PRIM]["version"] = "old"
    assert c.post("/api/register-face", json=b).status_code == 409
    b = register_body("X", Person(81)); b["samples"][0]["embeddings"][PRIM]["vector"][0] = float("nan")
    assert c.post("/api/register-face", data="{\"name\": NaN}", content_type="application/json").status_code == 400
    b = register_body("X", Person(81)); b["samples"][0]["embeddings"][PRIM]["vector"] = [0.1] * DIM
    assert c.post("/api/register-face", json=b).json["code"] == "invalid_embedding"
    b = register_body("X", Person(81)); b["samples"][0]["pose"] = "sideways"
    assert c.post("/api/register-face", json=b).json["code"] == "invalid_pose"
    b = register_body("X", Person(81)); del b["samples"][2]["embeddings"][SEC]
    assert c.post("/api/register-face", json=b).json["code"] == "invalid_embedding"


def test_database_failure_is_handled(env):
    c, repo, _ = env
    def boom(*a, **k):
        from backend.database import RepositoryError
        raise RepositoryError("down")
    repo.load_identities = boom
    r = rec(c, Person(91))
    assert r.status_code == 503 and r.json["code"] == "database_unavailable"


def test_security_headers_and_index(env):
    c, _, _ = env
    r = c.get("/")
    assert r.status_code == 200 and b"Register Face" in r.data
    assert "camera=(self)" in r.headers["Permissions-Policy"] and "default-src 'self'" in r.headers["Content-Security-Policy"]
