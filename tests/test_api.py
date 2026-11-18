import json
import unittest

from backend import create_app
from config import Config
from tests.fakes import FakeRepo, make_person, samples_of

CFG = Config(RATE_REGISTER_PER_HOUR=1000, RATE_RECOGNIZE_PER_MINUTE=1000, RATE_DEFAULT_PER_MINUTE=1000)


def reg(name, emb, **kw):
    b = {"name": name, "consent": True, "model_name": CFG.MODEL_NAME, "model_version": CFG.MODEL_VERSION,
         "embeddings": [e.tolist() for e in emb]}
    b.update(kw)
    return b


def rec(emb, **kw):
    b = {"consent": True, "model_name": CFG.MODEL_NAME, "model_version": CFG.MODEL_VERSION,
         "embedding": emb.tolist()}
    b.update(kw)
    return b


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.repo = FakeRepo()
        self.app = create_app(CFG, self.repo)
        self.c = self.app.test_client()
        self.alice = make_person(1)

    def post(self, path, body):
        return self.c.post(path, json=body)

    def test_index_and_headers(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Register Face", r.data)
        self.assertIn("camera=(self)", r.headers["Permissions-Policy"])
        self.assertIn("default-src 'self'", r.headers["Content-Security-Policy"])

    def test_config_endpoint(self):
        d = self.c.get("/api/config").get_json()
        self.assertEqual(d["embedding_dim"], 128)

    def test_health_ok_and_down(self):
        r = self.c.get("/api/health")
        self.assertEqual((r.status_code, r.get_json()["database"]), (200, "ok"))
        self.repo.down = True
        r = self.c.get("/api/health")
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("simulated", r.get_data(as_text=True))

    def test_register_then_duplicate_then_recognize(self):
        r = self.post("/api/register-face", reg("Alice", samples_of(self.alice, 5)))
        self.assertEqual((r.status_code, r.get_json()["status"]), (201, "registered"))
        self.assertEqual(r.get_json()["message"], "Face registered successfully.")
        self.assertEqual(len(self.repo.docs), 1)
        # duplicate under another name
        r = self.post("/api/register-face", reg("Mallory", samples_of(self.alice, 6)))
        j = r.get_json()
        self.assertEqual((r.status_code, j["status"], j["existing_name"]), (409, "duplicate", "Alice"))
        self.assertEqual(len(self.repo.docs), 1)
        # recognise
        r = self.post("/api/recognize-face", rec(samples_of(self.alice, 7, n=1)[0]))
        j = r.get_json()
        self.assertEqual((r.status_code, j["status"], j["name"]), (200, "match", "Alice"))
        self.assertIn("similarity_score", j)

    def test_unknown_face(self):
        self.post("/api/register-face", reg("Alice", samples_of(self.alice, 5)))
        r = self.post("/api/recognize-face", rec(make_person(321)))
        self.assertEqual(r.get_json()["status"], "unknown")
        self.assertEqual(r.get_json()["message"], "Unknown Face")

    def test_stored_data_has_no_images_and_response_has_no_embeddings(self):
        r = self.post("/api/register-face", reg("Alice", samples_of(self.alice, 5)))
        self.assertNotIn("embeddings", r.get_json())
        self.assertEqual(set(self.repo.docs[0]), {"name", "embeddings", "model_name", "model_version"})

    def test_invalid_requests(self):
        bad_dim = reg("A", samples_of(self.alice, 5))
        bad_dim["embeddings"][0] = bad_dim["embeddings"][0][:100]
        cases = [(reg("", samples_of(self.alice, 5)), 400), (bad_dim, 400),
                 (reg("A", samples_of(self.alice, 5), consent=False), 400),
                 (reg("A", samples_of(self.alice, 5), model_version="x"), 409)]
        for body, status in cases:
            self.assertEqual(self.post("/api/register-face", body).status_code, status)
        self.assertEqual(len(self.repo.docs), 0)
        r = self.c.post("/api/register-face", data="not json", content_type="application/json")
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/register-face", data="x", content_type="text/plain")
        self.assertEqual(r.status_code, 415)
        r = self.post("/api/register-face", [1, 2])
        self.assertEqual(r.status_code, 400)

    def test_nan_and_infinity_in_raw_json(self):
        good = rec(self.alice)
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            raw = json.dumps(good).replace(str(good["embedding"][0]), token, 1)
            r = self.c.post("/api/recognize-face", data=raw, content_type="application/json")
            self.assertEqual(r.status_code, 400, token)
            self.assertEqual(r.get_json()["code"], "invalid_embedding", token)

    def test_database_unavailable(self):
        self.repo.down = True
        r = self.post("/api/recognize-face", rec(self.alice))
        self.assertEqual((r.status_code, r.get_json()["code"]), (503, "database_unavailable"))
        r = self.post("/api/register-face", reg("A", samples_of(self.alice, 5)))
        self.assertEqual(r.status_code, 503)

    def test_unexpected_error_hides_details(self):
        self.repo.load_embeddings = lambda *a: 1 / 0
        r = self.post("/api/recognize-face", rec(self.alice))
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("division", r.get_data(as_text=True))

    def test_payload_too_large(self):
        r = self.c.post("/api/register-face", data="x" * 70000, content_type="application/json")
        self.assertEqual(r.status_code, 413)

    def test_capacity_limit(self):
        app = create_app(Config(MAX_RECORDS=0, RATE_REGISTER_PER_HOUR=1000), self.repo)
        r = app.test_client().post("/api/register-face", json=reg("A", samples_of(self.alice, 5)))
        self.assertEqual((r.status_code, r.get_json()["code"]), (503, "capacity_reached"))

    def test_rate_limiting(self):
        app = create_app(Config(RATE_RECOGNIZE_PER_MINUTE=3), self.repo)
        c = app.test_client()
        codes = [c.post("/api/recognize-face", json=rec(self.alice)).status_code for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])
        r = c.post("/api/recognize-face", json=rec(self.alice))
        self.assertIn("Retry-After", r.headers)

    def test_cors_only_for_allowed_origin(self):
        app = create_app(Config(ALLOWED_ORIGINS=("https://ok.example",), RATE_DEFAULT_PER_MINUTE=1000), self.repo)
        c = app.test_client()
        r = c.get("/api/health", headers={"Origin": "https://ok.example"})
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "https://ok.example")
        r = c.get("/api/health", headers={"Origin": "https://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", r.headers)
        r = self.c.get("/api/health", headers={"Origin": "https://ok.example"})
        self.assertNotIn("Access-Control-Allow-Origin", r.headers)

    def test_app_boots_without_database_config(self):
        app = create_app(Config(MONGODB_URI=""))
        r = app.test_client().get("/api/health")
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()
