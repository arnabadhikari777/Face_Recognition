import unittest

import numpy as np

from backend.validation import ApiError, parse_recognize, parse_register, validate_embedding, validate_name
from config import Config
from tests.fakes import make_person, samples_of

CFG = Config()


def reg_body(**kw):
    p = make_person(1)
    body = {"name": "Alice", "consent": True, "model_name": CFG.MODEL_NAME,
            "model_version": CFG.MODEL_VERSION,
            "embeddings": [e.tolist() for e in samples_of(p, 2)]}
    body.update(kw)
    return body


class ValidationTests(unittest.TestCase):
    def code(self, fn, *a):
        with self.assertRaises(ApiError) as cm:
            fn(*a)
        return cm.exception.code

    def test_valid_register(self):
        name, arr = parse_register(reg_body(name="  Alice   B "), CFG)
        self.assertEqual(name, "Alice B")
        self.assertEqual(arr.shape, (3, 128))

    def test_name_rules(self):
        self.assertEqual(self.code(validate_name, ""), "invalid_name")
        self.assertEqual(self.code(validate_name, "x" * 61), "invalid_name")
        self.assertEqual(self.code(validate_name, 5), "invalid_name")
        self.assertEqual(self.code(validate_name, "a\x00b"), "invalid_name")

    def test_consent_required(self):
        for v in (False, "true", 1, None):
            self.assertEqual(self.code(parse_register, reg_body(consent=v), CFG), "consent_required")

    def test_model_mismatch(self):
        self.assertEqual(self.code(parse_register, reg_body(model_version="other"), CFG), "model_mismatch")

    def test_wrong_dimension_and_count(self):
        self.assertEqual(self.code(validate_embedding, [0.1] * 127, 128), "invalid_embedding")
        b = reg_body()
        b["embeddings"] = b["embeddings"][:2]
        self.assertEqual(self.code(parse_register, b, CFG), "invalid_embedding")

    def test_nan_inf_and_types(self):
        good = make_person(3).tolist()
        for bad in (float("nan"), float("inf"), float("-inf"), "0.1", None, True):
            e = list(good)
            e[5] = bad
            self.assertEqual(self.code(validate_embedding, e, 128), "invalid_embedding")

    def test_not_normalised(self):
        self.assertEqual(self.code(validate_embedding, (make_person(3) * 3).tolist(), 128), "invalid_embedding")
        self.assertEqual(self.code(validate_embedding, [0.0] * 128, 128), "invalid_embedding")

    def test_inconsistent_samples(self):
        b = reg_body()
        b["embeddings"] = [make_person(s).tolist() for s in (10, 11, 12)]
        self.assertEqual(self.code(parse_register, b, CFG), "inconsistent_samples")

    def test_recognize(self):
        e = make_person(4).tolist()
        ok = parse_recognize({"consent": True, "model_name": CFG.MODEL_NAME,
                              "model_version": CFG.MODEL_VERSION, "embedding": e}, CFG)
        self.assertTrue(np.isclose(np.linalg.norm(ok), 1.0))
        self.assertEqual(self.code(parse_recognize, {"consent": False, "embedding": e}, CFG), "consent_required")


if __name__ == "__main__":
    unittest.main()
