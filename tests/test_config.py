import os
import tempfile
import unittest

from config import Config, load_dotenv_file


class DotenvTests(unittest.TestCase):
    def test_loads_without_overriding(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# comment\nMONGODB_URI=\"mongodb+srv://u:p@h/?x=1\"\nTRUST_PROXY=true\nBAD LINE\nKEEP=file\n")
            env = {"KEEP": "real"}
            self.assertEqual(load_dotenv_file(path, env), 2)
            self.assertEqual(env["MONGODB_URI"], "mongodb+srv://u:p@h/?x=1")
            self.assertEqual(env["KEEP"], "real")

    def test_missing_file_is_fine(self):
        self.assertEqual(load_dotenv_file("/nonexistent/.env", {}), 0)

    def test_from_env(self):
        cfg = Config.from_env({"MONGODB_URI": "x", "TRUST_PROXY": "true", "ALLOWED_ORIGINS": "https://a.example, https://b.example"})
        self.assertTrue(cfg.TRUST_PROXY)
        self.assertEqual(cfg.ALLOWED_ORIGINS, ("https://a.example", "https://b.example"))
        self.assertEqual(cfg.EMBEDDING_DIM, 128)


if __name__ == "__main__":
    unittest.main()
