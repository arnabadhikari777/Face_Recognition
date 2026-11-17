import unittest

import numpy as np

from backend import face_matching as fm
from config import Config
from tests.fakes import make_person, samples_of, unit

CFG = Config()


def rec(i, name, emb):
    return {"id": str(i), "name": name, "embeddings": emb}


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = make_person(1), make_person(2)
        self.records = [rec(0, "A", samples_of(self.a, 10)), rec(1, "B", samples_of(self.b, 11))]

    def test_same_person_is_match(self):
        q = samples_of(self.a, 99, n=1)[0]
        ranked = fm.rank_candidates(q, self.records, CFG.MATCH_THRESHOLD)
        verdict, best = fm.classify_recognition(ranked, CFG)
        self.assertEqual((verdict, best.name), ("match", "A"))

    def test_different_person_is_unknown(self):
        q = make_person(500)
        ranked = fm.rank_candidates(q, self.records, CFG.MATCH_THRESHOLD)
        self.assertEqual(fm.classify_recognition(ranked, CFG)[0], "unknown")

    def test_empty_database(self):
        self.assertEqual(fm.classify_recognition([], CFG)[0], "unknown")
        self.assertEqual(fm.classify_duplicate([], CFG, 3)[0], "new")

    def _at_similarity(self, base, s, seed):
        """Vector with cosine similarity exactly s to base (synthetic)."""
        r = unit(np.random.default_rng(seed).normal(size=base.size))
        r = unit(r - (r @ base) * base)
        return s * base + np.sqrt(1 - s * s) * r

    def test_threshold_behaviour(self):
        recs = [rec(0, "A", self.a[None, :])]
        for s, expected in ((0.50, "match"), (0.37, "match"), (0.33, "uncertain"), (0.10, "unknown")):
            q = self._at_similarity(self.a, s, 7)
            verdict, best = fm.classify_recognition(fm.rank_candidates(q, recs, CFG.MATCH_THRESHOLD), CFG)
            self.assertEqual(verdict, expected, f"similarity {s}")
            self.assertAlmostEqual(best.score, s, places=6)

    def test_ambiguous_two_people_is_uncertain(self):
        mid = unit(self.a + self.b)
        recs = [rec(0, "A", self.a[None, :]), rec(1, "B", self.b[None, :])]
        q = unit(mid)  # equally similar to both
        ranked = fm.rank_candidates(q, recs, 0.0)
        ranked[0].score = ranked[1].score + 0.01
        ranked[0].score = max(ranked[0].score, CFG.MATCH_THRESHOLD)
        ranked[1].score = ranked[0].score - 0.01
        self.assertEqual(fm.classify_recognition(ranked, CFG)[0], "uncertain")

    def test_duplicate_needs_supporting_samples(self):
        same = samples_of(self.a, 55)
        ranked = fm.rank_candidates(same, self.records, CFG.MATCH_THRESHOLD)
        verdict, best = fm.classify_duplicate(ranked, CFG, 3)
        self.assertEqual((verdict, best.name), ("duplicate", "A"))
        self.assertGreaterEqual(best.supporting, 2)
        other = samples_of(make_person(777), 56)
        ranked = fm.rank_candidates(other, self.records, CFG.MATCH_THRESHOLD)
        self.assertEqual(fm.classify_duplicate(ranked, CFG, 3)[0], "new")

    def test_one_lucky_sample_is_not_duplicate(self):
        # only 1 of 3 samples close to A, mean score below threshold
        q = np.stack([self._at_similarity(self.a, 0.9, 1001), make_person(901), make_person(902)])
        ranked = fm.rank_candidates(q, [rec(0, "A", self.a[None, :])], CFG.MATCH_THRESHOLD)
        self.assertNotEqual(fm.classify_duplicate(ranked, CFG, 3)[0], "duplicate")


if __name__ == "__main__":
    unittest.main()
