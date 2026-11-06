import unittest
from news_research.ranking import bm25_rank, reciprocal_rank_fusion


class RankingTests(unittest.TestCase):
    def test_exact_identifier_and_empty_match(self):
        texts = ["Bank revenue rose", "MicroLeaf ML-42 sensor chip consumes 2 milliwatts", "Solar plant opened"]
        self.assertEqual(bm25_rank(texts, "ML-42", 2), [1])
        self.assertEqual(bm25_rank(texts, "unmentioned", 4), [])
        self.assertEqual(bm25_rank([], "question", 4), [])

    def test_fusion_agreement_dedup_and_limit(self):
        self.assertEqual(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], 2)[0], "b")
        self.assertEqual(reciprocal_rank_fusion([["a", "a"], ["b"]], 2), ["a", "b"])
