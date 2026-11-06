import unittest
from evaluation.metrics import score_ranking
from evaluation.run import load_dataset


class EvaluationTests(unittest.TestCase):
    def test_known_ranking(self):
        result = score_ranking(["x", "a", "b", "y"], {"a", "b", "c"}, 4)
        self.assertEqual(result["precision"], .5)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["mrr"], .5)
        self.assertEqual(result["complete_evidence"], 0)

    def test_unanswerable_excluded_and_duplicates_not_counted(self):
        self.assertIsNone(score_ranking(["a"], [], 4))
        self.assertEqual(score_ranking(["a", "a"], ["a"], 4)["precision"], .25)

    def test_dataset_references_and_holdout(self):
        data = load_dataset()
        ids = {a["id"] for a in data["articles"]}
        self.assertGreaterEqual(len(data["questions"]), 40)
        self.assertEqual({q["split"] for q in data["questions"]}, {"dev", "test"})
        for query in data["questions"]:
            self.assertTrue(set(query["evidence"]) <= ids)
            self.assertEqual(bool(query["evidence"]), bool(query["answer_points"]))
