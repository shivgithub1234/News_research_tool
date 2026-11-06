import json
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from news_research.config import Settings
from news_research.comparison import compare_articles, comparison_excerpts


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.docs = [Document(page_content="Initial profit was 18 million rupees.", metadata={"article_id": "a", "source": "https://example.com/a"}),
                     Document(page_content="Revised profit was 16 million rupees.", metadata={"article_id": "b", "source": "https://example.com/b"})]
        self.excerpts = comparison_excerpts(self.docs, "profit")

    def payload(self):
        return {"status": "compared", "rows": [{"topic": "Profit revision", "relationship": "differing_reports",
            "reports": [{"article_id": doc.metadata["article_id"], "text": doc.page_content,
                         "citations": [{"evidence_id": doc.metadata["chunk_id"], "quote": doc.page_content}]}
                        for doc in self.excerpts]}]}

    @patch("news_research.comparison.invoke_json")
    def test_valid_comparison_preserves_separate_reports(self, invoke):
        invoke.return_value = json.dumps(self.payload()), 10, {}
        result = compare_articles("profit", self.docs, Settings("test"))
        self.assertEqual(result.status, "compared")
        self.assertEqual(result.rows[0]["relationship"], "differing_reports")
        self.assertIn("18 million", result.rows[0]["reports"][0]["text"])

    @patch("news_research.comparison.invoke_json")
    def test_cross_article_citation_is_rejected(self, invoke):
        payload = self.payload()
        payload["rows"][0]["reports"][0]["citations"] = payload["rows"][0]["reports"][1]["citations"]
        invoke.return_value = json.dumps(payload), 10, {}
        self.assertEqual(compare_articles("profit", self.docs, Settings("test")).status, "invalid_evidence")

    @patch("news_research.comparison.invoke_json")
    def test_missing_coverage_is_explicit_and_uncited(self, invoke):
        payload = self.payload()
        payload["rows"][0]["relationship"] = "partial_coverage"
        payload["rows"][0]["reports"][1] = {"article_id": "b", "text": "Not found in supplied excerpts", "missing_from_excerpts": True, "citations": []}
        invoke.return_value = json.dumps(payload), 10, {}
        self.assertEqual(compare_articles("profit", self.docs, Settings("test")).status, "compared")

    @patch("news_research.comparison.invoke_json")
    def test_selection_validation_before_api(self, invoke):
        with self.assertRaises(ValueError):
            compare_articles("profit", self.docs[:1], Settings("test"))
        invoke.assert_not_called()
