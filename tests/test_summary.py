import unittest
import httpx
from groq import BadRequestError
from unittest.mock import patch
from langchain_core.documents import Document
from news_research.config import Settings
from news_research.summary import summarize_article, is_summary_request


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.article = Document(page_content="First fact. Second fact. Third fact. Fourth fact. Final fact.",
                                metadata={"article_id": "a", "source": "https://example.com/a"})

    @patch("news_research.summary.invoke_json", return_value=('{"ids":[0,4]}', 10, {}))
    def test_generic_summary_can_cite_end_of_article(self, provider):
        request = "summarize the article for me"
        self.assertTrue(is_summary_request(request))
        self.assertFalse(is_summary_request("What changed in the article?"))
        answer = summarize_article(self.article, request, Settings("test"))
        self.assertEqual([c["quote"] for c in answer.evidence], ["First fact.", "Final fact."])

    @patch("news_research.summary.invoke_json", return_value=('{"ids":[99]}', 10, {}))
    def test_invalid_selection_labels_actual_opening_passages(self, provider):
        answer = summarize_article(self.article, "Summarize", Settings("test"))
        self.assertTrue(answer.text.startswith("Opening passages"))
        self.assertTrue(all(c["quote"] in self.article.page_content for c in answer.evidence))

    @patch("news_research.summary.invoke_json")
    def test_provider_json_rejection_falls_back_without_a_second_request(self, provider):
        response = httpx.Response(400, request=httpx.Request("POST", "https://example.com"))
        provider.side_effect = BadRequestError("Invalid JSON", response=response, body={"error": {"code": "json_validate_failed"}})
        answer = summarize_article(self.article, "Summarize", Settings("test"))
        self.assertTrue(answer.text.startswith("Opening passages"))
        provider.assert_called_once()
