import json
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from news_research.config import Settings
from news_research.generation import generate_answer, invoke_json


class EvidenceTests(unittest.TestCase):
    @patch("news_research.generation.ChatGroq")
    def test_unconfirmed_billing_cannot_call_provider(self, llm):
        with self.assertRaisesRegex(ValueError, "free-tier"):
            invoke_json("system", {}, Settings("test", free_tier_confirmed=False))
        llm.assert_not_called()

    def setUp(self):
        self.docs = [Document(page_content="Revenue increased by 10 percent. Profit was unchanged.",
                             metadata={"source": "https://example.com/a", "chunk_id": "a", "title": "Earnings"})]

    def response(self, quote="Revenue increased by 10 percent.", ident="a"):
        return json.dumps({"status": "answered", "claims": [{"text": "Revenue increased by 10 percent.",
            "citations": [{"evidence_id": ident, "quote": quote}]}]})

    @patch("news_research.generation.invoke_json")
    def test_valid_answer_has_quote_metadata_and_usage(self, invoke):
        invoke.return_value = self.response(), 100, {"total_tokens": 42}
        result = generate_answer("What changed?", self.docs, Settings("test"))
        self.assertEqual(result.status, "answered")
        self.assertIn("[1]", result.text)
        self.assertEqual(result.evidence[0]["title"], "Earnings")
        self.assertEqual(result.tokens["total_tokens"], 42)
        self.assertIn("untrusted", invoke.call_args.args[0])

    @patch("news_research.generation.invoke_json")
    def test_fabricated_quote_unknown_id_and_invalid_json_fail_closed(self, invoke):
        for raw in [self.response(quote="Profit doubled."), self.response(ident="invented"), "not JSON",
                    '{"status":"answered","claims":[]}']:
            with self.subTest(raw=raw):
                invoke.return_value = raw, 10, {}
                result = generate_answer("What changed?", self.docs, Settings("test"))
                self.assertEqual(result.status, "invalid_evidence")
                self.assertEqual(result.sources, [])

    @patch("news_research.generation.invoke_json")
    def test_abstention_is_distinct_from_invalid_response(self, invoke):
        invoke.return_value = '{"status":"insufficient_evidence","claims":[]}', 10, {}
        self.assertEqual(generate_answer("Who is CEO?", self.docs, Settings("test")).status, "insufficient_evidence")

    @patch("news_research.generation.invoke_json")
    def test_empty_evidence_never_uses_api(self, invoke):
        self.assertEqual(generate_answer("Question", [], Settings("test")).status, "insufficient_evidence")
        invoke.assert_not_called()
