"""Offline regression coverage for configuration and pipeline failure paths."""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from news_research.config import Settings, load_settings
from news_research.generation import generate_answer
from news_research.ingestion import extract_article_text, load_articles
from news_research.retrieval import index_articles, retrieve_documents


class ConfigurationTests(unittest.TestCase):
    @patch("news_research.config.load_dotenv")
    def test_only_groq_key_is_required(self, _):
        with patch.dict(os.environ, {"GROQ_API_KEY": " test-key "}, clear=True):
            self.assertEqual(load_settings().groq_api_key, "test-key")

    @patch("news_research.config.load_dotenv")
    def test_missing_or_blank_key_has_actionable_error(self, _):
        for env in ({}, {"GROQ_API_KEY": "  "}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "Missing GROQ_API_KEY"):
                    load_settings()


class IngestionTests(unittest.TestCase):
    def test_extraction_preserves_article_and_removes_navigation(self):
        paragraph = "The company reported higher revenue following increased demand in its main markets."
        html = f"<nav>{'Navigation ' * 10}</nav><article><p>{paragraph}</p><script>{'junk' * 20}</script></article>"
        self.assertEqual(extract_article_text(html), paragraph)

    @patch("news_research.ingestion.requests.Session")
    def test_partial_failure_keeps_successful_article_and_real_status(self, factory):
        session = factory.return_value.__enter__.return_value
        failed = requests.Response()
        failed.status_code = 404
        success = Mock()
        success.text = "<article>" + "A valid article contains enough useful text for extraction. " * 3 + "</article>"
        session.get.side_effect = [requests.HTTPError(response=failed), success]
        result = load_articles(["https://example.com/missing", "https://example.com/article"])
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].metadata["source"], "https://example.com/article")
        self.assertIn("HTTP 404", result.warnings[0])

    @patch("news_research.ingestion.requests.Session")
    def test_empty_page_is_reported(self, factory):
        session = factory.return_value.__enter__.return_value
        session.get.return_value.text = "<html><p>Short</p></html>"
        result = load_articles(["https://example.com/empty"])
        self.assertFalse(result.documents)
        self.assertIn("No readable article", result.warnings[0])

    @patch("news_research.ingestion.requests.Session")
    def test_invalid_scheme_is_not_fetched(self, factory):
        result = load_articles(["file:///private.txt"])
        factory.return_value.__enter__.return_value.get.assert_not_called()
        self.assertFalse(result.documents)
        self.assertIn("Invalid article URL", result.warnings[0])


class PipelineTests(unittest.TestCase):
    def test_index_can_be_reopened_with_source_metadata(self):
        from langchain_community.vectorstores import Chroma

        class LocalTestEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                return [1.0, 0.0]

        # Chroma can retain Windows file handles until process exit.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            embeddings = LocalTestEmbeddings()
            settings = Settings("test", persist_directory=directory)
            doc = Document(page_content="Revenue grew by ten percent.", metadata={"source": "https://example.com/a"})
            index_articles([doc], embeddings, settings)
            reopened = Chroma(persist_directory=directory, embedding_function=embeddings)
            retrieved = retrieve_documents(reopened, "Revenue?", k=1)
            self.assertEqual(retrieved[0].page_content, doc.page_content)
            self.assertEqual(retrieved[0].metadata["source"], doc.metadata["source"])

    @patch("news_research.retrieval.Chroma.from_documents")
    def test_empty_input_never_reaches_database(self, create):
        with self.assertRaisesRegex(ValueError, "No readable articles"):
            index_articles([], Mock(), Settings("test"))
        create.assert_not_called()

    @patch("news_research.generation.ChatGroq")
    def test_no_evidence_does_not_call_llm(self, llm):
        answer = generate_answer("What happened?", [], Settings("test"))
        llm.assert_not_called()
        self.assertEqual(answer.sources, [])
        self.assertIn("No relevant information", answer.text)



if __name__ == "__main__":
    unittest.main()
