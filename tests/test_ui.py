"""Exercise the Streamlit app without network requests or model downloads."""
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from news_research.generation import Answer
from news_research.ingestion import IngestionResult

ENTRY_POINT = str(Path(__file__).resolve().parents[1] / "main.py")


class AppTests(unittest.TestCase):
    @patch("news_research.config.load_dotenv")
    def test_missing_key_shows_setup_error(self, _):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            app = AppTest.from_file(ENTRY_POINT).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("Missing GROQ_API_KEY", app.error[0].value)

    @patch("news_research.config.load_dotenv")
    def test_groq_only_starts_without_loading_embeddings(self, _):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test", "HUGGINGFACEHUB_API_TOKEN": ""}), patch("news_research.ui.create_embeddings") as embeddings:
            app = AppTest.from_file(ENTRY_POINT).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        embeddings.assert_not_called()

    @patch("news_research.config.load_dotenv")
    def test_failed_ingestion_does_not_index_or_report_success(self, _):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test", "HUGGINGFACEHUB_API_TOKEN": ""}), patch("news_research.ui.load_articles", return_value=IngestionResult()), patch("news_research.ui.index_articles") as index:
            app = AppTest.from_file(ENTRY_POINT).run(timeout=30)
            app.sidebar.text_input[0].set_value("https://example.com/article")
            app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.success)
        self.assertIn("No articles could be processed", app.error[0].value)
        index.assert_not_called()

    @patch("news_research.config.load_dotenv")
    def test_question_submission_displays_answer_and_sources(self, _):
        docs = [Document(page_content="Revenue grew.", metadata={"source": "https://example.com/article"})]
        with patch.dict(os.environ, {"GROQ_API_KEY": "test", "HUGGINGFACEHUB_API_TOKEN": ""}), patch("news_research.ui.load_articles", return_value=IngestionResult(documents=docs)), patch("news_research.ui.cached_embeddings"), patch("news_research.ui.index_articles", return_value=Mock()), patch("news_research.ui.retrieve_documents", return_value=docs), patch("news_research.ui.generate_answer", return_value=Answer("Revenue grew.", ["https://example.com/article"])) as generate:
            app = AppTest.from_file(ENTRY_POINT).run(timeout=30)
            app.sidebar.text_input[0].set_value("https://example.com/article")
            app.sidebar.button[0].click().run()
            generate.assert_not_called()
            app.text_input[0].set_value("What changed?")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Revenue grew." in item.value for item in app.markdown))
            self.assertTrue(any("https://example.com/article" in item.value for item in app.markdown))
            self.assertEqual(generate.call_count, 1)
            app.run()
            self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
