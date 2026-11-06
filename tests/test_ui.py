import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from langchain_core.documents import Document
from news_research.ingestion import IngestionResult

ENTRY = str(Path(__file__).resolve().parents[1] / "main.py")


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {"NEWS_RESEARCH_DATA_DIR": self.temp.name, "GROQ_API_KEY": "test", "HUGGINGFACEHUB_API_TOKEN": ""})
        env.start()
        self.addCleanup(env.stop)
        dotenv = patch("news_research.config.load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)

    def test_starts_with_groq_only_and_creates_collection(self):
        with patch("news_research.ui.create_embeddings") as embeddings:
            app = AppTest.from_file(ENTRY).run(timeout=30)
            app.sidebar.text_input[0].set_value("Energy")
            app.sidebar.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.sidebar.selectbox[0].options), 2)
        embeddings.assert_not_called()

    def test_ingestion_failure_and_success(self):
        app = AppTest.from_file(ENTRY).run(timeout=30)
        app.sidebar.text_input[1].set_value("https://example.com/story")
        with patch("news_research.ui.load_articles", return_value=IngestionResult()):
            app.sidebar.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertIn("No articles", app.error[0].value)
        result = IngestionResult(documents=[Document(page_content="A bank story", metadata={"source": "https://example.com/story"})])
        with patch("news_research.ui.load_articles", return_value=result):
            app.sidebar.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
