import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from langchain_core.documents import Document
from news_research.ingestion import IngestionResult
from news_research.generation import Answer
from news_research.storage import ResearchStore
from news_research.comparison import Comparison

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

    def test_answer_submission_renders_evidence_without_rerun_api_call(self):
        app = AppTest.from_file(ENTRY).run(timeout=30)
        store = ResearchStore(self.temp.name)
        collection = store.collections()[0]["id"]
        doc = Document(page_content="Revenue grew.", metadata={"source": "https://example.com/a"})
        store.save_document(collection, doc)
        answer = Answer("Revenue grew. [1]", ["https://example.com/a"], evidence=[
            {"number": 1, "quote": "Revenue grew.", "passage": "Revenue grew.", "source": "https://example.com/a", "title": "Earnings"}])
        with patch("news_research.ui.open_collection_index"), patch("news_research.ui.cached_embeddings"), patch("news_research.ui.prune_indexes"), patch("news_research.ui.retrieve_documents", return_value=[doc]), patch("news_research.ui.generate_answer", return_value=answer) as generate:
            app.run()
            next(t for t in app.text_input if t.label == "Your question").set_value("What changed?")
            next(b for b in app.button if b.label == "Ask").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Revenue grew." in t.value for t in app.text))
            self.assertTrue(any("Earnings" in e.label for e in app.expander))
            app.run()
            self.assertEqual(generate.call_count, 1)

    def test_comparison_submission_renders_reports(self):
        app = AppTest.from_file(ENTRY).run(timeout=30)
        store = ResearchStore(self.temp.name)
        collection = store.collections()[0]["id"]
        ids = [store.save_document(collection, Document(page_content=text, metadata={"source": f"https://example.com/{i}", "title": f"Report {i}"}))
               for i, text in enumerate(["Initial profit was 18 million.", "Revised profit was 16 million."])]
        docs = store.documents(collection)
        evidence = {doc.metadata["article_id"]: {**doc.metadata, "passage": doc.page_content} for doc in docs}
        reports = [{"article_id": doc.metadata["article_id"], "text": doc.page_content, "citations": []} for doc in docs]
        comparison = Comparison("compared", [{"topic": "Profit", "relationship": "differing_reports", "reports": reports}], evidence)
        with patch("news_research.ui.compare_articles", return_value=comparison) as compare:
            app.run()
            app.multiselect[0].set_value(ids)
            next(t for t in app.text_input if t.label == "Comparison topic").set_value("Profit")
            next(b for b in app.button if b.label == "Compare coverage").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("18 million" in item.value for item in app.text))
            self.assertTrue(any("16 million" in item.value for item in app.text))
            app.run()
            self.assertEqual(compare.call_count, 1)
