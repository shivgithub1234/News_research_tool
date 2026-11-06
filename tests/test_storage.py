import tempfile
import unittest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from news_research.config import Settings
from news_research.storage import ResearchStore, normalize_url, open_collection_index, prune_indexes
from news_research.retrieval import retrieve_documents
from tests.chroma_cleanup import close_chroma


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float("bank" in t.lower()), float("solar" in t.lower()), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(close_chroma, self.temp.name)
        self.store = ResearchStore(self.temp.name)
        self.a = self.store.create_collection("Banks")
        self.b = self.store.create_collection("Energy")
        self.settings = Settings("test")

    def test_reload_dedup_and_isolation(self):
        doc = Document(page_content="Bank revenue increased.", metadata={"source": "https://example.com/bank?utm_source=x"})
        first = self.store.save_document(self.a, doc)
        self.assertEqual(first, self.store.save_document(self.a, doc))
        self.store.save_document(self.a, Document(page_content=doc.page_content, metadata={"source": "https://example.com/copy"}))
        reopened = ResearchStore(self.temp.name)
        self.assertEqual(len(reopened.documents(self.a)), 1)
        self.assertEqual(reopened.documents(self.b), [])
        self.assertEqual(normalize_url(doc.metadata["source"]), "https://example.com/bank")

    def test_vectors_isolated_and_deleted(self):
        first = self.store.save_document(self.a, Document(page_content="Bank revenue.", metadata={"source": "https://example.com/a"}))
        self.store.save_document(self.b, Document(page_content="Solar capacity.", metadata={"source": "https://example.com/b"}))
        bank = open_collection_index(self.store, self.a, TinyEmbeddings(), self.settings)
        solar = open_collection_index(self.store, self.b, TinyEmbeddings(), self.settings)
        self.assertEqual(bank.similarity_search("solar", k=1)[0].page_content, "Bank revenue.")
        self.assertEqual(solar.similarity_search("bank", k=1)[0].page_content, "Solar capacity.")
        for strategy in ["keyword", "hybrid"]:
            self.assertEqual(retrieve_documents(bank, "bank", 2, strategy)[0].page_content, "Bank revenue.")
        self.assertEqual(retrieve_documents(bank, "unmentioned", 2, "keyword"), [])
        self.store.delete_article(self.a, first)
        prune_indexes(self.store, self.a)
        self.assertIsNone(open_collection_index(self.store, self.a, TinyEmbeddings(), self.settings))
        self.assertEqual(len(solar.get()["ids"]), 1)

    def test_failed_ingestion_and_refresh(self):
        ident = self.store.mark_pending(self.a, "https://example.com/a")
        self.store.mark_failed(ident, "timeout")
        self.assertEqual(self.store.articles(self.a)[0]["status"], "failed")
        self.store.save_document(self.a, Document(page_content="Bank story", metadata={"source": "https://example.com/a"}))
        self.store.mark_failed(ident, "refresh timeout")
        self.assertEqual(len(self.store.documents(self.a)), 1)

    def test_refresh_to_duplicate_removes_old_content(self):
        self.store.save_document(self.a, Document(page_content="First report", metadata={"source": "https://example.com/a"}))
        self.store.save_document(self.a, Document(page_content="Second report", metadata={"source": "https://example.com/b"}))
        self.store.save_document(self.a, Document(page_content="Second report", metadata={"source": "https://example.com/a"}))
        self.assertEqual([d.page_content for d in self.store.documents(self.a)], ["Second report"])

    def test_changed_article_creates_new_snapshot(self):
        doc = Document(page_content="Bank first report", metadata={"source": "https://example.com/a"})
        self.store.save_document(self.a, doc)
        first = open_collection_index(self.store, self.a, TinyEmbeddings(), self.settings)
        doc.page_content = "Bank revised report"
        self.store.save_document(self.a, doc)
        new = open_collection_index(self.store, self.a, TinyEmbeddings(), self.settings)
        self.assertNotEqual(first._collection.name, new._collection.name)
        self.assertEqual(new.similarity_search("bank", k=1)[0].page_content, doc.page_content)
