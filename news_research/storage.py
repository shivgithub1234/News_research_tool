"""SQLite is the source of truth; vector indexes are rebuildable snapshots."""
import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from langchain_core.documents import Document


def normalize_url(url):
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("Use an HTTP or HTTPS article URL.")
    if parts.username or parts.password:
        raise ValueError("URLs containing credentials are not accepted.")
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port and (parts.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}))
    return urlunsplit((parts.scheme.lower(), host, parts.path or "/", query, ""))


class ResearchStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "research.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                    url TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', content_hash TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(collection_id, url));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_collection(self, name):
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("Collection names must contain 1–80 characters.")
        ident = uuid.uuid4().hex
        with self.connect() as db:
            try:
                db.execute("INSERT INTO collections(id,name) VALUES (?,?)", (ident, name))
            except sqlite3.IntegrityError as exc:
                raise ValueError("A collection with this name already exists.") from exc
        return ident

    def collections(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM collections ORDER BY created_at,id")]

    def articles(self, collection_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM articles WHERE collection_id=? ORDER BY updated_at,id", (collection_id,))]

    def mark_pending(self, collection_id, url):
        url = normalize_url(url)
        ident = hashlib.sha256(f"{collection_id}:{url}".encode()).hexdigest()
        with self.connect() as db:
            # Keep an existing processed article available if refreshing it fails.
            db.execute("INSERT OR IGNORE INTO articles(id,collection_id,url,status) VALUES (?,?,?,'pending')",
                       (ident, collection_id, url))
        return ident

    def mark_failed(self, article_id, error):
        with self.connect() as db:
            db.execute("UPDATE articles SET status=CASE WHEN content='' THEN 'failed' ELSE 'processed' END, "
                       "error=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (error, article_id))

    def save_document(self, collection_id, document):
        content = document.page_content.strip()
        if not content:
            raise ValueError("Cannot save an empty article.")
        url = normalize_url(document.metadata["source"])
        digest = hashlib.sha256(content.encode()).hexdigest()
        ident = hashlib.sha256(f"{collection_id}:{url}".encode()).hexdigest()
        metadata = {**document.metadata, "source": url, "article_id": ident}
        with self.connect() as db:
            duplicate = db.execute("SELECT id FROM articles WHERE collection_id=? AND content_hash=? AND id<>?",
                                   (collection_id, digest, ident)).fetchone()
            if duplicate:
                db.execute("DELETE FROM articles WHERE id=?", (ident,))
                return duplicate["id"]
            db.execute("""INSERT INTO articles(id,collection_id,url,content,content_hash,metadata,status)
                VALUES (?,?,?,?,?,?,'processed') ON CONFLICT(id) DO UPDATE SET
                content=excluded.content,content_hash=excluded.content_hash,metadata=excluded.metadata,
                status='processed',error='',updated_at=CURRENT_TIMESTAMP""",
                       (ident, collection_id, url, content, digest, json.dumps(metadata)))
        return ident

    def documents(self, collection_id):
        return [Document(page_content=row["content"], metadata=json.loads(row["metadata"]))
                for row in self.articles(collection_id) if row["status"] == "processed"]

    def delete_article(self, collection_id, article_id):
        with self.connect() as db:
            db.execute("DELETE FROM articles WHERE collection_id=? AND id=?", (collection_id, article_id))


def open_collection_index(store, collection_id, embeddings, settings):
    """A new content revision gets a new index; failed builds never reuse stale data.

    Callers prune older snapshots after successful creation. SQLite can
    reconstruct the index after a crash or an interrupted embedding operation.
    """
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    documents = store.documents(collection_id)
    if not documents:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = splitter.split_documents(documents)
    ids = []
    for i, chunk in enumerate(chunks):
        ident = hashlib.sha256((chunk.metadata["article_id"] + str(i) + chunk.page_content).encode()).hexdigest()
        chunk.metadata["chunk_id"] = ident
        ids.append(ident)
    version = hashlib.sha256((settings.embedding_model + "|normalized-v1|" + "|".join(ids)).encode()).hexdigest()[:16]
    prefix = "research_" + collection_id + "_"
    index = Chroma(collection_name=prefix + version, embedding_function=embeddings,
                   persist_directory=str(store.directory / "vectors"))
    existing = set(index.get()["ids"])
    if existing != set(ids):
        index.add_documents(chunks, ids=ids)
        extra = existing - set(ids)
        if extra:
            index.delete(ids=list(extra))
    return index


def prune_indexes(store, collection_id, keep_name=None):
    """Delete obsolete snapshots, including all vectors after the last article is removed."""
    import chromadb
    client = chromadb.PersistentClient(path=str(store.directory / "vectors"))
    prefix = "research_" + collection_id + "_"
    for collection in client.list_collections():
        name = collection if isinstance(collection, str) else collection.name
        if name.startswith(prefix) and name != keep_name:
            client.delete_collection(name)
