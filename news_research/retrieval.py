"""Embedding creation, chunking, indexing, and retrieval."""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from news_research.config import Settings


def create_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})


def index_articles(documents: list[Document], embeddings, settings: Settings):
    if not documents:
        raise ValueError("No readable articles to index.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks to index.")
    return Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=settings.persist_directory,
    )


def retrieve_documents(vectorstore, question: str, k: int, strategy="semantic") -> list[Document]:
    if k < 1:
        raise ValueError("k must be positive")
    if strategy == "semantic":
        return vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    if strategy not in {"keyword", "hybrid"}:
        raise ValueError("Unknown retrieval strategy")
    from news_research.ranking import bm25_rank, reciprocal_rank_fusion
    rows = vectorstore.get()
    documents = [Document(page_content=text, metadata=metadata or {})
                 for text, metadata in zip(rows["documents"], rows["metadatas"])]
    ids = rows["ids"]
    lookup = dict(zip(ids, documents))
    lexical = [ids[i] for i in bm25_rank([doc.page_content for doc in documents], question, max(k * 3, 12))]
    if strategy == "keyword":
        return [lookup[ident] for ident in lexical[:k]]
    semantic_docs = vectorstore.similarity_search(question, k=min(len(documents), max(k * 3, 12))) if documents else []
    semantic = [doc.metadata["chunk_id"] for doc in semantic_docs]
    return [lookup[ident] for ident in reciprocal_rank_fusion([semantic, lexical], k)]
