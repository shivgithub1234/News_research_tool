"""Embedding creation, chunking, indexing, and retrieval."""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from news_research.config import Settings


def create_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


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


def retrieve_documents(vectorstore, question: str, k: int) -> list[Document]:
    return vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
