

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain core
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.schema import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Groq LLM
from langchain_groq import ChatGroq

# Stable HF embeddings (remote)
# from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# Vector DB
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma


# Windows-safe loader
# from langchain_community.document_loaders import WebBaseLoader

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# ---------------- ENV ----------------
load_dotenv()





GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GROQ_API_KEY or not HF_TOKEN:
    st.error("❌ Missing GROQ_API_KEY or HUGGINGFACEHUB_API_TOKEN")
    st.stop()

# ---------------- UI ----------------
st.set_page_config(page_title="Finance Buddy 📰", layout="wide")
st.title("😶‍🌫️My News Research Tool")
st.caption("Ask questions from finance news — Pure RAG (no auto summaries)")

# ---------------- Sidebar ----------------
st.sidebar.header("🔗 Enter News URLs")




def load_urls_with_bs4(urls):
    documents = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove unwanted tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            text = "\n".join(
                line.strip() for line in text.splitlines() if len(line.strip()) > 50
            )

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url}
                    )
                )

        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")

    return documents





urls = []
for i in range(3):
    url = st.sidebar.text_input(f"News URL {i+1}")
    if url:
        urls.append(url)

process_btn = st.sidebar.button("🚀 Process Articles")

# ---------------- LLM ----------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.2
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------- Process URLs ----------------
if process_btn:
    if not urls:
        st.warning("⚠️ Please enter at least one URL.")
    else:
        with st.spinner("📥 Fetching and indexing articles..."):
            # loader = WebBaseLoader(urls)
            # documents = loader.load()
            documents = load_urls_with_bs4(urls)


            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            vectorstore.persist()
            st.session_state["vectorstore"] = vectorstore


        st.success("✅ Articles indexed successfully!")
# ---------------- Q&A ----------------
st.subheader("💬 Ask a question from the news")

query = st.text_input("Example: How will interest rate changes affect banks?")

if query:
    if "vectorstore" not in st.session_state:
        st.warning("⚠️ Please process news URLs first.")
    else:
        retriever = st.session_state["vectorstore"].as_retriever(
            search_kwargs={"k": 4}
        )

        with st.spinner("🧠 Analyzing..."):
            # 1. Retrieve relevant documents
            docs = retriever.invoke(query)

            if not docs:
                st.write("❌ No relevant information found in the articles.")
            else:
                context = "\n\n".join(doc.page_content for doc in docs)
                sources = list({
                    doc.metadata.get("source", "Unknown source")
                    for doc in docs
                })
                # 2. Prompt
                prompt = ChatPromptTemplate.from_template(
                    """Answer the question ONLY using the context below.
If the answer is not present, say so clearly.

Context:
{context}

Question:
{question}
"""
                )

                chain = (
                    prompt
                    | llm
                    | StrOutputParser()
                )

                answer = chain.invoke(
                    {"context": context, "question": query}
                )

                st.subheader("🧠 Answer")
                st.write(answer)
                st.subheader("🔗 Sources")
                for src in sources:
                    st.markdown(f"- [{src}]({src})")
