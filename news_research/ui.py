"""Streamlit presentation and session state."""
import streamlit as st

from news_research.config import load_settings
from news_research.generation import generate_answer
from news_research.ingestion import load_articles
from news_research.retrieval import create_embeddings, index_articles, retrieve_documents


@st.cache_resource
def cached_embeddings(model_name: str):
    return create_embeddings(model_name)


def run():
    st.set_page_config(page_title="My News Research Tool", layout="wide")
    st.title("My News Research Tool")
    st.caption("Ask questions from finance news using retrieved article passages.")
    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.header("Enter News URLs")
    urls = [st.sidebar.text_input(f"News URL {i + 1}").strip() for i in range(3)]
    urls = [url for url in urls if url]
    if st.sidebar.button("Process Articles"):
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            st.session_state.pop("vectorstore", None)
            with st.spinner("Fetching and indexing articles..."):
                result = load_articles(urls)
                for warning in result.warnings:
                    st.warning(warning)
                if not result.documents:
                    st.error("No articles could be processed. Check the URLs and try again.")
                else:
                    try:
                        embeddings = cached_embeddings(settings.embedding_model)
                        st.session_state["vectorstore"] = index_articles(result.documents, embeddings, settings)
                    except Exception:
                        st.error("Article indexing failed. Check the embedding model download and database access, then retry.")
                    else:
                        st.success(f"Indexed {len(result.documents)} article(s).")

    st.subheader("Ask a question from the news")
    with st.form("question_form"):
        query = st.text_input("Example: How will interest rate changes affect banks?")
        submitted = st.form_submit_button("Ask")
    if submitted:
        if not query.strip():
            st.warning("Please enter a question.")
        elif "vectorstore" not in st.session_state:
            st.warning("Please process news URLs first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    documents = retrieve_documents(st.session_state["vectorstore"], query.strip(), settings.retrieval_k)
                    answer = generate_answer(query.strip(), documents, settings)
                except Exception:
                    st.error("Could not answer the question. Check your Groq API key, service availability, and network connection, then retry.")
                else:
                    st.subheader("Answer")
                    st.write(answer.text)
                    if answer.sources:
                        st.subheader("Retrieved sources")
                        for source in answer.sources:
                            st.markdown(f"- [{source}]({source})")
