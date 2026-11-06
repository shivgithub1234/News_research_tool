"""Streamlit research workspace."""
import streamlit as st

from news_research.config import load_settings
from news_research.generation import generate_answer
from news_research.ingestion import load_articles
from news_research.retrieval import create_embeddings, retrieve_documents
from news_research.storage import ResearchStore, open_collection_index, prune_indexes


@st.cache_resource
def cached_embeddings(model_name):
    return create_embeddings(model_name)


def run():
    st.set_page_config(page_title="News Research Workspace", layout="wide")
    st.title("News Research Workspace")
    st.caption("Save articles, ask questions, and inspect your sources.")
    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    store = ResearchStore(settings.data_directory)
    if not store.collections():
        store.create_collection("My research")
    with st.sidebar.form("new_collection"):
        name = st.text_input("New collection name")
        create = st.form_submit_button("Create collection")
    if create:
        try:
            st.session_state["collection"] = store.create_collection(name)
            st.session_state.pop("answer", None)
        except ValueError as exc:
            st.sidebar.error(str(exc))
    collections = store.collections()
    names = {row["id"]: row["name"] for row in collections}
    collection_id = st.sidebar.selectbox("Research collection", list(names), format_func=names.get, key="collection")
    if st.session_state.get("answer_collection") != collection_id:
        st.session_state.pop("answer", None)
    st.sidebar.header("Add articles")
    urls = [st.sidebar.text_input(f"News URL {i+1}").strip() for i in range(3)]
    if st.sidebar.button("Process Articles"):
        urls = [url for url in urls if url]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            st.session_state.pop("answer", None)
            saved = 0
            with st.spinner("Fetching articles..."):
                for url in urls:
                    try:
                        article_id = store.mark_pending(collection_id, url)
                    except ValueError as exc:
                        st.warning(str(exc))
                        continue
                    result = load_articles([url])
                    for warning in result.warnings:
                        st.warning(warning)
                    if result.documents:
                        store.save_document(collection_id, result.documents[0])
                        saved += 1
                    else:
                        store.mark_failed(article_id, "Article extraction failed; retry or remove this URL.")
            if saved:
                st.success(f"Processed {saved} article URL(s). Duplicate content is stored once.")
            else:
                st.error("No articles could be processed. Check the URLs and try again.")

    articles = store.articles(collection_id)
    with st.expander(f"Articles in this collection ({len(articles)})"):
        for article in articles:
            st.write(f"{article['url']} — {article['status']}")
            if article["error"]:
                st.caption(article["error"])
            if st.button("Remove article", key="delete_" + article["id"]):
                store.delete_article(collection_id, article["id"])
                prune_indexes(store, collection_id)
                st.session_state.pop("answer", None)
                st.rerun()
    st.subheader("Ask about this collection")
    with st.form("question_form"):
        query = st.text_input("Your question")
        submitted = st.form_submit_button("Ask")
    if submitted:
        st.session_state.pop("answer", None)
        if not query.strip():
            st.warning("Please enter a question.")
        elif not store.documents(collection_id):
            st.warning("Please process news URLs first.")
        else:
            with st.spinner("Retrieving evidence and preparing an answer..."):
                try:
                    index = open_collection_index(store, collection_id, cached_embeddings(settings.embedding_model), settings)
                    documents = retrieve_documents(index, query.strip(), settings.retrieval_k)
                    answer = generate_answer(query.strip(), documents, settings)
                    prune_indexes(store, collection_id, index._collection.name)
                    st.session_state["answer"] = answer
                    st.session_state["answer_collection"] = collection_id
                except Exception:
                    st.error("Could not answer. Check model availability, your API key, and network access, then retry.")
    if st.session_state.get("answer"):
        answer = st.session_state["answer"]
        st.subheader("Answer")
        st.text(answer.text)
        for citation in answer.evidence:
            with st.expander(f"[{citation['number']}] {citation.get('title') or 'Supporting passage'}"):
                st.text(citation["quote"])
                st.caption(f"Publisher: {citation.get('publisher') or 'Unknown'} | Published: {citation.get('published') or 'Unknown'}")
                st.text(citation["passage"])
                st.link_button("Open article", citation["source"])
        st.caption("Citation IDs and quotes are checked. Whether each quote supports its claim still requires review.")
        if answer.tokens:
            st.caption(f"Generation: {answer.latency_ms / 1000:.2f}s | Tokens: {answer.tokens.get('total_tokens', 'unavailable')}")
        for source in answer.sources:
            st.markdown(f"- [{source}]({source})")
