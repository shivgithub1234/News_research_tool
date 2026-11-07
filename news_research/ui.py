"""Streamlit research workspace."""
import json
import logging
import streamlit as st

from news_research.generation import generate_answer
from news_research.ingestion import load_articles
from news_research.retrieval import create_embeddings, retrieve_documents
from news_research.storage import ResearchStore, open_collection_index, prune_indexes
from news_research.comparison import compare_articles


@st.cache_resource
def cached_embeddings(model_name):
    return create_embeddings(model_name)


def run_private(settings):
    st.caption("Private single-owner workspace. Only explicitly submitted questions or comparisons call Groq.")
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
    if st.session_state.get("comparison_collection") != collection_id:
        st.session_state.pop("comparison", None)
    st.sidebar.header("Add articles")
    urls = [st.sidebar.text_input(f"News URL {i+1}").strip() for i in range(3)]
    if st.sidebar.button("Process Articles"):
        urls = [url for url in urls if url]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            st.session_state.pop("answer", None)
            st.session_state.pop("comparison", None)
            saved = 0
            with st.spinner("Fetching articles..."):
                for url in urls:
                    if len(store.articles(collection_id)) >= 50:
                        st.warning("Collection limit reached (50 articles). Remove an article before adding more.")
                        break
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
                st.session_state.pop("comparison", None)
                st.rerun()
    st.subheader("Ask about this collection")
    with st.expander("Retrieval options"):
        strategy = st.selectbox("Search method", ["semantic", "keyword", "hybrid"])
        retrieval_k = st.slider("Evidence passages", min_value=2, max_value=8, value=settings.retrieval_k)
    with st.form("question_form"):
        query = st.text_input("Your question", max_chars=500)
        submitted = st.form_submit_button("Ask")
    if submitted:
        st.session_state.pop("answer", None)
        if not query.strip():
            st.warning("Please enter a question.")
        elif not store.documents(collection_id):
            st.warning("Please process news URLs first.")
        elif not settings.groq_api_key or not settings.free_tier_confirmed:
            st.warning("Live generation is disabled. Configure a Groq key and confirm your account is on the free tier before enabling it.")
        else:
            with st.spinner("Retrieving evidence and preparing an answer..."):
                try:
                    index = open_collection_index(store, collection_id, cached_embeddings(settings.embedding_model), settings)
                    documents = retrieve_documents(index, query.strip(), retrieval_k, strategy)
                    answer = generate_answer(query.strip(), documents, settings)
                    prune_indexes(store, collection_id, index._collection.name)
                    st.session_state["answer"] = answer
                    st.session_state["answer_collection"] = collection_id
                    logging.getLogger(__name__).info("answer_complete status=%s duration_ms=%.1f tokens=%s", answer.status, answer.latency_ms, answer.tokens.get("total_tokens", 0))
                except Exception as exc:
                    logging.getLogger(__name__).warning("answer_failed type=%s", type(exc).__name__)
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

    st.divider()
    st.subheader("Compare coverage")
    ready = {row["id"]: row for row in articles if row["status"] == "processed"}
    labels = {ident: json.loads(row["metadata"]).get("title") or row["url"] for ident, row in ready.items()}
    with st.form("comparison_form"):
        selected = st.multiselect("Articles to compare (2–4)", list(ready), format_func=labels.get, max_selections=4)
        topic = st.text_input("Comparison topic", placeholder="For example: profit figures and reporting dates", max_chars=500)
        compare = st.form_submit_button("Compare coverage")
    if compare:
        st.session_state.pop("comparison", None)
        if len(selected) < 2 or not topic.strip():
            st.warning("Choose two to four articles and enter a comparison topic.")
        elif not settings.groq_api_key or not settings.free_tier_confirmed:
            st.warning("Live comparison is disabled until a free-tier Groq key is configured and confirmed.")
        else:
            documents = [doc for doc in store.documents(collection_id) if doc.metadata["article_id"] in selected]
            with st.spinner("Comparing supplied evidence..."):
                try:
                    st.session_state["comparison"] = compare_articles(topic.strip(), documents, settings)
                    st.session_state["comparison_collection"] = collection_id
                except Exception as exc:
                    logging.getLogger(__name__).warning("comparison_failed type=%s", type(exc).__name__)
                    st.error("Comparison failed. Check API availability and try again.")
    if st.session_state.get("comparison"):
        render_comparison(st.session_state["comparison"])


def render_comparison(result):
    if result.status != "compared":
        st.info("Insufficient evidence for comparison." if result.status == "insufficient_evidence"
                else "The comparison failed evidence validation. No comparison is shown.")
        return
    st.caption("Comparison uses up to three excerpts per article. Missing coverage means not found in these excerpts. Labels require human review.")
    for row in result.rows:
        st.text(f"{row['topic']} — {row['relationship'].replace('_', ' ')}")
        columns = st.columns(len(row["reports"]))
        for column, report in zip(columns, row["reports"]):
            with column:
                article = next(item for item in result.evidence.values() if item["article_id"] == report["article_id"])
                st.text(article.get("title") or article["source"])
                st.caption(article.get("published") or "Publication date unknown")
                st.text(report["text"])
                for citation in report["citations"]:
                    source = result.evidence[citation["evidence_id"]]
                    with st.expander("Supporting evidence"):
                        st.text(citation["quote"])
                        st.text(source["passage"])
                        st.link_button("Open source", source["source"])
