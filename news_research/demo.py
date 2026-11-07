"""Public, read-only demo: no credentials, network fetches, or LLM calls."""
import streamlit as st
from evaluation.run import load_dataset
from news_research.ranking import bm25_rank


def run_demo():
    st.info("Free offline demo · synthetic articles · no API calls · no private workspace access")
    data = load_dataset()
    articles = data["articles"]
    with st.expander("Read the sample articles"):
        for article in articles:
            st.subheader(article["title"])
            st.caption(article["published"] + " · Synthetic News Fixture")
            st.text(article["text"])
    with st.form("demo_search"):
        question = st.text_input("Search sample evidence", value="How did Harbor Bank profit change?", max_chars=500)
        search = st.form_submit_button("Find evidence")
    if search:
        results = bm25_rank([article["text"] for article in articles], question, 4)
        st.caption("These are keyword-retrieved passages, not a generated or verified answer.")
        if not results:
            st.info("No matching sample evidence found.")
        for number, index in enumerate(results, 1):
            article = articles[index]
            st.subheader(f"[{number}] {article['title']}")
            st.caption(article["published"])
            st.text(article["text"])
    st.subheader("Coverage comparison example")
    st.caption("Authored illustration of the private comparison feature, not live model output.")
    left, right = st.columns(2)
    with left:
        st.text("Initial report · January 10, 2026")
        st.text("Profit: 18 million rupees. Revenue: 120 million rupees.")
    with right:
        st.text("Revision · January 12, 2026")
        st.text("Profit: 16 million rupees after a 2 million provisioning adjustment. Revenue unchanged.")
    st.caption("Different reporting dates explain this revision; differing figures alone do not establish a contradiction.")
    st.link_button("Read the code and measured evaluations", "https://github.com/shivgithub1234/News_research_tool/tree/codex/organize-stage-one")
