"""Lightweight public entry point: private dependencies are imported after login."""
import os
import streamlit as st
from news_research.config import load_settings
from news_research.access import private_access
from news_research.demo import run_demo


def run():
    st.set_page_config(page_title="News Research Workspace", layout="wide")
    st.title("News Research Workspace")
    st.caption("Explore reporting, compare accounts, and inspect the evidence.")
    settings = load_settings(require_api_key=False)
    mode = os.getenv("NEWS_RESEARCH_MODE", "demo").lower()
    if mode == "demo":
        run_demo()
    elif mode == "private":
        if private_access():
            try:
                from news_research.ui import run_private
            except ModuleNotFoundError:
                st.error("Private mode requires the packages in requirements-private.txt.")
                return
            run_private(settings)
    else:
        st.error("NEWS_RESEARCH_MODE must be demo or private.")
