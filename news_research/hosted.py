"""A hosted workspace uses the visitor's key and a separate collection directory."""
import hashlib
from dataclasses import replace
from pathlib import Path

import streamlit as st
from groq import Groq, APIError, AuthenticationError


def hosted_settings(settings, key):
    namespace = hashlib.sha256(b"news-research:workspace:v1:" + key.encode()).hexdigest()
    return replace(settings, groq_api_key=key, free_tier_confirmed=True,
                   embedding_model="onnx/all-MiniLM-L6-v2",
                   data_directory=str(Path(settings.data_directory) / "hosted" / namespace))


def connect_workspace(settings):
    key = st.session_state.get("hosted_groq_key")
    if key:
        if st.sidebar.button("Disconnect workspace"):
            st.session_state.clear()
            st.rerun()
        return hosted_settings(settings, key)

    st.subheader("Open your research workspace")
    st.write("Connect your free Groq key to add news URLs, ask questions with citations, and compare coverage.")
    with st.form("connect_workspace", clear_on_submit=True):
        entered = st.text_input("Groq API key", type="password", max_chars=256).strip()
        free = st.checkbox("My Groq account is on the free tier")
        submit = st.form_submit_button("Open workspace")
    st.caption("The app keeps your key in this browser session's server memory and sends it to Groq. Reconnect with the same key to reopen collections while the hosting disk retains them.")
    if submit:
        if not entered or not free:
            st.warning("Enter your key and confirm your free-tier account.")
            return None
        try:
            with Groq(api_key=entered, max_retries=0, timeout=12) as client:
                client.models.retrieve(settings.llm_model)
        except AuthenticationError:
            st.error("Groq rejected the key. Check the key and try again.")
        except APIError:
            st.error("Groq could not verify model access. Check your account or try again later.")
        else:
            st.session_state["hosted_groq_key"] = entered
            st.rerun()
    return None
