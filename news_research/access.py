"""Single-owner access gate. Use host-level private sharing as well when deploying."""
import hashlib
import hmac
import os
import streamlit as st


def private_access():
    password = os.getenv("APP_PASSWORD", "")
    if len(password) < 12:
        st.error("Private mode requires APP_PASSWORD with at least 12 characters in environment or hosting secrets.")
        return False
    fingerprint = hashlib.sha256(password.encode()).hexdigest()
    if st.session_state.get("authenticated") == fingerprint:
        if st.sidebar.button("Log out"):
            st.session_state.clear()
            st.rerun()
        return True
    with st.form("login"):
        entered = st.text_input("Workspace password", type="password")
        login = st.form_submit_button("Unlock workspace")
    if login:
        if hmac.compare_digest(entered, password):
            st.session_state["authenticated"] = fingerprint
            st.rerun()
        else:
            st.error("Incorrect workspace password.")
    return False
