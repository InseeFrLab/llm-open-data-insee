import streamlit as st


def initialize_session_state(defaults):
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_session_state(values: dict):
    for key, value in values.items():
        st.session_state[key] = value() if callable(value) else value
