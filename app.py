import streamlit as st

st.set_page_config(page_title="insee.fr assistant — Demo Disabled")

# Warning-style message
st.markdown("""
    <div style="background-color:#fff3cd; color:#856404; padding: 1rem; border-left: 6px solid #ffeeba; border-radius: 4px; margin-bottom: 2rem;">
        🚫 <strong>This demo instance is no longer active.</strong><br>
        It required significant resources and has been disabled.<br><br>
        👉 You can still explore the project and run it locally via the GitHub repository below.
    </div>
""", unsafe_allow_html=True)

# Stylish GitHub button
st.markdown("""
    <a href="https://github.com/InseeFrLab/llm-open-data-insee/tree/main" target="_blank">
        <button style="
            background-color: #24292e;
            color: white;
            padding: 0.6rem 1rem;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
        ">
            🚀 View on GitHub
        </button>
    </a>
""", unsafe_allow_html=True)


# Optional: short project description
st.markdown("""
#### About the Project

This project explores how large language models (LLMs) can be used to interact with open data published by Insee (France's National Institute of Statistics and Economic Studies).
See [this presentation](https://linogaliana.github.io/slides-workshopgenAI-unece2025/).
""")
