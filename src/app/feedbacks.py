from datetime import datetime

import streamlit as st
from streamlit_feedback import streamlit_feedback

css_annotation_title = "text-align: right; font-weight: bold; font-style: italic;"


def handle_feedback(response, index, history, unique_id, feedback_type="retriever"):
    st.toast("✔️ Feedback received!")
    message = history[index]["content"]
    question = history[index - 1]["content"]
    submission_time = datetime.now().strftime("%H:%M:%S")  # Only the time as string

    st.session_state.feedback.append(
        {
            "discussion_index": index,
            "evaluation": response["score"],
            "evaluation_text": response["text"],
            "question": question,
            "answer": message,
            "type": feedback_type,
            "submitted_at": submission_time,
            "unique_id": unique_id,
        }
    )
    # st.write(st.session_state.feedback)


def render_feedback_section(index, message, title, optional_text, key_prefix, unique_id, feedback_type):
    with st.container(key=f"{key_prefix}-{index}"):
        st.markdown(f"<p style='{css_annotation_title}'>{title}</p>", unsafe_allow_html=True)
        return streamlit_feedback(
            on_submit=lambda response, idx=index, msg=message: handle_feedback(
                response, idx, st.session_state.history, unique_id=unique_id, feedback_type=feedback_type
            ),
            feedback_type="faces",
            optional_text_label=optional_text,
            key=f"{key_prefix}_{index}",
        )


feedback_titles = [
    {
        "title": "Evaluation de la pertinence des documents renvoyés",
        "optional_text": "Pertinence des documents",
        "key_prefix": "feedback-retriever",
        "feedback_type": "retriever",
    },
    {
        "title": "Evaluation de la qualité de la réponse à l'aune du contexte fourni (fond):",
        "optional_text": "Qualité du fond",
        "key_prefix": "feedback-generation",
        "feedback_type": "generation_fond",
    },
    {
        "title": "Evaluation de la qualité de la réponse (style, mise en forme, etc.):",
        "optional_text": "Qualité de la forme",
        "key_prefix": "feedback-generation-mef",
        "feedback_type": "generation_forme",
    },
]
