from datetime import datetime

import streamlit as st
from streamlit_feedback import streamlit_feedback

css_annotation_title = "text-align: right; font-weight: bold; font-style: italic;"


def handle_feedback(response, index, history, unique_id, db_collection, feedback_type="retriever"):
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
            "collection_used": db_collection
        }
    )
    # st.write(st.session_state.feedback)


def render_feedback_section(index, message, title, optional_text, key_prefix, unique_id, db_collection, feedback_type):
    with st.container(key=f"{key_prefix}-{index}"):
        st.markdown(f"<p style='{css_annotation_title}'>{title}</p>", unsafe_allow_html=True)
        return streamlit_feedback(
            on_submit=lambda response, idx=index, msg=message: handle_feedback(
                response, idx, st.session_state.history, unique_id=unique_id,
                db_collection=db_collection, feedback_type=feedback_type
            ),
            feedback_type="thumbs",
            optional_text_label=optional_text,
            key=f"{key_prefix}_{index}",
        )


feedback_titles = [
    {
        "title": "Evaluation de la pertinence des documents renvoyés",
        "optional_text": "Les sources renvoyées sont-elles pertinentes ? Des sources plus pertinentes auraient-elles pu être citées?",
        "key_prefix": "feedback-retriever",
        "feedback_type": "retriever",
    },
    {
        "title": "Qualité de la réponse sur le fond et la forme",
        "optional_text": "Les critères d'évaluation sont nombreux (structure de la réponse, mise en forme, etc.), n'hésitez pas à préciser les dimensions satisfaisantes comme insatisfaisantes. Si une réponse plus pertinente aurait pu être faite, n'hésitez pas à la proposer.",
        "key_prefix": "feedback-generation",
        "feedback_type": "generation_fond",
    }
]
