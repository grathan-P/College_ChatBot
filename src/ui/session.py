import streamlit as st

from src.llm.memory import ConversationMemory


def initialize_session() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(
            max_turns=5
        )

    if "current_plan" not in st.session_state:
        st.session_state.current_plan = None

    if "messages" not in st.session_state:
        st.session_state.messages = []