import streamlit as st

from src.rag.retriever import Retriever
from src.workflow.graph import build_workflow


@st.cache_resource
def load_workflow(_retriever: Retriever):
    return build_workflow(_retriever)