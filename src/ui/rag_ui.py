from pathlib import Path

import streamlit as st

from src.rag.embeddings import EmbeddingModel
from src.rag.retriever import Retriever
from src.rag.vector_store import FAISSVectorStore


# Project root:
# College_ChatBot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local FAISS index:
# College_ChatBot/data/faiss_index/
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index"


@st.cache_resource
def load_rag_retriever() -> Retriever:
    index_file = FAISS_INDEX_PATH / "index.faiss"
    documents_file = FAISS_INDEX_PATH / "documents.json"

    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_file}"
        )

    if not documents_file.exists():
        raise FileNotFoundError(
            f"FAISS documents file not found: {documents_file}"
        )

    embedding_model = EmbeddingModel()

    vector_store = FAISSVectorStore.load(
        FAISS_INDEX_PATH
    )

    return Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )