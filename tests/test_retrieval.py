
from dataclasses import dataclass

from langchain_core.documents import Document

from src.rag.embeddings import EmbeddingModel
from src.rag.retriever import Retriever
from src.rag.vector_store import FAISSVectorStore


@dataclass
class RetrievalTestCase:
    question: str
    expected_source: str
    expected_keywords: list[str]


def build_test_retriever() -> Retriever:
    """
    Build a small in-memory retriever for evaluation tests.
    """

    documents = [
        Document(
            page_content=(
                "Students must complete the required internship "
                "during the prescribed semester. Students must "
                "satisfy all internship eligibility requirements."
            ),
            metadata={
                "source": "internship_guidelines.pdf",
                "page": 5,
            },
        ),
        Document(
            page_content=(
                "Students must satisfy the required attendance "
                "percentage for each course according to the "
                "academic regulations."
            ),
            metadata={
                "source": "academic_regulations.pdf",
                "page": 12,
            },
        ),
        Document(
            page_content=(
                "The college library provides students with "
                "academic books, journals and digital resources."
            ),
            metadata={
                "source": "student_facilities.pdf",
                "page": 3,
            },
        ),
        Document(
            page_content=(
                "Students should follow the examination "
                "registration procedure before the specified "
                "deadline."
            ),
            metadata={
                "source": "examination_guidelines.pdf",
                "page": 8,
            },
        ),
    ]

    embedding_model = EmbeddingModel()

    texts = [
        document.page_content
        for document in documents
    ]

    embeddings = embedding_model.embed_documents(texts)

    vector_store = FAISSVectorStore(
        dimension=embedding_model.dimension
    )

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
    )

    return Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


def test_internship_retrieval() -> None:
    retriever = build_test_retriever()

    results = retriever.retrieve(
        "What are the internship eligibility requirements?",
        top_k=3,
    )

    assert len(results) > 0

    assert results[0].source == "internship_guidelines.pdf"


def test_attendance_retrieval() -> None:
    retriever = build_test_retriever()

    results = retriever.retrieve(
        "What are the attendance requirements?",
        top_k=3,
    )

    assert len(results) > 0

    sources = [
        result.source
        for result in results
    ]

    assert "academic_regulations.pdf" in sources


def test_examination_retrieval() -> None:
    retriever = build_test_retriever()

    results = retriever.retrieve(
        "How do students register for examinations?",
        top_k=3,
    )

    assert len(results) > 0

    sources = [
        result.source
        for result in results
    ]

    assert "examination_guidelines.pdf" in sources


def test_empty_query_is_rejected() -> None:
    retriever = build_test_retriever()

    try:
        retriever.retrieve("")
        assert False, "Expected ValueError"
    except ValueError:
        pass
