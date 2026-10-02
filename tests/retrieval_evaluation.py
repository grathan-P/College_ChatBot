
from dataclasses import dataclass

from langchain_core.documents import Document

from src.rag.embeddings import EmbeddingModel
from src.rag.retriever import Retriever
from src.rag.vector_store import FAISSVectorStore


@dataclass
class EvaluationCase:
    question: str
    expected_source: str
    expected_keywords: list[str]


def build_retriever() -> Retriever:
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

    embeddings = embedding_model.embed_documents(
        [document.page_content for document in documents]
    )

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


def evaluate(
    retriever: Retriever,
    test_cases: list[EvaluationCase],
    k_values: list[int],
) -> None:

    total = len(test_cases)

    for k in k_values:
        hits = 0

        for case in test_cases:
            results = retriever.retrieve(
                case.question,
                top_k=k,
            )

            sources = [
                result.source
                for result in results
            ]

            if case.expected_source in sources:
                hits += 1

        hit_rate = hits / total

        print(
            f"Hit@{k}: "
            f"{hits}/{total} "
            f"({hit_rate:.2%})"
        )


if __name__ == "__main__":

    test_cases = [
        EvaluationCase(
            question=(
                "What are the internship eligibility requirements?"
            ),
            expected_source="internship_guidelines.pdf",
            expected_keywords=[
                "internship",
                "eligibility",
            ],
        ),
        EvaluationCase(
            question=(
                "What are the attendance requirements?"
            ),
            expected_source="academic_regulations.pdf",
            expected_keywords=[
                "attendance",
            ],
        ),
        EvaluationCase(
            question=(
                "How do students register for examinations?"
            ),
            expected_source="examination_guidelines.pdf",
            expected_keywords=[
                "examination",
                "registration",
            ],
        ),
        EvaluationCase(
            question=(
                "Where can students access academic resources?"
            ),
            expected_source="student_facilities.pdf",
            expected_keywords=[
                "library",
                "resources",
            ],
        ),
    ]

    retriever = build_retriever()

    evaluate(
        retriever=retriever,
        test_cases=test_cases,
        k_values=[1, 3, 5],
    )
