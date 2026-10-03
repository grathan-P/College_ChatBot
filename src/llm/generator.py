
"""Grounded answer generation using retrieved college documents."""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.config import LLMConfig
from src.llm.context import format_retrieved_context
from src.llm.model import create_chat_model
from src.llm.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.rag.retriever import RetrievalResult


@dataclass
class AnswerSource:
    """Source information attached to a generated answer."""

    source: str
    page: int | None
    academic_year: str | None
    document_type: str | None
    document_status: str | None


@dataclass
class GeneratedAnswer:
    """Stable output returned by the LLM module."""

    answer: str
    sources: list[AnswerSource]


def _build_sources(
    retrieval_results: list[RetrievalResult],
) -> list[AnswerSource]:
    """Build trustworthy source information from retrieval results."""

    return [
        AnswerSource(
            source=result.source,
            page=result.page,
            academic_year=result.academic_year,
            document_type=result.document_type,
            document_status=result.document_status,
        )
        for result in retrieval_results
    ]


def generate_answer(
    question: str,
    retrieval_results: list[RetrievalResult],
    config: LLMConfig | None = None,
) -> GeneratedAnswer:
    """Generate a grounded answer from retrieved college documents."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not retrieval_results:
        return GeneratedAnswer(
            answer=(
                "The information is not available in the provided "
                "college documents."
            ),
            sources=[],
        )

    llm_config = config or LLMConfig.from_environment()

    context = format_retrieved_context(retrieval_results)
    user_prompt = build_rag_prompt(question, context)

    model = create_chat_model(llm_config)

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
  answer_text = str(response.content).strip()

if not answer_text:
    raise RuntimeError(
        "The LLM returned an empty response. Please try again."
    )
    return GeneratedAnswer(
        answer=str(response.content).strip(),
        sources=_build_sources(retrieval_results),
    )
