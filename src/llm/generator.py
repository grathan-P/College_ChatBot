
"""Grounded answer generation using retrieved college documents."""

from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from src.llm.config import LLMConfig
from src.llm.context import format_retrieved_context
from src.llm.memory import ConversationMemory
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
    """Build unique source information from retrieval results."""

    sources: list[AnswerSource] = []
    seen_sources: set[tuple] = set()

    for result in retrieval_results:
        source_key = (
            result.source,
            result.page,
            result.academic_year,
            result.document_type,
            result.document_status,
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            AnswerSource(
                source=result.source,
                page=result.page,
                academic_year=result.academic_year,
                document_type=result.document_type,
                document_status=result.document_status,
            )
        )

    return sources


def generate_answer(
    question: str,
    retrieval_results: list[RetrievalResult],
    config: LLMConfig | None = None,
    memory: ConversationMemory | None = None,
) -> GeneratedAnswer:
    """Generate a grounded answer from retrieved college documents."""

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not retrieval_results:
        answer_text = (
            "The information is not available in the provided "
            "college documents."
        )

        if memory is not None:
            memory.add_user_message(question)
            memory.add_assistant_message(answer_text)

        return GeneratedAnswer(
            answer=answer_text,
            sources=[],
        )

    llm_config = config or LLMConfig.from_environment()

    context = format_retrieved_context(retrieval_results)
    user_prompt = build_rag_prompt(question, context)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    if memory is not None:
        for role, content in memory.get_messages():
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_prompt))

    model = create_chat_model(llm_config)

    response = model.invoke(messages)

    answer_text = str(response.content).strip()

    if not answer_text:
        raise RuntimeError(
            "The LLM returned an empty response. Please try again."
        )

    if memory is not None:
        memory.add_user_message(question)
        memory.add_assistant_message(answer_text)

    return GeneratedAnswer(
        answer=answer_text,
        sources=_build_sources(retrieval_results),
    )
