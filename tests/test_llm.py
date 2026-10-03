
"""Tests for the Person 2 LLM module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.llm.context import format_retrieved_context
from src.llm.generator import GeneratedAnswer, generate_answer
from src.llm.memory import ConversationMemory
from src.rag.retriever import RetrievalResult


def make_retrieval_result() -> RetrievalResult:
    """Create reusable mock retrieval evidence."""
    return RetrievalResult(
        content="Mock programme requirement: 165 credits.",
        source="Mock Regulations.pdf",
        page=22,
        score=0.90,
        metadata={
            "academic_year": "2026-27",
            "document_type": "regulations",
            "document_status": "current",
        },
    )


def test_context_formatting():
    result = make_retrieval_result()

    context = format_retrieved_context([result])

    assert "Mock Regulations.pdf" in context
    assert "Page: 22" in context
    assert "Academic Year: 2026-27" in context
    assert "Document Type: regulations" in context
    assert "Document Status: current" in context
    assert "165 credits" in context


def test_context_empty_results():
    assert format_retrieved_context([]) == ""


def test_generate_answer_without_results():
    generated = generate_answer(
        question="What is the requirement?",
        retrieval_results=[],
    )

    assert isinstance(generated, GeneratedAnswer)
    assert "not available" in generated.answer.lower()
    assert generated.sources == []


def test_empty_question_raises_error():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_answer(
            question="   ",
            retrieval_results=[],
        )


@patch("src.llm.generator.create_chat_model")
def test_generated_answer_contains_source_metadata(mock_create_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(
        content="The programme requires 165 credits."
    )
    mock_create_model.return_value = mock_model

    generated = generate_answer(
        question="How many credits are required?",
        retrieval_results=[make_retrieval_result()],
    )

    assert generated.answer == "The programme requires 165 credits."
    assert len(generated.sources) == 1

    source = generated.sources[0]

    assert source.source == "Mock Regulations.pdf"
    assert source.page == 22
    assert source.academic_year == "2026-27"
    assert source.document_type == "regulations"
    assert source.document_status == "current"


def test_memory_trimming():
    memory = ConversationMemory(max_turns=2)

    memory.add_user_message("Question 1")
    memory.add_assistant_message("Answer 1")

    memory.add_user_message("Question 2")
    memory.add_assistant_message("Answer 2")

    memory.add_user_message("Question 3")
    memory.add_assistant_message("Answer 3")

    messages = memory.get_messages()

    assert len(messages) == 4
    assert messages[0] == ("user", "Question 2")
    assert messages[-1] == ("assistant", "Answer 3")


@patch("src.llm.generator.create_chat_model")
def test_generator_updates_memory(mock_create_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(
        content="The programme requires 165 credits."
    )
    mock_create_model.return_value = mock_model

    memory = ConversationMemory()

    generate_answer(
        question="How many credits are required?",
        retrieval_results=[make_retrieval_result()],
        memory=memory,
    )

    messages = memory.get_messages()

    assert messages == [
        ("user", "How many credits are required?"),
        ("assistant", "The programme requires 165 credits."),
    ]


@patch("src.llm.generator.create_chat_model")
def test_empty_llm_response_raises_error(mock_create_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="")
    mock_create_model.return_value = mock_model

    with pytest.raises(
        RuntimeError,
        match="The LLM returned an empty response",
    ):
        generate_answer(
            question="How many credits are required?",
            retrieval_results=[make_retrieval_result()],
        )
