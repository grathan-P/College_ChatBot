
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


# ---------------------------------------------------------------------------
# Study Planner tests
# ---------------------------------------------------------------------------

from src.llm.planner import (
    StudyDay,
    StudyPlan,
    StudyPlanRequest,
    StudySession,
    generate_study_plan,
    modify_study_plan,
    parse_study_plan_response,
    validate_generated_study_plan,
    validate_study_plan_request,
)


def make_study_plan_request() -> StudyPlanRequest:
    """Create reusable valid study-plan constraints."""
    return StudyPlanRequest(
        subjects=["Mathematics", "DBMS"],
        exam_dates={
            "Mathematics": "2026-11-10",
            "DBMS": "2026-11-14",
        },
        available_hours_per_day=3.0,
        start_date="2026-10-05",
        weak_subjects=["Mathematics"],
        preferred_study_times=["evening"],
    )


def make_study_plan() -> StudyPlan:
    """Create a reusable valid study plan."""
    return StudyPlan(
        summary="Prioritize Mathematics while maintaining DBMS revision.",
        days=[
            StudyDay(
                date="2026-10-05",
                sessions=[
                    StudySession(
                        subject="Mathematics",
                        duration_minutes=120,
                        topic_or_goal="Review calculus fundamentals",
                    ),
                    StudySession(
                        subject="DBMS",
                        duration_minutes=60,
                        topic_or_goal="Review normalization",
                    ),
                ],
            )
        ],
        recommendations=[
            "Review weak Mathematics topics regularly."
        ],
    )


def test_study_plan_request_validation():
    request = make_study_plan_request()

    validate_study_plan_request(request)


def test_study_plan_request_rejects_invalid_exam_date():
    request = StudyPlanRequest(
        subjects=["Mathematics"],
        exam_dates={
            "Mathematics": "10-11-2026",
        },
        available_hours_per_day=3.0,
        start_date="2026-10-05",
    )

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_study_plan_request(request)


def test_parse_study_plan_response():
    response = """
    {
      "summary": "Prioritize Mathematics.",
      "days": [
        {
          "date": "2026-10-05",
          "sessions": [
            {
              "subject": "Mathematics",
              "duration_minutes": 120,
              "topic_or_goal": "Review calculus",
              "notes": null
            }
          ]
        }
      ],
      "recommendations": [
        "Review weak topics regularly."
      ]
    }
    """

    plan = parse_study_plan_response(response)

    assert plan.summary == "Prioritize Mathematics."
    assert len(plan.days) == 1
    assert plan.days[0].date == "2026-10-05"
    assert plan.days[0].sessions[0].subject == "Mathematics"
    assert plan.days[0].sessions[0].duration_minutes == 120


def test_parse_study_plan_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_study_plan_response("This is not JSON.")


def test_generated_plan_rejects_daily_limit_violation():
    request = make_study_plan_request()

    invalid_plan = StudyPlan(
        summary="Invalid plan.",
        days=[
            StudyDay(
                date="2026-10-05",
                sessions=[
                    StudySession(
                        subject="Mathematics",
                        duration_minutes=200,
                        topic_or_goal="Review calculus",
                    )
                ],
            )
        ],
        recommendations=[],
    )

    with pytest.raises(ValueError, match="daily study-time limit"):
        validate_generated_study_plan(
            invalid_plan,
            request,
        )


@patch("src.llm.planner.create_chat_model")
def test_generate_study_plan(mock_create_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(
        content="""
        {
          "summary": "Prioritize Mathematics while revising DBMS.",
          "days": [
            {
              "date": "2026-10-05",
              "sessions": [
                {
                  "subject": "Mathematics",
                  "duration_minutes": 120,
                  "topic_or_goal": "Review calculus",
                  "notes": null
                },
                {
                  "subject": "DBMS",
                  "duration_minutes": 60,
                  "topic_or_goal": "Review normalization",
                  "notes": null
                }
              ]
            }
          ],
          "recommendations": [
            "Review Mathematics weak areas."
          ]
        }
        """
    )
    mock_create_model.return_value = mock_model

    plan = generate_study_plan(
        make_study_plan_request()
    )

    assert isinstance(plan, StudyPlan)
    assert len(plan.days) == 1
    assert len(plan.days[0].sessions) == 2
    assert plan.days[0].sessions[0].subject == "Mathematics"


@patch("src.llm.planner.create_chat_model")
def test_modify_study_plan(mock_create_model):
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(
        content="""
        {
          "summary": "DBMS was moved to October 6.",
          "days": [
            {
              "date": "2026-10-05",
              "sessions": [
                {
                  "subject": "Mathematics",
                  "duration_minutes": 120,
                  "topic_or_goal": "Review calculus",
                  "notes": null
                }
              ]
            },
            {
              "date": "2026-10-06",
              "sessions": [
                {
                  "subject": "DBMS",
                  "duration_minutes": 60,
                  "topic_or_goal": "Review normalization",
                  "notes": null
                }
              ]
            }
          ],
          "recommendations": [
            "Continue regular revision."
          ]
        }
        """
    )
    mock_create_model.return_value = mock_model

    modified_plan = modify_study_plan(
        current_plan=make_study_plan(),
        modification_request=(
            "Move DBMS from October 5 to October 6."
        ),
        request=make_study_plan_request(),
    )

    assert isinstance(modified_plan, StudyPlan)
    assert len(modified_plan.days) == 2
    assert modified_plan.days[1].date == "2026-10-06"
    assert modified_plan.days[1].sessions[0].subject == "DBMS"


@patch("src.llm.planner.create_chat_model")
def test_empty_study_plan_llm_response_raises_error(
    mock_create_model,
):
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="")
    mock_create_model.return_value = mock_model

    with pytest.raises(
        RuntimeError,
        match="study-plan LLM returned an empty response",
    ):
        generate_study_plan(
            make_study_plan_request()
        )


def test_empty_modification_request_raises_error():
    with pytest.raises(
        ValueError,
        match="Modification request cannot be empty",
    ):
        modify_study_plan(
            current_plan=make_study_plan(),
            modification_request="   ",
            request=make_study_plan_request(),
        )
