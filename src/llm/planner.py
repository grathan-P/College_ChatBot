"""LLM-side study plan generation and modification."""

from dataclasses import asdict, dataclass, field
import json

from langchain_core.messages import HumanMessage, SystemMessage
from datetime import date

from src.llm.config import LLMConfig
from src.llm.model import create_chat_model


@dataclass
class StudyPlanRequest:
    """Student constraints used to generate a personalized study plan."""

    subjects: list[str]
    exam_dates: dict[str, str]
    available_hours_per_day: float
    start_date: str
    weak_subjects: list[str] = field(default_factory=list)
    preferred_study_times: list[str] = field(default_factory=list)
    additional_instructions: str | None = None


@dataclass
class StudySession:
    """One study session inside a study day."""

    subject: str
    duration_minutes: int
    topic_or_goal: str
    notes: str | None = None


@dataclass
class StudyDay:
    """Study sessions scheduled for one date."""

    date: str
    sessions: list[StudySession]


@dataclass
class StudyPlan:
    """Structured study plan returned by the LLM module."""

    summary: str
    days: list[StudyDay]
    recommendations: list[str]


def _parse_iso_date(value: str, field_name: str) -> date:
    """Parse a YYYY-MM-DD date or raise a clear validation error."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format."
        ) from exc


def validate_study_plan_request(request: StudyPlanRequest) -> None:
    """Validate required study-plan inputs."""

    if not request.subjects:
        raise ValueError("At least one subject is required.")

    if any(not subject.strip() for subject in request.subjects):
        raise ValueError("Subject names cannot be empty.")

    if len(set(request.subjects)) != len(request.subjects):
        raise ValueError("Subjects cannot contain duplicates.")

    if request.available_hours_per_day <= 0:
        raise ValueError(
            "Available study hours per day must be greater than zero."
        )

    if not request.start_date.strip():
        raise ValueError("Start date is required.")

    start_date = _parse_iso_date(
        request.start_date,
        "Start date",
    )

    unknown_exam_subjects = set(request.exam_dates) - set(request.subjects)

    if unknown_exam_subjects:
        raise ValueError(
            "Exam-date subjects must also be present in the subjects list."
        )

    unknown_weak_subjects = set(request.weak_subjects) - set(request.subjects)

    if unknown_weak_subjects:
        raise ValueError(
            "Weak subjects must also be present in the subjects list."
        )

    for subject, exam_date_value in request.exam_dates.items():
        exam_date = _parse_iso_date(
            exam_date_value,
            f"Exam date for {subject}",
        )

        if exam_date < start_date:
            raise ValueError(
                f"Exam date for {subject} cannot be before the start date."
            )



def parse_study_plan_response(response_text: str) -> StudyPlan:
    """Convert an LLM JSON response into a structured StudyPlan.

    Raises:
        ValueError: If the response is empty, invalid JSON, or does not
            contain the required study-plan structure.
    """
    response_text = response_text.strip()

    if not response_text:
        raise ValueError("Study-plan LLM response cannot be empty.")

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Study-plan LLM response is not valid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Study-plan LLM response must be a JSON object."
        )

    summary = data.get("summary")
    days_data = data.get("days")
    recommendations = data.get("recommendations")

    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(
            "Study plan must contain a non-empty summary."
        )

    if not isinstance(days_data, list):
        raise ValueError(
            "Study plan must contain a days list."
        )

    if not isinstance(recommendations, list) or not all(
        isinstance(item, str) for item in recommendations
    ):
        raise ValueError(
            "Study plan recommendations must be a list of strings."
        )

    days: list[StudyDay] = []

    for day_data in days_data:
        if not isinstance(day_data, dict):
            raise ValueError(
                "Each study day must be a JSON object."
            )

        day_date = day_data.get("date")
        sessions_data = day_data.get("sessions")

        if not isinstance(day_date, str):
            raise ValueError(
                "Each study day must contain a date."
            )

        _parse_iso_date(day_date, "Study day date")

        if not isinstance(sessions_data, list):
            raise ValueError(
                "Each study day must contain a sessions list."
            )

        sessions: list[StudySession] = []

        for session_data in sessions_data:
            if not isinstance(session_data, dict):
                raise ValueError(
                    "Each study session must be a JSON object."
                )

            subject = session_data.get("subject")
            duration_minutes = session_data.get("duration_minutes")
            topic_or_goal = session_data.get("topic_or_goal")
            notes = session_data.get("notes")

            if not isinstance(subject, str) or not subject.strip():
                raise ValueError(
                    "Each study session must contain a subject."
                )

            if (
                not isinstance(duration_minutes, int)
                or isinstance(duration_minutes, bool)
                or duration_minutes <= 0
            ):
                raise ValueError(
                    "Study-session duration must be a positive integer."
                )

            if (
                not isinstance(topic_or_goal, str)
                or not topic_or_goal.strip()
            ):
                raise ValueError(
                    "Each study session must contain a topic or goal."
                )

            if notes is not None and not isinstance(notes, str):
                raise ValueError(
                    "Study-session notes must be a string or null."
                )

            sessions.append(
                StudySession(
                    subject=subject,
                    duration_minutes=duration_minutes,
                    topic_or_goal=topic_or_goal,
                    notes=notes,
                )
            )

        days.append(
            StudyDay(
                date=day_date,
                sessions=sessions,
            )
        )

    return StudyPlan(
        summary=summary,
        days=days,
        recommendations=recommendations,
    )


def validate_generated_study_plan(
    plan: StudyPlan,
    request: StudyPlanRequest,
) -> None:
    """Validate that a generated plan respects student constraints.

    Raises:
        ValueError: If the generated plan violates the request.
    """
    allowed_subjects = set(request.subjects)
    start_date = _parse_iso_date(request.start_date, "Start date")
    daily_limit_minutes = round(request.available_hours_per_day * 60)

    parsed_exam_dates = {
        subject: _parse_iso_date(
            exam_date,
            f"Exam date for {subject}",
        )
        for subject, exam_date in request.exam_dates.items()
    }

    seen_dates: set[str] = set()

    for day in plan.days:
        day_date = _parse_iso_date(day.date, "Study day date")

        if day_date < start_date:
            raise ValueError(
                "Generated study plan contains a day before the start date."
            )

        if day.date in seen_dates:
            raise ValueError(
                f"Generated study plan contains duplicate date: {day.date}."
            )

        seen_dates.add(day.date)

        total_minutes = 0

        for session in day.sessions:
            if session.subject not in allowed_subjects:
                raise ValueError(
                    f"Generated study plan contains unknown subject: "
                    f"{session.subject}."
                )

            exam_date = parsed_exam_dates.get(session.subject)

            if exam_date is not None and day_date > exam_date:
                raise ValueError(
                    f"{session.subject} was scheduled after its exam date."
                )

            total_minutes += session.duration_minutes

        if total_minutes > daily_limit_minutes:
            raise ValueError(
                f"Generated study plan exceeds the daily study-time limit "
                f"on {day.date}."
            )

def generate_study_plan(
    request: StudyPlanRequest,
    config: LLMConfig | None = None,
) -> StudyPlan:
    """Generate a structured personalized study plan using the LLM."""

    validate_study_plan_request(request)

    # Imported here to avoid a circular import:
    # planner_prompts imports StudyPlanRequest from this module.
    from src.llm.planner_prompts import (
        STUDY_PLANNER_SYSTEM_PROMPT,
        build_study_plan_prompt,
    )

    llm_config = config or LLMConfig.from_environment()

    user_prompt = build_study_plan_prompt(request)

    model = create_chat_model(llm_config)

    response = model.invoke(
        [
            SystemMessage(content=STUDY_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    response_text = str(response.content).strip()

    if not response_text:
        raise RuntimeError(
            "The study-plan LLM returned an empty response."
        )

    plan = parse_study_plan_response(response_text)

    validate_generated_study_plan(
        plan=plan,
        request=request,
    )

    return plan


def modify_study_plan(
    current_plan: StudyPlan,
    modification_request: str,
    request: StudyPlanRequest,
    config: LLMConfig | None = None,
) -> StudyPlan:
    """Modify an existing study plan while preserving its constraints."""

    validate_study_plan_request(request)
    validate_generated_study_plan(current_plan, request)

    modification_request = modification_request.strip()

    if not modification_request:
        raise ValueError("Modification request cannot be empty.")

    from src.llm.planner_prompts import (
        STUDY_PLAN_MODIFICATION_SYSTEM_PROMPT,
        build_study_plan_modification_prompt,
    )

    llm_config = config or LLMConfig.from_environment()

    user_prompt = build_study_plan_modification_prompt(
        request=request,
        current_plan=asdict(current_plan),
        modification_request=modification_request,
    )

    model = create_chat_model(llm_config)

    response = model.invoke(
        [
            SystemMessage(
                content=STUDY_PLAN_MODIFICATION_SYSTEM_PROMPT
            ),
            HumanMessage(content=user_prompt),
        ]
    )

    response_text = str(response.content).strip()

    if not response_text:
        raise RuntimeError(
            "The study-plan modification LLM returned an empty response."
        )

    modified_plan = parse_study_plan_response(response_text)

    validate_generated_study_plan(
        plan=modified_plan,
        request=request,
    )

    return modified_plan
