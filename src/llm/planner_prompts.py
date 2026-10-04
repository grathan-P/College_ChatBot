"""Prompt construction for personalized study-plan generation."""

import json

from src.llm.planner import StudyPlanRequest


STUDY_PLANNER_SYSTEM_PROMPT = """
You are a study-planning assistant for college students.

Create practical personalized study plans using only the student constraints
provided in the request.

Rules:
1. Respect the supplied subjects, exam dates, start date, and available
   study hours.
2. Never invent or change an exam date.
3. Do not schedule study sessions after that subject's exam date.
4. Give reasonable additional attention to subjects marked as weak.
5. Keep the total planned study time for each day within the student's
   available study hours.
6. Treat the generated plan as a recommendation, not an official college
   timetable.
7. Do not invent official college rules, schedules, deadlines, or policies.
8. Return valid JSON only.
9. Do not wrap the JSON in Markdown code fences.
"""


def build_study_plan_prompt(request: StudyPlanRequest) -> str:
    """Build the user prompt for structured study-plan generation."""

    request_data = {
        "subjects": request.subjects,
        "exam_dates": request.exam_dates,
        "available_hours_per_day": request.available_hours_per_day,
        "start_date": request.start_date,
        "weak_subjects": request.weak_subjects,
        "preferred_study_times": request.preferred_study_times,
        "additional_instructions": request.additional_instructions,
    }

    output_schema = {
        "summary": "Short explanation of the study-plan strategy.",
        "days": [
            {
                "date": "YYYY-MM-DD",
                "sessions": [
                    {
                        "subject": "Subject name from the request",
                        "duration_minutes": 60,
                        "topic_or_goal": "Study goal for this session",
                        "notes": "Optional note or null",
                    }
                ],
            }
        ],
        "recommendations": [
            "Short practical recommendation"
        ],
    }

    return (
        "STUDENT STUDY CONSTRAINTS:\n"
        f"{json.dumps(request_data, indent=2)}\n\n"
        "REQUIRED OUTPUT JSON STRUCTURE:\n"
        f"{json.dumps(output_schema, indent=2)}\n\n"
        "Generate a personalized study plan that follows all constraints. "
        "Return JSON only."
    )


STUDY_PLAN_MODIFICATION_SYSTEM_PROMPT = """
You are a study-planning assistant modifying an existing student study plan.

Rules:
1. Apply the student's requested modification where possible.
2. Continue respecting the original subjects, exam dates, start date,
   and available study hours.
3. Never invent or change an exam date.
4. Never add subjects that are not in the original request.
5. Do not schedule study sessions after that subject's exam date.
6. Keep each day's total study time within the student's available hours.
7. Preserve reasonable parts of the existing plan that do not need changing.
8. Treat the plan as a recommendation, not an official college timetable.
9. Return valid JSON only.
10. Do not wrap the JSON in Markdown code fences.
"""


def build_study_plan_modification_prompt(
    request: StudyPlanRequest,
    current_plan: dict,
    modification_request: str,
) -> str:
    """Build the prompt for modifying an existing study plan."""

    request_data = {
        "subjects": request.subjects,
        "exam_dates": request.exam_dates,
        "available_hours_per_day": request.available_hours_per_day,
        "start_date": request.start_date,
        "weak_subjects": request.weak_subjects,
        "preferred_study_times": request.preferred_study_times,
        "additional_instructions": request.additional_instructions,
    }

    return (
        "ORIGINAL STUDENT CONSTRAINTS:\n"
        f"{json.dumps(request_data, indent=2)}\n\n"
        "CURRENT STUDY PLAN:\n"
        f"{json.dumps(current_plan, indent=2)}\n\n"
        "STUDENT MODIFICATION REQUEST:\n"
        f"{modification_request}\n\n"
        "Return the complete modified study plan using exactly the same "
        "JSON structure as the current plan. Return JSON only."
    )
