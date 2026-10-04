def route_request(question: str) -> str:
    """
    Determine the type of user request.

    Returns:
        "academic" for academic questions.
        "study_plan_create" for new study-plan requests.
        "study_plan_modify" for requests to modify an existing plan.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    normalized_question = question.lower().strip()

    modification_keywords = [
        "modify my study plan",
        "modify the study plan",
        "change my study plan",
        "change the study plan",
        "update my study plan",
        "update the study plan",
        "edit my study plan",
        "edit the study plan",
        "adjust my study plan",
        "adjust the study plan",
        "change my schedule",
        "update my schedule",
        "modify my schedule",
        "give more time to",
        "give less time to",
        "add more time",
        "reduce time",
    ]

    if any(
        keyword in normalized_question
        for keyword in modification_keywords
    ):
        return "study_plan_modify"

    creation_keywords = [
        "study plan",
        "study schedule",
        "study timetable",
        "study planner",
        "prepare for exam",
        "exam preparation",
        "make a plan",
        "create a plan",
        "create a study plan",
        "make a study plan",
    ]

    if any(
        keyword in normalized_question
        for keyword in creation_keywords
    ):
        return "study_plan_create"

    return "academic"