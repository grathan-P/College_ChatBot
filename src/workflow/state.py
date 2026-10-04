from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    # User's current message
    question: str

    # Determined by the router
    intent: str

    # Retrieved RAG documents
    retrieved_documents: list[Any]

    # Final academic answer
    answer: str

    # Sources returned by the LLM
    sources: list[Any]

    # Conversation memory object
    memory: Any

    # Study-plan request
    study_plan_request: Any

    # Current study plan, if one exists
    current_plan: Any

    # Newly generated/modified study plan
    study_plan: Any

    # Error information
    error: str | None